package main

import (
	"context"
	"decentralized-api/apiconfig"
	"decentralized-api/broker"
	"decentralized-api/chainphase"
	"decentralized-api/cosmosclient"
	"decentralized-api/internal"
	"decentralized-api/internal/bls"
	"decentralized-api/internal/event_listener"
	"decentralized-api/internal/modelmanager"
	"decentralized-api/internal/nats/server"
	adminserver "decentralized-api/internal/server/admin"
	mlserver "decentralized-api/internal/server/mlnode"
	pserver "decentralized-api/internal/server/public"
	"decentralized-api/mlnodeclient"
	"decentralized-api/payloadstorage"
	"decentralized-api/poc"
	"decentralized-api/poc/artifacts"
	"decentralized-api/poc/propagation"
	"net"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/productscience/inference/api/inference/inference"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"

	"decentralized-api/internal/validation"
	"decentralized-api/logging"
	"decentralized-api/participant"
	"decentralized-api/training"
	"encoding/json"
	"fmt"
	"log"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/productscience/inference/x/inference/types"
)

type cosmosHeaderSigner struct {
	client cosmosclient.CosmosMessageClient
}

func (s *cosmosHeaderSigner) Sign(msg []byte) ([]byte, error) {
	return s.client.SignBytes(msg)
}

type chainPubKeyProvider struct {
	queryClient types.QueryClient
}

func (p *chainPubKeyProvider) GetPubKey(participantAddr string) (string, error) {
	resp, err := p.queryClient.Participant(context.Background(), &types.QueryGetParticipantRequest{Index: participantAddr})
	if err != nil {
		return "", fmt.Errorf("query participant %s: %w", participantAddr, err)
	}
	if resp.Participant.WorkerPublicKey == "" {
		return "", fmt.Errorf("public key not found for address %s", participantAddr)
	}
	return resp.Participant.WorkerPublicKey, nil
}

func main() {
	if len(os.Args) >= 2 && os.Args[1] == "status" {
		logging.WithNoopLogger(func() (interface{}, error) {
			config, err := apiconfig.LoadDefaultConfigManager()
			if err != nil {
				log.Fatalf("Error loading config: %v", err)
			}
			returnStatus(config)
			return nil, nil
		})

		return
	}
	if len(os.Args) >= 2 && os.Args[1] == "pre-upgrade" {
		os.Exit(1)
	}

	config, err := apiconfig.LoadDefaultConfigManager()
	if err != nil {
		log.Fatalf("Error loading config: %v", err)
	}

	if config.GetApiConfig().TestMode {
		slog.SetLogLoggerLevel(slog.LevelDebug)
	}

	natssrv := server.NewServer(config.GetNatsConfig())
	if err := natssrv.Start(); err != nil {
		panic(err)
	}

	recorder, err := cosmosclient.NewInferenceCosmosClientWithRetry(
		context.Background(),
		"gonka",
		20,
		5*time.Second,
		config,
	)
	if err != nil {
		panic(err)
	}

	// Version sync is handled later in the event processing loop when blockchain is fully ready
	// This prevents EOF errors during startup from breaking the entire application

	chainPhaseTracker := chainphase.NewChainPhaseTracker()
	// NOTE: getParams is waiting for rpc to be ready, don't add request before it
	params, err := getParams(context.Background(), *recorder)
	if err != nil {
		logging.Error("Failed to get params", types.System, "error", err)
		return
	}
	chainPhaseTracker.UpdateEpochParams(*params.Params.EpochParams)

	participantInfo, err := participant.NewCurrentParticipantInfo(recorder)
	if err != nil {
		logging.Error("Failed to get participant info", types.Participants, "error", err)
		return
	}
	chainBridge := broker.NewBrokerChainBridgeImpl(recorder, config.GetChainNodeConfig().Url)
	nodeBroker := broker.NewBroker(chainBridge, chainPhaseTracker, participantInfo, config.GetApiConfig().PoCCallbackUrl, &mlnodeclient.HttpClientFactory{}, config)

	nodes := config.GetNodes()
	for _, node := range nodes {
		responseChan := nodeBroker.LoadNodeToBroker(&node)
		if responseChan != nil {
			response := <-responseChan
			if response.Error != nil {
				logging.Error("Failed to load node to broker. Skipping", types.Nodes, "node_id", node.Id, "error", response.Error)
			} else if response.Node == nil {
				logging.Error("Failed to load node to broker, response.Node == nil and response.Error == nil. Skipping", types.Nodes, "node_id", node.Id)
			} else {
				logging.Info("Successfully loaded node to broker", types.Nodes, "node_id", response.Node.Id)
			}
		}
	}

	if err := participant.RegisterParticipantIfNeeded(recorder, config); err != nil {
		logging.Error("Failed to register participant", types.Participants, "error", err)
		return
	}

	tendermintClient := cosmosclient.TendermintClient{
		ChainNodeUrl: config.GetChainNodeConfig().Url,
	}
	// Create a cancellable context for the entire system
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel() // Ensure resources are cleaned up

	// Start periodic config auto-flush of dynamic data to DB
	config.StartAutoFlush(ctx, 60*time.Second)

	training.NewAssigner(recorder, &tendermintClient, ctx)
	trainingExecutor := training.NewExecutor(ctx, nodeBroker, recorder)

	validator := validation.NewInferenceValidator(nodeBroker, config, recorder, chainPhaseTracker)
	blsManager := bls.NewBlsManager(*recorder)

	// Shared managed artifact store for off-chain PoC (used by both mlnode and public servers)
	// Manages per-height directories with automatic pruning (retains last 10)
	artifactStore := artifacts.NewManagedArtifactStore("/root/.dapi/data/poc-artifacts", 10)
	defer artifactStore.Close()

	// Initialize propagation infrastructure (if enabled)
	var propagationCache *propagation.Cache
	var propagationBundler *propagation.Bundler
	var propagationTransport *propagation.HTTPTransport
	var propagationReceiver *propagation.Receiver
	var propagationHandlers *pserver.PropagationHandlers
	var propagationPool *pgxpool.Pool
	var treeManager *propagation.TreeManager

	propConfig := config.GetConfig().PocPropagation
	if propConfig.Enabled {
		logging.Info("Initializing off-chain propagation system", types.PoC,
			"trees", propConfig.Trees, "fanout", propConfig.Fanout)

		var bundleStorage propagation.BundleStorage
		bundleStorage, propagationPool = propagation.NewBundleStorage(ctx, propConfig.StorageDir, participantInfo.GetAddress())
		propagationCache = propagation.NewCache(bundleStorage)

		propagationTransport = propagation.NewHTTPTransport(
			participantInfo.GetAddress(),
			30*time.Second,
		)

		treeCount := propConfig.Trees
		if treeCount <= 0 {
			treeCount = 1
		}
		treeFanout := propConfig.Fanout
		if treeFanout <= 0 {
			treeFanout = 1
		}

		epochGroupDataCache := internal.NewEpochGroupDataCache(recorder)
		treeManager = propagation.NewTreeManager(epochGroupDataCache, treeCount, treeFanout)

		bootstrapTrees := []*propagation.Tree{}

		pubKeyProvider := &chainPubKeyProvider{queryClient: recorder.NewInferenceQueryClient()}

		propagationReceiver = propagation.NewReceiver(
			propagationCache,
			bootstrapTrees,
			pubKeyProvider,
			participantInfo.GetAddress(),
			propagationTransport,
		)

		propagationTransport.RegisterReceiver(participantInfo.GetAddress(), propagationReceiver)

		signer := &cosmosHeaderSigner{client: recorder}
		propagationBundler = propagation.NewBundler(
			signer,
			propagationCache,
			bootstrapTrees,
			propagationTransport,
			participantInfo.GetAddress(),
		)

		propagationHandlers = pserver.NewPropagationHandlers(propagationTransport)

		logging.Info("Propagation system initialized with TreeManager", types.PoC,
			"note", "Trees will be rebuilt dynamically based on previous epoch weights")
	} else {
		logging.Info("Off-chain propagation disabled", types.PoC)
	}

	if propagationPool != nil {
		defer propagationPool.Close()
	}

	logging.Debug("Initializing PoC orchestrator",
		types.PoC, "name", recorder.GetApiAccount().SignerAccount.Name,
		"address", participantInfo.GetAddress(),
		"pubkey", participantInfo.GetPubKey())

	pocOrchestrator := poc.NewOrchestrator(
		participantInfo.GetPubKey(),
		nodeBroker,
		config.GetApiConfig().PoCCallbackUrl,
		config.GetChainNodeConfig().Url,
		recorder,
		chainPhaseTracker,
		propagationCache,
	)
	logging.Info("PoC orchestrator initialized", types.PoC)

	listener := event_listener.NewEventListener(config, pocOrchestrator, nodeBroker, validator, *recorder, trainingExecutor, chainPhaseTracker, cancel, blsManager)

	if propConfig.Enabled && treeManager != nil {
		listener.SetPropagationComponents(treeManager, propagationReceiver, propagationBundler, propagationTransport, recorder.NewInferenceQueryClient())
		logging.Info("Propagation components wired to event listener", types.PoC)
	}

	go listener.Start(ctx)

	mlnodeBackgroundManager := modelmanager.NewMLNodeBackgroundManager(
		config,
		chainPhaseTracker,
		nodeBroker,
		&mlnodeclient.HttpClientFactory{},
		30*time.Minute,
	)
	go mlnodeBackgroundManager.Start(ctx)

	addr := fmt.Sprintf(":%v", config.GetApiConfig().PublicServerPort)
	logging.Info("start public server on addr", types.Server, "addr", addr)

	// Bridge external block queue
	blockQueue := pserver.NewBlockQueue(recorder)

	// Shared payload storage for both public and admin servers
	// Uses PostgreSQL if PGHOST is set and accessible, otherwise file-based
	// ManagedStorage provides read caching + automatic epoch pruning (retains last 3 epochs)
	payloadStore := payloadstorage.NewManagedStorage(
		payloadstorage.NewPayloadStorage(ctx, "/root/.dapi/data/inference"),
		3,             // retain current + 2 previous epochs
		3*time.Minute, // cache TTL
	)

	// Create commit worker for time-based artifact commits and weight distribution
	// Worker owns flush lifecycle, commits periodically (not per-request), and handles distribution
	batchingCfg := config.GetTxBatchingConfig()
	commitInterval := time.Duration(batchingCfg.PocCommitIntervalSeconds) * time.Second

	commitWorker := poc.NewCommitWorker(
		artifactStore,
		recorder,
		chainPhaseTracker,
		participantInfo.GetAddress(),
		commitInterval,
		propConfig.Enabled,
		propagationBundler,
	)
	defer commitWorker.Close()

	// Build server options
	serverOpts := []pserver.ServerOption{pserver.WithArtifactStore(artifactStore)}
	if propagationHandlers != nil {
		serverOpts = append(serverOpts, pserver.WithPropagationHandlers(propagationHandlers))
	}

	publicServer := pserver.NewServer(nodeBroker, config, recorder, trainingExecutor, blockQueue, chainPhaseTracker, payloadStore, serverOpts...)
	publicServer.Start(addr)

	addr = fmt.Sprintf(":%v", config.GetApiConfig().MLServerPort)
	logging.Info("start ml server on addr", types.Server, "addr", addr)
	mlServer := mlserver.NewServer(recorder, nodeBroker, mlserver.WithArtifactStore(artifactStore))
	mlServer.Start(addr)

	addr = fmt.Sprintf(":%v", config.GetApiConfig().AdminServerPort)
	logging.Info("start admin server on addr", types.Server, "addr", addr)
	adminServer := adminserver.NewServer(recorder, nodeBroker, config, validator, blockQueue, payloadStore)
	adminServer.Start(addr)

	mlGrpcServerPort := config.GetApiConfig().MlGrpcServerPort
	if mlGrpcServerPort == 0 {
		mlGrpcServerPort = 9300
		logging.Info("ml grpc server port not set, using default port 9300", types.Server)
	}
	addr = fmt.Sprintf(":%v", mlGrpcServerPort)
	logging.Info("start training server on addr", types.Server, "addr", addr)
	grpcServer := grpc.NewServer()
	trainingServer := training.NewServer(recorder, trainingExecutor)
	inference.RegisterNetworkNodeServiceServer(grpcServer, trainingServer)
	reflection.Register(grpcServer)
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	go func() {
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatalf("failed to serve: %v", err)
		}
	}()

	logging.Info("Servers started", types.Server, "addr", addr)

	<-ctx.Done()

	ctxFlush, cancelFlush := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancelFlush()
	logging.Info("Flushing config to the DB on app exit", types.Config)
	_ = config.FlushNow(ctxFlush)

	// Close DB gracefully
	if db := config.SqlDb().GetDb(); db != nil {
		_ = db.Close()
	}

	os.Exit(1) // Exit with an error for cosmovisor to restart the process
}

func returnStatus(config *apiconfig.ConfigManager) {
	height := config.GetHeight()
	status := map[string]interface{}{
		"sync_info": map[string]string{
			"latest_block_height": strconv.FormatInt(height, 10),
		},
	}
	jsonData, err := json.MarshalIndent(status, "", "  ")
	if err != nil {
		panic(err)
	}
	fmt.Println(string(jsonData))
	os.Exit(0)
}

func getParams(ctx context.Context, transactionRecorder cosmosclient.InferenceCosmosClient) (*types.QueryParamsResponse, error) {
	var params *types.QueryParamsResponse
	var err error
	for i := 0; i < 10; i++ {
		params, err = transactionRecorder.NewInferenceQueryClient().Params(ctx, &types.QueryParamsRequest{})
		if err == nil {
			return params, nil
		}

		if strings.HasPrefix(err.Error(), "rpc error: code = Unknown desc = inference is not ready") {
			logging.Info("Inference not ready, retrying...", types.System, "attempt", i+1, "error", err)
			time.Sleep(2 * time.Second) // Try a longer wait for specific inference delays
			continue
		}
		// If not an RPC error, log and return early
		logging.Error("Failed to get chain params", types.System, "error", err)
		return nil, err
	}
	logging.Error("Exhausted all retries to get chain params", types.System, "error", err)
	return nil, err
}
