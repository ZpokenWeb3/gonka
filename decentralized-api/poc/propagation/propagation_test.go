package propagation

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"decentralized-api/poc/artifacts"

	"github.com/cosmos/cosmos-sdk/crypto/keys/secp256k1"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/modules/postgres"
	"github.com/testcontainers/testcontainers-go/wait"
)

func setupPropagationPostgres(t *testing.T) (*pgxpool.Pool, func()) {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping postgres tests in -short mode")
	}
	ctx := context.Background()
	container, err := postgres.Run(ctx,
		"postgres:18.1-bookworm",
		postgres.WithDatabase("testdb"),
		postgres.WithUsername("testuser"),
		postgres.WithPassword("testpass"),
		testcontainers.WithWaitStrategy(
			wait.ForLog("database system is ready to accept connections").
				WithOccurrence(2).
				WithStartupTimeout(60*time.Second),
		),
	)
	require.NoError(t, err)
	connString, err := container.ConnectionString(ctx, "sslmode=disable")
	require.NoError(t, err)
	pool, err := pgxpool.New(ctx, connString)
	require.NoError(t, err)
	cleanup := func() {
		pool.Close()
		container.Terminate(ctx)
	}
	return pool, cleanup
}

type bundleStorageFactory func(t *testing.T, tempDir, addr string) (BundleStorage, error)

func testSmallPropagation(t *testing.T, storageFactory bundleStorageFactory) {
	numParticipants := 5
	numTrees := 1
	fanout := 2

	participants := make([]string, numParticipants)
	privKeys := make(map[string][]byte)
	pubKeys := make(map[string]string)

	for i := 0; i < numParticipants; i++ {
		addr := fmt.Sprintf("participant%d", i)
		participants[i] = addr

		privKey := secp256k1.GenPrivKey()
		privKeys[addr] = privKey.Key
		pubKeys[addr] = hex.EncodeToString(privKey.PubKey().Bytes())
	}

	blockHash := sha256.Sum256([]byte("test-block"))
	pocHeight := int64(1000)

	trees := BuildTrees(participants, blockHash[:], numTrees, fanout)

	transport := NewMockTransport()
	pubKeyProvider := NewMockPubKeyProvider()
	for addr, pubKey := range pubKeys {
		pubKeyProvider.RegisterKey(addr, pubKey)
	}

	tempDir := t.TempDir()

	caches := make(map[string]*Cache)
	receivers := make(map[string]*Receiver)
	bundlers := make(map[string]*Bundler)
	stores := make(map[string]*artifacts.ArtifactStore)

	signers := make(map[string]HeaderSigner)

	for i, addr := range participants {
		storage, err := storageFactory(t, tempDir, addr)
		require.NoError(t, err)
		cache := NewCache(storage)
		caches[addr] = cache

		receiver := NewReceiver(cache, trees, pubKeyProvider, addr, transport)
		receivers[addr] = receiver
		transport.RegisterReceiver(addr, receiver)

		storeDir := filepath.Join(tempDir, addr, "store")
		if err := os.MkdirAll(storeDir, 0755); err != nil {
			t.Fatalf("failed to create store dir for %s: %v", addr, err)
		}
		store, err := artifacts.Open(storeDir)
		if err != nil {
			t.Fatalf("failed to create store for %s: %v", addr, err)
		}
		stores[addr] = store

		for j := 0; j < 10; j++ {
			nonce := int32(i*1000 + j)
			vector := []byte(fmt.Sprintf("vector-%d-%d", i, j))
			if err := store.Add(nonce, vector); err != nil {
				t.Fatalf("failed to add artifact: %v", err)
			}
		}

		if err := store.Flush(); err != nil {
			t.Fatalf("failed to flush store for %s: %v", addr, err)
		}

		signers[addr] = &testKeySigner{key: privKeys[addr]}
		bundler := NewBundler(signers[addr], cache, trees, transport, addr)
		bundlers[addr] = bundler
	}

	sender := trees[0].Shuffled[0]

	senderCount := stores[sender].Count()
	senderRoot := stores[sender].GetRoot()
	if err := bundlers[sender].Publish(pocHeight, blockHash[:], sender, senderCount, senderRoot); err != nil {
		t.Fatalf("failed to publish: %v", err)
	}

	bundleID := MakeBundleID(sender, pocHeight, stores[sender].GetRoot(), stores[sender].Count(), 1)

	receivedCount := 0
	for _, addr := range participants {
		if addr == sender {
			continue
		}

		header, err := caches[addr].GetHeader(bundleID)
		if err != nil {
			continue
		}

		receivedCount++

		if header.Participant != sender {
			t.Errorf("participant %s: wrong sender in header: got %s, want %s",
				addr, header.Participant, sender)
		}
	}

	if receivedCount != numParticipants-1 {
		t.Errorf("Not all participants received the bundle: got %d, want %d", receivedCount, numParticipants-1)
	}

	for _, store := range stores {
		store.Close()
	}
}

func TestSmallPropagation(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping postgres tests in -short mode")
	}

	pool, cleanup := setupPropagationPostgres(t)
	defer cleanup()

	testSmallPropagation(t, func(t *testing.T, tempDir, addr string) (BundleStorage, error) {
		return NewPostgresBundleStorage(context.Background(), pool, addr)
	})
}

func TestSmallPropagationWithFileStorage(t *testing.T) {
	testSmallPropagation(t, func(t *testing.T, tempDir, addr string) (BundleStorage, error) {
		storageDir := filepath.Join(tempDir, addr, "bundles")
		return NewFileBundleStorage(storageDir)
	})
}

func testAllBundlesForHeightAfterPropagation(t *testing.T, storageFactory bundleStorageFactory) {
	numParticipants := 5
	fanout := 2

	participants := make([]string, numParticipants)
	privKeys := make(map[string][]byte)
	pubKeys := make(map[string]string)

	for i := 0; i < numParticipants; i++ {
		addr := fmt.Sprintf("participant%d", i)
		participants[i] = addr

		privKey := secp256k1.GenPrivKey()
		privKeys[addr] = privKey.Key
		pubKeys[addr] = hex.EncodeToString(privKey.PubKey().Bytes())
	}

	blockHash := sha256.Sum256([]byte("test-block"))
	pocHeight := int64(2000)

	trees := make([]*Tree, numParticipants)
	for i := 0; i < numParticipants; i++ {
		rotated := make([]string, numParticipants)
		for j := 0; j < numParticipants; j++ {
			rotated[j] = participants[(i+j)%numParticipants]
		}
		trees[i] = buildTree(i, rotated, fanout)
	}

	transport := NewMockTransport()
	pubKeyProvider := NewMockPubKeyProvider()
	for addr, pubKey := range pubKeys {
		pubKeyProvider.RegisterKey(addr, pubKey)
	}

	tempDir := t.TempDir()

	caches := make(map[string]*Cache)
	bundlers := make(map[string]*Bundler)
	stores := make(map[string]*artifacts.ArtifactStore)

	for i, addr := range participants {
		storage, err := storageFactory(t, tempDir, addr)
		require.NoError(t, err)
		cache := NewCache(storage)
		caches[addr] = cache

		receiver := NewReceiver(cache, trees, pubKeyProvider, addr, transport)
		transport.RegisterReceiver(addr, receiver)

		storeDir := filepath.Join(tempDir, addr, "store")
		require.NoError(t, os.MkdirAll(storeDir, 0755))
		store, err := artifacts.Open(storeDir)
		require.NoError(t, err)
		stores[addr] = store

		for j := 0; j < 10; j++ {
			nonce := int32(i*1000 + j)
			vector := []byte(fmt.Sprintf("vector-%d-%d", i, j))
			require.NoError(t, store.Add(nonce, vector))
		}
		require.NoError(t, store.Flush())

		bundler := NewBundler(&testKeySigner{key: privKeys[addr]}, cache, trees, transport, addr)
		bundlers[addr] = bundler
	}

	type publishedMeta struct {
		participant string
		count       uint32
		rootHash    []byte
	}
	published := make(map[string]publishedMeta, numParticipants)

	for _, addr := range participants {
		count := stores[addr].Count()
		root := stores[addr].GetRoot()
		require.NoError(t, bundlers[addr].Publish(pocHeight, blockHash[:], addr, count, root))
		published[addr] = publishedMeta{
			participant: addr,
			count:       count,
			rootHash:    root,
		}
	}

	for _, addr := range participants {
		bundles := caches[addr].AllBundlesForHeight(pocHeight)

		ownFound := false
		for _, h := range bundles {
			if h.Participant == addr {
				ownFound = true
			}
		}
		require.True(t, ownFound,
			"participant %s: own header missing from AllBundlesForHeight", addr)

		seen := make(map[string]bool)
		for _, h := range bundles {
			require.False(t, seen[h.Participant],
				"participant %s: duplicate header for %s", addr, h.Participant)
			seen[h.Participant] = true

			require.Equal(t, pocHeight, h.PocHeight)

			meta, ok := published[h.Participant]
			require.True(t, ok,
				"participant %s: header from unknown participant %s", addr, h.Participant)
			require.Equal(t, meta.count, h.Count,
				"participant %s: wrong count for %s", addr, h.Participant)
			require.Equal(t, meta.rootHash, h.RootHash,
				"participant %s: wrong root hash for %s", addr, h.Participant)

			expectedID := MakeBundleID(h.Participant, pocHeight, meta.rootHash, meta.count, 1)
			require.Equal(t, expectedID, h.BundleID,
				"participant %s: wrong bundle ID for %s", addr, h.Participant)
		}
	}

	for _, addr := range participants {
		bundles := caches[addr].AllBundlesForHeight(pocHeight)
		require.Len(t, bundles, numParticipants,
			"participant %s: expected %d bundles, got %d", addr, numParticipants, len(bundles))
	}

	for _, store := range stores {
		store.Close()
	}
}

func TestAllBundlesForHeightAfterPropagation(t *testing.T) {
	testAllBundlesForHeightAfterPropagation(t, func(t *testing.T, tempDir, addr string) (BundleStorage, error) {
		storageDir := filepath.Join(tempDir, addr, "bundles")
		return NewFileBundleStorage(storageDir)
	})
}

type testKeySigner struct {
	key []byte
}

func (s *testKeySigner) Sign(msg []byte) ([]byte, error) {
	k := &secp256k1.PrivKey{Key: s.key}
	return k.Sign(msg)
}

