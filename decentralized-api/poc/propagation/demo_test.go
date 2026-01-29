package propagation

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"decentralized-api/poc/artifacts"

	"github.com/cosmos/cosmos-sdk/crypto/keys/secp256k1"
	"github.com/stretchr/testify/require"
)

type loggingSender struct {
	from    string
	dst     Sender
	records *[]string
	mu      sync.Mutex
}

func newLoggingSender(from string, dst Sender, records *[]string) *loggingSender {
	return &loggingSender{from: from, dst: dst, records: records}
}

func (l *loggingSender) SendHeader(treeIdx int, to string, h BundleHeader) error {
	l.mu.Lock()
	if l.records != nil {
		*l.records = append(*l.records, fmt.Sprintf("header: %s -> %s tree=%d bundle=%x", l.from, to, treeIdx, h.BundleID[:8]))
	}
	l.mu.Unlock()
	return l.dst.SendHeader(treeIdx, to, h)
}

type treeTrackingSender struct {
	from      string
	dst       Sender
	records   *[]string
	treeUsage map[int]int
	mu        *sync.Mutex
}

func newTreeTrackingSender(from string, dst Sender, records *[]string, treeUsage map[int]int, mu *sync.Mutex) *treeTrackingSender {
	return &treeTrackingSender{
		from:      from,
		dst:       dst,
		records:   records,
		treeUsage: treeUsage,
		mu:        mu,
	}
}

func (s *treeTrackingSender) SendHeader(treeIdx int, to string, h BundleHeader) error {
	s.mu.Lock()
	if s.records != nil {
		*s.records = append(*s.records, fmt.Sprintf("header: %s -> %s tree=%d bundle=%x", s.from, to, treeIdx, h.BundleID[:8]))
	}
	s.treeUsage[treeIdx]++
	s.mu.Unlock()
	return s.dst.SendHeader(treeIdx, to, h)
}

type propagationStorageFactory func(t *testing.T, tempDir, addr string) (BundleStorage, error)

func testPropagationDemo(t *testing.T, numParticipants int, storageFactory propagationStorageFactory) {
	numTrees := 6
	fanout := 32

	weightedParticipants := make([]WeightedParticipant, numParticipants)
	privKeys := make(map[string][]byte)
	pubKeys := make(map[string]string)

	for i := 0; i < numParticipants; i++ {
		addr := fmt.Sprintf("participant%d", i)
		weight := uint64(100 + i)
		weightedParticipants[i] = WeightedParticipant{
			Address: addr,
			Weight:  weight,
		}

		privKey := secp256k1.GenPrivKey()
		privKeys[addr] = privKey.Key
		pubKeys[addr] = hex.EncodeToString(privKey.PubKey().Bytes())
	}

	blockHash := sha256.Sum256([]byte("test-block"))
	pocHeight := int64(1000)

	trees := BuildTreesWithWeights(weightedParticipants, blockHash[:], numTrees, fanout)

	participants := make([]string, numParticipants)
	for i, wp := range weightedParticipants {
		participants[i] = wp.Address
	}

	transport := NewMockTransport()
	pubKeyProvider := NewMockPubKeyProvider()
	for addr, pubKey := range pubKeys {
		pubKeyProvider.RegisterKey(addr, pubKey)
	}

	var sendLogs []string
	tempDir := t.TempDir()

	caches := make(map[string]*Cache)
	receivers := make(map[string]*Receiver)
	bundlers := make(map[string]*Bundler)
	stores := make(map[string]*artifacts.ArtifactStore)

	for i, addr := range participants {
		storage, err := storageFactory(t, tempDir, addr)
		require.NoError(t, err)
		cache := NewCache(storage)
		caches[addr] = cache

		sender := newLoggingSender(addr, transport, &sendLogs)
		receiver := NewReceiver(cache, trees, pubKeyProvider, addr, sender)
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

		for j := 0; j < 100; j++ {
			nonce := int32(i*1000 + j)
			vector := []byte(fmt.Sprintf("vector-%d-%d", i, j))
			if err := store.Add(nonce, vector); err != nil {
				t.Fatalf("failed to add artifact: %v", err)
			}
		}

		if err := store.Flush(); err != nil {
			t.Fatalf("failed to flush store for %s: %v", addr, err)
		}

		signer := &testKeySigner{key: privKeys[addr]}
		bundler := NewBundler(signer, cache, trees, sender, addr)
		bundlers[addr] = bundler
	}

	sender := trees[0].Shuffled[0]

	senderCount := stores[sender].Count()
	senderRoot := stores[sender].GetRoot()
	if err := bundlers[sender].Publish(pocHeight, blockHash[:], sender, senderCount, senderRoot); err != nil {
		t.Fatalf("failed to publish: %v", err)
	}

	bundleID := MakeBundleID(sender, pocHeight, senderRoot, senderCount, 1)

	receivedCount := 0
	for _, addr := range participants {
		if addr == sender {
			continue
		}

		header, err := caches[addr].GetHeader(bundleID)
		if err != nil {
			continue
		}

		if header.Participant != sender {
			t.Errorf("participant %s: wrong sender in header: got %s, want %s",
				addr, header.Participant, sender)
		}

		if header.Count != senderCount {
			t.Errorf("participant %s: wrong count in header: got %d, want %d",
				addr, header.Count, senderCount)
		}

		receivedCount++
	}

	for _, store := range stores {
		store.Close()
	}

	t.Logf("- Trees: %d (fanout %d)", numTrees, fanout)
	t.Logf("- Participants: %d", numParticipants)
	t.Logf("- Participants who received: %d out of %d", receivedCount, numParticipants-1)

	if receivedCount != numParticipants-1 {
		t.Errorf("Not all participants received the bundle: got %d, want %d", receivedCount, numParticipants-1)
	}
}

func TestPropagationDemo(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping postgres tests in -short mode")
	}

	pool, cleanup := setupPropagationPostgres(t)
	defer cleanup()

	testPropagationDemo(t, 1000, func(t *testing.T, tempDir, addr string) (BundleStorage, error) {
		return NewPostgresBundleStorage(context.Background(), pool, addr)
	})
}

func TestPropagationDemoSmallPostgres(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping postgres tests in -short mode")
	}

	pool, cleanup := setupPropagationPostgres(t)
	defer cleanup()

	testPropagationDemo(t, 100, func(t *testing.T, tempDir, addr string) (BundleStorage, error) {
		return NewPostgresBundleStorage(context.Background(), pool, addr)
	})
}

func TestPropagationDemoWithFileStorage(t *testing.T) {
	testPropagationDemo(t, 100, func(t *testing.T, tempDir, addr string) (BundleStorage, error) {
		storageDir := filepath.Join(tempDir, addr, "bundles")
		return NewFileBundleStorage(storageDir)
	})
}

func TestPropagationDemoLargeWithFileStorage(t *testing.T) {
	testPropagationDemo(t, 1000, func(t *testing.T, tempDir, addr string) (BundleStorage, error) {
		storageDir := filepath.Join(tempDir, addr, "bundles")
		return NewFileBundleStorage(storageDir)
	})
}

func testMultiPublisherPropagation(t *testing.T, numParticipants int, storageFactory propagationStorageFactory) {
	numTrees := 6
	fanout := 4

	weightedParticipants := make([]WeightedParticipant, numParticipants)
	privKeys := make(map[string][]byte)
	pubKeys := make(map[string]string)

	for i := 0; i < numParticipants; i++ {
		addr := fmt.Sprintf("participant%d", i)
		weight := uint64(50 + i*10)
		weightedParticipants[i] = WeightedParticipant{
			Address: addr,
			Weight:  weight,
		}

		privKey := secp256k1.GenPrivKey()
		privKeys[addr] = privKey.Key
		pubKeys[addr] = hex.EncodeToString(privKey.PubKey().Bytes())
	}

	blockHash := sha256.Sum256([]byte("test-block"))
	pocHeight := int64(1000)

	trees := BuildTreesWithWeights(weightedParticipants, blockHash[:], numTrees, fanout)

	participants := make([]string, numParticipants)
	for i, wp := range weightedParticipants {
		participants[i] = wp.Address
	}

	transport := NewMockTransport()
	pubKeyProvider := NewMockPubKeyProvider()
	for addr, pubKey := range pubKeys {
		pubKeyProvider.RegisterKey(addr, pubKey)
	}

	treeUsage := make(map[int]int)
	var sendLogs []string
	var treeUsageMu sync.Mutex
	tempDir := t.TempDir()

	caches := make(map[string]*Cache)
	receivers := make(map[string]*Receiver)
	bundlers := make(map[string]*Bundler)
	stores := make(map[string]*artifacts.ArtifactStore)

	for i, addr := range participants {
		storage, err := storageFactory(t, tempDir, addr)
		require.NoError(t, err)
		cache := NewCache(storage)
		caches[addr] = cache

		sender := newTreeTrackingSender(addr, transport, &sendLogs, treeUsage, &treeUsageMu)
		receiver := NewReceiver(cache, trees, pubKeyProvider, addr, sender)
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

		for j := 0; j < 100; j++ {
			nonce := int32(i*1000 + j)
			vector := []byte(fmt.Sprintf("vector-%d-%d", i, j))
			if err := store.Add(nonce, vector); err != nil {
				t.Fatalf("failed to add artifact: %v", err)
			}
		}

		if err := store.Flush(); err != nil {
			t.Fatalf("failed to flush store for %s: %v", addr, err)
		}

		signer := &testKeySigner{key: privKeys[addr]}
		bundler := NewBundler(signer, cache, trees, sender, addr)
		bundlers[addr] = bundler
	}

	publishers := participants
	bundleIDs := make([][32]byte, len(publishers))

	for i, publisher := range publishers {
		bundler := bundlers[publisher]
		if bundler == nil {
			t.Fatalf("bundler missing for %s", publisher)
		}
		pubCount := stores[publisher].Count()
		pubRoot := stores[publisher].GetRoot()
		if err := bundler.Publish(pocHeight, blockHash[:], publisher, pubCount, pubRoot); err != nil {
			t.Fatalf("failed to publish from %s: %v", publisher, err)
		}
		bundleIDs[i] = MakeBundleID(publisher, pocHeight, pubRoot, pubCount, 1)
	}

	actualReceipts := 0

	for _, addr := range participants {
		for i, bundleID := range bundleIDs {
			publisher := publishers[i]
			if addr == publisher {
				continue
			}

			header, err := caches[addr].GetHeader(bundleID)
			if err != nil {
				continue
			}

			if header.Participant != publisher {
				t.Errorf("participant %s: wrong sender in header: got %s, want %s",
					addr, header.Participant, publisher)
			}

			actualReceipts++
		}
	}

	for _, store := range stores {
		store.Close()
	}

	t.Logf("\nTree Usage Statistics:")
	totalTreeSends := 0
	for i := 0; i < numTrees; i++ {
		count := treeUsage[i]
		totalTreeSends += count
		t.Logf("  Tree %d: %d sends", i, count)
	}

	t.Logf("\nPropagation Results:")
	t.Logf("- Participants: %d", numParticipants)
	t.Logf("- Publishers: %d", len(publishers))
	t.Logf("- Trees: %d (fanout %d)", numTrees, fanout)
	t.Logf("- Actual receipts: %d", actualReceipts)
}

func TestMultiPublisherPropagation(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping postgres tests in -short mode")
	}

	pool, cleanup := setupPropagationPostgres(t)
	defer cleanup()

	testMultiPublisherPropagation(t, 100, func(t *testing.T, tempDir, addr string) (BundleStorage, error) {
		return NewPostgresBundleStorage(context.Background(), pool, addr)
	})
}

func TestMultiPublisherPropagationWithFileStorage(t *testing.T) {
	testMultiPublisherPropagation(t, 50, func(t *testing.T, tempDir, addr string) (BundleStorage, error) {
		storageDir := filepath.Join(tempDir, addr, "bundles")
		return NewFileBundleStorage(storageDir)
	})
}

func TestTreeTopology(t *testing.T) {
	weightedParticipants := []WeightedParticipant{
		{Address: "A", Weight: 100},
		{Address: "B", Weight: 90},
		{Address: "C", Weight: 80},
		{Address: "D", Weight: 70},
		{Address: "E", Weight: 60},
		{Address: "F", Weight: 50},
		{Address: "G", Weight: 40},
		{Address: "H", Weight: 30},
		{Address: "I", Weight: 20},
		{Address: "J", Weight: 10},
	}
	blockHash := sha256.Sum256([]byte("test"))

	trees := BuildTreesWithWeights(weightedParticipants, blockHash[:], 4, 2)

	for _, tree := range trees {
		t.Logf("\nTree %d:", tree.Index)
		t.Logf("Order: %v", tree.Shuffled)

		for _, wp := range weightedParticipants {
			parent, children := tree.Role(wp.Address)
			t.Logf("  %s (weight=%d): parent=%s, children=%v", wp.Address, wp.Weight, parent, children)
		}
	}
}

func TestBundleSigning(t *testing.T) {
	privKey := secp256k1.GenPrivKey()
	pubKey := hex.EncodeToString(privKey.PubKey().Bytes())

	header := BundleHeader{
		BundleID:     [32]byte{1, 2, 3},
		Participant:  "test-participant",
		PocHeight:    1000,
		PocBlockHash: []byte("block-hash"),
		RootHash:     []byte("root-hash"),
		Count:        100,
		Version:      1,
		CreatedAt:    1234567890,
	}

	sig, err := SignHeader(header, privKey.Key)
	if err != nil {
		t.Fatalf("failed to sign header: %v", err)
	}

	header.Signature = sig

	if err := VerifyHeader(header, pubKey); err != nil {
		t.Fatalf("failed to verify header: %v", err)
	}

	header.Count = 999
	if err := VerifyHeader(header, pubKey); err == nil {
		t.Fatal("expected verification to fail after tampering")
	}

	t.Log("Bundle signing and verification works correctly")
}

func TestZeroWeightParticipantsExcluded(t *testing.T) {
	weightedParticipants := []WeightedParticipant{
		{Address: "legit1", Weight: 100},
		{Address: "legit2", Weight: 200},
		{Address: "legit3", Weight: 150},
		{Address: "attacker1", Weight: 0},
		{Address: "attacker2", Weight: 0},
		{Address: "legit4", Weight: 50},
	}

	blockHash := sha256.Sum256([]byte("test-block"))
	trees := BuildTreesWithWeights(weightedParticipants, blockHash[:], 3, 2)

	if len(trees) == 0 {
		t.Fatal("expected trees to be built")
	}

	for i, tree := range trees {
		if len(tree.Nodes) != 4 {
			t.Errorf("tree %d should have 4 nodes (excluding zero-weight), got %d", i, len(tree.Nodes))
		}

		if _, exists := tree.Nodes["attacker1"]; exists {
			t.Errorf("tree %d should not contain zero-weight attacker1", i)
		}

		if _, exists := tree.Nodes["attacker2"]; exists {
			t.Errorf("tree %d should not contain zero-weight attacker2", i)
		}

		if _, exists := tree.Nodes["legit1"]; !exists {
			t.Errorf("tree %d should contain legit1 with weight", i)
		}

		if _, exists := tree.Nodes["legit2"]; !exists {
			t.Errorf("tree %d should contain legit2 with weight", i)
		}

		t.Logf("Tree %d participants: %v", i, tree.Shuffled)
	}

	t.Log("Zero-weight participants successfully excluded from propagation trees")
}

func TestAttackerBlockingPrevention(t *testing.T) {
	legitimateParticipants := []WeightedParticipant{
		{Address: "validator1", Weight: 1000},
		{Address: "validator2", Weight: 800},
		{Address: "validator3", Weight: 600},
		{Address: "validator4", Weight: 400},
		{Address: "validator5", Weight: 200},
	}

	attackerNodes := []WeightedParticipant{
		{Address: "attacker1", Weight: 0},
		{Address: "attacker2", Weight: 0},
		{Address: "attacker3", Weight: 0},
		{Address: "attacker4", Weight: 0},
		{Address: "attacker5", Weight: 0},
		{Address: "attacker6", Weight: 0},
		{Address: "attacker7", Weight: 0},
		{Address: "attacker8", Weight: 0},
		{Address: "attacker9", Weight: 0},
		{Address: "attacker10", Weight: 0},
	}

	allParticipants := append(legitimateParticipants, attackerNodes...)

	blockHash := sha256.Sum256([]byte("test-block"))
	trees := BuildTreesWithWeights(allParticipants, blockHash[:], 3, 2)

	if len(trees) == 0 {
		t.Fatal("expected trees to be built")
	}

	for i, tree := range trees {
		if len(tree.Nodes) != len(legitimateParticipants) {
			t.Errorf("tree %d should have %d nodes (only legitimate participants), got %d",
				i, len(legitimateParticipants), len(tree.Nodes))
		}

		attackerFound := false
		for _, attacker := range attackerNodes {
			if _, exists := tree.Nodes[attacker.Address]; exists {
				attackerFound = true
				t.Errorf("tree %d contains attacker node %s", i, attacker.Address)
			}
		}

		if attackerFound {
			t.Errorf("tree %d contains attacker nodes - security vulnerability!", i)
		}

		legitimateCount := 0
		for _, legit := range legitimateParticipants {
			if _, exists := tree.Nodes[legit.Address]; exists {
				legitimateCount++
			}
		}

		if legitimateCount != len(legitimateParticipants) {
			t.Errorf("tree %d should contain all %d legitimate participants, got %d",
				i, len(legitimateParticipants), legitimateCount)
		}

		t.Logf("Tree %d: %d legitimate participants, 0 attackers (10 blocked)", i, legitimateCount)
	}

	t.Log("Attacker blocking prevention successful - new nodes without weights cannot participate")
}

func TestAllZeroWeightNoTreeCreated(t *testing.T) {
	allZeroWeight := []WeightedParticipant{
		{Address: "attacker1", Weight: 0},
		{Address: "attacker2", Weight: 0},
		{Address: "attacker3", Weight: 0},
	}

	blockHash := sha256.Sum256([]byte("test-block"))
	trees := BuildTreesWithWeights(allZeroWeight, blockHash[:], 3, 2)

	if len(trees) != 0 {
		t.Errorf("expected no trees when all weights are zero, got %d trees", len(trees))
	}

	t.Log("No trees created when all participants have zero weight - correct behavior")
}
