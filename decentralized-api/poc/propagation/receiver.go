package propagation

import (
	"bytes"
	"context"
	"fmt"
	"sync"
	"time"

	"decentralized-api/logging"

	"github.com/productscience/inference/x/inference/types"
)

type Receiver struct {
	cache    *Cache
	trees    []*Tree
	verifier PubKeyProvider
	myAddr   string
	sender   Sender

	mu               sync.RWMutex
	processedHeaders map[[32]byte]bool
	pendingHeaders   map[[32]byte]*BundleHeader
	lastHeaderTime   map[[32]byte]time.Time
}

type PubKeyProvider interface {
	GetPubKey(participantAddr string) (string, error)
}

func NewReceiver(cache *Cache, trees []*Tree, verifier PubKeyProvider, myAddr string, sender Sender) *Receiver {
	return &Receiver{
		cache:            cache,
		trees:            trees,
		verifier:         verifier,
		myAddr:           myAddr,
		sender:           sender,
		processedHeaders: make(map[[32]byte]bool),
		pendingHeaders:   make(map[[32]byte]*BundleHeader),
		lastHeaderTime:   make(map[[32]byte]time.Time),
	}
}

func (r *Receiver) OnHeader(h BundleHeader, treeIdx int) error {
	logging.Debug("Receiver: received header", types.PoC,
		"receiver", r.myAddr, "publisher", h.Participant, "pocHeight", h.PocHeight,
		"count", h.Count, "bundleID", fmt.Sprintf("%x", h.BundleID[:8]),
		"tree", treeIdx)

	r.mu.RLock()
	processed := r.processedHeaders[h.BundleID]
	r.mu.RUnlock()

	if processed {
		logging.Debug("Receiver: duplicate header ignored (already processed)", types.PoC,
			"receiver", r.myAddr, "bundleID", fmt.Sprintf("%x", h.BundleID[:8]))
		return nil
	}

	pubKey, err := r.verifier.GetPubKey(h.Participant)
	if err != nil {
		logging.Warn("Receiver: failed to get public key", types.PoC,
			"participant", h.Participant, "error", err)
		return fmt.Errorf("get pubkey: %w", err)
	}

	if err := VerifyHeader(h, pubKey); err != nil {
		logging.Warn("Receiver: header signature verification failed", types.PoC,
			"participant", h.Participant, "error", err)
		return fmt.Errorf("verify header: %w", err)
	}

	expectedID := MakeBundleID(h.Participant, h.PocHeight, h.RootHash, h.Count, h.Version)
	if !bytes.Equal(expectedID[:], h.BundleID[:]) {
		logging.Warn("Receiver: bundle ID mismatch", types.PoC,
			"expected", fmt.Sprintf("%x", expectedID[:8]),
			"got", fmt.Sprintf("%x", h.BundleID[:8]))
		return fmt.Errorf("bundle ID mismatch")
	}

	r.mu.Lock()
	if r.processedHeaders[h.BundleID] {
		r.mu.Unlock()
		logging.Info("Receiver: duplicate header ignored (race detected)", types.PoC,
			"receiver", r.myAddr, "bundleID", fmt.Sprintf("%x", h.BundleID[:8]))
		return nil
	}

	if err := r.cache.StoreHeader(context.Background(), h); err != nil {
		r.mu.Unlock()
		logging.Warn("Receiver: failed to store header", types.PoC,
			"bundleID", fmt.Sprintf("%x", h.BundleID[:8]), "error", err)
		return fmt.Errorf("store header: %w", err)
	}

	r.processedHeaders[h.BundleID] = true
	r.pendingHeaders[h.BundleID] = &h
	r.lastHeaderTime[h.BundleID] = time.Now()
	trees := r.trees
	r.mu.Unlock()

	logging.Info("Receiver: commit metadata verified and stored", types.PoC,
		"receiver", r.myAddr, "publisher", h.Participant, "pocHeight", h.PocHeight,
		"bundleID", fmt.Sprintf("%x", h.BundleID[:8]))

	r.forwardHeader(h, treeIdx, trees)

	return nil
}

func (r *Receiver) forwardHeader(h BundleHeader, treeIdx int, trees []*Tree) {
	if treeIdx >= len(trees) {
		return
	}

	tree := trees[treeIdx]
	node := tree.GetNode(r.myAddr)
	if node == nil {
		return
	}

	var wg sync.WaitGroup
	for _, child := range node.Children {
		wg.Add(1)
		go func(childAddr string) {
			defer wg.Done()
			if err := r.sender.SendHeader(treeIdx, childAddr, h); err != nil {
				logging.Debug("Receiver: failed to forward header", types.PoC,
					"tree", treeIdx, "child", childAddr, "error", err)
			}
		}(child.Address)
	}
	wg.Wait()
}

func (r *Receiver) SetTrees(trees []*Tree) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.trees = trees
}
