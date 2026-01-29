package propagation

import (
	"context"
	"fmt"
	"sync"
	"time"

	"decentralized-api/logging"

	"github.com/productscience/inference/x/inference/types"
)

type Bundler struct {
	signer HeaderSigner
	cache  *Cache
	trees  []*Tree
	mu     sync.RWMutex
	sender Sender
	myAddr string
}

func NewBundler(signer HeaderSigner, cache *Cache, trees []*Tree, sender Sender, myAddr string) *Bundler {
	return &Bundler{
		signer: signer,
		cache:  cache,
		trees:  trees,
		sender: sender,
		myAddr: myAddr,
	}
}

func (b *Bundler) Publish(pocHeight int64, blockHash []byte, participant string, count uint32, rootHash []byte) error {
	if count == 0 || rootHash == nil {
		logging.Debug("Bundler: no artifacts to publish", types.PoC,
			"pocHeight", pocHeight, "participant", participant)
		return nil
	}

	bundleID := MakeBundleID(participant, pocHeight, rootHash, count, 1)
	header := BundleHeader{
		BundleID:     bundleID,
		Participant:  participant,
		PocHeight:    pocHeight,
		PocBlockHash: blockHash,
		RootHash:     rootHash,
		Count:        count,
		Version:      1,
		CreatedAt:    time.Now().Unix(),
	}

	sig, err := SignHeaderWith(header, b.signer)
	if err != nil {
		return fmt.Errorf("sign header: %w", err)
	}
	header.Signature = sig

	if b.cache != nil {
		if err := b.cache.StoreHeader(context.Background(), header); err != nil {
			logging.Warn("Bundler: failed to cache own header", types.PoC, "error", err)
		}
	}

	logging.Info("Bundler: publishing commit metadata", types.PoC,
		"pocHeight", pocHeight, "participant", participant,
		"count", count, "bundleID", fmt.Sprintf("%x", bundleID[:8]))

	if err := b.sendHeader(header); err != nil {
		logging.Warn("Bundler: failed to send header", types.PoC, "error", err)
		return fmt.Errorf("send header: %w", err)
	}

	logging.Info("Bundler: publish complete", types.PoC,
		"pocHeight", pocHeight, "participant", participant)

	return nil
}

func (b *Bundler) sendHeader(h BundleHeader) error {
	b.mu.RLock()
	trees := b.trees
	b.mu.RUnlock()

	var wg sync.WaitGroup

	for _, tree := range trees {
		node := tree.GetNode(b.myAddr)
		if node == nil {
			continue
		}
		for _, child := range node.Children {
			wg.Add(1)
			go func(treeIndex int, childAddr string) {
				defer wg.Done()
				if err := b.sender.SendHeader(treeIndex, childAddr, h); err != nil {
					logging.Debug("Bundler: failed to send header to child", types.PoC,
						"tree", treeIndex, "child", childAddr, "error", err)
				}
			}(tree.Index, child.Address)
		}
	}

	wg.Wait()
	return nil
}

func (b *Bundler) SetTrees(trees []*Tree) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.trees = trees
}
