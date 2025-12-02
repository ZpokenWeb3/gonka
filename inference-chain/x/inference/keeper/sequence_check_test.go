package keeper

import (
    "testing"
)

func TestSequenceCheck_ValidArtifact(t *testing.T) {
    userSeed := "usersig-abc"
    inferenceId := "inf-123"
    runSeed := GenerateRunSeed(userSeed, inferenceId)

    // Build artifact positions with TopK and Chosen computed from our sampler
    positions := make([]ArtifactPosition, 3)
    for i := 0; i < 3; i++ {
        // simple top-k tokens for determinism
        topk := []string{"tokenA", "tokenB", "tokenC", "tokenD"}
        idx, err := sampleIndexForPosition(runSeed, uint64(i), len(topk))
        if err != nil {
            t.Fatalf("sampleIndexForPosition failed: %v", err)
        }
        positions[i] = ArtifactPosition{
            TopK:  topk,
            Chosen: topk[idx],
        }
    }
    art := ArtifactLite{Positions: positions}
    ok, err := SequenceCheck(art, runSeed)
    if err != nil {
        t.Fatalf("SequenceCheck returned error: %v", err)
    }
    if !ok {
        t.Fatalf("SequenceCheck expected true for valid artifact")
    }
}

func TestSequenceCheck_TamperedChosen(t *testing.T) {
    userSeed := "usersig-abc"
    inferenceId := "inf-123"
    runSeed := GenerateRunSeed(userSeed, inferenceId)

    topk := []string{"tokenA", "tokenB", "tokenC"}
    // deliberately pick wrong chosen value
    pos := ArtifactPosition{TopK: topk, Chosen: "not-in-topk"}
    art := ArtifactLite{Positions: []ArtifactPosition{pos}}
    ok, err := SequenceCheck(art, runSeed)
    if err != nil {
        t.Fatalf("SequenceCheck returned error: %v", err)
    }
    if ok {
        t.Fatalf("SequenceCheck expected false for tampered chosen")
    }
}

func TestSequenceCheck_EmptyTopK(t *testing.T) {
    runSeed := GenerateRunSeed("u", "i")
    art := ArtifactLite{Positions: []ArtifactPosition{{TopK: []string{}, Chosen: "x"}}}
    _, err := SequenceCheck(art, runSeed)
    if err == nil {
        t.Fatalf("SequenceCheck expected error for empty top_k")
    }
}
