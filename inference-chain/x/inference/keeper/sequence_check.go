package keeper

import (
    "crypto/sha256"
    "encoding/binary"
    "errors"
    "math/big"
)

// NOTE: Stage 1 Sequence Check helper (off-chain use)
//
// This file implements deterministic, platform-independent sampling for
// validators to reproduce a sampled sequence using the developer-provided
// user seed. The canonical user_seed is `types.RandomSeed.Signature`.
//
// The helper is intentionally implemented as pure SHA256-based sampling
// (no math/rand) so results are deterministic across OSes and Go versions.
//
// TODO: To enable fully on-chain Sequence Check later, `MsgValidation`
// (or a new message) must be extended to accept artifact payloads or
// artifact checksums. At that time this helper can be wired into
// `msgServer.Validation` for on-chain verification.

// ArtifactLite is a minimal representation of an artifact for SequenceCheck.
// This is used by unit tests and by off-chain validator code. When wiring
// on-chain, replace this with the canonical artifact proto type.
type ArtifactPosition struct {
    TopK  []string
    Chosen string
}

type ArtifactLite struct {
    Positions []ArtifactPosition
}

// GenerateRunSeed computes run_seed = SHA256(userSeed || inferenceId)
// where `userSeed` should be `types.RandomSeed.Signature` (string).
func GenerateRunSeed(userSeed string, inferenceId string) []byte {
    h := sha256.New()
    h.Write([]byte(userSeed))
    h.Write([]byte(inferenceId))
    return h.Sum(nil)
}

// sampleIndexForPosition deterministically samples an index in [0, topKLen)
// using seed_i = SHA256(runSeed || position) and then taking seed_i mod topKLen.
// This avoids PRNG state and is fully reproducible across platforms.
func sampleIndexForPosition(runSeed []byte, position uint64, topKLen int) (int, error) {
    if topKLen <= 0 {
        return 0, errors.New("topKLen must be > 0")
    }
    // prepare buffer = runSeed || position(8-byte big endian)
    posBytes := make([]byte, 8)
    binary.BigEndian.PutUint64(posBytes, position)
    buf := append(make([]byte, 0, len(runSeed)+8), runSeed...)
    buf = append(buf, posBytes...)

    h := sha256.Sum256(buf)
    // convert to big.Int and mod
    bi := new(big.Int).SetBytes(h[:])
    mod := big.NewInt(int64(topKLen))
    bi.Mod(bi, mod)
    return int(bi.Int64()), nil
}

// SequenceCheck verifies that for each position i in the artifact,
// artifact.Positions[i].Chosen == artifact.Positions[i].TopK[sampled_index]
// where sampled_index is computed via sampleIndexForPosition(runSeed, i).
// Returns (true, nil) if all positions match, (false, nil) if any mismatch,
// or (false, err) for malformed inputs.
func SequenceCheck(artifact ArtifactLite, runSeed []byte) (bool, error) {
    for i, pos := range artifact.Positions {
        if len(pos.TopK) == 0 {
            return false, errors.New("artifact position has empty top_k")
        }
        sampledIdx, err := sampleIndexForPosition(runSeed, uint64(i), len(pos.TopK))
        if err != nil {
            return false, err
        }
        if sampledIdx < 0 || sampledIdx >= len(pos.TopK) {
            return false, errors.New("sampled index out of bounds")
        }
        if pos.Chosen != pos.TopK[sampledIdx] {
            return false, nil
        }
    }
    return true, nil
}
