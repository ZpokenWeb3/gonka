package public

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"decentralized-api/logging"
	"github.com/productscience/inference/x/inference/types"
)

// GenerateRunSeed generates a deterministic run seed for Stage-1 Sequence Check validation.
// Formula: run_seed = SHA256(user_seed || inference_id)
// where user_seed is RandomSeed.Signature for the executor in the current epoch.
func GenerateRunSeed(userSeed, inferenceId string) string {
	h := sha256.New()
	h.Write([]byte(userSeed))
	h.Write([]byte(inferenceId))
	return hex.EncodeToString(h.Sum(nil))
}

// GetExecutorRandomSeed fetches the executor's RandomSeed.Signature from the chain
// for the current effective epoch. Returns empty string if not found (non-fatal).
func (s *Server) GetExecutorRandomSeed(ctx context.Context, executorAddress string) (string, error) {
	queryClient := s.recorder.NewInferenceQueryClient()

	// Get current effective epoch
	epochResp, err := queryClient.EffectiveEpoch(ctx, &types.QueryEffectiveEpochRequest{})
	if err != nil {
		logging.Warn("Failed to get effective epoch for random seed", types.Inferences,
			"executor", executorAddress,
			"error", err)
		return "", nil // Non-fatal: validation can still work without sequence check
	}

	if epochResp.Epoch == nil {
		logging.Warn("No effective epoch found", types.Inferences, "executor", executorAddress)
		return "", nil
	}

	epochId := epochResp.Epoch.Index

	// Get RandomSeed for executor in this epoch
	seedResp, err := queryClient.RandomSeed(ctx, &types.QueryRandomSeedRequest{
		EpochId:       epochId,
		ParticipantId: executorAddress,
	})

	if err != nil {
		logging.Debug("Random seed not found for executor (non-fatal)", types.Inferences,
			"executor", executorAddress,
			"epoch", epochId,
			"error", err)
		return "", nil // Non-fatal: executor may not have submitted seed yet
	}

	if seedResp.Seed == nil || seedResp.Seed.Signature == "" {
		logging.Debug("Empty random seed for executor", types.Inferences,
			"executor", executorAddress,
			"epoch", epochId)
		return "", nil
	}

	logging.Info("Retrieved random seed for deterministic sampling", types.Inferences,
		"executor", executorAddress,
		"epoch", epochId)

	return seedResp.Seed.Signature, nil
}

// PrepareVLLMRequestForSequenceCheck adds deterministic sampling parameters to vLLM request.
// This enables Stage-1 Sequence Check validation by:
// 1. Setting deterministic_seed=run_seed (full hex string, 64 chars)
// 2. Enabling logprobs=true, top_logprobs=5 for on-chain storage
//
// vLLM will use enforced type: concatenate seed with position and use as binary seed.
// Returns modified request body with deterministic sampling enabled.
func PrepareVLLMRequestForSequenceCheck(requestBody []byte, inferenceId string, runSeed string) ([]byte, error) {
	if runSeed == "" {
		logging.Debug("No run_seed available, skipping deterministic sampling setup", types.Inferences,
			"inferenceId", inferenceId)
		return requestBody, nil
	}

	// Parse existing request
	var req map[string]interface{}
	if err := json.Unmarshal(requestBody, &req); err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}

	// Add deterministic sampling parameters
	// Pass full hex run_seed (64 chars) as deterministic_seed string
	req["deterministic_seed"] = runSeed
	req["logprobs"] = true
	req["top_logprobs"] = 5

	logging.Info("Enabled deterministic sampling for inference", types.Inferences,
		"inferenceId", inferenceId,
		"deterministic_seed", runSeed[:16]+"...") // Log first 16 chars for debugging

	modifiedBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal modified request: %w", err)
	}

	return modifiedBody, nil
}
