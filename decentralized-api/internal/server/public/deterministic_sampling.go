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
// where user_seed is the seed provided by developer in the API request (e.g., seed: 42).
// Per proposal: https://github.com/ZpokenWeb3/gonka/blob/main/proposals/inference-validation/inference-validation.md
func GenerateRunSeed(userSeed int64, inferenceId string) string {
	h := sha256.New()
	h.Write([]byte(fmt.Sprintf("%d", userSeed)))
	h.Write([]byte(inferenceId))
	return hex.EncodeToString(h.Sum(nil))
}

// ExtractUserSeedFromRequest extracts the seed from the OpenAI request.
// Returns the seed value, or 0 if not provided (default value per proposal).
func ExtractUserSeedFromRequest(openAiRequest *ChatCompletionRequest) int64 {
	if openAiRequest.Seed != nil {
		return int64(*openAiRequest.Seed)
	}
	return 0 // Default user_seed when not provided
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
