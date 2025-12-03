package keeper

import (
	"encoding/json"
	"errors"

	"github.com/productscience/inference/x/inference/types"
)

// vLLM response structures for parsing logprobs
type vllmTokenLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int   `json:"bytes,omitempty"`
}

type vllmPositionLogprobs struct {
	Token       string             `json:"token"`
	Logprob     float64            `json:"logprob"`
	Bytes       []int              `json:"bytes,omitempty"`
	TopLogprobs []vllmTokenLogprob `json:"top_logprobs"`
}

type vllmLogprobs struct {
	Content []vllmPositionLogprobs `json:"content"`
}

type vllmChoice struct {
	Index        int           `json:"index"`
	Message      interface{}   `json:"message,omitempty"`
	Logprobs     *vllmLogprobs `json:"logprobs,omitempty"`
	FinishReason string        `json:"finish_reason,omitempty"`
}

type vllmResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []vllmChoice `json:"choices"`
}

// parseVLLMLogprobs parses vLLM response payload and extracts top-k logprobs
// for Stage-1 Sequence Check validation.
func parseVLLMLogprobs(responsePayload string) ([]*types.PositionLogprobs, error) {
	if responsePayload == "" {
		return nil, nil // No logprobs to parse
	}

	var response vllmResponse
	if err := json.Unmarshal([]byte(responsePayload), &response); err != nil {
		return nil, err
	}

	if len(response.Choices) == 0 {
		return nil, errors.New("no choices in vLLM response")
	}

	choice := response.Choices[0]
	if choice.Logprobs == nil || len(choice.Logprobs.Content) == 0 {
		return nil, nil // No logprobs requested or available
	}

	positions := make([]*types.PositionLogprobs, 0, len(choice.Logprobs.Content))
	for _, pos := range choice.Logprobs.Content {
		topK := make([]*types.TokenLogprob, 0, len(pos.TopLogprobs))
		for _, tl := range pos.TopLogprobs {
			topK = append(topK, &types.TokenLogprob{
				Token:   tl.Token,
				Logprob: tl.Logprob,
			})
		}

		positions = append(positions, &types.PositionLogprobs{
			TopK:        topK,
			ChosenToken: pos.Token, // The actually generated token
		})
	}

	return positions, nil
}
