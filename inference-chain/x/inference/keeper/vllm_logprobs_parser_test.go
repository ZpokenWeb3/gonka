package keeper

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestParseVLLMLogprobs(t *testing.T) {
	t.Run("valid vLLM response with logprobs", func(t *testing.T) {
		payload := `{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1677858242,
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "index": 0,
      "logprobs": {
        "content": [
          {
            "token": " world",
            "logprob": -0.5,
            "top_logprobs": [
              {"token": " world", "logprob": -0.5},
              {"token": " universe", "logprob": -1.2},
              {"token": " planet", "logprob": -2.3},
              {"token": " globe", "logprob": -3.1},
              {"token": " earth", "logprob": -4.0}
            ]
          },
          {
            "token": "!",
            "logprob": -0.1,
            "top_logprobs": [
              {"token": "!", "logprob": -0.1},
              {"token": ".", "logprob": -2.5},
              {"token": "?", "logprob": -3.8}
            ]
          }
        ]
      },
      "finish_reason": "stop"
    }
  ]
}`

		positions, err := parseVLLMLogprobs(payload)
		require.NoError(t, err)
		require.NotNil(t, positions)
		require.Len(t, positions, 2)

		// First position: " world"
		require.Equal(t, " world", positions[0].ChosenToken)
		require.Len(t, positions[0].TopK, 5)
		require.Equal(t, " world", positions[0].TopK[0].Token)
		require.InDelta(t, -0.5, positions[0].TopK[0].Logprob, 0.001)
		require.Equal(t, " universe", positions[0].TopK[1].Token)
		require.InDelta(t, -1.2, positions[0].TopK[1].Logprob, 0.001)

		// Second position: "!"
		require.Equal(t, "!", positions[1].ChosenToken)
		require.Len(t, positions[1].TopK, 3)
		require.Equal(t, "!", positions[1].TopK[0].Token)
		require.InDelta(t, -0.1, positions[1].TopK[0].Logprob, 0.001)
	})

	t.Run("empty payload", func(t *testing.T) {
		positions, err := parseVLLMLogprobs("")
		require.NoError(t, err)
		require.Nil(t, positions)
	})

	t.Run("no choices", func(t *testing.T) {
		payload := `{"id": "test", "choices": []}`
		positions, err := parseVLLMLogprobs(payload)
		require.Error(t, err)
		require.Contains(t, err.Error(), "no choices")
	})

	t.Run("no logprobs field", func(t *testing.T) {
		payload := `{
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop"
    }
  ]
}`
		positions, err := parseVLLMLogprobs(payload)
		require.NoError(t, err)
		require.Nil(t, positions)
	})

	t.Run("logprobs with empty content", func(t *testing.T) {
		payload := `{
  "choices": [
    {
      "index": 0,
      "logprobs": {
        "content": []
      }
    }
  ]
}`
		positions, err := parseVLLMLogprobs(payload)
		require.NoError(t, err)
		require.Nil(t, positions)
	})

	t.Run("invalid JSON", func(t *testing.T) {
		payload := `{"invalid json`
		positions, err := parseVLLMLogprobs(payload)
		require.Error(t, err)
		require.Nil(t, positions)
	})
}
