import os
import requests

from api.inference.top_tokens import TopLogProbsSequence, compare_token_sequences


VLLM_URL = os.getenv("VLLM_URL", "http://195.189.60.154:8000")
MODEL = os.getenv("MODEL", "facebook/opt-125m")
CHAT_TEMPLATE = "{{ messages[0].content }}"


def run_inference(
    prompt: str,
    seed: int = None,
    enforced_str: str = None,
    enforced_tokens: dict = None,
    max_tokens: int = 30
) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 5,
        "chat_template": CHAT_TEMPLATE
    }
    if seed is not None:
        payload["seed"] = seed
    if enforced_str is not None:
        payload["enforced_str"] = enforced_str
    if enforced_tokens is not None:
        payload["enforced_tokens"] = enforced_tokens

    response = requests.post(f"{VLLM_URL}/v1/chat/completions", json=payload)
    assert response.status_code == 200, f"Request failed: {response.text}"
    return response.json()


def get_content(response: dict) -> str:
    return response['choices'][0]['message']['content']


def get_logprobs_content(response: dict) -> list:
    return response['choices'][0]['logprobs']['content']


def build_enforced_tokens(response: dict) -> dict:
    content = get_logprobs_content(response)
    tokens = []
    for position in content:
        tokens.append({
            "token": position["token"],
            "top_tokens": [t["token"] for t in position["top_logprobs"]]
        })
    return {"tokens": tokens}


class TestSeedDeterminism:

    def test_same_seed_produces_same_output(self):
        response1 = run_inference("Hello world", seed=42, max_tokens=20)
        response2 = run_inference("Hello world", seed=42, max_tokens=20)
        assert get_content(response1) == get_content(response2)

    def test_different_seeds_produce_different_output(self):
        response1 = run_inference("Hello world", seed=111, max_tokens=20)
        response2 = run_inference("Hello world", seed=222, max_tokens=20)
        assert get_content(response1) != get_content(response2)

    def test_same_seed_produces_same_logprobs(self):
        response1 = run_inference("Test prompt", seed=999, max_tokens=15)
        response2 = run_inference("Test prompt", seed=999, max_tokens=15)

        seq1 = TopLogProbsSequence.from_json(response1)
        seq2 = TopLogProbsSequence.from_json(response2)
        matches = compare_token_sequences(seq1, seq2)

        assert all(matches)


class TestEnforcedStr:

    def test_enforced_str_produces_exact_tokens(self):
        enforced = "One two three four five"
        response = run_inference("Count", enforced_str=enforced, max_tokens=50)
        assert enforced in get_content(response)

    def test_enforced_str_returns_logprobs(self):
        response = run_inference("Say hello", enforced_str="Hello there", max_tokens=20)
        logprobs = get_logprobs_content(response)
        assert len(logprobs) > 0
        for position in logprobs:
            assert "top_logprobs" in position
            assert len(position["top_logprobs"]) >= 1


class TestEnforcedTokens:

    def test_enforced_tokens_reproduces_output(self):
        original = run_inference("What is AI", seed=123, max_tokens=20)
        original_content = get_content(original)

        enforced_tokens = build_enforced_tokens(original)
        validated = run_inference("What is AI", enforced_tokens=enforced_tokens, max_tokens=20)

        assert get_content(validated) == original_content

    def test_enforced_tokens_logprobs_match(self):
        original = run_inference("Explain briefly", seed=456, max_tokens=15)
        enforced_tokens = build_enforced_tokens(original)
        validated = run_inference("Explain briefly", enforced_tokens=enforced_tokens, max_tokens=15)

        orig_seq = TopLogProbsSequence.from_json(original)
        val_seq = TopLogProbsSequence.from_json(validated)
        matches = compare_token_sequences(orig_seq, val_seq)

        match_rate = sum(matches) / len(matches)
        assert match_rate >= 0.8, f"Expected >= 80% logprobs match, got {match_rate*100:.1f}%"

    def test_enforced_tokens_returns_top_tokens(self):
        original = run_inference("Generate", seed=789, max_tokens=10)
        enforced_tokens = build_enforced_tokens(original)
        validated = run_inference("Generate", enforced_tokens=enforced_tokens, max_tokens=10)

        val_logprobs = get_logprobs_content(validated)
        orig_logprobs = get_logprobs_content(original)

        for i, (orig, val) in enumerate(zip(orig_logprobs, val_logprobs)):
            orig_top_tokens = {t["token"] for t in orig["top_logprobs"]}
            val_top_tokens = {t["token"] for t in val["top_logprobs"]}
            assert orig_top_tokens.issubset(val_top_tokens), f"Position {i}: original top tokens not in validation"


class TestValidationWorkflow:

    def test_inference_then_validation_content_matches(self):
        prompt = "The quick brown fox"
        seed = 42

        inference = run_inference(prompt, seed=seed, max_tokens=15)
        enforced_tokens = build_enforced_tokens(inference)
        validation = run_inference(prompt, enforced_tokens=enforced_tokens, max_tokens=15)

        assert get_content(inference) == get_content(validation)

    def test_validation_returns_original_top_tokens(self):
        prompt = "Simple test"
        seed = 123

        inference = run_inference(prompt, seed=seed, max_tokens=10)
        enforced_tokens = build_enforced_tokens(inference)
        validation = run_inference(prompt, enforced_tokens=enforced_tokens, max_tokens=10)

        inf_lp = get_logprobs_content(inference)
        val_lp = get_logprobs_content(validation)

        for i, (inf, val) in enumerate(zip(inf_lp, val_lp)):
            inf_tokens = {t["token"] for t in inf["top_logprobs"]}
            val_tokens = {t["token"] for t in val["top_logprobs"]}
            assert inf_tokens.issubset(val_tokens), f"Position {i}: missing tokens"
