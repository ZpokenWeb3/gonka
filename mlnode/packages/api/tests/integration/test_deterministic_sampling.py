import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "benchmarks" / "src"))

from validation.utils import generate_and_validate, verify_artifacts
from validation.data import ModelInfo, RequestParams, ExperimentRequest


VLLM_URL = os.getenv("VLLM_URL", "http://195.189.60.154:8000")
MODEL = os.getenv("MODEL", "facebook/opt-125m")


def create_experiment_request(
    prompt: str,
    temperature: float = 0.7,
    seed: int = 42,
    max_tokens: int = 30,
    top_logprobs: int = 5,
) -> ExperimentRequest:
    model_info = ModelInfo(name=MODEL, url=VLLM_URL)
    request_params = RequestParams(
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        top_logprobs=top_logprobs,
    )
    return ExperimentRequest(
        prompt=prompt,
        inference_model=model_info,
        validation_model=model_info,
        request_params=request_params,
    )


class TestDeterministicSampling:

    def test_greedy_sampling(self):
        experiment = create_experiment_request(
            prompt="What is the capital of France?",
            temperature=0.0,
            seed=42,
            max_tokens=20,
        )
        result = generate_and_validate(experiment)
        assert result.inference_result.text == result.validation_result.text
        assert verify_artifacts(result.inference_result, result.validation_result)

    def test_temperature_sampling_low(self):
        experiment = create_experiment_request(
            prompt="Write a short sentence about the weather.",
            temperature=0.3,
            seed=123,
            max_tokens=25,
        )
        result = generate_and_validate(experiment)
        assert result.inference_result.text == result.validation_result.text
        assert verify_artifacts(result.inference_result, result.validation_result)

    def test_temperature_sampling_medium(self):
        experiment = create_experiment_request(
            prompt="Tell me something interesting.",
            temperature=0.7,
            seed=456,
            max_tokens=30,
        )
        result = generate_and_validate(experiment)
        assert result.inference_result.text == result.validation_result.text
        assert verify_artifacts(result.inference_result, result.validation_result)

    def test_temperature_sampling_high(self):
        experiment = create_experiment_request(
            prompt="Generate a creative story opening.",
            temperature=1.0,
            seed=789,
            max_tokens=35,
        )
        result = generate_and_validate(experiment)
        assert result.inference_result.text == result.validation_result.text
        assert verify_artifacts(result.inference_result, result.validation_result)

    def test_inference_and_validation_match(self):
        experiment = create_experiment_request(
            prompt="Hello world",
            seed=42,
            temperature=0.7,
            max_tokens=20
        )
        result = generate_and_validate(experiment)

        assert result.inference_result.text == result.validation_result.text
        assert verify_artifacts(result.inference_result, result.validation_result)

    def test_different_seeds_produce_different_output(self):
        prompt = "The weather today"
        experiment1 = create_experiment_request(prompt=prompt, seed=111, temperature=0.7, max_tokens=20)
        experiment2 = create_experiment_request(prompt=prompt, seed=222, temperature=0.7, max_tokens=20)

        result1 = generate_and_validate(experiment1)
        result2 = generate_and_validate(experiment2)

        assert result1.inference_result.text != result2.inference_result.text

    def test_validation_preserves_logprobs(self):
        experiment = create_experiment_request(
            prompt="Explain quantum computing",
            temperature=0.5,
            seed=999,
            max_tokens=15,
        )
        result = generate_and_validate(experiment)

        assert len(result.inference_result.results) == len(result.validation_result.results)

        for inf_pos, val_pos in zip(result.inference_result.results, result.validation_result.results):
            assert inf_pos.token == val_pos.token
            inf_top_tokens = set(inf_pos.logprobs.keys())
            val_top_tokens = set(val_pos.logprobs.keys())
            assert inf_top_tokens.issubset(val_top_tokens)

    def test_different_temperature_settings(self):
        prompts_and_temps = [
            ("Count to five", 0.0),
            ("Describe the sky", 0.2),
            ("What is AI?", 0.5),
            ("Tell me a joke", 0.8),
            ("Be creative", 1.2),
        ]

        for prompt, temp in prompts_and_temps:
            experiment = create_experiment_request(
                prompt=prompt,
                temperature=temp,
                seed=42,
                max_tokens=20,
            )
            result = generate_and_validate(experiment)
            assert result.inference_result.text == result.validation_result.text
            assert verify_artifacts(result.inference_result, result.validation_result)

    def test_validation_with_long_sequences(self):
        experiment = create_experiment_request(
            prompt="Write a paragraph about machine learning.",
            temperature=0.6,
            seed=555,
            max_tokens=50,
        )
        result = generate_and_validate(experiment)

        assert result.inference_result.text == result.validation_result.text
        assert len(result.inference_result.results) > 0
        assert len(result.validation_result.results) > 0
        assert verify_artifacts(result.inference_result, result.validation_result)
