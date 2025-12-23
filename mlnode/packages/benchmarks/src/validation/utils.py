import requests
import math
import json
from typing import (
    Dict,
    Any,
    List,
    Callable,
    Optional
)
from pathlib import Path

from pydantic import BaseModel


from typing import Any, Dict, List
from pydantic import BaseModel, Field

from validation.data import (
    ModelInfo,
    RequestParams,
    ExperimentRequest,
    ValidationItem,
    Result,
    PositionResult
)

from common.logger import create_logger


logger = create_logger(__name__)


class EnforcedToken(BaseModel):
    token: str
    top_tokens: List[str] = Field(default_factory=list)

class EnforcedTokens(BaseModel):
    tokens: List[EnforcedToken]

    @classmethod
    def from_content(cls, content: List[Dict[str, Any]]) -> "EnforcedTokens":
        tokens = []
        for position in content:
            token = position["token"]
            top_tokens = [x["token"] for x in position["top_logprobs"]]
            tokens.append(EnforcedToken(token=token, top_tokens=top_tokens))
        return cls(tokens=tokens)
    
    @classmethod
    def from_result(cls, result: Result) -> "EnforcedTokens":
        return cls(tokens=[EnforcedToken(token=r.token, top_tokens=list(r.logprobs.keys())) for r in result.results])

    
def _prepare_messages(
    prompt: str,
) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": "You are a helpful assistant. Response clear, correct and complete."},
        {"role": "user", "content": prompt}
    ]


def inference(
    model_info: ModelInfo,
    request_params: RequestParams,
    prompt: str,
    inference_id: str,
) -> Dict[str, Any]:
    url = f"{model_info.url}/v1/chat/completions"
    payload = {
        "model": model_info.name,
        "messages": _prepare_messages(prompt),
        "max_tokens": request_params.max_tokens,
        "temperature": request_params.temperature,
        "seed": request_params.seed,
        "stream": False,
        "logprobs": True,
        "n": 1,
        "top_logprobs": request_params.top_logprobs,
        "top_k": request_params.top_k,
        "skip_special_tokens": False,
        "repetition_penalty": 1.2,
        "chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}",
        "inference_id": inference_id,
    }

    response = requests.post(url, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Inference API request failed with status {response.status_code} {response.text}")
    return response.json()


def validation(
    model_info: ModelInfo,
    request_params: RequestParams,
    prompt: str,
    run_seed: int,
    enforced_tokens: EnforcedTokens,
) -> Dict[str, Any]:
    url = f"{model_info.url}/v1/chat/completions"
    payload = {
        "model": model_info.name,
        "messages": _prepare_messages(prompt),
        "max_tokens": request_params.max_tokens,
        "temperature": request_params.temperature,
        "seed": request_params.seed,
        "stream": False,
        "logprobs": True,
        "top_logprobs": request_params.top_logprobs,
        "top_k": request_params.top_k,
        "n": 1,
        "skip_special_tokens": False,
        "repetition_penalty": 1.2,
        "chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}",
        "run_seed": run_seed,
        "enforced_tokens": enforced_tokens.model_dump(),
    }

    response = requests.post(url, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Validation API request failed with status {response.status_code} {response.text}\n(enforced_tokens: {enforced_tokens})\n(payload: {payload})")

    return response.json()


def _extract_logprobs(resp) -> Result:
    logprobs = resp["choices"][0]["logprobs"]["content"]
    text = resp["choices"][0]["message"]["content"]
    results = []
    for position in logprobs:
        res = PositionResult(
            token=position["token"],
            logprobs={logprob["token"]: logprob["logprob"] for logprob in position["top_logprobs"]}
        )
        results.append(res)

    return Result(text=text, results=results)


def _extract_enforced_tokens(resp) -> EnforcedTokens:
    return EnforcedTokens.from_content(resp["choices"][0]["logprobs"]["content"])


def _extract_run_seed(resp) -> int:
    return resp["choices"][0]["run_seed"]


def _generate_inference_id() -> str:
    import uuid
    return str(uuid.uuid4())


def verify_artifacts(inf_result: Result, val_result: Result, prob_tolerance: float = 0.0) -> bool:
    if len(inf_result.results) != len(val_result.results):
        return False

    for i, (inf_pos, val_pos) in enumerate(zip(inf_result.results, val_result.results)):
        if inf_pos.token != val_pos.token:
            logger.error(f"Position {i}: token mismatch {inf_pos.token} vs {val_pos.token}")
            return False

        inf_top_tokens = set(inf_pos.logprobs.keys())
        val_top_tokens = set(val_pos.logprobs.keys())
        if not inf_top_tokens.issubset(val_top_tokens):
            logger.error(f"Position {i}: top tokens mismatch")
            return False

        if inf_pos.token not in val_top_tokens:
            logger.error(f"Position {i}: inference token {inf_pos.token} not in validation top-k")
            return False

        for token in inf_top_tokens:
            inf_logprob = inf_pos.logprobs[token]
            val_logprob = val_pos.logprobs[token]
            prob_diff = abs(inf_logprob - val_logprob)

            if prob_diff > prob_tolerance:
                logger.error(
                    f"Position {i}, token '{token}': probability mismatch. "
                    f"Inference: {inf_logprob:.6f}, Validation: {val_logprob:.6f}, "
                    f"Difference: {prob_diff:.6f}"
                )
                return False
            else:
                logger.debug(
                    f"Position {i}, token '{token}': probabilities match. "
                    f"Inference: {inf_logprob:.6f}, Validation: {val_logprob:.6f}"
                )

    return True


def generate_and_validate(
    experiment_request: ExperimentRequest,
    save_inference_resp_to: Optional[str] = None,
    inference_id: Optional[str] = None,
) -> ValidationItem:
    if inference_id is None:
        inference_id = _generate_inference_id()

    inference_resp = inference(
        experiment_request.inference_model,
        experiment_request.request_params,
        experiment_request.prompt,
        inference_id=inference_id
    )
    
    if save_inference_resp_to:
        save_inference_response(inference_resp, save_inference_resp_to)
    
    inference_result = _extract_logprobs(inference_resp)
    enforced_tokens = _extract_enforced_tokens(inference_resp)
    run_seed = _extract_run_seed(inference_resp)

    validation_resp = validation(
        experiment_request.validation_model,
        experiment_request.request_params,
        experiment_request.prompt,
        run_seed=run_seed,
        enforced_tokens=enforced_tokens,
    )
    validation_result = _extract_logprobs(validation_resp)

    if not verify_artifacts(inference_result, validation_result):
        logger.error(
            f"Artifact verification failed\n"
            f"inference: {[r.token for r in inference_result.results]}\n"
            f"validation: {[r.token for r in validation_result.results]}"
        )

    return experiment_request.to_result(
        inference_result,
        validation_result
    )


def save_inference_response(
    inference_resp: Dict[str, Any],
    filepath: str
) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(inference_resp, f, indent=2)
    logger.info(f"Inference response saved to {filepath}")


def load_inference_response(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        inference_resp = json.load(f)
    logger.info(f"Inference response loaded from {filepath}")
    return inference_resp


def generate_and_validate_from_file(
    experiment_request: ExperimentRequest,
    inference_resp_filepath: str
) -> ValidationItem:
    inference_resp = load_inference_response(inference_resp_filepath)
    inference_result = _extract_logprobs(inference_resp)
    enforced_tokens = _extract_enforced_tokens(inference_resp)
    run_seed = _extract_run_seed(inference_resp)

    validation_resp = validation(
        experiment_request.validation_model,
        experiment_request.request_params,
        experiment_request.prompt,
        run_seed=run_seed,
        enforced_tokens=enforced_tokens,
    )
    validation_result = _extract_logprobs(validation_resp)

    if not verify_artifacts(inference_result, validation_result):
        logger.error(
            f"Artifact verification failed\n"
            f"inference: {[r.token for r in inference_result.results]}\n"
            f"validation: {[r.token for r in validation_result.results]}"
        )

    return experiment_request.to_result(
        inference_result,
        validation_result
    )


def token_distance(
    inf_position_logprobs: PositionResult,
    val_position_logprobs: PositionResult
):
    dist = 0
    n_matches = 0
    for k, v in inf_position_logprobs.logprobs.items():
        if k in val_position_logprobs.logprobs:
            n_matches += 1
            dist += abs(v - val_position_logprobs.logprobs[k]) / (1e-10 + abs(v) + abs(val_position_logprobs.logprobs[k])) / 2.
    return dist, n_matches



def _check_match(
    inf_result: Result,
    val_result: Result,
):
    if [r.token for r in inf_result.results] != [r.token for r in val_result.results]:
        logger.debug(
            f"tokens sequences don't match\n" +
            f"inference:\n {[r.token for r in inf_result.results]}\n" +
            f"{'-'*10}\n" +
            f"validation:\n {[r.token for r in val_result.results]}\n" +
            f"{'-'*100}"
        )
        return False
    return True

def distance(
    inf_result: Result,
    val_result: Result,
    distance_func: Callable = token_distance
):

    if not _check_match(inf_result, val_result):
        return -1, -1

    total_dist = 0
    total_n_matches = 0
    for inf_position, val_position in zip(inf_result.results, val_result.results):
        dist, n_matches = distance_func(inf_position, val_position)
        total_dist += dist
        total_n_matches += n_matches
    
    matches_ratio = total_n_matches / (len(inf_result.results)*len(inf_result.results[0].logprobs))
    total_dist /= (len(inf_result.results)*len(inf_result.results[0].logprobs))
    return total_dist, matches_ratio


def token_distance2(
    inf_position_logprobs: PositionResult,
    val_position_logprobs: PositionResult
):
    dist = 0.0
    n_matches = 0

    if not val_position_logprobs.logprobs:
        return len(inf_position_logprobs.logprobs), 0

    sorted_logprobs = sorted(val_position_logprobs.logprobs.values())
    
    if len(sorted_logprobs) >= 2:
        min_val_logprob_1 = sorted_logprobs[0]
        min_val_logprob_2 = sorted_logprobs[1]
    else:
        min_val_logprob_1 = sorted_logprobs[0]
        min_val_logprob_2 = min_val_logprob_1 - 1.0

    for token, inf_logprob in inf_position_logprobs.logprobs.items():
        if token in val_position_logprobs.logprobs:
            val_logprob = val_position_logprobs.logprobs[token]
            n_matches += 1
        else:
            val_logprob = min_val_logprob_1 - (min_val_logprob_2 - min_val_logprob_1)

        denom = 1e-10 + abs(inf_logprob) + abs(val_logprob)
        dist += abs(inf_logprob - val_logprob) / denom / 2.0

    return dist, n_matches


def similarity2(
    inf_result: Result,
    val_result: Result,
):
    dist, matches_ratio = distance2(inf_result, val_result)
    if dist == -1:
        return -1, -1
    return 1 - dist, matches_ratio


def distance2(inf_result: Result, val_result: Result):
    if not _check_match(inf_result, val_result):
        return -1, -1

    total_dist = 0
    total_n_matches = 0
    for inf_position, val_position in zip(inf_result.results, val_result.results):
        dist, n_matches = token_distance2(inf_position, val_position)
        total_dist += dist
        total_n_matches += n_matches
    
    matches_ratio = total_n_matches / (len(inf_result.results)*len(inf_result.results[0].logprobs))
    total_dist = (total_dist + 1.0) / (max(100, len(inf_result.results))*len(inf_result.results[0].logprobs) + 1.0)
    return total_dist, matches_ratio



import numpy as np
from typing import List, Dict
from validation.data import Result

BAD_LOGP = -10.0

def _clean_logprob(lp: float, floor: float = BAD_LOGP) -> float:
    return lp if lp is not None and lp > floor else floor


def get_metric(logprobs: List[float]) -> float:
    if not logprobs:
        return 0.0
    return float(np.exp(np.mean(logprobs)))


def get_metric_from_result(inf_result: Result) -> float:
    per_token_lp: List[float] = []

    for r in inf_result.results:
        lp = r.logprobs.get(r.token, BAD_LOGP)
        per_token_lp.append(_clean_logprob(lp))

    return get_metric(per_token_lp)
