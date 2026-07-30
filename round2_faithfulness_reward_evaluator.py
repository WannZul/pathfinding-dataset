import json
import re
from typing import Any, Dict, List, Optional


DIRECTIONS = {"up", "down", "left", "right"}


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(block.get("text", block))
            if isinstance(block, dict)
            else str(block)
            for block in content
        )
    if content is None:
        return ""
    return json.dumps(content) if isinstance(content, dict) else str(content)


def _extract_response(sample: Dict[str, Any]) -> str:
    messages = sample.get("messages", sample.get("prompt", []))
    response = ""
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "assistant":
                response = _content_to_text(message.get("content", ""))
    if response:
        return response

    for key in ("response", "completion", "generated_output", "output"):
        if key in sample:
            return _content_to_text(sample.get(key))
    return ""


def _ground_truth(sample: Dict[str, Any]) -> Dict[str, Any]:
    reward_model = sample.get("reward_model", {})
    value: Any = reward_model.get("ground_truth", "") if isinstance(
        reward_model,
        dict,
    ) else ""

    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def extract_output(response: str) -> Optional[Dict[str, Any]]:
    """Extract a Pathfinding result object from model output."""
    if not response:
        return None

    candidates = [response.strip()]
    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        for position, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[position:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "path" in parsed:
                return parsed

    # Partial-credit fallback for a bare direction array.
    array_match = re.search(
        r'\[(?:\s*"(?:up|down|left|right)"\s*,?\s*)*\]',
        response,
        re.IGNORECASE,
    )
    if array_match:
        try:
            path = json.loads(array_match.group(0))
            return {"path": path}
        except json.JSONDecodeError:
            pass

    return None


def _valid_path(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    path = [str(move).lower() for move in value]
    return path if all(move in DIRECTIONS for move in path) else []


def reward_function(sample: Dict[str, Any], index: int) -> Dict[str, Any]:
    response = _extract_response(sample)
    expected = _ground_truth(sample)
    predicted = extract_output(response)

    expected_path = _valid_path(expected.get("path", []))
    predicted_path = _valid_path(predicted.get("path", [])) if predicted else []
    expected_steps = expected.get("steps", len(expected_path))
    expected_start = expected.get("start_position", [])

    has_object = predicted is not None
    has_path = bool(predicted_path)
    path_exact = predicted_path == expected_path and has_path
    steps_match = bool(predicted) and predicted.get("steps") == expected_steps
    start_match = bool(predicted) and predicted.get("start_position") == expected_start
    contract_exact = path_exact and steps_match and start_match

    predicted_length = len(predicted_path)
    expected_length = len(expected_path)
    length_match = predicted_length == expected_length and has_path
    compared_length = min(predicted_length, expected_length)
    matches = sum(
        predicted_path[position] == expected_path[position]
        for position in range(compared_length)
    )
    element_accuracy = matches / expected_length if expected_length else 0.0
    dropped_steps = max(0, expected_length - predicted_length)
    added_steps = max(0, predicted_length - expected_length)

    if contract_exact:
        aggregate_reward = 1.0
    elif path_exact:
        aggregate_reward = 0.85 + 0.075 * float(steps_match) + 0.075 * float(start_match)
    elif has_path and expected_path:
        length_ratio = min(predicted_length, expected_length) / max(
            predicted_length,
            expected_length,
        )
        if length_match:
            aggregate_reward = 0.50 * element_accuracy
        else:
            aggregate_reward = 0.20 * element_accuracy * length_ratio
    elif has_object:
        aggregate_reward = 0.10
    elif any(direction in response.lower() for direction in DIRECTIONS):
        aggregate_reward = 0.05
    else:
        aggregate_reward = 0.0

    aggregate_reward = round(max(0.0, min(1.0, aggregate_reward)), 6)
    metrics = [
        {"name": "contract_exact", "value": float(contract_exact), "type": "Reward"},
        {"name": "path_exact", "value": float(path_exact), "type": "Reward"},
        {"name": "steps_match", "value": float(steps_match), "type": "Metric"},
        {"name": "start_position_match", "value": float(start_match), "type": "Metric"},
        {"name": "has_object", "value": float(has_object), "type": "Metric"},
        {"name": "length_match", "value": float(length_match), "type": "Metric"},
        {"name": "element_accuracy", "value": float(element_accuracy), "type": "Metric"},
        {"name": "expected_steps", "value": float(expected_length), "type": "Metric"},
        {"name": "predicted_steps", "value": float(predicted_length), "type": "Metric"},
        {"name": "dropped_steps", "value": float(dropped_steps), "type": "Metric"},
        {"name": "added_steps", "value": float(added_steps), "type": "Metric"},
        {"name": "response_length", "value": float(len(response)), "type": "Metric"},
    ]

    extra_info = sample.get("extra_info", {})
    sample_id = sample.get(
        "id",
        extra_info.get("index", f"sample-{index:03d}")
        if isinstance(extra_info, dict)
        else f"sample-{index:03d}",
    )
    return {
        "id": str(sample_id),
        "aggregate_reward_score": aggregate_reward,
        "metrics_list": metrics,
    }


# Keep SageMaker Studio's generated handler unchanged.
def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda Handler for reward function."""
    try:
        batch = event.get("input", event) if isinstance(event, dict) else event
        if "batch" in event:
            batch = event.get("batch", [])
        elif "body" in event:
            body = json.loads(event.get("body", "{}"))
            batch = body.get("batch", [])

        if not batch:
            return {"error": "Missing or empty batch"}

        results = []
        for i, sample in enumerate(batch):
            try:
                result = reward_function(sample, i)
                results.append(result)
            except Exception as error:
                return {"error": str(error)}

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(results),
        }
    except Exception as error:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(error)}),
        }
