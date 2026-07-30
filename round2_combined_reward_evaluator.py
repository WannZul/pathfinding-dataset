import json
import re
from json import JSONDecoder
from typing import Any, Dict, List, Optional

import boto3


PATHFINDING_FUNCTION_NAME = "AgentCoreGatewayTool-Pathfinding"
DIRECTIONS = {"up", "down", "left", "right"}
lambda_client = boto3.client("lambda", region_name="us-east-1")


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(block.get("text", block)) if isinstance(block, dict) else str(block)
            for block in content
        )
    if content is None:
        return ""
    return json.dumps(content) if isinstance(content, dict) else str(content)


def _response(sample: Dict[str, Any]) -> str:
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
        reward_model, dict
    ) else ""
    if not value:
        value = sample.get("reference_answer", "")
        if isinstance(value, dict):
            value = value.get("text", value)
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _arguments(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _tool_call(response: str) -> Optional[Dict[str, Any]]:
    if not response:
        return None
    candidates = []
    for pattern, output_format in (
        (r"<tool_call>\s*(.*?)\s*</tool_call>", "qwen_native"),
        (r"\[TOOL_CALL\]\s*(.*?)\s*\[/TOOL_CALL\]", "legacy_tags"),
    ):
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            candidates.append((match.group(1).strip(), output_format))
    decoder = JSONDecoder()
    for position, character in enumerate(response):
        if character == "{":
            try:
                _, end = decoder.raw_decode(response[position:])
                candidates.append((response[position:position + end], "raw_json"))
            except json.JSONDecodeError:
                pass
    for candidate, output_format in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        name = parsed.get("name", parsed.get("tool"))
        if name is None:
            continue
        return {
            "tool": str(name),
            "parameters": _arguments(
                parsed.get("arguments", parsed.get("parameters", {}))
            ),
            "format": output_format,
        }
    return None


def _invoke(arguments: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = lambda_client.invoke(
            FunctionName=PATHFINDING_FUNCTION_NAME,
            InvocationType="RequestResponse",
            Payload=json.dumps(arguments).encode("utf-8"),
        )
        result = json.loads(response["Payload"].read().decode("utf-8"))
        if response.get("FunctionError"):
            return {"error": "Pathfinding Lambda function error", "details": result}
        return result if isinstance(result, dict) else {"error": "Invalid response"}
    except Exception as error:
        return {"error": str(error)}


def _body(output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    body = output.get("body", {})
    if isinstance(body, dict):
        return body
    if isinstance(body, str):
        try:
            parsed = json.loads(body)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _outputs_match(predicted: Dict[str, Any], expected: Dict[str, Any]) -> float:
    if str(predicted.get("statusCode")) != str(expected.get("statusCode")):
        return 0.0
    predicted_body = _body(predicted)
    expected_body = _body(expected)
    if predicted_body is None or expected_body is None:
        return 0.0
    return float(all(
        predicted_body.get(field) == expected_body.get(field)
        for field in ("path", "steps", "start_position")
    ))


def _sample_id(sample: Dict[str, Any], index: int) -> str:
    extra_info = sample.get("extra_info", {})
    fallback = (
        extra_info.get("index", f"sample-{index:03d}")
        if isinstance(extra_info, dict)
        else f"sample-{index:03d}"
    )
    return str(sample.get("id", fallback))


def _tool_reward(
    sample: Dict[str, Any],
    index: int,
    expected: Dict[str, Any],
) -> Dict[str, Any]:
    response = _response(sample)
    expected_function = expected.get("function", {})
    expected_tool = str(expected_function.get("name", "pathfinding_lambda")).lower()
    expected_input = _arguments(expected_function.get("arguments", {}))
    expected_output = expected.get("output", {})
    predicted = _tool_call(response)

    tool_score = format_score = map_score = pos_score = strategy_score = 0.0
    lambda_success = output_match = 0.0
    if predicted:
        format_score = 1.0
        if predicted["tool"].lower() == expected_tool:
            tool_score = 1.0
        parameters = predicted["parameters"]
        if tool_score and expected_input:
            map_score = float(parameters.get("game_map") == expected_input.get("game_map"))
            pos_score = float(parameters.get("start_pos") == expected_input.get("start_pos"))
            strategy_score = float(
                str(parameters.get("strategy", "")).lower()
                == str(expected_input.get("strategy", "")).lower()
            )
        parameter_score = map_score * 0.50 + pos_score * 0.25 + strategy_score * 0.25
        if tool_score and parameter_score > 0.5:
            actual_output = _invoke(parameters)
            lambda_success = float(
                "error" not in actual_output and actual_output.get("statusCode") == 200
            )
            if expected_output:
                output_match = _outputs_match(actual_output, expected_output)
        reward = (
            0.20 + parameter_score * 0.50 + format_score * 0.10
            + lambda_success * 0.10 + output_match * 0.10
        ) if tool_score else 0.0
    else:
        lower = response.lower()
        reward = min(
            0.10,
            0.03 * float("pathfinding_lambda" in lower)
            + 0.02 * float(any(s in lower for s in ("quickest", "coins_first", "score_hunter")))
            + 0.02 * float("game_map" in lower or "start_pos" in lower)
            + 0.03 * float("{" in response and "}" in response),
        )
        parameter_score = 0.0

    reward = round(max(0.0, min(1.0, reward)), 6)
    metrics = [
        {"name": "tool_task", "value": 1.0, "type": "Metric"},
        {"name": "correct_tool", "value": tool_score, "type": "Reward"},
        {"name": "game_map_match", "value": map_score, "type": "Reward"},
        {"name": "start_pos_match", "value": pos_score, "type": "Reward"},
        {"name": "strategy_match", "value": strategy_score, "type": "Reward"},
        {"name": "format_valid", "value": format_score, "type": "Metric"},
        {"name": "lambda_success", "value": lambda_success, "type": "Metric"},
        {"name": "output_match", "value": output_match, "type": "Metric"},
        {"name": "response_length", "value": float(len(response)), "type": "Metric"},
    ]
    return {
        "id": _sample_id(sample, index),
        "aggregate_reward_score": reward,
        "metrics_list": metrics,
    }


def _result_object(response: str) -> Optional[Dict[str, Any]]:
    if not response:
        return None
    candidates = [response.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    decoder = JSONDecoder()
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
    return None


def _valid_path(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    path = [str(move).lower() for move in value]
    return path if all(move in DIRECTIONS for move in path) else []


def _faithfulness_reward(
    sample: Dict[str, Any],
    index: int,
    expected: Dict[str, Any],
) -> Dict[str, Any]:
    response = _response(sample)
    predicted = _result_object(response)
    expected_path = _valid_path(expected.get("path", []))
    predicted_path = _valid_path(predicted.get("path", [])) if predicted else []
    path_exact = bool(predicted_path) and predicted_path == expected_path
    steps_match = bool(predicted) and predicted.get("steps") == expected.get("steps", len(expected_path))
    start_match = bool(predicted) and predicted.get("start_position") == expected.get("start_position", [])
    contract_exact = path_exact and steps_match and start_match
    compared = min(len(predicted_path), len(expected_path))
    element_accuracy = (
        sum(predicted_path[i] == expected_path[i] for i in range(compared))
        / len(expected_path)
        if expected_path else 0.0
    )
    length_match = bool(predicted_path) and len(predicted_path) == len(expected_path)
    if contract_exact:
        reward = 1.0
    elif path_exact:
        reward = 0.85 + 0.075 * float(steps_match) + 0.075 * float(start_match)
    elif predicted_path and expected_path:
        ratio = min(len(predicted_path), len(expected_path)) / max(len(predicted_path), len(expected_path))
        reward = (0.50 * element_accuracy) if length_match else (0.20 * element_accuracy * ratio)
    elif predicted:
        reward = 0.10
    else:
        reward = 0.05 * float(any(direction in response.lower() for direction in DIRECTIONS))
    metrics = [
        {"name": "faithfulness_task", "value": 1.0, "type": "Metric"},
        {"name": "contract_exact", "value": float(contract_exact), "type": "Reward"},
        {"name": "path_exact", "value": float(path_exact), "type": "Reward"},
        {"name": "steps_match", "value": float(steps_match), "type": "Metric"},
        {"name": "start_position_match", "value": float(start_match), "type": "Metric"},
        {"name": "element_accuracy", "value": float(element_accuracy), "type": "Metric"},
        {"name": "response_length", "value": float(len(response)), "type": "Metric"},
    ]
    return {
        "id": _sample_id(sample, index),
        "aggregate_reward_score": round(max(0.0, min(1.0, reward)), 6),
        "metrics_list": metrics,
    }


def reward_function(sample: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Dispatch each mixed record to its matching deterministic reward."""
    expected = _ground_truth(sample)
    if isinstance(expected.get("function"), dict):
        return _tool_reward(sample, index, expected)
    if all(field in expected for field in ("path", "steps", "start_position")):
        return _faithfulness_reward(sample, index, expected)
    return {
        "id": _sample_id(sample, index),
        "aggregate_reward_score": 0.0,
        "metrics_list": [
            {"name": "recognized_task", "value": 0.0, "type": "Reward"}
        ],
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda Handler with SageMaker Studio test-wrapper support."""
    try:
        if isinstance(event, str):
            event = json.loads(event)
        if isinstance(event, dict):
            if "batch" in event:
                batch = event.get("batch", [])
            elif "body" in event:
                body = event.get("body", {})
                if isinstance(body, str):
                    body = json.loads(body)
                batch = body.get("batch", body.get("input", body)) if isinstance(body, dict) else body
            else:
                batch = event.get("input", event)
        else:
            batch = event
        if isinstance(batch, str):
            batch = json.loads(batch)
        if isinstance(batch, dict):
            batch = batch.get("batch", batch.get("input", [batch]))
        if isinstance(batch, str):
            batch = json.loads(batch)
        if not isinstance(batch, list) or not batch:
            return {"error": "Missing or empty batch"}
        results = []
        for index, sample in enumerate(batch):
            if isinstance(sample, str):
                sample = json.loads(sample)
            if not isinstance(sample, dict):
                return {"error": f"Each batch item must be an object; received {type(sample).__name__}"}
            results.append(reward_function(sample, index))
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
