import json
import re
from json import JSONDecoder
from typing import Any, Dict, Optional

import boto3


PATHFINDING_FUNCTION_NAME = "AgentCoreGatewayTool-Pathfinding"
lambda_client = boto3.client("lambda", region_name="us-east-1")


def _normalize_arguments(arguments: Any) -> Dict[str, Any]:
    """Return tool arguments as a dictionary."""
    if isinstance(arguments, dict):
        return arguments

    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    return {}


def _normalized_tool_call(parsed: Any, output_format: str) -> Optional[Dict[str, Any]]:
    """Normalize supported tool-call JSON shapes."""
    if not isinstance(parsed, dict):
        return None

    name = parsed.get("name", parsed.get("tool", ""))
    arguments = parsed.get("arguments", parsed.get("parameters", {}))

    return {
        "tool": str(name),
        "parameters": _normalize_arguments(arguments),
        "format": output_format,
    }


def extract_tool_call(response: str) -> Optional[Dict[str, Any]]:
    """Extract Qwen, legacy, or raw-JSON tool calls from model output."""
    if not response:
        return None

    match = re.search(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        try:
            return _normalized_tool_call(
                json.loads(match.group(1).strip()),
                "qwen_native",
            )
        except json.JSONDecodeError:
            pass

    match = re.search(
        r"\[TOOL_CALL\]\s*(.*?)\s*\[/TOOL_CALL\]",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        try:
            return _normalized_tool_call(
                json.loads(match.group(1).strip()),
                "legacy_tags",
            )
        except json.JSONDecodeError:
            pass

    # Decode complete JSON objects instead of using a flat regex that breaks
    # when the arguments object contains nested maps and arrays.
    decoder = JSONDecoder()
    for position, character in enumerate(response):
        if character != "{":
            continue

        try:
            parsed, _ = decoder.raw_decode(response[position:])
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict) and ("name" in parsed or "tool" in parsed):
            return _normalized_tool_call(parsed, "raw_json")

    return None


def extract_text_signals(response: str) -> Dict[str, bool]:
    """Extract gradual-reward signals when no valid tool call is found."""
    signals = {
        "mentions_tool_name": False,
        "mentions_strategy": False,
        "mentions_params": False,
        "has_json": False,
        "has_tool_tags": False,
    }
    if not response:
        return signals

    lower = response.lower()
    signals["mentions_tool_name"] = "pathfinding_lambda" in lower
    signals["mentions_strategy"] = any(
        strategy in lower
        for strategy in ("quickest", "coins_first", "score_hunter")
    )
    signals["mentions_params"] = (
        "game_map" in lower or "start_pos" in lower
    )
    signals["has_json"] = "{" in response and "}" in response
    signals["has_tool_tags"] = (
        "<tool_call>" in lower or "[tool_call]" in lower
    )
    return signals


def invoke_lambda(function_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke the live Pathfinding Lambda and return its decoded response."""
    try:
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload).encode("utf-8"),
        )
        result = json.loads(response["Payload"].read().decode("utf-8"))

        if response.get("FunctionError"):
            return {
                "error": "Pathfinding Lambda returned a function error",
                "details": result,
            }

        return result if isinstance(result, dict) else {"error": "Invalid Lambda response"}
    except Exception as error:
        return {"error": str(error)}


def _parse_body(output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Decode an API-style Lambda response body."""
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


def compare_lambda_outputs(
    predicted_output: Dict[str, Any],
    expected_output: Dict[str, Any],
) -> float:
    """Return full credit only when the complete Pathfinding contract matches."""
    if str(predicted_output.get("statusCode")) != str(
        expected_output.get("statusCode")
    ):
        return 0.0

    predicted_body = _parse_body(predicted_output)
    expected_body = _parse_body(expected_output)
    if predicted_body is None or expected_body is None:
        return 0.0

    required_fields = ("path", "steps", "start_position")
    return float(
        all(
            predicted_body.get(field) == expected_body.get(field)
            for field in required_fields
        )
    )


def _content_to_text(content: Any) -> str:
    """Normalize common assistant-content representations to text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block)))
            else:
                parts.append(str(block))
        return " ".join(parts)

    if content is None:
        return ""

    return json.dumps(content) if isinstance(content, dict) else str(content)


def _extract_response(sample: Dict[str, Any]) -> str:
    """Extract the generated assistant response from an evaluator sample."""
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


def _parse_ground_truth(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the dataset's reward_model.ground_truth value."""
    reward_model = sample.get("reward_model", {})
    value: Any = reward_model.get("ground_truth", "") if isinstance(
        reward_model, dict
    ) else ""

    if not value:
        reference = sample.get("reference_answer", "")
        if isinstance(reference, dict):
            value = reference.get("text", reference)
        else:
            value = reference

    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def reward_function(sample: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Score Pathfinding tool-call format, arguments, and live execution."""
    response = _extract_response(sample)
    ground_truth = _parse_ground_truth(sample)

    expected_function = ground_truth.get("function", {})
    expected_tool = str(expected_function.get("name", "pathfinding_lambda")).lower()
    expected_input = _normalize_arguments(expected_function.get("arguments", {}))
    if not expected_input and isinstance(ground_truth.get("input"), dict):
        expected_input = ground_truth["input"]
    expected_output = ground_truth.get("output", {})

    predicted = extract_tool_call(response)
    text_signals = extract_text_signals(response)

    text_reward = 0.0
    tool_score = 0.0
    format_score = 0.0
    map_score = 0.0
    pos_score = 0.0
    strategy_score = 0.0
    param_score = 0.0
    lambda_success_score = 0.0
    output_match_score = 0.0

    if not predicted:
        if text_signals["has_tool_tags"]:
            text_reward = 0.15
        else:
            if text_signals["mentions_tool_name"]:
                text_reward += 0.03
            if text_signals["mentions_strategy"]:
                text_reward += 0.02
            if text_signals["mentions_params"]:
                text_reward += 0.02
            if text_signals["has_json"]:
                text_reward += 0.03
            text_reward = min(text_reward, 0.10)
    else:
        format_score = 1.0
        tool_name = predicted.get("tool", "").lower()
        if tool_name == expected_tool:
            tool_score = 1.0

        predicted_parameters = predicted.get("parameters", {})
        if tool_score > 0 and expected_input:
            if predicted_parameters.get("game_map") == expected_input.get("game_map"):
                map_score = 1.0
            if predicted_parameters.get("start_pos") == expected_input.get("start_pos"):
                pos_score = 1.0

            predicted_strategy = str(
                predicted_parameters.get("strategy", "")
            ).lower()
            expected_strategy = str(expected_input.get("strategy", "")).lower()
            if predicted_strategy == expected_strategy:
                strategy_score = 1.0

            param_score = (
                map_score * 0.50
                + pos_score * 0.25
                + strategy_score * 0.25
            )

        if tool_score > 0 and param_score > 0.5:
            lambda_output = invoke_lambda(
                PATHFINDING_FUNCTION_NAME,
                predicted_parameters,
            )
            if (
                "error" not in lambda_output
                and lambda_output.get("statusCode") == 200
            ):
                lambda_success_score = 1.0

            if expected_output:
                output_match_score = compare_lambda_outputs(
                    lambda_output,
                    expected_output,
                )

    if not predicted:
        aggregate_reward = text_reward
    elif tool_score == 0.0:
        aggregate_reward = 0.0
    else:
        aggregate_reward = (
            0.20
            + param_score * 0.50
            + format_score * 0.10
            + lambda_success_score * 0.10
            + output_match_score * 0.10
        )

    aggregate_reward = round(
        max(0.0, min(1.0, aggregate_reward)),
        6,
    )

    metrics = [
        {"name": "correct_tool", "value": float(tool_score), "type": "Reward"},
        {"name": "game_map_match", "value": float(map_score), "type": "Reward"},
        {"name": "start_pos_match", "value": float(pos_score), "type": "Reward"},
        {"name": "strategy_match", "value": float(strategy_score), "type": "Reward"},
        {"name": "format_valid", "value": float(format_score), "type": "Metric"},
        {
            "name": "lambda_success",
            "value": float(lambda_success_score),
            "type": "Metric",
        },
        {
            "name": "output_match",
            "value": float(output_match_score),
            "type": "Metric",
        },
        {"name": "text_reward", "value": float(text_reward), "type": "Metric"},
        {
            "name": "response_length",
            "value": float(len(response)),
            "type": "Metric",
        },
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
        "aggregate_reward_score": float(aggregate_reward),
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
