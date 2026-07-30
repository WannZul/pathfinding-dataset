"""Generate deterministic Round 2 RLVR datasets from the verified Round 1 corpus."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pathfinding_lambda


ROOT = Path(__file__).resolve().parent
TOOL_TRAIN_SOURCE = ROOT / "training-dataset.jsonl"
TOOL_VALIDATION_SOURCE = ROOT / "validation-dataset.jsonl"
TOOL_TRAIN_OUTPUT = ROOT / "round2-training-dataset.jsonl"
TOOL_VALIDATION_OUTPUT = ROOT / "round2-validation-dataset.jsonl"
FAITHFULNESS_TRAIN_OUTPUT = ROOT / "round2-faithfulness-training.jsonl"
FAITHFULNESS_VALIDATION_OUTPUT = ROOT / "round2-faithfulness-validation.jsonl"

TOOL_SYSTEM_PROMPT = """Output ONLY a tool call to find a safe path on the map:

<tool_call>
{"name": "pathfinding_lambda", "arguments": {"game_map": <2d_array>, "start_pos": [row,col], "strategy": "quickest|coins_first|score_hunter"}}
</tool_call>"""

TOOL_DESCRIPTION = (
    "AWS Lambda pathfinding function for a 2D game map. Strategies: "
    "'quickest' finds the shortest safe route to treasure; 'coins_first' "
    "collects coins and standard challenges before treasure; 'score_hunter' "
    "collects all reachable rewards, obtains red key c40 before red door c30, "
    "obtains green key c41 before green door c31, minimizes spike c8 "
    "crossings, and enters treasure last. Cells may include start, normal, "
    "wall, treasure, c1-c8, "
    "c17, c30, c31, c40, and c41."
)

FAITHFULNESS_SYSTEM_PROMPT = (
    "You are a tool-output relay. Return the supplied JSON object exactly as-is. "
    "Do not drop, add, reorder, explain, summarize, or wrap any field."
)

Coordinate = Tuple[int, int]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_source_arguments(record: Dict[str, Any]) -> Dict[str, Any]:
    ground_truth = json.loads(record["reward_model"]["ground_truth"])
    arguments = ground_truth["function"]["arguments"]
    return json.loads(arguments) if isinstance(arguments, str) else arguments


def trace_path(
    board: Sequence[Sequence[str]],
    start: Coordinate,
    path: Sequence[str],
) -> Tuple[Coordinate, List[Coordinate], List[str]]:
    position = start
    positions: List[Coordinate] = []
    cells: List[str] = []

    for move in path:
        row_change, column_change = pathfinding_lambda.MOVE_DELTAS[move]
        position = (
            position[0] + row_change,
            position[1] + column_change,
        )
        positions.append(position)
        cells.append(str(board[position[0]][position[1]]).lower().strip())

    return position, positions, cells


def find_treasure(board: Sequence[Sequence[str]]) -> Coordinate:
    for row_index, row in enumerate(board):
        for column_index, value in enumerate(row):
            if str(value).lower().strip() == "treasure":
                return row_index, column_index
    raise ValueError("Map has no treasure")


def lambda_result(arguments: Dict[str, Any]) -> Dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        result = pathfinding_lambda.lambda_handler(arguments, None)
    if result.get("statusCode") != 200:
        raise ValueError(f"Pathfinding failed: {result}")
    return result


def result_body(result: Dict[str, Any]) -> Dict[str, Any]:
    body = result.get("body", {})
    return json.loads(body) if isinstance(body, str) else body


def reaches_treasure(arguments: Dict[str, Any], result: Dict[str, Any]) -> bool:
    board = arguments["game_map"]
    start = tuple(arguments["start_pos"])
    path = result_body(result).get("path", [])
    if not path:
        return start == find_treasure(board)

    final_position, _, _ = trace_path(board, start, path)
    return final_position == find_treasure(board)


def base_score_hunter_positions(
    board: Sequence[Sequence[str]],
    start: Coordinate,
) -> List[Coordinate]:
    arguments = {
        "game_map": copy.deepcopy(board),
        "start_pos": list(start),
        "strategy": "score_hunter",
    }
    result = lambda_result(arguments)
    path = result_body(result).get("path", [])
    _, positions, _ = trace_path(board, start, path)

    candidates: List[Coordinate] = []
    seen = set()
    for position in positions:
        if position in seen:
            continue
        seen.add(position)
        if str(board[position[0]][position[1]]).lower().strip() == "normal":
            candidates.append(position)
    return candidates


def augment_round2_map(
    board: Sequence[Sequence[str]],
    start: Coordinate,
    seed: int,
    include_keyed_doors: bool,
) -> List[List[str]]:
    original = [list(row) for row in board]
    candidates = base_score_hunter_positions(original, start)

    if not candidates:
        return original

    randomizer = random.Random(seed)
    middle = candidates[1:-1] if len(candidates) > 2 else candidates[:]
    randomizer.shuffle(middle)

    if not include_keyed_doors:
        augmented = [row[:] for row in original]
        row, column = middle[0] if middle else candidates[0]
        augmented[row][column] = "c17"
        return augmented

    if len(middle) < 5:
        return original

    # Keys are selected from the first half of the original safe traversal and
    # doors from the second half. Candidate permutations are validated later.
    ordered = [position for position in candidates if position in set(middle)]
    split = max(2, len(ordered) // 2)
    early = ordered[:split]
    late = ordered[split:]
    if len(early) < 2 or len(late) < 3:
        return original

    for _ in range(40):
        key_positions = randomizer.sample(early, 2)
        late_positions = randomizer.sample(late, 3)
        augmented = [row[:] for row in original]

        placements = {
            key_positions[0]: "c40",
            key_positions[1]: "c41",
            late_positions[0]: "c30",
            late_positions[1]: "c31",
            late_positions[2]: "c17",
        }
        for (row, column), value in placements.items():
            augmented[row][column] = value

        arguments = {
            "game_map": augmented,
            "start_pos": list(start),
            "strategy": "score_hunter",
        }
        result = lambda_result(arguments)
        if not reaches_treasure(arguments, result):
            continue

        path = result_body(result)["path"]
        _, positions, crossed_cells = trace_path(augmented, start, path)
        first_visit = {
            cell: crossed_cells.index(cell)
            for cell in ("c40", "c41", "c30", "c31", "c17")
            if cell in crossed_cells
        }
        if len(first_visit) != 5:
            continue
        if first_visit["c40"] >= first_visit["c30"]:
            continue
        if first_visit["c41"] >= first_visit["c31"]:
            continue
        if positions[-1] != find_treasure(augmented):
            continue
        return augmented

    # Keep c17 coverage even when the topology cannot safely support both
    # keyed-door pairs without creating an unreachable map.
    augmented = [row[:] for row in original]
    row, column = middle[0]
    augmented[row][column] = "c17"
    return augmented


def strategy_for_index(index: int) -> str:
    # Half the corpus teaches the score-maximizing Round 2 strategy, while the
    # other half preserves shortest-path and coins-first generalization.
    return ("score_hunter", "score_hunter", "quickest", "coins_first")[index % 4]


def position_label(start: Sequence[int]) -> str:
    row, column = start
    return f"{chr(ord('A') + column)}{row + 1}"


def build_user_prompt(
    board: Sequence[Sequence[str]],
    start: Sequence[int],
    strategy: str,
) -> str:
    position = position_label(start)
    return (
        f"Find a safe path from position {position}, where the position is "
        "formatted as {column}{row}. {column} is a letter starting with A, "
        "and {row} is a number starting with 1. Map coordinates use "
        "[rowIndex,columnIndex], both starting with 0. The path should process "
        "the map according to the requested strategy and reach treasure on "
        f"this map: {json.dumps(board)}. Use strategy: {strategy}."
    )


def tool_schema() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "pathfinding_lambda",
                "description": TOOL_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "game_map": {
                            "type": "array",
                            "description": (
                                "2D map containing start, normal, wall, treasure, "
                                "c1-c8, c17, c30, c31, c40, and c41 cells"
                            ),
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "start_pos": {
                            "type": "array",
                            "description": "Starting [row, column] coordinates",
                            "items": {"type": "number"},
                        },
                        "strategy": {
                            "type": "string",
                            "description": "Navigation strategy",
                            "enum": [
                                "quickest",
                                "coins_first",
                                "score_hunter",
                            ],
                        },
                    },
                    "required": ["game_map", "start_pos", "strategy"],
                },
            },
        }
    ]


def build_tool_record(
    source_record: Dict[str, Any],
    index: int,
    split: str,
    variant: int,
) -> Dict[str, Any]:
    source_arguments = parse_source_arguments(source_record)
    start = list(source_arguments["start_pos"])
    strategy = strategy_for_index(index)
    board = augment_round2_map(
        source_arguments["game_map"],
        tuple(start),
        seed=(index + 1) * 10_007 + variant * 97 + (0 if split == "train" else 1),
        include_keyed_doors=strategy == "score_hunter",
    )

    arguments = {
        "game_map": board,
        "start_pos": start,
        "strategy": strategy,
    }
    output = lambda_result(arguments)
    if not reaches_treasure(arguments, output):
        raise ValueError("Generated route does not safely reach treasure")

    ground_truth = {
        "tool_call_id": f"round2-{split}-{index:04d}",
        "type": "function",
        "function": {
            "name": "pathfinding_lambda",
            "arguments": json.dumps(arguments),
        },
        "output": output,
    }

    return {
        "data_source": "agentcore_gateway_tools_round2",
        "prompt": [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(board, start, strategy),
            },
        ],
        "ability": "tool_use",
        "reward_model": {
            "ground_truth": json.dumps(ground_truth),
            "style": "rule",
        },
        "extra_info": {
            "index": index,
            "split": split,
            "tool": "pathfinding_lambda",
            "map_size": len(board),
            "strategy": strategy,
            "position": position_label(start),
            "round": 2,
        },
        "tools": tool_schema(),
    }


def build_tool_split(
    source_records: Sequence[Dict[str, Any]],
    size: int,
    split: str,
) -> List[Dict[str, Any]]:
    generated: List[Dict[str, Any]] = []
    identities = set()

    for index in range(size):
        last_error: Exception | None = None
        for variant in range(len(source_records) * 2):
            source = source_records[(index + variant) % len(source_records)]
            try:
                record = build_tool_record(source, index, split, variant)
                ground_truth = json.loads(
                    record["reward_model"]["ground_truth"]
                )
                arguments = json.loads(
                    ground_truth["function"]["arguments"]
                )
                identity = json.dumps(
                    {
                        "game_map": arguments["game_map"],
                        "start_pos": arguments["start_pos"],
                        "strategy": arguments["strategy"],
                    },
                    sort_keys=True,
                )
                if identity in identities:
                    continue

                identities.add(identity)
                generated.append(record)
                break
            except (ValueError, IndexError) as error:
                last_error = error
        else:
            raise RuntimeError(
                f"Could not generate {split} record {index}: {last_error}"
            )

    return generated


def build_faithfulness_record(
    tool_record: Dict[str, Any],
    index: int,
    split: str,
) -> Dict[str, Any]:
    ground_truth = json.loads(tool_record["reward_model"]["ground_truth"])
    output = result_body(ground_truth["output"])
    exact_output = {
        "path": output.get("path", []),
        "steps": output.get("steps", 0),
        "start_position": output.get("start_position", []),
    }

    source_info = tool_record["extra_info"]
    return {
        "data_source": "pathfinding_faithfulness_round2",
        "prompt": [
            {"role": "system", "content": FAITHFULNESS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Return this tool output exactly:\n"
                    + json.dumps(exact_output, separators=(",", ":"))
                ),
            },
        ],
        "ability": "tool_output_faithfulness",
        "reward_model": {
            "ground_truth": json.dumps(exact_output, separators=(",", ":")),
            "style": "exact_match",
        },
        "extra_info": {
            "index": index,
            "source_index": source_info["index"],
            "split": split,
            "tool": "pathfinding_lambda",
            "map_size": source_info["map_size"],
            "strategy": source_info["strategy"],
            "position": source_info["position"],
            "round": 2,
        },
    }


def choose_faithfulness_records(
    records: Sequence[Dict[str, Any]],
    size: int,
    split: str,
) -> List[Dict[str, Any]]:
    # Interleave strategies and path lengths rather than selecting only the
    # first records. Paths longer than 250 moves are excluded to stay safely
    # under the workshop's 500-token filtering guidance.
    eligible = []
    for record in records:
        ground_truth = json.loads(record["reward_model"]["ground_truth"])
        path = result_body(ground_truth["output"]).get("path", [])
        if path and len(path) <= 250:
            eligible.append((len(path), record))

    eligible.sort(key=lambda item: item[0])
    if len(eligible) < size:
        raise RuntimeError(
            f"Only {len(eligible)} eligible faithfulness records for {size} requested"
        )

    step = len(eligible) / size
    selected = [eligible[int(offset * step)][1] for offset in range(size)]
    return [
        build_faithfulness_record(record, index, split)
        for index, record in enumerate(selected)
    ]


def main() -> None:
    train_source = load_jsonl(TOOL_TRAIN_SOURCE)
    validation_source = load_jsonl(TOOL_VALIDATION_SOURCE)

    tool_train = build_tool_split(train_source, 500, "train")
    tool_validation = build_tool_split(validation_source, 100, "validation")
    faithfulness_train = choose_faithfulness_records(tool_train, 403, "train")
    faithfulness_validation = choose_faithfulness_records(
        tool_validation,
        81,
        "validation",
    )

    write_jsonl(TOOL_TRAIN_OUTPUT, tool_train)
    write_jsonl(TOOL_VALIDATION_OUTPUT, tool_validation)
    write_jsonl(FAITHFULNESS_TRAIN_OUTPUT, faithfulness_train)
    write_jsonl(FAITHFULNESS_VALIDATION_OUTPUT, faithfulness_validation)

    for path, records in (
        (TOOL_TRAIN_OUTPUT, tool_train),
        (TOOL_VALIDATION_OUTPUT, tool_validation),
        (FAITHFULNESS_TRAIN_OUTPUT, faithfulness_train),
        (FAITHFULNESS_VALIDATION_OUTPUT, faithfulness_validation),
    ):
        print(f"{path.name}: {len(records)}")


if __name__ == "__main__":
    main()
