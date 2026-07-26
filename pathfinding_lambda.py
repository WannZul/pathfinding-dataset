import json
import re
from collections import deque

DIRECTIONS = [
    (-1, 0, "up"),
    (1, 0, "down"),
    (0, -1, "left"),
    (0, 1, "right"),
]

MOVE_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

def _parse_start(position):
    """Parse A1, [row, col], and other common coordinate formats."""
    try:
        if isinstance(position, (list, tuple)):
            if len(position) == 1:
                return _parse_start(position[0])

            if len(position) >= 2:
                first = re.sub(
                    r"[^A-Za-z0-9]",
                    "",
                    str(position[0]),
                )
                second = re.sub(
                    r"[^A-Za-z0-9]",
                    "",
                    str(position[1]),
                )

                if first.isalpha():
                    return (
                        int(second) - 1,
                        ord(first.upper()) - ord("A"),
                    )

                return (int(first), int(second))

        value = re.sub(
            r"[^A-Za-z0-9]",
            "",
            str(position),
        )

        match = re.match(
            r"([A-Za-z])([0-9]+)",
            value,
        )

        if match:
            return (
                int(match.group(2)) - 1,
                ord(match.group(1).upper()) - ord("A"),
            )

        numbers = re.findall(
            r"[0-9]+",
            value,
        )

        if len(numbers) >= 2:
            return (
                int(numbers[0]),
                int(numbers[1]),
            )

    except (ValueError, TypeError, IndexError):
        pass

    return (0, 0)

def _cell(board, row, column):
    """Return a normalized cell name."""
    return str(board[row][column]).lower().strip()

def _is_reward(cell):
    """Treat every challenge and coin except c8 as a reward."""
    return cell.startswith("c") and cell != "c8"

def _bfs(board, start, goal, blocked):
    """Find the shortest route between two coordinates."""
    rows = len(board)
    columns = len(board[0])

    queue = deque([
        (start[0], start[1], [])
    ])
    visited = {start}

    while queue:
        row, column, path = queue.popleft()

        if (row, column) == goal:
            return path

        for row_change, column_change, move in DIRECTIONS:
            next_row = row + row_change
            next_column = column + column_change
            next_position = (next_row, next_column)

            if not (
                0 <= next_row < rows
                and 0 <= next_column < columns
            ):
                continue

            if next_position in visited:
                continue

            if _cell(
                board,
                next_row,
                next_column,
            ) in blocked:
                continue

            visited.add(next_position)

            queue.append(
                (
                    next_row,
                    next_column,
                    path + [move],
                )
            )

    return None

def _nearest(board, start, target_test, blocked):
    """Find the nearest reachable cell matching target_test."""
    rows = len(board)
    columns = len(board[0])

    queue = deque([
        (start[0], start[1], [])
    ])
    visited = {start}

    while queue:
        row, column, path = queue.popleft()

        current_cell = _cell(
            board,
            row,
            column,
        )

        if (
            (row, column) != start
            and target_test(current_cell)
        ):
            return path, (row, column)

        for row_change, column_change, move in DIRECTIONS:
            next_row = row + row_change
            next_column = column + column_change
            next_position = (next_row, next_column)

            if not (
                0 <= next_row < rows
                and 0 <= next_column < columns
            ):
                continue

            if next_position in visited:
                continue

            if _cell(
                board,
                next_row,
                next_column,
            ) in blocked:
                continue

            visited.add(next_position)

            queue.append(
                (
                    next_row,
                    next_column,
                    path + [move],
                )
            )

    return None

def _walk_and_clear(board, start, path):
    """Apply a path and clear rewards crossed along the route."""
    row, column = start
    found_key = False

    for move in path:
        row_change, column_change = MOVE_DELTAS[move]

        row += row_change
        column += column_change

        current_cell = _cell(
            board,
            row,
            column,
        )

        if current_cell == "c40":
            found_key = True

        if _is_reward(current_cell):
            board[row][column] = "normal"

    return (row, column), found_key

def swift_path(board, start, treasure):
    """Take the shortest route to treasure."""
    return _bfs(
        board,
        start,
        treasure,
        {"wall"},
    ) or []

def get_coins_path(board, start, treasure):
    """Collect reachable c7 coins safely, then enter treasure."""
    working_board = [
        row[:]
        for row in board
    ]

    position = start
    complete_path = []

    maximum_iterations = (
        len(working_board)
        * len(working_board[0])
    )

    for _ in range(maximum_iterations):
        target = _nearest(
            working_board,
            position,
            lambda cell: cell == "c7",
            {
                "wall",
                "c8",
                "treasure",
                "c30",
            },
        )

        if not target:
            break

        path, _ = target
        complete_path.extend(path)

        position, _ = _walk_and_clear(
            working_board,
            position,
            path,
        )

    final_path = _bfs(
        working_board,
        position,
        treasure,
        {
            "wall",
            "c8",
            "c30",
        },
    )

    if final_path is not None:
        complete_path.extend(final_path)

    return complete_path

def score_hunter_path(board, start, treasure):
    """
    Collect every safe challenge and coin.

    Spikes and treasure stay blocked during collection. The red door stays
    blocked until the red key has been collected. Treasure is entered last.
    """
    working_board = [
        row[:]
        for row in board
    ]

    position = start
    complete_path = []

    have_key = (
        _cell(
            working_board,
            position[0],
            position[1],
        )
        == "c40"
    )

    starting_cell = _cell(
        working_board,
        position[0],
        position[1],
    )

    if _is_reward(starting_cell):
        working_board[
            position[0]
        ][
            position[1]
        ] = "normal"

    maximum_iterations = (
        len(working_board)
        * len(working_board[0])
        * 2
    )

    # Collect every reward reachable without crossing the red door.
    for _ in range(maximum_iterations):
        target = _nearest(
            working_board,
            position,
            lambda cell: (
                _is_reward(cell)
                and cell != "c30"
            ),
            {
                "wall",
                "c8",
                "treasure",
                "c30",
            },
        )

        if not target:
            break

        path, _ = target
        complete_path.extend(path)

        position, found_key = _walk_and_clear(
            working_board,
            position,
            path,
        )

        have_key = have_key or found_key

    # Open every reachable red door after obtaining the key.
    door_opened = False

    if have_key:
        for _ in range(maximum_iterations):
            target = _nearest(
                working_board,
                position,
                lambda cell: cell == "c30",
                {
                    "wall",
                    "c8",
                    "treasure",
                },
            )

            if not target:
                break

            path, _ = target
            complete_path.extend(path)

            position, _ = _walk_and_clear(
                working_board,
                position,
                path,
            )

            door_opened = True

    # Collect rewards behind the door and any remaining rewards.
    collection_blocks = {
        "wall",
        "c8",
        "treasure",
    }

    if not door_opened:
        collection_blocks.add("c30")

    for _ in range(maximum_iterations):
        target = _nearest(
            working_board,
            position,
            lambda cell: (
                _is_reward(cell)
                and cell != "c30"
            ),
            collection_blocks,
        )

        if not target:
            break

        path, _ = target
        complete_path.extend(path)

        position, found_key = _walk_and_clear(
            working_board,
            position,
            path,
        )

        have_key = have_key or found_key

    # Enter treasure only after no safe reward remains.
    final_blocks = {
        "wall",
        "c8",
    }

    if not door_opened:
        final_blocks.add("c30")

    final_path = _bfs(
        working_board,
        position,
        treasure,
        final_blocks,
    )

    if final_path is not None:
        complete_path.extend(final_path)

    return complete_path

def _error(status_code, message):
    return {
        "statusCode": status_code,
        "body": json.dumps({
            "error": message,
        }),
    }

def lambda_handler(event, context):
    """Pathfinding Lambda with selectable navigation strategies."""
    try:
        if not isinstance(event, dict):
            return _error(
                400,
                "Event must be a JSON object",
            )

        body = event.get(
            "body",
            event,
        )

        if isinstance(body, str):
            body = json.loads(body)

        if not isinstance(body, dict):
            return _error(
                400,
                "Request body must be a JSON object",
            )

        game_map = body.get(
            "game_map",
            [],
        )

        if not game_map:
            return _error(
                400,
                "Missing game_map",
            )

        # Preserve the behavior of the previously working version.
        maximum_columns = max(
            len(row)
            for row in game_map
        )

        game_map = [
            row
            + ["normal"]
            * (maximum_columns - len(row))
            for row in game_map
        ]

        map_config = body.get(
            "map_config",
            {},
        )

        if not isinstance(
            map_config,
            dict,
        ):
            map_config = {}

        player_start = (
            map_config.get("playerStart")
            or body.get("playerStart")
            or {}
        )

        if isinstance(
            player_start,
            str,
        ):
            start_position = _parse_start(
                player_start
            )

        elif (
            isinstance(player_start, dict)
            and player_start
        ):
            start_position = (
                int(
                    player_start.get(
                        "row",
                        0,
                    )
                ),
                int(
                    player_start.get(
                        "col",
                        0,
                    )
                ),
            )

        else:
            start_position = _parse_start(
                body.get("start_pos")
                or body.get("start")
                or body.get("position")
                or [0, 0]
            )

        rows = len(game_map)
        columns = len(game_map[0])

        if not (
            0 <= start_position[0] < rows
            and 0 <= start_position[1] < columns
        ):
            start_position = (0, 0)

        treasure = None

        for row in range(rows):
            for column in range(columns):
                if _cell(
                    game_map,
                    row,
                    column,
                ) == "treasure":
                    treasure = (
                        row,
                        column,
                    )
                    break

            if treasure is not None:
                break

        if treasure is None:
            return _error(
                400,
                "No treasure found on map",
            )

        requested_strategy = str(
            body.get(
                "strategy",
                "swift",
            )
        ).lower().strip()

        if any(
            phrase in requested_strategy
            for phrase in (
                "score_hunter",
                "score hunter",
                "score",
                "maximize",
                "all rewards",
                "all challenges",
                "safe",
            )
        ):
            strategy = "score_hunter"

            path = score_hunter_path(
                game_map,
                start_position,
                treasure,
            )

        elif "coin" in requested_strategy:
            strategy = "get_coins"

            path = get_coins_path(
                game_map,
                start_position,
                treasure,
            )

        else:
            strategy = "swift"

            path = swift_path(
                game_map,
                start_position,
                treasure,
            )

        result = {
            "path": path,
            "steps": len(path),
            "start_position": list(
                start_position
            ),
            "strategy": strategy,
        }

        print(
            "RESULT: "
            f"strategy={strategy} "
            f"steps={len(path)} "
            f"start={list(start_position)}"
        )

        return {
            "statusCode": 200,
            "body": json.dumps(result),
        }

    except json.JSONDecodeError as error:
        return _error(
            400,
            f"Invalid JSON body: {error}",
        )

    except Exception as error:
        print(
            "ERROR: "
            f"{type(error).__name__}: {error}"
        )

        return _error(
            500,
            f"{type(error).__name__}: {error}",
        )
