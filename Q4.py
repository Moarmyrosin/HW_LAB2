import ast
import heapq
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


Point = Tuple[int, int]


def astar_cost(start: Point, goal: Point, grid: List[List[object]]) -> int:
    rows, cols = len(grid), len(grid[0])
    weights = [cell for row in grid for cell in row if isinstance(cell, (int, float))]
    if any(weight < 0 for weight in weights):
        raise ValueError("地形代价不能为负数")
    minimum_weight = min([1, *weights])

    def heuristic(point: Point) -> float:
        return (abs(point[0] - goal[0]) + abs(point[1] - goal[1])) * minimum_weight

    def entry_cost(cell: object) -> float:
        return float(cell) if isinstance(cell, (int, float)) else 1

    queue = [(heuristic(start), 0, start)]
    distances = {start: 0}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        _, cost, point = heapq.heappop(queue)
        if cost != distances.get(point):
            continue
        if point == goal:
            return int(cost)
        for dr, dc in directions:
            neighbor = (point[0] + dr, point[1] + dc)
            row, col = neighbor
            if not (0 <= row < rows and 0 <= col < cols):
                continue
            cell = grid[row][col]
            if cell == "X":
                continue
            next_cost = cost + entry_cost(cell)
            if next_cost < distances.get(neighbor, math.inf):
                distances[neighbor] = next_cost
                heapq.heappush(queue, (next_cost + heuristic(neighbor), next_cost, neighbor))
    return -1


def build_distances(
    positions: Dict[str, Point], grid: List[List[object]]
) -> Dict[Tuple[str, str], int]:
    return {
        (source, destination): astar_cost(positions[source], positions[destination], grid)
        for source in positions
        for destination in positions
    }


@dataclass(frozen=True)
class State:
    position_a: str
    position_b: str
    remaining: Tuple[str, ...]
    utility: int
    maximizing: bool


def minimax(
    state: State,
    depth: int,
    distances: Dict[Tuple[str, str], int],
    values: Dict[str, int],
) -> Tuple[int, Optional[str], int]:
    if depth == 0 or not state.remaining:
        return state.utility, None, 1

    best_value = float("-inf") if state.maximizing else float("inf")
    best_target = None
    visited = 1
    current_position = state.position_a if state.maximizing else state.position_b
    for target in state.remaining:
        gain = values[target] - distances[(current_position, target)]
        next_state = State(
            target if state.maximizing else state.position_a,
            target if not state.maximizing else state.position_b,
            tuple(item for item in state.remaining if item != target),
            state.utility + (gain if state.maximizing else -gain),
            not state.maximizing,
        )
        value, _, child_visited = minimax(next_state, depth - 1, distances, values)
        visited += child_visited
        if (state.maximizing and value > best_value) or (
            not state.maximizing and value < best_value
        ):
            best_value, best_target = value, target
    return int(best_value), best_target, visited


def alpha_beta(
    state: State,
    depth: int,
    distances: Dict[Tuple[str, str], int],
    values: Dict[str, int],
) -> Tuple[int, Optional[str], int, int]:
    visited = 1
    pruned = 0

    def search(
        current: State,
        remaining_depth: int,
        alpha: float,
        beta: float,
    ) -> Tuple[int, Optional[str]]:
        nonlocal visited, pruned
        if remaining_depth == 0 or not current.remaining:
            return current.utility, None

        maximizing = current.maximizing
        best_value = float("-inf") if maximizing else float("inf")
        best_target = None
        current_position = current.position_a if maximizing else current.position_b
        for index, target in enumerate(current.remaining):
            visited += 1
            gain = values[target] - distances[(current_position, target)]
            next_state = State(
                target if maximizing else current.position_a,
                target if not maximizing else current.position_b,
                tuple(item for item in current.remaining if item != target),
                current.utility + (gain if maximizing else -gain),
                not maximizing,
            )
            value, _ = search(next_state, remaining_depth - 1, alpha, beta)
            if (maximizing and value > best_value) or (
                not maximizing and value < best_value
            ):
                best_value, best_target = value, target
            if maximizing:
                alpha = max(alpha, best_value)
            else:
                beta = min(beta, best_value)
            if alpha >= beta:
                pruned += len(current.remaining) - index - 1
                break
        return int(best_value), best_target

    value, target = search(state, depth, float("-inf"), float("inf"))
    return value, target, visited, pruned


def solve(
    depth: int,
    positions: Dict[str, Point],
    values: Dict[str, int],
    distances: Dict[Tuple[str, str], int],
) -> dict:
    targets = tuple(sorted(
        (name for name in positions if name.startswith("T")),
        key=lambda name: int(name[1:]),
    ))
    initial = State("A", "B", targets, 0, True)
    value, target, minimax_nodes = minimax(initial, depth, distances, values)
    ab_value, ab_target, ab_nodes, pruned = alpha_beta(
        initial, depth, distances, values
    )
    return {
        "H": depth,
        "best_target": target,
        "best_value": value,
        "minimax_nodes": minimax_nodes,
        "alpha_beta_target": ab_target,
        "alpha_beta_value": ab_value,
        "alpha_beta_nodes": ab_nodes,
        "alpha_beta_pruned": pruned,
    }


def read_input() -> Tuple[List[List[object]], Dict[str, Point], Dict[str, int], List[int]]:
    print("请输入综合题地图二维列表（障碍用 'X'，A/B 和宝藏格用 T1、T2 等标签）：")
    lines = []
    bracket_depth = 0
    while True:
        line = input()
        lines.append(line)
        bracket_depth += line.count('[') - line.count(']')
        if lines[0].lstrip().startswith('[') and bracket_depth == 0:
            break
    grid = ast.literal_eval("\n".join(lines))
    if not isinstance(grid, list) or not grid or not all(isinstance(row, list) for row in grid):
        raise ValueError("地图格式应为二维列表")
    if any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("地图每一行的长度必须相同")

    positions = {}
    for row_index, row in enumerate(grid):
        for col_index, cell in enumerate(row):
            if cell in {"A", "B"} or (isinstance(cell, str) and cell.startswith("T")):
                if cell in positions:
                    raise ValueError(f"地图中的 {cell} 标签重复")
                positions[cell] = (row_index, col_index)
    if "A" not in positions or "B" not in positions:
        raise ValueError("地图必须包含 A 和 B")

    values_text = input("请输入宝藏价值字典（如 {'T1': 9, 'T2': 6}）: ").strip()
    values = ast.literal_eval(values_text)
    targets = {name for name in positions if name.startswith("T")}
    if not isinstance(values, dict) or set(values) != targets:
        raise ValueError("宝藏价值字典必须为地图中的每个宝藏提供且只提供一个价值")
    if any(not isinstance(value, (int, float)) for value in values.values()):
        raise ValueError("宝藏价值必须是数字")
    values = {name: int(value) for name, value in values.items()}

    depths_text = input("请输入要计算的搜索深度，以逗号分隔（如 2,4）: ").strip()
    depths = [int(item.strip()) for item in depths_text.split(",") if item.strip()]
    if not depths or any(depth < 1 for depth in depths):
        raise ValueError("搜索深度必须是正整数")
    return grid, positions, values, depths


if __name__ == "__main__":
    try:
        grid, positions, values, depths = read_input()
    except EOFError:
        raise SystemExit("输入未完成，请提供地图、宝藏价值和搜索深度。")
    distances = build_distances(positions, grid)
    labels = ["A", "B"] + sorted(
        (name for name in positions if name.startswith("T")),
        key=lambda name: int(name[1:]),
    )
    print(json.dumps({
        "labels": labels,
        "distance_matrix": [
            [distances[(source, destination)] for destination in labels]
            for source in labels
        ],
        "results": [solve(depth, positions, values, distances) for depth in depths],
    }, ensure_ascii=False, indent=2))
