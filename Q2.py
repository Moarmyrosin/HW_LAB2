import ast
import heapq
import math
from typing import List, Tuple, Union


Cell = Union[str, int, float]
Point = Tuple[int, int]


def minimum_cost(start: Point, goal: Point, grid: List[List[Cell]]) -> int:
    """Return the minimum cost to reach goal, or -1 if it is unreachable."""
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        return -1

    rows, cols = len(grid), len(grid[0])

    def in_bounds(point: Point) -> bool:
        row, col = point
        return 0 <= row < rows and 0 <= col < cols

    def traversable(point: Point) -> bool:
        return in_bounds(point) and grid[point[0]][point[1]] != "X"

    if not traversable(start) or not traversable(goal):
        return -1
    if start == goal:
        return 0

    weights = [
        float(cell)
        for row in grid
        for cell in row
        if cell != "X" and str(cell).upper() not in {"S", "E"}
    ]
    if any(weight < 0 for weight in weights):
        raise ValueError("地形代价不能为负数")
    minimum_weight = min(weights, default=0)

    def heuristic(point: Point) -> float:
        steps_before_goal = max(
            0, abs(point[0] - goal[0]) + abs(point[1] - goal[1]) - 1
        )
        return steps_before_goal * minimum_weight

    open_heap = [(heuristic(start), 0, start)]
    best_cost = {start: 0}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        if current_cost != best_cost.get(current):
            continue
        if current == goal:
            return int(current_cost)

        for row_delta, col_delta in directions:
            neighbor = (current[0] + row_delta, current[1] + col_delta)
            if not traversable(neighbor):
                continue
            cell = grid[neighbor[0]][neighbor[1]]
            entry_cost = 0 if str(cell).upper() in {"S", "E"} else float(cell)
            if entry_cost < 0:
                raise ValueError("地形代价不能为负数")
            new_cost = current_cost + entry_cost
            if new_cost < best_cost.get(neighbor, math.inf):
                best_cost[neighbor] = new_cost
                heapq.heappush(open_heap, (new_cost + heuristic(neighbor), new_cost, neighbor))

    return -1


def read_input() -> Tuple[Point, Point, List[List[Cell]]]:
    def read_point(prompt: str) -> Point:
        while True:
            text = input(prompt).strip()
            if text:
                parts = text.replace(',', ' ').replace('(', ' ').replace(')', ' ').split()
                if len(parts) != 2:
                    raise ValueError("坐标格式应为 (row,col)")
                return int(parts[0]), int(parts[1])
            prompt = ""

    start = read_point("请输入起点坐标 (row,col): ")
    goal = read_point("请输入终点坐标 (row,col): ")
    print("请输入地图二维列表（障碍用 'X'，数字表示进入代价，可用 S/E 标记起终点）：")

    lines = []
    bracket_depth = 0
    while True:
        line = input()
        lines.append(line)
        bracket_depth += line.count('[') - line.count(']')
        if lines[0].lstrip().startswith('[') and bracket_depth == 0:
            break

    class MarkerNames(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id in {"S", "E", "X"}:
                return ast.copy_location(ast.Constant(node.id), node)
            return node

    map_expression = ast.parse('\n'.join(lines), mode="eval")
    map_expression = MarkerNames().visit(map_expression)
    grid = ast.literal_eval(ast.fix_missing_locations(map_expression))
    if not isinstance(grid, list) or not grid or not all(isinstance(row, list) for row in grid):
        raise ValueError("地图格式应为二维列表")
    if any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("地图每一行的长度必须相同")
    return start, goal, grid


if __name__ == "__main__":
    try:
        start, goal, grid = read_input()
    except EOFError:
        raise SystemExit("输入未完成，请提供起点、终点和完整的地图二维列表。")
    print(minimum_cost(start, goal, grid))