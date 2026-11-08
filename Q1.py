import ast
import heapq
import sys
from typing import List, Tuple

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return dx + dy


def astar(start: Tuple[int, int], goal: Tuple[int, int],
          grid: List[List[str]]) -> List[Tuple[int, int]]:
    return astar_with_stats(start, goal, grid)[0]


def astar_with_stats(start: Tuple[int, int], goal: Tuple[int, int],
                     grid: List[List[str]], use_heuristic: bool = True
                     ) -> Tuple[List[Tuple[int, int]], int]:
    """Return the path and number of expanded non-goal nodes."""
    if not grid or not grid[0]:
        return [], 0

    rows, cols = len(grid), len(grid[0])

    def valid(pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < rows and 0 <= c < cols and grid[r][c] != 'X'

    if not valid(start) or not valid(goal):
        return [], 0

    if start == goal:
        return [start], 0

    start_h = manhattan(start, goal) if use_heuristic else 0
    open_heap = [(start_h, 0, start)]
    g_score = {start: 0}
    came_from = {}
    closed = set()
    expanded = 0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, expanded

        expanded += 1

        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            if not valid(neighbor) or neighbor in closed:
                continue
            tentative_g = g + 1
            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                h = manhattan(neighbor, goal) if use_heuristic else 0
                f = tentative_g + h
                heapq.heappush(open_heap, (f, tentative_g, neighbor))

    return [], expanded


def parse_point(text: str) -> Tuple[int, int]:
    parts = text.replace(',', ' ').replace('(', ' ').replace(')', ' ').split()
    if len(parts) != 2:
        raise ValueError("坐标格式应为: row col")
    return int(parts[0]), int(parts[1])


def parse_map(line: str) -> List[str]:
    tokens = line.split()
    if len(tokens) > 1:
        return tokens
    return list(line.strip())


def read_nonempty_line(prompt: str) -> str:
    while True:
        line = input(prompt).strip()
        if line:
            return line
        prompt = ""


def read_input() -> Tuple[Tuple[int, int], Tuple[int, int], List[List[str]]]:
    start = parse_point(read_nonempty_line("请输入起点坐标 (row,col): "))
    goal = parse_point(read_nonempty_line("请输入终点坐标 (row,col): "))

    map_lines = []
    bracket_depth = 0
    map_started = False
    while True:
        line = input("请输入地图二维列表: " if not map_started else "")
        map_lines.append(line)
        bracket_depth += line.count('[') - line.count(']')
        map_started = map_started or '[' in line
        if map_started and bracket_depth == 0:
            break

    grid = ast.literal_eval('\n'.join(map_lines))
    if not isinstance(grid, list) or not grid or not all(
        isinstance(row, list) for row in grid
    ):
        raise ValueError("地图格式应为二维列表")
    if any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("地图每一行的长度必须相同")
    return start, goal, grid


if __name__ == "__main__":
    try:
        start, goal, grid = read_input()
    except EOFError:
        raise SystemExit("输入未完成，请提供起点、终点和完整的地图二维列表。")
    path = astar(start, goal, grid)
    if path:
        print("最短路径:")
        for r, c in path:
            print(f"({r}, {c})")
    else:
        print("[]")
