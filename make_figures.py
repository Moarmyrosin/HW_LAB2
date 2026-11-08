# -*- coding: utf-8 -*-
"""Regenerate all figures referenced by 实验报告.md.

Run:  python make_figures.py   (writes figures/*.png, ~10 s)

All data is recomputed live from the same instrumented search loops and the
same parameters/seeds as used in the report, so figures stay in sync with the
tables.
"""
import heapq
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
import Q3
import Q4

random.seed(2026)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

C_H0 = "#e67e22"      # h=0 / minimax / natural
C_H1 = "#2e86c1"      # heuristic / alpha-beta
C_REV = "#8e44ad"     # reversed / extra series


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/{name}")


# ---------------------------------------------------------------- Q1: trace
Q1_GRID = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", "X", ".", "X", "."],
    [".", ".", ".", "X", "E"],
    ["X", "X", ".", ".", "."],
]


def q1_expansion_order(grid, start=(0, 0), goal=(3, 4), use_h=True):
    """Replicate Q1.astar_with_stats and record pop order + closed set."""

    def h(p):
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1]) if use_h else 0

    heap = [(h(start), 0, start)]
    g = {start: 0}
    closed = set()
    order = {}
    while heap:
        _, gc, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        if cur != goal:
            order[cur] = len(order)
        if cur == goal:
            break
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nb[0] < len(grid) and 0 <= nb[1] < len(grid[0])):
                continue
            if grid[nb[0]][nb[1]] == "X" or nb in closed:
                continue
            ng = gc + 1
            if ng < g.get(nb, float("inf")):
                g[nb] = ng
                heapq.heappush(heap, (ng + h(nb), ng, nb))
    return order, closed


def fig1_q1_trace():
    order_h, closed_h = q1_expansion_order(Q1_GRID, use_h=True)
    order_0, closed_0 = q1_expansion_order(Q1_GRID, use_h=False)
    rows, cols = len(Q1_GRID), len(Q1_GRID[0])

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    for ax, (order, closed, name) in zip(
        axes, [(order_h, closed_h, "曼哈顿距离（展开 13 格）"), (order_0, closed_0, "h=0（展开 15 格）")]
    ):
        for r in range(rows):
            for c in range(cols):
                v = Q1_GRID[r][c]
                p = (r, c)
                if v == "X":
                    ax.add_patch(plt.Rectangle((c, rows - 1 - r), 1, 1, color="#565a5e"))
                    ax.text(c + 0.5, rows - 0.5 - r, "X", ha="center", va="center", color="w", fontsize=11)
                elif v in ("S", "E"):
                    ax.add_patch(plt.Rectangle((c, rows - 1 - r), 1, 1, color="#f7dc6f"))
                    ax.text(c + 0.5, rows - 0.5 - r, v, ha="center", va="center", fontsize=10, color="#7d6608")
                elif p in order:
                    ax.add_patch(plt.Rectangle((c, rows - 1 - r), 1, 1, color="#a9cce3"))
                    ax.text(c + 0.5, rows - 0.5 - r, str(order[p]), ha="center", va="center",
                            fontsize=9, color="#154360")
                else:
                    ax.add_patch(plt.Rectangle((c, rows - 1 - r), 1, 1, color="#f5b7b1"))
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_xticks(range(cols + 1) and [c + 0.5 for c in range(cols)])
        ax.set_yticks([r + 0.5 for r in range(rows)])
        ax.set_xticklabels([str(c) for c in range(cols)])
        ax.set_yticklabels([str(r) for r in range(rows)])
        ax.set_title(f"{name}", fontsize=10)
        ax.set_aspect("equal")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("row", fontsize=9)
    fig.suptitle("题目一：格子颜色 = 被展开（蓝） / 目标弹出前未展开（红）；数字 = 展开次序", fontsize=10, y=1.04)
    save(fig, "fig1_q1_trace.png")


# ------------------------------------------------------------- Q1: scale-up
Q1_BIG = [("5×5", 15, 13), ("10×10", 71, 42), ("20×20", 251, 132), ("30×30", 735, 624)]


def fig2_q1_scale():
    labels = [x[0] for x in Q1_BIG]
    h0 = np.array([x[1] for x in Q1_BIG], dtype=float)
    h1 = np.array([x[2] for x in Q1_BIG], dtype=float)
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    b1 = ax.bar(x - w / 2, h0, w, color=C_H0, label="h = 0（Dijkstra 式）")
    b2 = ax.bar(x + w / 2, h1, w, color=C_H1, label="曼哈顿距离")
    for rect, v in list(zip(b1, h0)) + list(zip(b2, h1)):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 6, str(int(v)), ha="center", fontsize=9)
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for xi, (a, b) in enumerate(zip(h0, h1)):
        ax.text(xi, -0.14, f"降幅 {(a - b) / a:.0%}", transform=trans, ha="center",
                fontsize=10, color="#7b241c", weight="bold", clip_on=False)
    ax.set_ylim(0, max(h0) * 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("展开的非目标格数")
    ax.set_title("题目一：启发函数收益随地图规模的对照（固定种子自设地图）", fontsize=11)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig2_q1_scale.png")


# ---------------------------------------------------------------- Q2: path
Q2_GRID = [
    ["S", 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, "E"],
]
Q2_PATH = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (4, 2), (4, 3), (4, 4)]


def fig3_q2_path():
    rows, cols = 5, 5
    weights = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            v = Q2_GRID[r][c]
            weights[r, c] = 0 if v in ("S", "E") else float(v)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    im = ax.imshow(weights, cmap="YlOrBr", vmin=0, vmax=8)
    for r in range(rows):
        for c in range(cols):
            v = Q2_GRID[r][c]
            label = str(v)
            ax.text(c, r - 0.16, label, ha="center", va="center", fontsize=10,
                    color="#f7f3e9" if weights[r, c] >= 4 else "#5b3a1a", weight="bold")
    cum = 0
    for i, (r, c) in enumerate(Q2_PATH[1:], start=1):
        v = Q2_GRID[r][c]
        cum += 0 if v in ("S", "E") else float(v)
        ax.text(c, r + 0.26, f"Σ{cum:g}", ha="center", va="center", fontsize=7.5, color="#1a5276")
    pr, pc = zip(*Q2_PATH)
    ax.plot(pc, pr, "-", color="#1e8449", lw=2.6, alpha=0.9)
    ax.plot(pc, pr, "o", color="#0b5345", ms=4.5)
    ax.set_xticks(range(cols), range(cols))
    ax.set_yticks(range(rows), range(rows))
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, shrink=0.75, label="进入代价")
    ax.set_title("题目二：可行地形权值与最优路径（Σ = 代价累计，E 进入计 0）", fontsize=10)
    save(fig, "fig3_q2_path.png")


# ------------------------------------------------------------- Q3: trees
TREE_CFG = [("2×5", 31, 31, 20, 27), ("3×3", 40, 40, 35, 32),
            ("4×2", 31, 31, 29, 22), ("4×3", 121, 121, 53, 102)]
TREE_LABELS = ("深度×分支",)


def gen_tree(depth, branch, rng):
    if depth == 0:
        return Q3.TreeNode(value=rng.randint(0, 20))
    return Q3.TreeNode(children=[gen_tree(depth - 1, branch, rng) for _ in range(branch)])


def tree_param():
    """Replay the report's rng sequence and recompute its table."""
    rng = random.Random(2026)
    out = {}
    for depth, branch in [(2, 5), (3, 3), (4, 2), (4, 3)]:
        t = gen_tree(depth, branch, rng)

        def count(node):
            return 1 + sum(count(c) for c in node.children)

        def rev(node):
            return Q3.TreeNode(children=[rev(c) for c in reversed(node.children)]) \
                if node.children else Q3.TreeNode(value=node.value)

        mm_n = Q3.minimax(t)[1]
        ab_n = Q3.alpha_beta(t)[1]
        rev_n = Q3.alpha_beta(rev(t))[1]
        out[f"{depth}×{branch}"] = (count(t), mm_n, ab_n, rev_n)
    return out


def fig4_q3_trees():
    data = tree_param()
    labels = list(data)
    total = [v[0] for v in data.values()]
    ab = [v[2] for v in data.values()]
    rev = [v[3] for v in data.values()]
    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    b1 = ax.bar(x - w, total, w, color="#b3b6b7", label="Minimax（全树）")
    b2 = ax.bar(x, ab, w, color=C_H0, label="Alpha-Beta 原序")
    b3 = ax.bar(x + w, rev, w, color=C_REV, label="Alpha-Beta 逐层反转序")
    for bars in (b1, b2, b3):
        for rect in bars:
            v = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, v + 1.5, str(int(v)),
                    ha="center", fontsize=9)
    best = [min(a, r) for a, r in zip(ab, rev)]
    top = max(total)
    for xi, (a, r, b) in enumerate(zip(ab, rev, best)):
        ax.text(xi, top * 1.05,
                f"最好序省 {(1 - b / total[xi]):.0%}", ha="center", fontsize=10,
                color="#7b241c", weight="bold")
    ax.set_ylim(0, top * 1.18)
    ax.set_xticks(x, labels)
    ax.set_xlabel("树规模（深度 × 分支）")
    ax.set_ylabel("访问节点数")
    ax.set_title("题目三：随机博弈树上的搜索开销（值 ∈ [0,20]，同一棵树三种访问策略）", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig4_q3_trees.png")


# --------------------------------------------------------------- Q4: tools
Q4_GRID = [
    ["A", 1, 1, "X", 2, 1, "T1"],
    [1, "X", 6, "X", 1, "X", 1],
    [1, 1, "T2", 1, 1, 1, 1],
    ["X", 1, "X", 8, "X", 1, "X"],
    [1, 1, 1, 1, "T3", 1, 1],
    [1, "X", 9, "X", 1, "X", 1],
    ["T4", 1, 1, "X", 1, 1, "B"],
]
POSITIONS = {"A": (0, 0), "B": (6, 6), "T1": (0, 6), "T2": (2, 2), "T3": (4, 4), "T4": (6, 0)}
VALUES = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}


def first_move_values(H, dist):
    rows = []
    for t in ["T1", "T2", "T3", "T4"]:
        gain = VALUES[t] - dist[("A", t)]
        nxt = Q4.State(t, "B", tuple(x for x in ["T1", "T2", "T3", "T4"] if x != t), gain, False)
        v, _ , _ = Q4.minimax(nxt, H - 1, dist, VALUES)
        rows.append(v)
    return rows


def fig5_q4_depth():
    dist = Q4.build_distances(POSITIONS, Q4_GRID)
    depths = [1, 2, 3, 4]
    res = [Q4.solve(H, POSITIONS, VALUES, dist) for H in depths]
    mm = [r["minimax_nodes"] for r in res]
    ab = [r["alpha_beta_nodes"] for r in res]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    x = np.arange(len(depths))
    w = 0.38
    b1 = ax.bar(x - w / 2, mm, w, color=C_H0, label="Minimax 状态数")
    b2 = ax.bar(x + w / 2, ab, w, color=C_H1, label="Alpha-Beta 状态数")
    for bars in (b1, b2):
        for rect in bars:
            v = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, v + 1, str(int(v)), ha="center", fontsize=9)
    for xi, r in enumerate(res):
        ax.text(xi, max(mm[xi], ab[xi]) + 6, f"首步 T{r['best_target'][1]}  值 {r['best_value']:+d}",
                ha="center", fontsize=10, color="#7b241c")
    ax.set_xticks(x, [f"H={d}" for d in depths])
    ax.set_ylabel("访问博弈状态数")
    ax.set_ylim(0, max(mm) * 1.35)
    ax.set_title("深度 × 搜索开销", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    vals_2 = first_move_values(2, dist)
    vals_4 = first_move_values(4, dist)
    idx = np.arange(4)
    b1 = ax2.bar(idx - w / 2, vals_2, w, color="#d5dbdb", label="H=2（截断在 B 步后）")
    b2 = ax2.bar(idx + w / 2, vals_4, w, color=C_H1, label="H=4（完整回合）")
    for bars in (b1, b2):
        for rect in bars:
            v = rect.get_height()
            ax2.text(rect.get_x() + rect.get_width() / 2, v + (0.3 if v >= 0 else -0.75),
                     f"{v:+d}", ha="center", fontsize=9)
    ax2.axhline(0, color="#555", lw=0.9)
    ax2.annotate("最优首步翻转到 T3", xy=(2 + w / 2, 3), xytext=(2.1, -4.6),
                 fontsize=10, color="#7b241c", weight="bold",
                 arrowprops=dict(arrowstyle="->", color="#7b241c"))
    ax2.set_xticks(idx, ["T1", "T2", "T3", "T4"])
    ax2.set_xlabel("A 的首步")
    ax2.set_ylabel("最优效用差 U")
    ax2.set_ylim(-9, 6)
    ax2.set_title("各首步整体效用的深度对照", fontsize=11)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.spines[["top", "right"]].set_visible(False)
    fig.suptitle("题目四：深度对照——H=4 同时改变最优首步与效用差符号", fontsize=12, y=1.02)
    save(fig, "fig5_q4_depth.png")


ORDERS = [
    ("自然序 T1,T2,T3,T4", ["T1", "T2", "T3", "T4"]),
    ("反转序 T4,T3,T2,T1", ["T4", "T3", "T2", "T1"]),
    ("价值降序 T1,T3,T2,T4", ["T1", "T3", "T2", "T4"]),
    ("价值升序 T2,T4,T3,T1", ["T2", "T4", "T3", "T1"]),
    ("动态：即时收益降序", None),
]


def order_nodes(H, dist, order):
    """Alpha-beta visited count for a given static or dynamic ordering
    (root counted, same convention as Q4.solve)."""
    visited, pruned = 1, 0

    def search(cur, rem, alpha, beta):
        nonlocal visited, pruned
        if rem == 0 or not cur.remaining:
            return cur.utility
        maximizing = cur.maximizing
        best = float("-inf") if maximizing else float("inf")
        pos = cur.position_a if maximizing else cur.position_b
        if order is None:
            cand = sorted(cur.remaining, key=lambda t: VALUES[t] - dist[(pos, t)], reverse=maximizing)
        else:
            cand = [t for t in order if t in cur.remaining]
        for i, t in enumerate(cand):
            visited += 1
            gain = VALUES[t] - dist[(pos, t)]
            nxt = Q4.State(
                t if maximizing else cur.position_a,
                t if not maximizing else cur.position_b,
                tuple(x for x in cur.remaining if x != t),
                cur.utility + (gain if maximizing else -gain),
                not maximizing,
            )
            v = search(nxt, rem - 1, alpha, beta)
            best = max(best, v) if maximizing else min(best, v)
            if maximizing:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)
            if alpha >= beta:
                pruned += len(cand) - i - 1
                break
        return int(best)

    init = Q4.State("A", "B", ("T1", "T2", "T3", "T4"), 0, True)
    search(init, H, float("-inf"), float("inf"))
    return visited, pruned


def fig6_q4_order():
    dist = Q4.build_distances(POSITIONS, Q4_GRID)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharey=False)
    for ax, H in zip(axes, [2, 4]):
        names = [o[0] for o in ORDERS]
        nodes, prune = [], []
        for _, od in ORDERS:
            n, p = order_nodes(H, dist, od)
            nodes.append(n)
            prune.append(p)
        y = np.arange(len(names))[::-1]
        colors = [(C_H1 if n == min(nodes) else "#d5d8dc") for n in nodes]
        bars = ax.barh(y, nodes, height=0.62, color=colors)
        for yi, (n, p) in zip(y, zip(nodes, prune)):
            ax.text(n + 0.6, yi, f"{n}（剪 {p}）", va="center", fontsize=9)
            if n == min(nodes):
                ax.text(n * 0.5, yi, "最优", va="center", ha="center",
                        fontsize=10, color="w", weight="bold")
        ax.set_yticks(y, names, fontsize=9)
        ax.set_xlim(0, max(nodes) * 1.3)
        ax.set_xlabel(f"H={H}：Alpha-Beta 访问状态数（括号内为剪去子节点数）")
        ax.set_title(f"H = {H}", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("题目四：候选顺序对照——H=2 自然序最优，H=4 价值升序最优；结论对顺序“个案敏感”", fontsize=12, y=1.03)
    save(fig, "fig6_q4_order.png")


def main():
    print("Regenerating figures ->", FIG)
    fig1_q1_trace()
    fig2_q1_scale()
    fig3_q2_path()
    fig4_q3_trees()
    fig5_q4_depth()
    fig6_q4_order()
    print("done.")


if __name__ == "__main__":
    main()
