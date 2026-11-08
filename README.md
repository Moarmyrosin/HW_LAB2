# 当代人工智能实验二：搜索算法

## 文件

- `Q1.py`：迷宫 A*；`astar` 返回路径，`astar_with_stats` 可切换曼哈顿启发式并返回展开数。
- `Q2.py`：带地形权值的 A*，`minimum_cost(start, goal, grid)` 返回最小代价或 `-1`。
- `Q3.py`：博弈树 Minimax 与 Alpha-Beta。支持从标准输入读取题目给出的 JSON。
- `Q4.py`：综合题的 A* 距离矩阵、Minimax、Alpha-Beta 与访问顺序实验。
- `实验报告.md`：算法说明、验收结果和性能对照。
- `make_figures.py`：重新实测并生成报告引用的全部图表到 `figures/`（运行 `python make_figures.py`，约 10 秒）。

## 运行

在此目录执行：

```powershell
python Q1.py
python Q2.py
python Q3.py
python Q4.py
```

`Q1.py` 提示输入起点、终点和多行地图列表，无需输入地图行数。`Q2.py` 提示输入起点、终点和加权地图（S/E 标记起终点，数字为进入代价）。`Q3.py` 提示粘贴博弈树 JSON，叶节点使用 `value`，非叶节点使用 `children`；粘贴结束后按 Ctrl+Z 再回车。`Q4.py` 提示输入综合题地图、宝藏价值字典和搜索深度（逗号分隔）。

## 复现实验样例

题目一/二输入：起点 `(0,0)`，终点 `(3,4)` / `(4,4)`，地图见实验要求 PDF（Q2 的 E 格进入代价为 0）。预期输出分别为坐标序列和 `7`。

题目三输入（粘贴后 Ctrl+Z 回车）：

```json
{"tree_structure": {"children": [
  {"children": [{"children": [{"value": 3}, {"value": 5}]}, {"children": [{"value": 6}, {"value": 9}]}]},
  {"children": [{"children": [{"value": 6}, {"value": 9}]}, {"children": [{"value": 1}, {"value": 2}]}]}
]}}
```

预期：minimax=5、alpha_beta=5、节点数 15/14、剪枝 1。

题目四输入：

```text
地图：
['A', 1, 1, 'X', 2, 1, 'T1']
[1, 'X', 6, 'X', 1, 'X', 1]
[1, 1, 'T2', 1, 1, 1, 1]
['X', 1, 'X', 8, 'X', 1, 'X']
[1, 1, 1, 1, 'T3', 1, 1]
[1, 'X', 9, 'X', 1, 'X', 1]
['T4', 1, 1, 'X', 1, 1, 'B']
宝藏价值：{'T1': 9, 'T2': 6, 'T3': 7, 'T4': 7}
深度：2,4
```

预期：H=2 输出 `T2/-1`，H=4 输出 `T3/3`；6×6 距离矩阵见实验报告 §7.1。

Q4 距离矩阵行列顺序为 `A, B, T1, T2, ...`。核心函数仍可直接导入调用；命令行运行不再自动使用题目测试样例。
