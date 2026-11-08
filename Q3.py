import json
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TreeNode:
    value: Optional[float] = None
    children: List["TreeNode"] = field(default_factory=list)


def minimax(node: TreeNode, maximizing: bool = True) -> Tuple[float, int]:
    """Return the minimax score and number of visited nodes."""
    visited = 1
    if not node.children:
        if node.value is None:
            raise ValueError("叶子节点必须提供 value")
        return node.value, visited

    child_scores = []
    for child in node.children:
        score, child_visited = minimax(child, not maximizing)
        child_scores.append(score)
        visited += child_visited
    return (max(child_scores) if maximizing else min(child_scores)), visited


def alpha_beta(
    node: TreeNode,
    maximizing: bool = True,
    alpha: float = float("-inf"),
    beta: float = float("inf"),
) -> Tuple[float, int, int]:
    """Return score, visited node count, and pruned child count."""
    visited = 1
    pruned = 0
    if not node.children:
        if node.value is None:
            raise ValueError("叶子节点必须提供 value")
        return node.value, visited, pruned

    if maximizing:
        best_score = float("-inf")
        for index, child in enumerate(node.children):
            score, child_visited, child_pruned = alpha_beta(
                child, False, alpha, beta
            )
            visited += child_visited
            pruned += child_pruned
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if alpha >= beta:
                pruned += len(node.children) - index - 1
                break
    else:
        best_score = float("inf")
        for index, child in enumerate(node.children):
            score, child_visited, child_pruned = alpha_beta(
                child, True, alpha, beta
            )
            visited += child_visited
            pruned += child_pruned
            best_score = min(best_score, score)
            beta = min(beta, best_score)
            if alpha >= beta:
                pruned += len(node.children) - index - 1
                break
    return best_score, visited, pruned


def tree_from_dict(data: dict) -> TreeNode:
    if "value" in data:
        return TreeNode(value=data["value"])
    return TreeNode(children=[tree_from_dict(child) for child in data.get("children", [])])


if __name__ == "__main__":
    print("请输入博弈树 JSON（叶节点用 value，非叶节点用 children；输入结束后按 Ctrl+Z 再回车）：")
    payload_text = sys.stdin.read()
    if not payload_text.strip():
        raise SystemExit("未读取到博弈树 JSON。")
    payload = json.loads(payload_text)
    root = tree_from_dict(payload.get("tree_structure", payload))

    minimax_score, minimax_nodes = minimax(root)
    alpha_beta_score, alpha_beta_nodes, pruned_children = alpha_beta(root)
    print(json.dumps({
        "minimax_value": minimax_score,
        "alpha_beta_value": alpha_beta_score,
        "minimax_nodes": minimax_nodes,
        "alpha_beta_nodes": alpha_beta_nodes,
        "pruned_children": pruned_children,
    }, ensure_ascii=False))