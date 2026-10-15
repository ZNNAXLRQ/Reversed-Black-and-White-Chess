"""课程 Agent 接口：固定基准权重 + Negamax 搜索。

candidate_list 行为（课程最终依据）：
- 先 append 全部合法走法；
- 再把选中的走法 append 到末尾（平台取最后一个元素作为决策）；
- 无合法走法时返回空列表。

go() 中绝不运行 NCS 或任何多局优化。
"""
import numpy as np

from .rules import legal_moves
from .search import best_move_iterative

# 固定基准权重（项目无既有基准权重，按 Plan 0 结论使用；仅用于对照，非最优值）。
BASELINE_WEIGHTS = np.array([-1.0, 1.0, 1.0, 1.0, -1.0], dtype=np.float64)

# 迭代加深的绝对深度上限（实际深度由 time_out 时间预算动态决定，不会超过此值）。
# 设为 4：与训练/验证/H2H 统一深度一致，保证每步 <5s、整局 <180s。
SEARCH_DEPTH = 4


class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.weights = BASELINE_WEIGHTS.copy()
        self.depth = SEARCH_DEPTH

    def go(self, chessboard):
        self.candidate_list.clear()

        moves = legal_moves(chessboard, self.color)
        if not moves:
            return []

        # 返回完整合法走法集合
        for move in moves:
            self.candidate_list.append(move)

        # 迭代加深搜索：在 time_out 预算内尽量搜深，选中的走法放在最后一个元素。
        budget = self.time_out if (self.time_out and self.time_out > 0) else None
        chosen = best_move_iterative(chessboard, self.color, self.depth, self.weights, time_budget=budget)
        if chosen is None:  # 防御性兜底，理论上 moves 非空时不会发生
            chosen = moves[0]
        self.candidate_list.append(chosen)

        return self.candidate_list
