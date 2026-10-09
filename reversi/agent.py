"""课程 Agent 接口：固定基准权重 + Negamax 搜索。

candidate_list 行为（课程最终依据）：
- 先 append 全部合法走法；
- 再把选中的走法 append 到末尾（平台取最后一个元素作为决策）；
- 无合法走法时返回空列表。

go() 中绝不运行 NCS 或任何多局优化。
"""
import numpy as np

from .rules import legal_moves
from .search import best_move

# 固定基准权重（项目无既有基准权重，按 Plan 0 结论使用；仅用于对照，非最优值）。
BASELINE_WEIGHTS = np.array([-1.0, 1.0, 1.0, 1.0, -1.0], dtype=np.float64)

# 权重实验统一搜索深度（搜索深度不进入 NCS 权重向量）。
SEARCH_DEPTH = 3


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

        # 搜索选中的走法放在最后一个元素
        chosen = best_move(chessboard, self.color, self.depth, self.weights)
        if chosen is None:  # 防御性兜底，理论上 moves 非空时不会发生
            chosen = moves[0]
        self.candidate_list.append(chosen)

        return self.candidate_list

