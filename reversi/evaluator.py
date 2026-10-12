"""Reversed Reversi 评估器：5 个固定特征 + 终局效用。

特征顺序严格固定：disc_diff, position_diff, mobility_diff, corner_diff, frontier_diff。
所有特征从 perspective 视角计算为「己方特征 - 对方特征」。
"""
import numpy as np

from .board import BOARD_SIZE, COLOR_NONE, DIRECTIONS, count, opponent, shift
from .rules import count_legal_moves

# 固定位置表 P（8x8），row 0 在顶部；归一化分母固定为 64*120。
POSITION_TABLE = np.array([
    [120, -20,  20,   5,   5,  20, -20, 120],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [120, -20,  20,   5,   5,  20, -20, 120],
], dtype=np.float64)

_CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]


def _count_frontier(board, color):
    """frontier disc：8 邻域内至少相邻一个空格的棋子，每枚棋子只计一次。"""
    disc = board == color
    empty = board == COLOR_NONE
    frontier = np.zeros_like(board, dtype=bool)
    for dr, dc in DIRECTIONS:
        frontier |= disc & shift(empty, dr, dc)
    return int(np.count_nonzero(frontier))


def features(board, perspective):
    """返回从 perspective 视角计算的 5 特征向量，形状 (5,)。"""
    opp = opponent(perspective)
    own_mask = (board == perspective)
    opp_mask = (board == opp)
    own_count = int(np.count_nonzero(own_mask))
    opp_count = int(np.count_nonzero(opp_mask))

    # 1. disc_diff = (own - opp) / 64
    disc_diff = (own_count - opp_count) / 64.0

    # 2. position_diff = (sum(P[own]) - sum(P[opp])) / (64 * 120)
    pos_own = float(np.sum(POSITION_TABLE[own_mask]))
    pos_opp = float(np.sum(POSITION_TABLE[opp_mask]))
    position_diff = (pos_own - pos_opp) / (64.0 * 120.0)

    # 3. mobility_diff = (own_legal - opp_legal) / max(1, own_legal + opp_legal)
    own_moves = count_legal_moves(board, perspective)
    opp_moves = count_legal_moves(board, opp)
    mobility_diff = (own_moves - opp_moves) / max(1, own_moves + opp_moves)

    # 4. corner_diff = (own_corner - opp_corner) / 4
    own_corners = sum(1 for r, c in _CORNERS if board[r, c] == perspective)
    opp_corners = sum(1 for r, c in _CORNERS if board[r, c] == opp)
    corner_diff = (own_corners - opp_corners) / 4.0

    # 5. frontier_diff = (own_frontier - opp_frontier) / 64
    frontier_diff = (_count_frontier(board, perspective) - _count_frontier(board, opp)) / 64.0

    return np.array(
        [disc_diff, position_diff, mobility_diff, corner_diff, frontier_diff],
        dtype=np.float64,
    )


def evaluate(board, perspective, weights):
    """非终局线性评估 E(s; w) = dot(w, features)，权重形状必须为 (5,)。"""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (5,):
        raise ValueError(f"weights must have shape (5,), got {weights.shape}")
    return float(np.dot(weights, features(board, perspective)))


def terminal_utility(board, perspective):
    """反转黑白棋终局效用（从 perspective 视角）：己方棋子更少者获胜。"""
    own = count(board, perspective)
    opp = count(board, opponent(perspective))
    if own < opp:
        return 1000.0 + (opp - own)
    if own > opp:
        return -1000.0 + (opp - own)
    return 0.0

