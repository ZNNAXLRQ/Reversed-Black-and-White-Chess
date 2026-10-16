"""Reversed Reversi 评估器：5 个固定特征 + 终局效用（位棋盘 + Numba）。

特征顺序严格固定：disc_diff, position_diff, mobility_diff, corner_diff, frontier_diff。
所有特征从 perspective 视角计算为「己方特征 - 对方特征」。
棋盘表示为位棋盘 [black, white]。
"""
import numba
import numpy as np

from .board import (
    BOARD_SIZE, COLOR_BLACK, CORNER_MASK, FULL_MASK, board_from_array, count, opponent,
)
from .rules import _DIR_PRE, _DIR_SHIFTS, _legal_mask_nb, _sh

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

# row-major 展平后的 64 个位置权重，bit 索引 = row*8+col。
_POSITION_WEIGHTS = POSITION_TABLE.ravel()


@numba.njit
def _popcount(x):
    """uint64 的 popcount（numba 无内置 bit_count，用 SWAR 实现）。返回 int64。"""
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = x + (x >> np.uint64(8))
    x = x + (x >> np.uint64(16))
    x = x + (x >> np.uint64(32))
    return np.int64(x & np.uint64(0x7F))


@numba.njit
def _frontier_count_nb(disc, empty):
    """frontier disc：8 邻域内至少相邻一个空格的棋子，每枚只计一次。"""
    frontier = np.uint64(0)
    for d in range(8):
        s = _DIR_SHIFTS[d]
        pre = _DIR_PRE[d]
        frontier |= _sh(empty & pre, s) & disc
    return _popcount(frontier)


@numba.njit
def _evaluate_nb(me, opp, weights):
    """Numba 编译的位棋盘评估：返回 dot(weights, features)。me/opp 为 perspective 视角掩码。"""
    own = _popcount(me)
    opp_n = _popcount(opp)

    # 1. disc_diff = (own - opp) / 64
    disc_diff = (own - opp_n) / 64.0

    # 2. position_diff = (sum(P[own]) - sum(P[opp])) / (64 * 120)
    pos_own = 0.0
    pos_opp = 0.0
    for i in range(64):
        bit = np.uint64(1) << np.uint64(i)
        if (me & bit) != np.uint64(0):
            pos_own += _POSITION_WEIGHTS[i]
        if (opp & bit) != np.uint64(0):
            pos_opp += _POSITION_WEIGHTS[i]
    position_diff = (pos_own - pos_opp) / (64.0 * 120.0)

    # 3. mobility_diff = (own_legal - opp_legal) / max(1, own_legal + opp_legal)
    own_moves = _popcount(_legal_mask_nb(me, opp))
    opp_moves = _popcount(_legal_mask_nb(opp, me))
    mobility_diff = (own_moves - opp_moves) / max(1, own_moves + opp_moves)

    # 4. corner_diff = (own_corner - opp_corner) / 4
    own_corners = _popcount(me & CORNER_MASK)
    opp_corners = _popcount(opp & CORNER_MASK)
    corner_diff = (own_corners - opp_corners) / 4.0

    # 5. frontier_diff = (own_frontier - opp_frontier) / 64
    empty = ~(me | opp) & FULL_MASK
    frontier_diff = (_frontier_count_nb(me, empty) - _frontier_count_nb(opp, empty)) / 64.0

    return (
        weights[0] * disc_diff
        + weights[1] * position_diff
        + weights[2] * mobility_diff
        + weights[3] * corner_diff
        + weights[4] * frontier_diff
    )


def _masks(bb, perspective):
    if perspective == COLOR_BLACK:
        return np.uint64(bb[0]), np.uint64(bb[1])
    return np.uint64(bb[1]), np.uint64(bb[0])


def features(board, perspective):
    """返回从 perspective 视角计算的 5 特征向量，形状 (5,)。支持位棋盘或 numpy 数组。"""
    if isinstance(board, np.ndarray) and board.ndim == 2:
        board = board_from_array(board)
    me, opp = _masks(board, perspective)
    own = int(me).bit_count()
    opp_n = int(opp).bit_count()

    disc_diff = (own - opp_n) / 64.0

    pos_own = 0.0
    pos_opp = 0.0
    for i in range(64):
        bit = np.uint64(1) << np.uint64(i)
        if me & bit:
            pos_own += float(_POSITION_WEIGHTS[i])
        if opp & bit:
            pos_opp += float(_POSITION_WEIGHTS[i])
    position_diff = (pos_own - pos_opp) / (64.0 * 120.0)

    own_moves = int(_legal_mask_nb(me, opp)).bit_count()
    opp_moves = int(_legal_mask_nb(opp, me)).bit_count()
    mobility_diff = (own_moves - opp_moves) / max(1, own_moves + opp_moves)

    own_corners = int(me & CORNER_MASK).bit_count()
    opp_corners = int(opp & CORNER_MASK).bit_count()
    corner_diff = (own_corners - opp_corners) / 4.0

    empty = ~(me | opp) & FULL_MASK
    own_frontier = _frontier_count_nb(me, empty)
    opp_frontier = _frontier_count_nb(opp, empty)
    frontier_diff = (own_frontier - opp_frontier) / 64.0

    return np.array(
        [disc_diff, position_diff, mobility_diff, corner_diff, frontier_diff],
        dtype=np.float64,
    )


def evaluate(board, perspective, weights):
    """非终局线性评估 E(s; w) = dot(w, features)，权重形状必须为 (5,)。支持位棋盘或 numpy 数组。"""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (5,):
        raise ValueError(f"weights must have shape (5,), got {weights.shape}")
    if isinstance(board, np.ndarray) and board.ndim == 2:
        board = board_from_array(board)
    me, opp = _masks(board, perspective)
    return float(_evaluate_nb(me, opp, weights))


def terminal_utility(bb, perspective):
    """反转黑白棋终局效用（从 perspective 视角）：己方棋子更少者获胜。"""
    own = count(bb, perspective)
    opp = count(bb, opponent(perspective))
    if own < opp:
        return 1000.0 + (opp - own)
    if own > opp:
        return -1000.0 + (opp - own)
    return 0.0

