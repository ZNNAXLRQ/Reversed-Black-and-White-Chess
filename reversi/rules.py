"""Reversed Reversi 规则：走法生成、翻转、换手与终局判定。

走法合法性与翻转规则与标准 Reversi 完全一致；仅终局目标反转（己方棋子更少者获胜）。

棋盘核心表示：位棋盘 [black, white]（两个 np.uint64）。为兼容棋谱解析等既有代码，
对外接口同时接受位棋盘与 numpy 8x8 数组（自动转换）。
"""
import numba
import numpy as np

from .board import (
    BOARD_SIZE, COLOR_BLACK, COLOR_WHITE,
    COL0_MASK, COL7_MASK, FULL_MASK, ROW0_MASK, ROW7_MASK,
    board_from_array, board_to_array, opponent,
)

# 8 个方向的 bit 位移量（E/W/S/N/SE/NW/SW/NE）与「源位」前置掩码（防止跨行列环绕）。
_DIR_SHIFTS = np.array([1, -1, 8, -8, 9, -9, 7, -7], dtype=np.int64)
_DIR_PRE = np.array([
    FULL_MASK ^ COL7_MASK,                    # E:  源 col != 7
    FULL_MASK ^ COL0_MASK,                    # W:  源 col != 0
    FULL_MASK ^ ROW7_MASK,                    # S:  源 row != 7
    FULL_MASK ^ ROW0_MASK,                    # N:  源 row != 0
    FULL_MASK ^ (ROW7_MASK | COL7_MASK),      # SE
    FULL_MASK ^ (ROW0_MASK | COL0_MASK),      # NW
    FULL_MASK ^ (ROW7_MASK | COL0_MASK),      # SW
    FULL_MASK ^ (ROW0_MASK | COL7_MASK),      # NE
], dtype=np.uint64)


@numba.njit
def _sh(x, s):
    if s > 0:
        return x << np.uint64(s)
    return x >> np.uint64(-s)


@numba.njit
def _legal_mask_nb(me, opp):
    """me 的合法落点位掩码。"""
    empty = ~(me | opp) & FULL_MASK
    legal = np.uint64(0)
    for d in range(8):
        s = _DIR_SHIFTS[d]
        pre = _DIR_PRE[d]
        candidates = _sh(me & pre, s) & opp
        while candidates != np.uint64(0):
            candidates = _sh(candidates & pre, s)
            legal |= candidates & empty
            candidates &= opp
    return legal


@numba.njit
def _flips_mask_nb(me, opp, pos):
    """落子 pos 后被翻转的 opp 位掩码。"""
    flipped = np.uint64(0)
    for d in range(8):
        s = _DIR_SHIFTS[d]
        pre = _DIR_PRE[d]
        m = _sh(pos & pre, s)
        line = np.uint64(0)
        while (m & opp) != np.uint64(0):
            line |= m
            m = _sh(m & pre, s)
        if (m & me) != np.uint64(0):
            flipped |= line
    return flipped


@numba.njit
def _apply_move_nb(me, opp, pos):
    flipped = _flips_mask_nb(me, opp, pos)
    me_new = me | pos | flipped
    opp_new = opp & ~flipped
    return me_new, opp_new


def _is_array(board):
    return isinstance(board, np.ndarray) and board.ndim == 2


def _to_bb(board):
    if _is_array(board):
        return board_from_array(board)
    return board


def _color_mask(bb, color):
    if color == COLOR_BLACK:
        return np.uint64(bb[0])
    return np.uint64(bb[1])


def legal_mask(bb, color):
    """color 的合法落点位掩码（供评估/搜索复用）。要求位棋盘输入。"""
    me = _color_mask(bb, color)
    opp = _color_mask(bb, -color)
    return np.uint64(_legal_mask_nb(me, opp))


def _mask_to_moves(mask):
    moves = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if mask & (np.uint64(1) << np.uint64(r * 8 + c)):
                moves.append((r, c))
    return moves


def legal_moves(board, color):
    """返回 color 的所有合法走法 (row, col) 列表。支持位棋盘或 numpy 数组。"""
    bb = _to_bb(board)
    mask = legal_mask(bb, color)
    return _mask_to_moves(mask)


def has_legal_move(board, color):
    bb = _to_bb(board)
    return legal_mask(bb, color) != np.uint64(0)


def count_legal_moves(board, color):
    bb = _to_bb(board)
    return int(legal_mask(bb, color)).bit_count()


def flips(board, color, row, col):
    """返回在 (row, col) 落 color 子会翻转的对方棋子坐标列表（不修改棋盘）。"""
    bb = _to_bb(board)
    me = _color_mask(bb, color)
    opp = _color_mask(bb, -color)
    pos = np.uint64(1) << np.uint64(row * 8 + col)
    flipped = np.uint64(_flips_mask_nb(me, opp, pos))
    return _mask_to_moves(flipped)


def apply_move(board, color, move):
    """返回落子后的新棋盘（原棋盘不变）。输入 numpy 则返回 numpy，输入位棋盘则返回位棋盘。"""
    as_array = _is_array(board)
    bb = _to_bb(board)
    row, col = move
    pos = np.uint64(1) << np.uint64(row * 8 + col)
    me = _color_mask(bb, color)
    opp = _color_mask(bb, -color)
    me_new, opp_new = _apply_move_nb(me, opp, pos)
    if color == COLOR_BLACK:
        new_bb = np.array([me_new, opp_new], dtype=np.uint64)
    else:
        new_bb = np.array([opp_new, me_new], dtype=np.uint64)
    if as_array:
        return board_to_array(new_bb)
    return new_bb


def is_terminal(board):
    """双方均无合法走法时为终局。"""
    return not has_legal_move(board, COLOR_BLACK) and not has_legal_move(board, COLOR_WHITE)
