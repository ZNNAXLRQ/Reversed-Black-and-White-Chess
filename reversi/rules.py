"""Reversed Reversi 规则：合法走法、翻转、换手与终局判定。

走法合法性与翻转规则与标准 Reversi 完全一致；仅终局目标反转（己方棋子更少者获胜）。
"""
import numba
import numpy as np

from .board import (
    BOARD_SIZE, COLOR_BLACK, COLOR_NONE, COLOR_WHITE, DIRECTIONS,
    in_bounds, opponent,
)

_DIRECTIONS_NB = np.array(DIRECTIONS, dtype=np.int8)


@numba.njit
def _legal_mask_nb(board, color):
    """Numba 加速的合法走法掩码（纯循环，编译为机器码）。"""
    n = board.shape[0]
    opp = -color
    legal = np.zeros((n, n), dtype=np.bool_)
    for r in range(n):
        for c in range(n):
            if board[r, c] != 0:
                continue
            for d in range(8):
                dr = _DIRECTIONS_NB[d, 0]
                dc = _DIRECTIONS_NB[d, 1]
                rr = r + dr
                cc = c + dc
                if rr < 0 or rr >= n or cc < 0 or cc >= n:
                    continue
                if board[rr, cc] != opp:
                    continue
                rr += dr
                cc += dc
                found = False
                while 0 <= rr < n and 0 <= cc < n:
                    if board[rr, cc] == color:
                        found = True
                        break
                    if board[rr, cc] != opp:
                        break
                    rr += dr
                    cc += dc
                if found:
                    legal[r, c] = True
                    break
    return legal


def flips(board, color, row, col):
    """返回在 (row, col) 落 color 子会翻转的对方棋子坐标列表（不修改棋盘）。"""
    if board[row, col] != COLOR_NONE:
        return []
    captured = []
    for dr, dc in DIRECTIONS:
        r, c = row + dr, col + dc
        line = []
        while in_bounds(r, c) and board[r, c] == opponent(color):
            line.append((r, c))
            r += dr
            c += dc
        if line and in_bounds(r, c) and board[r, c] == color:
            captured.extend(line)
    return captured


def _legal_mask(board, color):
    """返回 color 的合法走法布尔掩码（8x8）。"""
    return _legal_mask_nb(board, color)


def legal_moves(board, color):
    """返回颜色 color 的所有合法走法 (row, col) 列表。"""
    return [(int(r), int(c)) for r, c in zip(*np.nonzero(_legal_mask(board, color)))]


def has_legal_move(board, color):
    return bool(_legal_mask(board, color).any())


def count_legal_moves(board, color):
    return int(np.count_nonzero(_legal_mask(board, color)))


def apply_move(board, color, move):
    """返回落子后的新棋盘（原棋盘不变）。"""
    row, col = move
    new_board = board.copy()
    new_board[row, col] = color
    for r, c in flips(board, color, row, col):
        new_board[r, c] = color
    return new_board


def is_terminal(board):
    """双方均无合法走法时为终局。"""
    return not _legal_mask(board, COLOR_BLACK).any() and not _legal_mask(board, COLOR_WHITE).any()

