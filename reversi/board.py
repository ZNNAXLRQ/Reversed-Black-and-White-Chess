"""Reversed Reversi 棋盘常量与基础工具。"""
import numpy as np

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0

BOARD_SIZE = 8

# 8 个方向增量 (dr, dc)
DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def opponent(color):
    return -color


def in_bounds(row, col):
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def initial_board():
    """标准 Reversi 初始四子布局（黑先手）。"""
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    board[3, 3] = COLOR_WHITE
    board[4, 4] = COLOR_WHITE
    board[3, 4] = COLOR_BLACK
    board[4, 3] = COLOR_BLACK
    return board


def count(board, color):
    return int(np.count_nonzero(board == color))


def clone(board):
    return board.copy()


def shift(board, dr, dc):
    """沿 (dr, dc) 移动棋盘，越界位置填 0：s[r, c] = board[r+dr, c+dc]。"""
    n = board.shape[0]
    s = np.zeros_like(board)
    r0 = max(0, -dr)
    r1 = min(n, n - dr)
    c0 = max(0, -dc)
    c1 = min(n, n - dc)
    s[r0:r1, c0:c1] = board[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return s


