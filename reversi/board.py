"""Reversed Reversi 位棋盘表示与基础工具。

棋盘用两个 np.uint64 位掩码表示：black（黑子）、white（白子）。
bit 索引 = row * 8 + col，row 0 在顶部、col 0 在左；bit 0 = 左上角 (0,0)，
bit 63 = 右下角 (7,7)。位棋盘数组形状为 (2,)，依次为 [black, white]。
"""
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

FULL_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
COL0_MASK = np.uint64(0x0101010101010101)
COL7_MASK = np.uint64(0x8080808080808080)
ROW0_MASK = np.uint64(0x00000000000000FF)
ROW7_MASK = np.uint64(0xFF00000000000000)
CORNER_MASK = np.uint64(0x8100000000000081)  # (0,0),(0,7),(7,0),(7,7)


def opponent(color):
    return -color


def _bit(row, col):
    return np.uint64(1) << np.uint64(row * 8 + col)


def initial_board():
    """标准 Reversi 初始四子布局（黑先手），返回位棋盘 [black, white]。"""
    black = _bit(3, 4) | _bit(4, 3)
    white = _bit(3, 3) | _bit(4, 4)
    return np.array([black, white], dtype=np.uint64)


def initial_board_array():
    """标准初始四子布局的 numpy 8x8 int8 数组（供平台接口/测试参照）。"""
    return board_to_array(initial_board())


def board_from_array(arr):
    """numpy 8x8 int8 数组 -> 位棋盘 [black, white]。"""
    black = np.uint64(0)
    white = np.uint64(0)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = int(arr[r, c])
            if v == COLOR_BLACK:
                black |= _bit(r, c)
            elif v == COLOR_WHITE:
                white |= _bit(r, c)
    return np.array([black, white], dtype=np.uint64)


def board_to_array(bb):
    """位棋盘 [black, white] -> numpy 8x8 int8 数组。"""
    arr = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    black, white = np.uint64(bb[0]), np.uint64(bb[1])
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            b = _bit(r, c)
            if black & b:
                arr[r, c] = COLOR_BLACK
            elif white & b:
                arr[r, c] = COLOR_WHITE
    return arr


def count(bb, color):
    """color 的棋子数。同时支持位棋盘 [black, white] 与 numpy 8x8 数组。"""
    if isinstance(bb, np.ndarray) and bb.ndim == 2:
        return int(np.count_nonzero(bb == color))
    if color == COLOR_BLACK:
        return int(bb[0]).bit_count()
    return int(bb[1]).bit_count()


def empty_count(bb):
    """剩余空格数。同时支持位棋盘与 numpy 8x8 数组。"""
    if isinstance(bb, np.ndarray) and bb.ndim == 2:
        return int(np.count_nonzero(bb == COLOR_NONE))
    return 64 - int(bb[0] | bb[1]).bit_count()


def clone(bb):
    return bb.copy()



