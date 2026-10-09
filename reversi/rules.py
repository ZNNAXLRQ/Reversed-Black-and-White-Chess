"""Reversed Reversi 规则：合法走法、翻转、换手与终局判定。

走法合法性与翻转规则与标准 Reversi 完全一致；仅终局目标反转（己方棋子更少者获胜）。
"""
from .board import (
    BOARD_SIZE, COLOR_BLACK, COLOR_NONE, COLOR_WHITE, DIRECTIONS,
    in_bounds, opponent,
)


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


def legal_moves(board, color):
    """返回颜色 color 的所有合法走法 (row, col) 列表。"""
    moves = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row, col] == COLOR_NONE and flips(board, color, row, col):
                moves.append((row, col))
    return moves


def has_legal_move(board, color):
    return len(legal_moves(board, color)) > 0


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
    return (not has_legal_move(board, COLOR_BLACK)) and (not has_legal_move(board, COLOR_WHITE))

