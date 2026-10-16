"""棋盘规则单元测试：合法走法、翻转、换手、终局。"""
import numpy as np

from reversi.board import (
    COLOR_BLACK, COLOR_NONE, COLOR_WHITE, board_from_array, board_to_array,
    count, initial_board,
)
from reversi.rules import apply_move, has_legal_move, is_terminal, legal_moves


def test_initial_legal_moves():
    board = initial_board()
    moves = legal_moves(board, COLOR_BLACK)
    assert set(moves) == {(2, 3), (3, 2), (4, 5), (5, 4)}


def test_apply_move_flips():
    board = initial_board()
    new_board = apply_move(board, COLOR_BLACK, (2, 3))
    arr = board_to_array(new_board)
    assert arr[2, 3] == COLOR_BLACK
    assert arr[3, 3] == COLOR_BLACK  # 白被翻黑
    assert arr[3, 4] == COLOR_BLACK
    assert arr[4, 3] == COLOR_BLACK
    assert arr[4, 4] == COLOR_WHITE
    # 原棋盘不变
    orig = board_to_array(board)
    assert orig[2, 3] == COLOR_NONE
    assert orig[3, 3] == COLOR_WHITE
    assert orig[3, 4] == COLOR_BLACK
    assert orig[4, 3] == COLOR_BLACK
    assert orig[4, 4] == COLOR_WHITE


def test_apply_move_returns_new_board():
    board = initial_board()
    new_board = apply_move(board, COLOR_BLACK, (2, 3))
    assert new_board is not board


def test_not_terminal_initial():
    board = initial_board()
    assert has_legal_move(board, COLOR_BLACK)
    assert has_legal_move(board, COLOR_WHITE)
    assert not is_terminal(board)


def test_no_legal_move_pass_situation():
    # 黑无合法走法、但白有合法走法的局面
    arr = np.full((8, 8), COLOR_WHITE, dtype=np.int8)
    arr[0, 0] = COLOR_NONE
    arr[1, 1] = COLOR_BLACK
    board = board_from_array(arr)
    assert not has_legal_move(board, COLOR_BLACK)
    assert has_legal_move(board, COLOR_WHITE)
    assert not is_terminal(board)


def test_full_game_reaches_terminal():
    board = initial_board()
    color = COLOR_BLACK
    rng = np.random.default_rng(0)
    steps = 0
    while not is_terminal(board) and steps < 200:
        moves = legal_moves(board, color)
        if moves:
            move = moves[rng.integers(len(moves))]
            board = apply_move(board, color, move)
        color = -color
        steps += 1
    assert is_terminal(board)
    assert count(board, COLOR_BLACK) + count(board, COLOR_WHITE) <= 64
