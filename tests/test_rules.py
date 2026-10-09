"""棋盘规则单元测试：合法走法、翻转、换手、终局。"""
import numpy as np

from reversi.board import COLOR_BLACK, COLOR_NONE, COLOR_WHITE, initial_board
from reversi.rules import apply_move, has_legal_move, is_terminal, legal_moves


def test_initial_legal_moves():
    board = initial_board()
    moves = legal_moves(board, COLOR_BLACK)
    assert set(moves) == {(2, 3), (3, 2), (4, 5), (5, 4)}


def test_apply_move_flips():
    board = initial_board()
    new_board = apply_move(board, COLOR_BLACK, (2, 3))
    assert new_board[2, 3] == COLOR_BLACK
    assert new_board[3, 3] == COLOR_BLACK  # 白被翻黑
    assert new_board[3, 4] == COLOR_BLACK
    assert new_board[4, 3] == COLOR_BLACK
    assert new_board[4, 4] == COLOR_WHITE
    # 原棋盘不变
    assert board[2, 3] == COLOR_NONE
    assert board[3, 3] == COLOR_WHITE


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
    board = np.full((8, 8), COLOR_WHITE, dtype=np.int8)
    board[0, 0] = COLOR_NONE
    board[1, 1] = COLOR_BLACK
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
    assert int(np.count_nonzero(board != COLOR_NONE)) <= 64

