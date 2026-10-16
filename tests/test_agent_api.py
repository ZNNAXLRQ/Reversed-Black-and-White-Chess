"""Agent 接口单元测试：candidate_list 行为、无合法走法、签名、完整对局。"""
import numpy as np

from reversi.agent import AI
from reversi.board import (
    COLOR_BLACK, COLOR_NONE, COLOR_WHITE, board_from_array, board_to_array,
    initial_board, initial_board_array,
)
from reversi.rules import apply_move, is_terminal, legal_moves


def test_signature_and_init():
    ai = AI(chessboard_size=8, color=COLOR_BLACK, time_out=5.0)
    assert ai.chessboard_size == 8
    assert ai.color == COLOR_BLACK
    assert ai.time_out == 5.0
    assert ai.candidate_list == []


def test_candidate_list_all_legal_moves_and_chosen_last():
    ai = AI(8, COLOR_BLACK, 5.0)
    arr = initial_board_array()
    result = ai.go(arr)
    moves = legal_moves(board_from_array(arr), COLOR_BLACK)
    assert result == ai.candidate_list
    assert len(result) == len(moves) + 1
    assert set(result[:-1]) == set(moves)
    assert result[-1] in moves
    assert all(isinstance(m, tuple) and len(m) == 2 for m in result)


def test_no_legal_move_returns_empty():
    ai = AI(8, COLOR_BLACK, 5.0)
    board = np.full((8, 8), COLOR_WHITE, dtype=np.int8)
    board[0, 0] = COLOR_NONE
    board[1, 1] = COLOR_BLACK
    assert ai.go(board) == []
    assert ai.candidate_list == []


def test_full_game_smoke():
    # 两个 AI 交替下到终局，验证每步合法且能完成完整对局。
    # 冒烟只验证对局闭环与接口协作，故使用浅深度 1 与较小 time_out 以控制测试时长。
    black_ai = AI(8, COLOR_BLACK, 0.1)
    white_ai = AI(8, COLOR_WHITE, 0.1)
    black_ai.depth = 1
    white_ai.depth = 1
    board = initial_board()
    color = COLOR_BLACK
    steps = 0
    while not is_terminal(board) and steps < 200:
        ai = black_ai if color == COLOR_BLACK else white_ai
        moves = legal_moves(board, color)
        if moves:
            cand = ai.go(board_to_array(board))
            assert cand, "有合法走法时不应返回空列表"
            chosen = cand[-1]
            assert chosen in moves
            board = apply_move(board, color, chosen)
        color = -color
        steps += 1
    assert is_terminal(board)
