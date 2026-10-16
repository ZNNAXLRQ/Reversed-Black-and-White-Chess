"""评估器单元测试：5 特征、视角对称、终局效用、维度检查。"""
import numpy as np
import pytest

from reversi.board import (
    COLOR_BLACK, COLOR_WHITE, board_from_array, initial_board,
)
from reversi.evaluator import evaluate, features, terminal_utility


def _make(*cells):
    """cells 为 (r, c, color) 三元组，返回位棋盘。"""
    arr = np.zeros((8, 8), dtype=np.int8)
    for r, c, color in cells:
        arr[r, c] = color
    return board_from_array(arr)


def test_features_shape():
    board = initial_board()
    f = features(board, COLOR_BLACK)
    assert f.shape == (5,)


def test_disc_diff_value():
    board = _make((0, 0, COLOR_BLACK), (0, 1, COLOR_BLACK), (0, 2, COLOR_WHITE))
    f = features(board, COLOR_BLACK)
    assert f[0] == pytest.approx((2 - 1) / 64.0)


def test_corner_diff_value():
    board = _make((0, 0, COLOR_BLACK), (0, 7, COLOR_BLACK), (7, 0, COLOR_WHITE))
    f = features(board, COLOR_BLACK)
    assert f[3] == pytest.approx((2 - 1) / 4.0)


def test_initial_symmetric_features_zero():
    board = initial_board()
    f = features(board, COLOR_BLACK)
    # 初始局面双方完全对称：disc/position/mobility/corner/frontier 差分均为 0
    assert f[0] == pytest.approx(0.0)
    assert f[1] == pytest.approx(0.0)
    assert f[2] == pytest.approx(0.0)
    assert f[3] == pytest.approx(0.0)
    assert f[4] == pytest.approx(0.0)


def test_viewpoint_antisymmetry():
    board = _make((0, 0, COLOR_BLACK), (0, 1, COLOR_BLACK), (0, 2, COLOR_WHITE))
    weights = np.array([-1.0, 1.0, 1.0, 1.0, -1.0])
    assert evaluate(board, COLOR_BLACK, weights) == pytest.approx(
        -evaluate(board, COLOR_WHITE, weights)
    )


def test_terminal_utility_fewer_discs_wins():
    arr = np.zeros((8, 8), dtype=np.int8)
    arr.flat[0:10] = COLOR_BLACK
    arr.flat[10:30] = COLOR_WHITE
    board = board_from_array(arr)
    # 黑 10 白 20：黑更少 → 黑视角为正
    assert terminal_utility(board, COLOR_BLACK) == pytest.approx(1000.0 + (20 - 10))
    assert terminal_utility(board, COLOR_WHITE) == pytest.approx(-1000.0 + (10 - 20))


def test_terminal_utility_tie():
    arr = np.zeros((8, 8), dtype=np.int8)
    arr.flat[0:10] = COLOR_BLACK
    arr.flat[10:20] = COLOR_WHITE
    board = board_from_array(arr)
    assert terminal_utility(board, COLOR_BLACK) == 0.0
    assert terminal_utility(board, COLOR_WHITE) == 0.0


def test_weights_affect_evaluation():
    board = _make((0, 0, COLOR_BLACK), (0, 1, COLOR_BLACK), (0, 2, COLOR_WHITE))
    w1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    w2 = np.array([2.0, 0.0, 0.0, 0.0, 0.0])
    e1 = evaluate(board, COLOR_BLACK, w1)
    e2 = evaluate(board, COLOR_BLACK, w2)
    assert e2 == pytest.approx(2 * e1)
    assert e1 != e2


def test_weights_shape_check():
    board = initial_board()
    with pytest.raises(ValueError):
        evaluate(board, COLOR_BLACK, np.array([1.0, 2.0]))
