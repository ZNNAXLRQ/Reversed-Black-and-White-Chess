"""ReversiObjective 与 MatchRunner 集成测试。

注意：为控制测试时长，使用 depth=1；正式实验的搜索深度 3 在 Plan 6 端到端运行。
"""
import numpy as np
import pytest

from experiments.match_runner import (
    GreedyAgent, H2HResult, MatchRunner, MatchSummary, WeightedAgent,
)
from experiments.reversi_objective import ReversiObjective
from optimization.objective import EvaluationResult
from reversi.board import COLOR_BLACK, count, initial_board
from reversi.rules import apply_move, legal_moves

BASELINE = np.array([-1.0, 1.0, 1.0, 1.0, -1.0], dtype=np.float64)


def test_greedy_agent_picks_fewest_discs():
    board = initial_board()
    moves = legal_moves(board, COLOR_BLACK)
    agent = GreedyAgent()
    move = agent.select_move(board, COLOR_BLACK, moves, seed=0)
    assert move in moves
    chosen_count = count(apply_move(board, COLOR_BLACK, move), COLOR_BLACK)
    for m in moves:
        assert chosen_count <= count(apply_move(board, COLOR_BLACK, m), COLOR_BLACK)


def test_objective_returns_evaluation_result():
    obj = ReversiObjective(BASELINE, depth=1)
    z = np.array([0.5] * 5)  # w = [0]*5
    result = obj(z)
    assert isinstance(result, EvaluationResult)
    assert 0.0 <= result.cost <= 1.0
    m = result.metrics
    assert m["wins"] + m["draws"] + m["losses"] == 6
    assert "mean_u" in m
    assert len(m["per_game"]) == 6


def test_objective_cost_bounds():
    obj = ReversiObjective(BASELINE, depth=1)
    for z in ([0.0] * 5, [0.5] * 5, [1.0] * 5):
        assert 0.0 <= obj(np.array(z)).cost <= 1.0


def test_objective_decodes_z_to_w():
    obj = ReversiObjective(BASELINE, depth=1)
    z = np.array([0.0, 0.5, 1.0, 0.25, 0.75])
    w = obj.decode(z)
    assert np.allclose(w, [-10.0, 0.0, 10.0, -5.0, 5.0])


def test_objective_invalid_weights_raise():
    obj = ReversiObjective(BASELINE, depth=1)
    with pytest.raises(ValueError):
        obj(np.array([0.5] * 4))
    with pytest.raises(ValueError):
        obj(np.array([0.5, 0.5, 0.5, 0.5, 1.5]))


def test_play_game_returns_result():
    runner = MatchRunner(depth=1)
    r = runner.play_game(WeightedAgent(BASELINE, 1), GreedyAgent(), seed=101)
    assert r.black_count + r.white_count <= 64
    assert r.steps > 0


def test_h2h_two_games_and_antisymmetry():
    runner = MatchRunner(depth=1)
    wA = np.array([-1.0, 1.0, 1.0, 1.0, -1.0])
    wB = np.array([1.0, -1.0, -1.0, -1.0, 1.0])
    r1 = runner.head_to_head(wA, wB)
    r2 = runner.head_to_head(wB, wA)
    assert isinstance(r1, H2HResult)
    assert len(r1.games) == 2 and len(r1.u_values) == 2
    assert r1.mean_u == pytest.approx(-r2.mean_u)


def test_h2h_challenger_wins_flag():
    runner = MatchRunner(depth=1)
    r = runner.head_to_head(BASELINE, BASELINE)
    # 相同权重 -> mean_u == 0 -> 不判胜
    assert r.mean_u == pytest.approx(0.0)
    assert r.challenger_wins is False

