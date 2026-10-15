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


def test_evaluate_candidate_weighted_mean_u():
    """self-play 加权目标：baseline_weight=2、greedy_weight=1 时 mean_u 按 4:2 加权。"""
    runner = MatchRunner(depth=1)
    weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    summary = runner.evaluate_candidate(
        weights, BASELINE, greedy_seeds=(101,), baseline_weight=2.0, greedy_weight=1.0
    )
    assert len(summary.games) == 4
    assert summary.wins + summary.draws + summary.losses == 4
    u = summary.u_values
    expected = (2.0 * (u[0] + u[1]) + 1.0 * (u[2] + u[3])) / 6.0
    assert summary.mean_u == pytest.approx(expected)


def test_objective_weighted_selfplay_ratio():
    """ReversiObjective 透传加权参数，cost = (1 - weighted_mean_u)/2。"""
    obj = ReversiObjective(
        BASELINE, depth=1, greedy_seeds=(101,), baseline_weight=2.0, greedy_weight=1.0
    )
    z = np.array([0.5] * 5)
    result = obj(z)
    m = result.metrics
    assert len(m["per_game"]) == 4
    u = [pg["u"] for pg in m["per_game"]]
    expected_mean_u = (2.0 * (u[0] + u[1]) + 1.0 * (u[2] + u[3])) / 6.0
    assert m["mean_u"] == pytest.approx(expected_mean_u)
    assert result.cost == pytest.approx((1.0 - expected_mean_u) / 2.0)


def test_validate_weights_opponent_param():
    """A1：validate_weights 支持 opponent_weights；对手=候选自身时前两局互为镜像（u0 == -u1）。"""
    from experiments.plan9 import validate_weights

    runner = MatchRunner(depth=1)
    weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    r = validate_weights(weights, runner, (603, 604), opponent_weights=weights)
    assert len(r["u_values"]) == 6
    assert r["u_values"][0] == pytest.approx(-r["u_values"][1])
    assert "cost" in r and "mean_u" in r
