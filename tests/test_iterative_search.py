"""Budgeted search regressions shared by training and single-file deployment."""
import importlib
import time
from functools import lru_cache

import numpy as np
import pytest

from reversi import search
from reversi.board import COLOR_BLACK, empty_count, initial_board
from reversi.evaluator import evaluate, terminal_utility
from reversi.rules import apply_move, is_terminal, legal_moves

WEIGHTS = np.array([-6.33247672352715, -3.629403162945928, 0.9349258943961338,
                    -9.421310195876254, -0.6242989774127423])


@pytest.fixture(params=['reversi.search', 'submission.agent'])
def engine(request):
    return importlib.import_module(request.param)


def position(empties, seed=19):
    rng = np.random.default_rng(seed)
    board, color = initial_board(), COLOR_BLACK
    while empty_count(board) > empties and not is_terminal(board):
        moves = legal_moves(board, color)
        if moves:
            board = apply_move(board, color, moves[int(rng.integers(len(moves)))])
        color = -color
    return board, color


@pytest.mark.parametrize('budget', [1, 3, 31, 2000, 5000])
@pytest.mark.parametrize('empties', [60, 30, 14, 13])
def test_strict_deterministic_budget_and_legal_fallback(engine, budget, empties):
    board, color = position(empties)
    before = board.copy()
    first, second = {}, {}
    a = engine.best_move_iterative(board, color, 10, WEIGHTS, node_limit=budget, stats=first)
    b = engine.best_move_iterative(board, color, 10, WEIGHTS, node_limit=budget, stats=second)
    assert a == b and first == second
    assert a in legal_moves(board, color)
    assert first['nodes'] <= budget
    assert first['fallback_used'] == (first['completed_depth'] == 0)
    assert first['terminal_solved'] is False
    np.testing.assert_array_equal(board, before)


def test_endgame_first_iteration_is_not_forced_to_solve(engine):
    board, color = position(14)
    stats = {}
    engine.best_move_iterative(board, color, 10, WEIGHTS, node_limit=5000, stats=stats)
    assert stats['completed_depth'] >= 1
    assert stats['fallback_used'] is False
    assert stats['target_depth'] == 14


def test_no_completed_iteration_has_explicit_fallback(engine):
    board = initial_board()
    stats = {'stale': True}
    move = engine.best_move_iterative(board, COLOR_BLACK, 10, WEIGHTS, node_limit=1, stats=stats)
    assert move == legal_moves(board, COLOR_BLACK)[0]
    assert stats['completed_depth'] == 0
    assert stats['fallback_used'] and stats['interrupted']
    assert stats['stop_reason'] == 'node_limit'
    assert 'stale' not in stats


def test_abort_preserves_last_completed_iteration(monkeypatch):
    calls = []

    def fake_run(state, me, opp, color, depth, solve_endgame=True):
        calls.append((depth, solve_endgame))
        state['nodes'][0] += 1
        if depth == 2:
            state['abort'][0] = 1
            return 19
        return 26

    monkeypatch.setattr(search, '_run', fake_run)
    stats = {}
    move = search.best_move_iterative(initial_board(), COLOR_BLACK, 4, WEIGHTS,
                                      node_limit=100, stats=stats)
    assert move == (3, 2)
    assert stats['completed_depth'] == 1
    assert calls == [(1, False), (2, False)]


def test_expired_deadline_is_observable(engine, monkeypatch):
    clock = iter([100.0, 102.0])
    monkeypatch.setattr(time, 'time', lambda: next(clock))
    stats = {}
    move = engine.best_move_iterative(initial_board(), COLOR_BLACK, 10, WEIGHTS,
                                      time_budget=1.0, stats=stats)
    assert move in legal_moves(initial_board(), COLOR_BLACK)
    assert stats['completed_depth'] == stats['nodes'] == 0
    assert stats['interrupted'] and stats['stop_reason'] == 'time_budget'


def test_pass_and_terminal_status(engine):
    terminal = np.array([np.uint64(1), np.uint64(0)], dtype=np.uint64)
    stats = {}
    assert engine.best_move_iterative(terminal, COLOR_BLACK, 10, WEIGHTS, stats=stats) is None
    assert stats['terminal_solved'] and stats['nodes'] == 0
    for seed in range(30):
        board, color = position(4, seed)
        for side in (color, -color):
            if not legal_moves(board, side) and legal_moves(board, -side):
                assert engine.best_move_iterative(board, side, 10, WEIGHTS, stats=stats) is None
                assert not stats['terminal_solved']
                return
    pytest.fail('pass fixture not found')


def test_small_endgame_matches_independent_exhaustive_search(engine):
    board, color = position(5, 23)

    @lru_cache(None)
    def exact(black, white, side):
        bb = np.array([black, white], dtype=np.uint64)
        moves = legal_moves(bb, side)
        if not moves:
            if not legal_moves(bb, -side):
                return terminal_utility(bb, side)
            return -exact(black, white, -side)
        return max(-exact(*(int(x) for x in apply_move(bb, side, m)), -side) for m in moves)

    stats = {}
    move = engine.best_move_iterative(board, color, 2, WEIGHTS, node_limit=100000, stats=stats)
    result = apply_move(board, color, move)
    assert -exact(int(result[0]), int(result[1]), -color) == exact(int(board[0]), int(board[1]), color)
    assert stats['terminal_solved']
    assert stats['completed_depth'] == empty_count(board)


@pytest.mark.parametrize('empties', [60, 32, 15, 14, 8])
@pytest.mark.parametrize('budget', [1, 73, 2000, 5000])
def test_training_deployment_identical_moves_and_stats(empties, budget):
    deployment = importlib.import_module('submission.agent')
    board, color = position(empties, 103)
    a, b = {}, {}
    assert search.best_move_iterative(board, color, 10, WEIGHTS, node_limit=budget, stats=a) == (
        deployment.best_move_iterative(board, color, 10, WEIGHTS, node_limit=budget, stats=b))
    assert a == b


def test_weighted_agent_uses_iterative_and_exposes_stats(monkeypatch):
    from experiments import match_runner

    def fake(bb, color, max_depth, weights, **kwargs):
        assert max_depth == 10 and kwargs['node_limit'] == 2000
        kwargs['stats']['completed_depth'] = 3
        return (2, 3)

    monkeypatch.setattr('experiments.match_runner.best_move_iterative', fake)
    agent = match_runner.WeightedAgent(WEIGHTS)
    assert agent.select_move(initial_board(), COLOR_BLACK, [(2, 3)]) == (2, 3)
    assert agent.last_search_stats['completed_depth'] == 3


def test_candidate_and_incumbent_receive_identical_search_settings(monkeypatch):
    from experiments.match_runner import GameResult, MatchRunner
    runner = MatchRunner(depth=7, node_limit=123)
    seen = []

    def play(black, white, seed=None):
        seen.extend([black, white])
        return GameResult(32, 32, 60, 0.0)

    monkeypatch.setattr(runner, 'play_game', play)
    runner.validate_protocol(WEIGHTS, -WEIGHTS, [WEIGHTS] * 3)
    weighted = [a for a in seen if hasattr(a, 'weights')]
    assert weighted and all(a.depth == 7 and a.node_limit == 123 for a in weighted)


@pytest.mark.parametrize('depth', [0, -1, 65])
def test_invalid_depth_is_rejected(engine, depth):
    with pytest.raises(ValueError, match='max_depth'):
        engine.best_move_iterative(initial_board(), COLOR_BLACK, depth, WEIGHTS)


def test_aborted_parent_never_writes_transposition_entry():
    board = initial_board()
    state = search._new_state()
    search._prepare(state, WEIGHTS, None, 1)
    me, opp = search._masks(board, COLOR_BLACK)
    search._negamax_nb(me, opp, COLOR_BLACK, 4, -1e300, 1e300, state['weights'],
                       state['tt_key'], state['tt_depth'], state['tt_flag'],
                       state['tt_val'], state['tt_move'], state['killers'],
                       state['history'], state['nodes'], state['abort'],
                       state['node_limit'], False)
    assert state['abort'][0] == 1 and state['nodes'][0] == 1
    assert np.all(state['tt_depth'] == -1)
    assert np.all(state['killers'] == -1)
    assert np.all(state['history'] == 0)


def test_default_search_settings_and_game_diagnostics():
    from experiments.match_runner import MatchRunner, WeightedAgent
    from experiments.plan9 import DEPTH
    from experiments.reversi_objective import ReversiObjective

    runner = MatchRunner()
    assert DEPTH == runner.depth == ReversiObjective(WEIGHTS).runner.depth == 10
    assert runner.node_limit == ReversiObjective(WEIGHTS).runner.node_limit == 2000
    game = runner.play_game(WeightedAgent(WEIGHTS), WeightedAgent(WEIGHTS))
    assert game.search_stats
    for stats in game.search_stats:
        assert stats['color'] in (-1, 1)
        assert stats['nodes'] <= 2000
        assert stats['completed_depth'] >= 1
        assert 'interrupted' in stats and 'terminal_solved' in stats


@pytest.mark.parametrize('depth', [1, 2, 3])
@pytest.mark.parametrize('seed', [29, 43])
def test_bounded_search_matches_independent_minimax(engine, depth, seed):
    board, side = position(35, seed)

    def reference(bb, color, remaining):
        moves = legal_moves(bb, color)
        if not moves:
            if not legal_moves(bb, -color):
                return terminal_utility(bb, color)
            return -reference(bb, -color, remaining)
        if remaining <= 0:
            return evaluate(bb, color, WEIGHTS)
        return max(-reference(apply_move(bb, color, m), -color, remaining - 1)
                   for m in moves)

    stats = {}
    move = engine.best_move_iterative(board, side, depth, WEIGHTS, stats=stats)
    expected = reference(board, side, depth)
    actual = -reference(apply_move(board, side, move), -side, depth - 1)
    assert actual == pytest.approx(expected)
    assert stats['completed_depth'] == depth and not stats['interrupted']