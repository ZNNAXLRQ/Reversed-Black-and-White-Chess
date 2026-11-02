"""Submission contract, recursive deadlines, total budget and isolated memory."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import pytest

from experiments.check_submission_resources import (
    apply_reference, initial_board, reference_moves,
)
from submission import agent


ROOT = Path(__file__).resolve().parents[1]


def assert_contract(ai, board):
    expected = reference_moves(board, ai.color)
    before = board.copy()
    result = ai.go(board)
    np.testing.assert_array_equal(board, before)
    assert result is ai.candidate_list
    assert set(result) == set(expected)
    assert all(type(move) is tuple and len(move) == 2
               and all(type(value) is int for value in move) for move in result)
    if expected:
        assert len(result) == len(expected) + 1
        assert set(result[:-1]) == set(expected) and result[-1] in expected
    else:
        assert result == []


def test_source_is_single_file_without_forbidden_or_jit_imports():
    source = (ROOT / "submission" / "agent.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"open", "exec", "eval", "__import__"}
    assert set(imports) == {"array", "time", "numpy"}
    assert tuple(inspect.signature(agent.AI.__init__).parameters) == (
        "self", "chessboard_size", "color", "time_out",
    )
    assert tuple(inspect.signature(agent.AI.go).parameters) == ("self", "chessboard")


def test_fast_kernels_match_independent_oracle_on_arbitrary_bitboards():
    rng = random.Random(17029)
    for _ in range(250):
        me = rng.getrandbits(64)
        opp = rng.getrandbits(64) & ~me
        board = np.zeros((8, 8), dtype=np.int8)
        for index in range(64):
            if me & (1 << index):
                board[index // 8, index % 8] = -1
            elif opp & (1 << index):
                board[index // 8, index % 8] = 1
        for color, own, other in ((-1, me, opp), (1, opp, me)):
            expected = reference_moves(board, color)
            mask = agent._legal_mask_nb(own, other)
            assert mask == sum(1 << (r * 8 + c) for r, c in expected)
            for (r, c), flips in expected.items():
                assert agent._flips_mask_nb(own, other, 1 << (r * 8 + c)) == sum(
                    1 << (rr * 8 + cc) for rr, cc in flips
                )
            assert agent._evaluate_nb(own, other, agent._FIXED_WEIGHTS, mask) == (
                agent._evaluate_nb(own, other, agent._FIXED_WEIGHTS)
            )
        expected_position = sum(agent._POSITION_WEIGHTS[i] for i in range(64) if me & (1 << i))
        assert agent._position_sum(me) == expected_position
        empty = agent.FULL_MASK ^ (me | opp)
        frontier = 0
        for r in range(8):
            for c in range(8):
                if board[r, c] != -1:
                    continue
                if any(0 <= r + dr < 8 and 0 <= c + dc < 8 and board[r + dr, c + dc] == 0
                       for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc):
                    frontier += 1
        assert agent._frontier_count_nb(me, empty) == frontier


def test_random_reachable_boards_and_flips_match_independent_oracle():
    ai = agent.AI(8, -1, 0.001)  # Contract checks need not spend time searching.
    rng = random.Random(20261102)
    for _ in range(6):
        board = initial_board(np)
        color = -1
        for ply in range(125):
            for side in (-1, 1):
                expected = reference_moves(board, side)
                bb = agent.board_from_array(board)
                assert set(agent.legal_moves(bb, side)) == set(expected)
                me, opp = agent._masks(bb, side)
                for move, flips in expected.items():
                    pos = 1 << (move[0] * 8 + move[1])
                    assert agent._flips_mask_nb(me, opp, pos) == sum(
                        1 << (r * 8 + c) for r, c in flips
                    )
                if ply % 8 == 0 or not expected:
                    ai.color = side
                    assert_contract(ai, board)
            moves = reference_moves(board, color)
            if not moves and not reference_moves(board, -color):
                break
            if moves:
                apply_reference(board, color, rng.choice(list(moves)), moves)
            color = -color
        else:
            pytest.fail("reference game failed to terminate")


def test_pass_clears_previous_candidates_and_updates_total_time():
    ai = agent.AI(8, -1, 5)
    ai.candidate_list[:] = [(2, 3), (3, 2)]
    ai.last_search_stats["stale"] = True
    board = np.ones((8, 8), dtype=np.int8)
    board[0, 0], board[1, 1] = 0, -1
    used = ai._time_used
    assert_contract(ai, board)
    assert ai.last_search_stats == {} and ai._time_used >= used


@pytest.mark.parametrize("requested", [5, 10, 180, 0, -1, None, float("inf"), float("nan")])
def test_go_always_caps_budget_even_with_invalid_or_large_timeout(monkeypatch, requested):
    ai = agent.AI(8, -1, requested)
    board = initial_board(np)
    # Few empty squares allow this test to exercise the per-turn cap, not only
    # the more conservative whole-game allocation.
    board[:] = 1
    board[0, 0] = board[0, 7] = 0
    board[0, 2] = board[0, 5] = -1
    captured = []

    def choose(bb, color, depth, weights, time_budget=None, stats=None):
        captured.append(time_budget)
        return agent.legal_moves(bb, color)[0]

    monkeypatch.setattr(agent, "best_move_iterative", choose)
    assert_contract(ai, board)
    assert len(captured) == 1 and 0 < captured[0] < 4.4


def test_exhausted_game_budget_never_starts_search(monkeypatch):
    ai = agent.AI(8, -1, 5)
    ai._time_used = agent._GAME_BUDGET_SECONDS

    def unexpected_search(*args, **kwargs):
        pytest.fail("search started after cumulative budget was exhausted")

    monkeypatch.setattr(agent, "best_move_iterative", unexpected_search)
    assert_contract(ai, initial_board(np))


def test_opening_allocation_recovers_time_without_increasing_game_cap(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(agent, "_clock", lambda: clock[0])
    ai = agent.AI(8, -1, 5)
    captured = []

    def choose(bb, color, depth, weights, time_budget=None, stats=None):
        captured.append(time_budget)
        return agent.legal_moves(bb, color)[0]

    monkeypatch.setattr(agent, "best_move_iterative", choose)
    assert_contract(ai, initial_board(np))
    assert agent._GAME_BUDGET_SECONDS == 75.0
    assert len(captured) == 1 and 2.0 < captured[0] < 2.2


def test_consecutive_turns_after_passes_cannot_overdraw_game_budget(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(agent, "_clock", lambda: clock[0])
    ai = agent.AI(8, -1, 5)
    requested = []

    def choose(bb, color, depth, weights, time_budget=None, stats=None):
        requested.append(time_budget)
        clock[0] += time_budget
        return agent.legal_moves(bb, color)[0]

    monkeypatch.setattr(agent, "best_move_iterative", choose)
    board = initial_board(np)
    for _ in range(200):
        assert_contract(ai, board)
    assert len(requested) > 60
    assert sum(requested) <= agent._GAME_BUDGET_SECONDS
    assert ai._time_used <= agent._GAME_BUDGET_SECONDS
    assert all(0 < budget < agent._MOVE_BUDGET_SECONDS for budget in requested)


def test_game_budget_includes_calls_and_does_not_reset_per_turn(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(agent, "_clock", lambda: clock[0])
    ai = agent.AI(8, -1, 180)
    ai._time_used = agent._GAME_BUDGET_SECONDS - 1.0
    board = initial_board(np)
    captured = []

    def choose(bb, color, depth, weights, time_budget=None, stats=None):
        captured.append(time_budget)
        clock[0] += time_budget + 0.001
        return agent.legal_moves(bb, color)[0]

    monkeypatch.setattr(agent, "best_move_iterative", choose)
    for _ in range(3):
        assert_contract(ai, board)
    assert len(captured) == 3 and captured[0] > captured[1] > captured[2] > 0
    assert ai._time_used == pytest.approx(
        agent._GAME_BUDGET_SECONDS - 1.0 + sum(captured) + 0.003,
    )
    ai._time_used = agent._GAME_BUDGET_SECONDS - 0.0001
    assert_contract(ai, board)
    assert len(captured) == 3  # Too little remaining time to spend on search.


def test_forced_single_move_returns_immediately_without_search(monkeypatch):
    ai = agent.AI(8, 1, 5)
    board = np.ones((8, 8), dtype=np.int8)
    board[0, 0], board[1, 1] = 0, -1

    def unexpected_search(*args, **kwargs):
        pytest.fail("there is no decision to search when only one move is legal")

    monkeypatch.setattr(agent, "best_move_iterative", unexpected_search)
    assert_contract(ai, board)
    assert ai.candidate_list == [(0, 0), (0, 0)]


def test_recursive_deadline_interrupts_during_a_depth_iteration(monkeypatch):
    board = agent.board_from_array(initial_board(np))
    clock = [100.0]
    real_evaluate = agent._evaluate_nb
    monkeypatch.setattr(agent, "_clock", lambda: clock[0])

    def slow_leaf(me, opp, weights, moves_mask=None):
        clock[0] += 0.02
        return real_evaluate(me, opp, weights, moves_mask)

    monkeypatch.setattr(agent, "_evaluate_nb", slow_leaf)
    stats = {}
    move = agent.best_move_iterative(board, -1, 10, agent._FIXED_WEIGHTS,
                                     time_budget=0.12, stats=stats)
    assert move in agent.legal_moves(board, -1)
    assert stats["interrupted"] and stats["stop_reason"] == "time_budget"
    assert stats["completed_depth"] == 1 and stats["nodes"] <= 32


def test_aborted_iteration_preserves_last_complete_move(monkeypatch):
    seen = []

    def run(state, me, opp, color, depth, solve_endgame=True):
        seen.append(depth)
        if depth == 2:
            state["abort"][0] = 2
            return 19
        return 26

    monkeypatch.setattr(agent, "_run", run)
    stats = {}
    move = agent.best_move_iterative(agent.board_from_array(initial_board(np)), -1, 10,
                                     agent._FIXED_WEIGHTS, stats=stats)
    assert seen == [1, 2] and move == (3, 2)
    assert stats["completed_depth"] == 1 and stats["stop_reason"] == "time_budget"


@pytest.mark.parametrize("budget", [0.0, -1.0, 0.01, 0.05])
def test_actual_search_respects_small_wallclock_budget(budget):
    stats = {}
    started = time.perf_counter()
    move = agent.best_move_iterative(agent.board_from_array(initial_board(np)), -1, 10,
                                     agent._FIXED_WEIGHTS, time_budget=budget, stats=stats)
    elapsed = time.perf_counter() - started
    assert move in {(2, 3), (3, 2), (4, 5), (5, 4)}
    assert elapsed < max(0.0, budget) + 0.15
    assert stats["interrupted"] and stats["stop_reason"] == "time_budget"


def test_single_file_import_and_os_peak_memory_in_fresh_process(tmp_path):
    report_path = tmp_path / "resources.json"
    result = subprocess.run([
        sys.executable, "-B", str(ROOT / "experiments" / "check_submission_resources.py"),
        "--games", "0", "--positions", "1", "--time-out", "0.02",
        "--report", str(report_path),
    ], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["source_sha256"] == hashlib.sha256(
        (ROOT / "submission" / "agent.py").read_bytes(),
    ).hexdigest()
    assert report["passed"] and report["peak_rss_bytes"] < 100000000
    assert not report["numba_loaded"]
    assert report["first_go_seconds"] < 5.0