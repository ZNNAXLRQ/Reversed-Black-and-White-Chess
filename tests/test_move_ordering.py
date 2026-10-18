"""Production/deployment ordering contracts, independent of experiment artifacts."""
import importlib

import numpy as np
import pytest

from reversi.board import initial_board
from reversi.rules import legal_moves


@pytest.fixture(params=["reversi.search", "submission.agent"])
def engine(request):
    return importlib.import_module(request.param)


@pytest.mark.parametrize("hint", [-1, 64, 100, 55, 41])
def test_hint_is_legal_and_stable(engine, hint):
    moves = np.array([8, 23, 41, 9, 63], dtype=np.int8)
    engine._promote_hint(moves, 4, hint)
    expected = [41, 8, 23, 9, 63] if hint == 41 else [8, 23, 41, 9, 63]
    np.testing.assert_array_equal(moves, expected)


def test_prepare_clears_previous_root_and_tt(engine):
    state = engine._new_state()
    state["root_pv"][0] = 26
    state["tt_move"].fill(26)
    engine._prepare(state, np.zeros(5), None, 500)
    assert state["root_pv"][0] == -1
    assert np.all(state["tt_move"] == -1)


def test_only_completed_root_becomes_next_hint(engine, monkeypatch):
    seen = []
    captured = []

    def run(state, me, opp, side, depth, solve_endgame=True):
        seen.append(int(state["root_pv"][0]))
        captured.append(state)
        if depth == 2:
            state["abort"][0] = 1
            return 19
        return 26

    monkeypatch.setitem(engine.__dict__, "_run", run)
    stats = {}
    assert engine.best_move_iterative(initial_board(), -1, 4, np.zeros(5), stats=stats) == (3, 2)
    assert seen == [-1, 26] and captured[-1]["root_pv"][0] == 26
    assert stats["completed_depth"] == 1


@pytest.mark.parametrize("collision,illegal", [(False, False), (True, False), (False, True)])
def test_shallow_tt_hint_needs_matching_key_and_legal_move(engine, collision, illegal):
    board = initial_board()
    state = engine._new_state()
    engine._prepare(state, np.zeros(5), None, None)
    me, opp = engine._masks(board, -1)
    key = engine._hash(-1, me, opp)
    index = int(key) & (engine._TT_SIZE - 1)
    state["tt_key"][index] = np.uint64(int(key) ^ (1 << 40) if collision else int(key))
    state["tt_depth"][index] = 0  # Hint usable, value is NOT a depth-1 exact hit.
    state["tt_move"][index] = 0 if illegal else 44
    state["tt_val"][index] = 987654.0
    value = engine._negamax_nb(me, opp, -1, 1, -1e300, -1e299, state["weights"],
                              state["tt_key"], state["tt_depth"], state["tt_flag"],
                              state["tt_val"], state["tt_move"], state["killers"],
                              state["history"], state["nodes"], state["abort"],
                              state["node_limit"], False)
    assert value == 0
    assert state["tt_move"][index] == (19 if collision or illegal else 44)


def test_root_hint_breaks_ties_without_changing_value(engine):
    board = initial_board()
    state = engine._new_state()
    engine._prepare(state, np.zeros(5), None, None)
    state["root_pv"][0] = 44
    me, opp = engine._masks(board, -1)
    assert engine._run(state, me, opp, -1, 1, solve_endgame=False) == 44
    assert (5, 4) in legal_moves(board, -1)


def test_aborted_parent_cannot_publish_tt_hint(engine):
    state = engine._new_state()
    engine._prepare(state, np.zeros(5), None, 1)
    me, opp = engine._masks(initial_board(), -1)
    engine._negamax_nb(me, opp, -1, 4, -1e300, 1e300, state["weights"],
                      state["tt_key"], state["tt_depth"], state["tt_flag"],
                      state["tt_val"], state["tt_move"], state["killers"],
                      state["history"], state["nodes"], state["abort"],
                      state["node_limit"], False)
    assert state["abort"][0] == 1
    assert np.all(state["tt_depth"] == -1) and np.all(state["tt_move"] == -1)