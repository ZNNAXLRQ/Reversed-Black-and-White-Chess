"""v2 protocol integrity, paired scoring, stopping rules and real replay tests."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import validation as v
from experiments.match_runner import GameResult, GreedyAgent, MatchRunner
from reversi.board import count, initial_board
from reversi.rules import apply_move, is_terminal, legal_moves

INCUMBENT = [0.0] * 5
HISTORY = [{"cycle_index": i, "weights": [float(i), 0, 0, 0, 0],
            "validation_cost": 1 - i / 10} for i in range(1, 8)]


@pytest.fixture
def frozen(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    return v.freeze_protocol(tmp_path, run, "unit_only_987654", INCUMBENT, HISTORY, 3, 31), run


def test_protocol_freezes_before_results_and_never_overwrites(frozen, tmp_path):
    protocol, run = frozen
    v.verify_protocol(protocol)
    assert len(protocol["validation"]) == 12 and len(protocol["holdout"]) == 24
    keys = [o["canonical_sha256"] for phase in ("validation", "holdout") for o in protocol[phase]]
    assert len(set(keys)) == 36
    assert protocol["games_per_model"] == {"validation": 120, "holdout": 240}
    assert [r["cycle_index"] for r in protocol["opponents"]] == [1, 4, 7]
    before = (run / "validation_protocol.json").read_bytes()
    with pytest.raises(FileExistsError):
        v.freeze_protocol(tmp_path, run, "unit_only_987654", INCUMBENT, HISTORY, 3, 31)
    assert before == (run / "validation_protocol.json").read_bytes()


def test_history_cost_does_not_change_opponents():
    changed = copy.deepcopy(HISTORY)
    for row in changed:
        row["validation_cost"] = float("nan")
    assert v.select_opponents(HISTORY, INCUMBENT) == v.select_opponents(changed, INCUMBENT)


@pytest.mark.parametrize("history", [HISTORY[:2], HISTORY[:2] + [HISTORY[0]],
                                    [{"cycle_index": 1, "weights": [float("nan")] * 5}]])
def test_invalid_pool_fails_closed(history):
    with pytest.raises(ValueError):
        v.select_opponents(history, INCUMBENT)


def test_prior_samples_and_dataset_boards_excluded(frozen, tmp_path):
    protocol, _ = frozen
    artifacts = tmp_path / "artifacts" / "old"
    artifacts.mkdir(parents=True)
    (artifacts / "samples.json").write_text(json.dumps({"old": protocol["validation"]}), encoding="utf-8")
    data = tmp_path / "training" / "datasets" / "old"
    data.mkdir(parents=True)
    from reversi.board import board_to_array
    (data / "test.jsonl").write_text(json.dumps({"board_before": board_to_array(initial_board()).tolist(),
                                               "unused_action_label": "do not use"}), encoding="utf-8")
    excluded = v.prior_positions(tmp_path)
    assert v.canonical(initial_board()) in excluded
    assert all(o["canonical_sha256"] in excluded for o in protocol["validation"])
    fresh = v.generate_openings(12, np.random.default_rng(900025), set(excluded))
    assert not excluded.intersection(o["canonical_sha256"] for o in fresh)


@pytest.mark.parametrize("mutation", ["hash", "prefix", "overlap", "count", "budget"])
def test_corrupt_protocol_rejected(frozen, mutation):
    original, _ = frozen
    protocol = copy.deepcopy(original)
    if mutation == "hash":
        protocol["sha256"] = "wrong"
    else:
        if mutation == "prefix":
            protocol["validation"][0]["prefix"][0][1] = [0, 0]
        elif mutation == "overlap":
            protocol["holdout"][0] = protocol["validation"][0]
        elif mutation == "count":
            protocol["validation"].pop()
        else:
            protocol["search_policy"] = "old"
        protocol["sha256"] = v.digest({k: x for k, x in protocol.items() if k != "sha256"})
    with pytest.raises(ValueError):
        v.verify_protocol(protocol)


def fake_evaluation(frozen, monkeypatch, phase="validation"):
    protocol, _ = frozen
    runner = MatchRunner(depth=3, node_limit=31)
    calls = []

    def play(black, white, seed=None, opening=None):
        calls.append((black, white, seed, opening))
        return GameResult(20, 44, 1, 0.0, seed)

    monkeypatch.setattr(runner, "play_game", play)
    result = v.evaluate_weights([9, 0, 0, 0, 0], runner, protocol, phase)
    return result, calls, runner


def test_shared_openings_color_balance_and_equal_settings(frozen, monkeypatch):
    result, calls, _ = fake_evaluation(frozen, monkeypatch)
    assert len(calls) == len(result["games"]) == 120
    assert len(result["self_play_u_values"]) == 24
    assert result["mean_u"] == result["self_play_mean_u"] == 0
    assert result["cost"] == .5 and result["opening_clusters"] == [0] * 12
    for i in range(0, len(calls), 2):
        first, second = calls[i:i + 2]
        assert first[3] is second[3] and first[2] == second[2] == 999
        assert first[0].weights.tolist() == second[1].weights.tolist() == [9, 0, 0, 0, 0]
        for agent in (*first[:2], *second[:2]):
            if hasattr(agent, "weights"):
                assert agent.depth == 3 and agent.node_limit == 31


def test_wrong_runner_budget_rejected(frozen):
    protocol, _ = frozen
    with pytest.raises(ValueError, match="budget"):
        v.evaluate_weights(INCUMBENT, MatchRunner(depth=3, node_limit=32), protocol)


def test_failure_consumes_once_only_reservation(frozen, monkeypatch):
    protocol, run = frozen
    runner = MatchRunner(depth=3, node_limit=31)

    def fail(*args, **kwargs):
        raise RuntimeError("simulated interrupted validation")

    monkeypatch.setattr(v, "evaluate_weights", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        v.evaluate_once(INCUMBENT, runner, protocol, run, "finalist", "holdout")
    with pytest.raises(FileExistsError):
        v.evaluate_once(INCUMBENT, runner, protocol, run, "finalist", "holdout")
    assert not (run / "holdout_finalist_result.json").exists()


@pytest.mark.parametrize("corruption", ["truncated", "duplicate", "summary", "counts"])
def test_corrupt_holdout_cannot_pass_gate(frozen, monkeypatch, corruption):
    candidate, _, _ = fake_evaluation(frozen, monkeypatch, "holdout")
    incumbent = copy.deepcopy(candidate)
    if corruption == "truncated":
        candidate["games"].pop()
    elif corruption == "duplicate":
        candidate["game_metadata"][1] = copy.deepcopy(candidate["game_metadata"][0])
    elif corruption == "summary":
        candidate["opening_clusters"][0] = .9
    else:
        candidate["games"][0]["black_count"] = 64
    with pytest.raises(ValueError):
        v.compare_holdout(candidate, incumbent)


def test_successful_reservation_still_cannot_repeat(frozen, monkeypatch):
    protocol, run = frozen
    _, _, runner = fake_evaluation(frozen, monkeypatch)
    v.evaluate_once(INCUMBENT, runner, protocol, run, "incumbent")
    assert json.loads((run / "validation_incumbent_usage.json").read_text())["status"] == "completed"
    with pytest.raises(FileExistsError):
        v.evaluate_once(INCUMBENT, runner, protocol, run, "incumbent")


def test_cluster_bootstrap_resamples_openings_not_games():
    assert v.cluster_interval([.125] * 12) == [.125, .125]
    assert v.cluster_interval([-.5, .5] * 6)[0] < 0
    assert v.cluster_interval([-.5, .5] * 6) == v.cluster_interval([-.5, .5] * 6)


def test_holdout_requires_both_lower_bounds_positive(frozen, monkeypatch):
    candidate, _, _ = fake_evaluation(frozen, monkeypatch, "holdout")
    incumbent = copy.deepcopy(candidate)

    def set_scores(panel, h2h):
        for row, game in zip(candidate["game_metadata"], candidate["games"]):
            row["u"] = (h2h[row["opening_index"]] if row["opponent_id"] == "incumbent"
                        else panel[row["opening_index"]])
            difference = int(row["u"] * 64) * (-row["candidate_color"])
            game["black_count"] = (64 - difference) // 2
            game["white_count"] = (64 + difference) // 2
        candidate["u_values"] = [r["u"] for r in candidate["game_metadata"]]
        candidate["opening_clusters"] = [float(np.mean([r["u"] for r in candidate["game_metadata"]
                                                       if r["opening_index"] == i])) for i in range(24)]
        candidate["self_play_clusters"] = h2h

    set_scores([.125] * 24, [.25] * 24)
    assert v.compare_holdout(candidate, incumbent)["passed"]
    set_scores([.125] * 24, [0] * 24)
    assert not v.compare_holdout(candidate, incumbent)["passed"]
    set_scores([-.5, .25] * 12, [.25] * 24)
    assert not v.compare_holdout(candidate, incumbent)["passed"]
    candidate["game_metadata"][0]["candidate_color"] *= -1
    with pytest.raises(ValueError, match="unpaired"):
        v.compare_holdout(candidate, incumbent)


def test_real_continuation_replays_legal_opening_and_full_trace(frozen):
    protocol, _ = frozen
    opening = protocol["validation"][0]
    game = MatchRunner(depth=1, node_limit=31).play_game(GreedyAgent(), GreedyAgent(), seed=900025,
                                                       opening=opening)
    board, color = v.replay_opening(opening)
    placed = 0
    for row in game.trace:
        assert row["color"] == color
        moves = legal_moves(board, color)
        if row["move"] is None:
            assert not moves and legal_moves(board, -color)
        else:
            move = tuple(row["move"])
            assert move in moves
            board = apply_move(board, color, move)
            placed += 1
        color = -color
    assert is_terminal(board) and placed == game.steps
    assert game.final_board == [int(x) for x in board]
    assert (game.black_count, game.white_count) == (count(board, -1), count(board, 1))


@pytest.mark.parametrize("qualified,holdout_pass", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("preference_available", [False, True])
def test_orchestration_never_writes_before_holdout(tmp_path, monkeypatch, qualified, holdout_pass,
                                                preference_available):
    """All optimizer/match/writeback operations are mocks; no training or promotion."""
    from experiments import plan9
    log = SimpleNamespace(log=lambda message: None)
    monkeypatch.setattr(plan9, "_TRAINING_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(plan9, "_SOURCE_FILES", [])
    monkeypatch.setattr(plan9, "load_submission_weights", lambda path: np.zeros(5))
    monkeypatch.setattr(plan9, "next_version_number", lambda: 1)
    monkeypatch.setattr(plan9, "backup_submission", lambda *a: {})
    monkeypatch.setattr(plan9, "load_champion_pool", lambda: HISTORY)
    preference = {"candidate_id": "mock_preference_only", "weights": [3, 0, 0, 0, 0]}
    monkeypatch.setattr(plan9, "select_preference_candidate",
                        lambda logger: preference if preference_available else None)
    mock_protocol = {"sha256": "mock", "games_per_model": {"validation": 120, "holdout": 240},
                     "stopping_rule": "once"}
    monkeypatch.setitem(plan9.__dict__, "freeze_protocol", lambda *a: mock_protocol)
    optimizer_result = SimpleNamespace(best_candidate=np.full(5, .6), best_cost=.25, metrics={},
                                      num_evaluations=0, num_iterations=0, stop_reason="mock", history=[])
    monkeypatch.setitem(plan9.__dict__, "NCSOptimizer",
                        lambda *args: SimpleNamespace(optimize=lambda *unused: optimizer_result))
    events = []

    def evaluate(weights, runner, protocol, run_dir, label, phase="validation"):
        events.append((phase, label))
        is_candidate = label != "incumbent"
        return {"cost": .25 if is_candidate and qualified else .5,
                "mean_u": .5 if is_candidate else 0,
                "self_play_mean_u": .25 if is_candidate else 0, "wins": 1, "draws": 0, "losses": 0}

    monkeypatch.setitem(plan9.__dict__, "evaluate_once", evaluate)
    monkeypatch.setitem(plan9.__dict__, "compare_holdout", lambda *a: {"passed": holdout_pass})
    monkeypatch.setattr(plan9, "update_submission_weights",
                        lambda *a: (events.append("writeback") or {"updated": True}))
    monkeypatch.setattr(plan9, "append_champion_to_pool", lambda *a, **kw: events.append("pool"))
    consumed = []
    monkeypatch.setattr(plan9, "record_preference_candidate_evaluated",
                        lambda candidate_id, outcome, logger: consumed.append((candidate_id, outcome)))
    report = plan9.run_plan9_cycle(987654, 900025, "mock_only", log, selfplay=True)
    assert report["promoted"] == (qualified and holdout_pass)
    if qualified:
        start = 3 if preference_available else 2
        assert events[start:start + 2] == [("holdout", "incumbent"), ("holdout", "finalist")]
        selection = json.loads((tmp_path / "mock_only" / "holdout_finalist.json").read_text())
        assert selection["kind"] == "match_ncs"  # Equal metrics: NCS wins the frozen tie rule.
        assert sum(event == ("holdout", "finalist") for event in events) == 1
    else:
        assert len(events) == (3 if preference_available else 2) and report["holdout"] is None
    assert ("writeback" in events) == (qualified and holdout_pass)
    if "writeback" in events:
        assert events[-2:] == ["writeback", "pool"]
    assert len(consumed) == int(preference_available)


def test_new_champion_metadata_never_rewrites_old_scores(tmp_path, monkeypatch):
    from experiments import plan9
    pool = tmp_path / "champion_pool.json"
    history = copy.deepcopy(HISTORY)
    monkeypatch.setattr(plan9, "_CHAMPION_POOL_PATH", str(pool))
    plan9.append_champion_to_pool(history, 100, [9, 0, 0, 0, 0], .25,
                                  SimpleNamespace(log=lambda msg: None), protocol={"sha256": "unit_only"})
    saved = json.loads(pool.read_text())
    assert saved["champions"][:-1] == HISTORY
    entry = saved["champions"][-1]
    assert entry["validation_cost"] == .25
    assert entry["validation_protocol"] == v.PROTOCOL_VERSION
    assert entry["validation_protocol_sha256"] == "unit_only"