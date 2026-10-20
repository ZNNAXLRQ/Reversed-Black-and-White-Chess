"""Isolated fixtures and tiny real continuations; never touch production training state."""
import copy
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import continuation_dataset as data
from experiments import plan9
from experiments.continuation_objective import ContinuationObjective
from experiments.match_runner import GameResult, GreedyAgent, MatchRunner
from experiments.validation import canonical, digest, replay_opening
from optimization.objective import EvaluationResult
from reversi.board import board_to_array, count, initial_board
from reversi.rules import apply_move, is_terminal, legal_moves

HISTORY = [dict(cycle_index=i, weights=[float(i), 0, 0, 0, 0], validation_cost=.5)
           for i in (1, 2, 3, 4)]


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding="utf-8")


def recorded_game(root, game, seed):
    board, side, rows = initial_board(), -1, []
    rng = np.random.default_rng(seed)
    source = digest(dict(game=game, fixture_seed=seed))
    while not is_terminal(board):
        moves = legal_moves(board, side)
        move = moves[int(rng.integers(len(moves)))] if moves else None
        after = apply_move(board, side, move) if move else board.copy()
        rows.append(dict(game_id=game, ply=len(rows) + 1, side_to_move=side,
                         board_before=board_to_array(board).tolist(),
                         board_after=board_to_array(after).tolist(), legal_moves=moves,
                         chosen_move=move, is_pass=not moves, source_sha256=source))
        board, side = after, -side
    directory = root / "training" / "processed" / game
    write_json(directory / "summary.json", dict(game_id=game, source_sha256=source,
                                                action_count=len(rows), terminal=True,
                                                rule_validation_status="ok"))
    (directory / "moves.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows),
                                           encoding="utf-8")
    return source


@pytest.fixture
def source_dataset(tmp_path):
    games = [f"game_{i}" for i in range(10)]
    hashes = [recorded_game(tmp_path, game, 900025 + i) for i, game in enumerate(games)]
    manifest = dict(dataset_id="unit_only", included_game_ids=games, source_sha256s=hashes,
                    split={phase: dict(game_ids=ids) for phase, ids in
                           (("train", games[:8]), ("validation", games[8:9]), ("test", games[9:]))})
    path = tmp_path / "training" / "datasets" / "unit_only" / "dataset_manifest.json"
    write_json(path, manifest)
    return tmp_path, path, manifest


@pytest.fixture
def frozen(source_dataset):
    root, path, _ = source_dataset
    run = root / "run"
    run.mkdir()
    validation = dict(validation=[], holdout=[], sha256="isolated_unit_protocol")
    protocol = data.freeze_training(root, run, path, 900025, 3, [0.] * 5, HISTORY,
                                    1, 8, validation, "champion_random")
    return protocol, run


def test_sample_bank_is_deterministic_isolated_and_stage_balanced(source_dataset):
    root, path, manifest = source_dataset
    before = {p: p.read_bytes() for p in (root / "training").rglob("*") if p.is_file()}
    a = data.build_position_bank(root, path, 900025, 12)
    b = data.build_position_bank(root, path, 900025, 12)
    assert a == b
    assert Counter(row["stage"] for row in a["positions"]) == dict(early=4, middle=4, late=4)
    assert len({row["canonical_sha256"] for row in a["positions"]}) == 12
    assert all(row["game_id"] in manifest["split"]["train"]["game_ids"] for row in a["positions"])
    for game in {row["game_id"] for row in a["positions"]}:
        plies = sorted(row["ply"] for row in a["positions"] if row["game_id"] == game)
        assert len(plies) <= 3
        assert all(b - a >= 8 for a, b in zip(plies, plies[1:]))
    for row in a["positions"]:
        board, side = replay_opening(row)
        assert side == row["color"] and canonical(board) == row["canonical_sha256"]
        assert "chosen_move" not in row  # No recorded future action in the training sample.
    assert before == {p: p.read_bytes() for p in (root / "training").rglob("*") if p.is_file()}


def test_historical_holdout_games_and_symmetric_boards_never_reclaimed(source_dataset):
    root, path, manifest = source_dataset
    prior = copy.deepcopy(manifest)
    prior["split"]["train"]["game_ids"].remove("game_0")
    prior["split"]["test"]["game_ids"].append("game_0")
    write_json(root / "training" / "datasets" / "old" / "dataset_manifest.json", prior)
    a = data.build_position_bank(root, path, 900025, 6)
    assert not any(row["game_id"] == "game_0" for row in a["positions"])
    rows, _ = data.replay_positions(root / "training" / "processed", "game_0")
    assert not set(row["canonical_sha256"] for row in rows).intersection(
        row["canonical_sha256"] for row in a["positions"])


def test_cross_source_duplicate_with_holdout_is_excluded(source_dataset):
    root, path, manifest = source_dataset
    directory = root / "training" / "processed"
    summary = data.read_json(directory / "game_0" / "summary.json")
    old_sha = summary["source_sha256"]
    summary["source_sha256"] = data.read_json(directory / "game_8" / "summary.json")["source_sha256"]
    write_json(directory / "game_0" / "summary.json", summary)
    moves = directory / "game_0" / "moves.jsonl"
    moves.write_text(moves.read_text().replace(old_sha, summary["source_sha256"]), encoding="utf-8")
    a = data.build_position_bank(root, path, 900025, 6)
    assert not any(row["game_id"] == "game_0" for row in a["positions"])


@pytest.mark.parametrize("number", [0, 1, 4, -3, 3.0])
def test_invalid_sample_budgets_fail(source_dataset, number):
    root, path, _ = source_dataset
    with pytest.raises(ValueError, match="multiple of three"):
        data.build_position_bank(root, path, 900025, number)


def test_overlap_and_insufficient_coverage_fail_closed(source_dataset):
    root, path, manifest = source_dataset
    with pytest.raises(ValueError, match="insufficient"):
        data.build_position_bank(root, path, 900025, 99)
    manifest["split"]["test"]["game_ids"].append("game_0")
    write_json(path, manifest)
    with pytest.raises(ValueError, match="disjoint"):
        data.build_position_bank(root, path, 900025, 3)


def test_frozen_evaluation_boards_excluded(source_dataset):
    root, path, _ = source_dataset
    a = data.build_position_bank(root, path, 900025, 3)
    keys = {row["canonical_sha256"] for row in a["positions"]}
    write_json(root / "artifacts" / "old" / "validation_protocol.json", {"validation": a["positions"]})
    b = data.build_position_bank(root, path, 900025, 3)
    assert not keys.intersection(row["canonical_sha256"] for row in b["positions"])


@pytest.mark.parametrize("field", ["side_to_move", "board_before", "board_after", "is_pass",
                                  "legal_moves", "source_sha256"])
def test_invalid_recorded_semantics_fail(source_dataset, field):
    root, _, _ = source_dataset
    path = root / "training" / "processed" / "game_0" / "moves.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = rows[0]
    row[field] = {"side_to_move": 1, "board_before": [[0] * 8] * 8,
                  "board_after": [[0] * 8] * 8, "is_pass": True,
                  "legal_moves": [], "source_sha256": "wrong"}[field]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        data.replay_positions(root / "training" / "processed", "game_0")


def test_protocol_exclusive_and_new_validation_excludes_training(frozen, source_dataset):
    protocol, run = frozen
    root, path, _ = source_dataset
    data.verify_training(protocol)
    assert protocol["games_per_evaluation"] == 24
    assert [row["weight"] for row in protocol["opponents"]] == [.5, 1 / 6, 1 / 6, 1 / 6]
    before = (run / "continuation_protocol.json").read_bytes()
    with pytest.raises(FileExistsError):
        data.freeze_training(root, run, path, 900025, 3, [0.] * 5, HISTORY, 1, 8,
                             dict(validation=[], holdout=[], sha256="unit"), "champion_random")
    assert (run / "continuation_protocol.json").read_bytes() == before
    from experiments.validation import prior_positions
    archive = root / "artifacts" / "old" / "continuation_protocol.json"
    write_json(archive, protocol)
    assert all(row["canonical_sha256"] in prior_positions(root) for row in protocol["positions"])


@pytest.mark.parametrize("mutation", ["hash", "budget", "duplicate", "prefix", "weights", "turn",
                                     "provenance", "stage_ranges"])
def test_corrupt_protocol_rejected_even_when_rehashed(frozen, mutation):
    protocol, _ = frozen
    p = copy.deepcopy(protocol)
    if mutation == "budget":
        p["node_limit"] = 0
    elif mutation == "duplicate":
        p["positions"][1] = p["positions"][0]
    elif mutation == "prefix":
        p["positions"][0]["prefix"][0][1] = [0, 0]
    elif mutation == "weights":
        p["opponents"][0]["weight"] = .9
    elif mutation == "turn":
        p["positions"][0]["color"] *= -1
    elif mutation == "provenance":
        p["positions"][0]["source_sha256"] = "wrong"
    elif mutation == "stage_ranges":
        p["sampling"]["stages"]["early"] = [0, 64]
    p["sha256"] = digest({k: v for k, v in p.items() if k != "sha256"})
    if mutation == "hash":
        p["sha256"] = "wrong"
    with pytest.raises(ValueError):
        data.verify_training(p)


def test_weighting_assignments_cache_and_private_freeze(frozen):
    protocol, _ = frozen
    obj = ContinuationObjective(protocol)
    frozen_copy = copy.deepcopy(protocol)
    protocol["opponents"][0]["weights"] = [9.] * 5
    protocol["positions"].clear()
    seen = []

    def play(black, white, opening):
        index = len(seen)
        color = -1 if index % 2 == 0 else 1
        opponent = frozen_copy["opponents"][index // 6]
        candidate = black if color == -1 else white
        other = white if color == -1 else black
        assert np.array_equal(candidate.weights, [2.] * 5)
        assert np.array_equal(other.weights, opponent["weights"])
        assert opening == frozen_copy["positions"][(index % 6) // 2]
        seen.append((opening["board"], opening["color"], color))
        # Incumbent losses (-.5), historical wins (+.5): weighted mean is exactly zero.
        own, opp = (48, 16) if opponent["id"] == "incumbent" else (16, 48)
        b, w = (own, opp) if color == -1 else (opp, own)
        return GameResult(b, w, 1, 0)

    obj.runner.play_game = play
    result = obj([.6] * 5)
    assert result.cost == pytest.approx(.5)
    assert result.metrics["incumbent_mean_u"] == -.5
    assert [row["mean_u"] for row in result.metrics["per_opponent"]] == [-.5, .5, .5, .5]
    assert result.metrics["games"] == obj.games_played == 24
    for a, b in zip(seen[::2], seen[1::2]):
        assert a[:2] == b[:2] and a[2] == -b[2]
    assert obj([.6] * 5) is result
    assert obj.cache_hits == 1 and obj.games_played == 24
    obj.runner.node_limit += 1
    with pytest.raises(ValueError, match="budget changed"):
        obj([.6] * 5)  # Even cache hits cannot hide changed conditions.


def test_real_continuations_replay_to_terminal_and_audit(frozen):
    protocol, run = frozen
    audit = run / "games.jsonl"
    obj = ContinuationObjective(protocol, audit)
    result = obj([.5] * 5)
    assert 0 <= result.cost <= 1 and obj.games_played == 24
    assert result.metrics["incumbent_mean_u"] == 0  # Identical policies, paired assignments.
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    assert len(records) == 25 and records[-1]["type"] == "completed"
    for row in records[:-1]:
        opening = protocol["positions"][row["position_index"]]
        board, side = replay_opening(opening)
        game = row["result"]
        for action in game["trace"]:
            assert side == action["color"] and not is_terminal(board)
            moves = legal_moves(board, side)
            if action["move"] is None:
                assert not moves
            else:
                move = tuple(action["move"])
                assert move in moves
                board = apply_move(board, side, move)
            side = -side
        assert is_terminal(board) and game["final_board"] == [int(x) for x in board]
        assert (game["black_count"], game["white_count"]) == (count(board, -1), count(board, 1))
        assert row["u"] == ((game["white_count"] - game["black_count"]) / 64 * -row["candidate_color"])
    assert obj.cached_result([.5] * 5) is result
    before = audit.read_bytes()
    obj([.5] * 5)
    assert audit.read_bytes() == before


def test_tiny_real_ncs_integration_keeps_panel_fixed_and_accounts_for_cache(frozen):
    from optimization.config import OptimizationConfig
    from optimization.ncs.optimizer import NCSConfig, NCSOptimizer
    from optimization.parameter_space import ParameterSpace
    protocol, run = frozen
    before = copy.deepcopy(protocol)
    objective = ContinuationObjective(protocol, run / "tiny_ncs_games.jsonl")
    optimizer = NCSOptimizer(NCSConfig(T_max=1, population_size=2,
                                      initial_candidates=[[.5] * 5]))
    result = optimizer.optimize(objective, ParameterSpace([0.] * 5, [1.] * 5),
                                OptimizationConfig(seed=900025))
    assert result.num_evaluations == 4 and result.num_iterations == 1
    assert result.metrics["seeded_processes"] == 1
    cached = objective.cached_result(result.best_candidate)
    assert result.best_cost == cached.cost
    assert objective.games_played == (4 - objective.cache_hits) * 24
    assert protocol == before
    records = [json.loads(line) for line in (run / "tiny_ncs_games.jsonl").read_text().splitlines()]
    assert sum(row["type"] == "game" for row in records) == objective.games_played
    assert all(row["metrics"]["protocol_sha256"] == protocol["sha256"]
               for row in records if row["type"] == "completed")


def test_pass_start_preserves_actual_side_and_is_not_terminal(source_dataset):
    root, _, _ = source_dataset
    pass_position = None
    for game in range(10):
        rows, _ = data.replay_positions(root / "training" / "processed", f"game_{game}")
        for row in rows:
            board, side = replay_opening(row)
            if not legal_moves(board, side):
                pass_position = row
                break
        if pass_position:
            break
    assert pass_position is not None
    board, side = replay_opening(pass_position)
    assert not legal_moves(board, side) and legal_moves(board, -side)
    game = MatchRunner(depth=1, node_limit=8).play_game(GreedyAgent(), GreedyAgent(), opening=pass_position)
    assert game.trace[0] == dict(color=side, move=None)
    assert game.trace[1]["color"] == -side


@pytest.mark.parametrize("qualified,holdout_pass", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("initialization", ["champion_random", "random"])
def test_new_orchestration_preserves_promotion_gates_and_skips_preferences(
        source_dataset, monkeypatch, qualified, holdout_pass, initialization):
    root, manifest, _ = source_dataset
    events = []
    monkeypatch.setattr(plan9, "_PROJECT_ROOT", str(root))
    monkeypatch.setattr(plan9, "_TRAINING_RUNS_DIR", str(root / "runs"))
    monkeypatch.setattr(plan9, "_SOURCE_FILES", [])
    monkeypatch.setattr(plan9, "load_submission_weights", lambda path: np.zeros(5))
    monkeypatch.setattr(plan9, "next_version_number", lambda: 1)
    monkeypatch.setattr(plan9, "backup_submission", lambda *a: {})
    monkeypatch.setattr(plan9, "load_champion_pool", lambda: HISTORY)

    def forbid(*args):
        pytest.fail("legacy preference pipeline must not be consulted or consumed")

    monkeypatch.setattr(plan9, "select_preference_candidate", forbid)
    monkeypatch.setattr(plan9, "record_preference_candidate_evaluated", forbid)

    class MockOptimizer:
        def __init__(self, config):
            if initialization == "champion_random":
                assert np.array_equal(config.initial_candidates[0], [.5] * 5)
            else:
                assert config.initial_candidates is None

        def optimize(self, objective, space, config):
            obj = objective.objective
            assert (root / "runs" / "unit_only" / "continuation_protocol.json").exists()
            # No expensive optimization: seed a reporting cache only.
            obj._cache[tuple([2.] * 5)] = EvaluationResult(.25, dict(incumbent_mean_u=.25))
            return SimpleNamespace(best_candidate=np.full(5, .6), best_cost=.25, metrics={},
                                   num_evaluations=0, num_iterations=0, stop_reason="mock", history=[])

    monkeypatch.setitem(plan9.__dict__, "NCSOptimizer", MockOptimizer)

    def evaluate(weights, runner, protocol, run_dir, label, phase="validation"):
        events.append((phase, label))
        candidate = label != "incumbent"
        return dict(cost=.25 if candidate and qualified else .5, mean_u=.5 if candidate else 0,
                    self_play_mean_u=.25 if candidate else 0, wins=1, draws=0, losses=0)

    monkeypatch.setitem(plan9.__dict__, "evaluate_once", evaluate)
    monkeypatch.setitem(plan9.__dict__, "compare_holdout", lambda *a: dict(passed=holdout_pass))
    monkeypatch.setattr(plan9, "update_submission_weights",
                        lambda *a: events.append("writeback") or dict(updated=True))
    monkeypatch.setattr(plan9, "append_champion_to_pool", lambda *a, **k: events.append("pool"))
    report = plan9.run_plan9_cycle(900025, 900025, "unit_only", SimpleNamespace(log=lambda msg: None),
                                   dataset_manifest=str(manifest), position_count=3,
                                   initialization=initialization)
    assert report["objective_mode"] == "continuation"
    assert report["preference_candidate_available"] is False
    assert report["promoted"] == (qualified and holdout_pass)
    assert events[:2] == [("validation", "incumbent"), ("validation", "match_ncs")]
    if qualified:
        assert events[2:4] == [("holdout", "incumbent"), ("holdout", "finalist")]
    else:
        assert len(events) == 2
    assert ("writeback" in events) == (qualified and holdout_pass)
    config = data.read_json(root / "runs" / "unit_only" / "config.json")
    assert config["objective_games"] == 5040 and config["games_per_evaluation"] == 24
    assert config["initialization"] == initialization


@pytest.mark.parametrize("arguments", [[], ["--cycle-index", "900025", "--seed", "900025"],
                                     ["--cycle-index", "900025", "--seed", "900025",
                                      "--selfplay", "--training-mode", "continuation"]])
def test_cli_fails_before_creating_run_for_missing_or_conflicting_inputs(tmp_path, monkeypatch, arguments):
    import sys
    run = tmp_path / "must_not_exist"
    monkeypatch.setattr(plan9, "_TRAINING_RUNS_DIR", str(run))
    monkeypatch.setattr(sys, "argv", ["plan9.py", *arguments])
    with pytest.raises(SystemExit) as error:
        plan9.main()
    assert error.value.code == 2
    assert not run.exists()