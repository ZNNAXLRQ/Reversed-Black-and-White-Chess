"""Reproducible local A/B: fixed-node speed, equal-time depth, paired games.

Both source files are loaded directly; no training package or Numba is imported.
The independent 2D oracle checks every candidate list and applies every move.
All game results use reversed Reversi: fewer discs wins. Small samples are not
an Elo estimate. The baseline source is an explicit input, never hidden data.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import statistics
import sys
import time

if __package__:
    from .check_submission_resources import (
        apply_reference, checked_go, initial_board, memory_snapshot, reference_moves,
    )
else:
    # Absolute script invocation also works with Python's isolated (-I) mode.
    helper_path = Path(__file__).resolve().with_name("check_submission_resources.py")
    helper_spec = importlib.util.spec_from_file_location("strength_resource_helper", str(helper_path))
    if helper_spec is None or helper_spec.loader is None:
        raise ValueError("Cannot load resource oracle: %s" % helper_path)
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    apply_reference, checked_go = helper.apply_reference, helper.checked_go
    initial_board, memory_snapshot = helper.initial_board, helper.memory_snapshot
    reference_moves = helper.reference_moves


def load_agent(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load agent: %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def position(np, empties, seed):
    rng = random.Random(seed)
    board = initial_board(np)
    side = -1
    for _ in range(125):
        if int(np.count_nonzero(board == 0)) <= empties:
            break
        moves = reference_moves(board, side)
        if moves:
            apply_reference(board, side, rng.choice(list(moves)), moves)
        elif not reference_moves(board, -side):
            break
        side = -side
    return board, side


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=20000)
    parser.add_argument("--position-seconds", type=float, default=0.25)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--game-time-out", type=float, default=0.15)
    parser.add_argument("--opening-seed", type=int, default=1009)
    parser.add_argument("--use-go-budget", action="store_true",
                        help="Also compare each engine's real go time allocator on fixtures")
    args = parser.parse_args()
    if args.pairs < 0 or args.nodes <= 0 or args.position_seconds <= 0 or args.game_time_out <= 0:
        parser.error("invalid nonpositive limit or negative pair count")
    engines = {
        "baseline": load_agent(args.baseline.resolve(), "strength_baseline"),
        "candidate": load_agent(args.candidate.resolve(), "strength_candidate"),
    }
    np = engines["candidate"].np
    report = {
        "python": sys.version,
        "sources": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for name, path in (("baseline", args.baseline), ("candidate", args.candidate))},
        "settings": {"nodes": args.nodes, "position_seconds": args.position_seconds,
                     "pairs": args.pairs, "game_time_out": args.game_time_out,
                     "opening_seed": args.opening_seed},
        "positions": [], "games": [],
    }

    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fixtures = [(60, 11), (44, 19), (36, 41), (32, 29), (24, 43), (16, 101), (12, 19), (8, 29)]
    for index, (empties, seed) in enumerate(fixtures):
        board, side = position(np, empties, seed)
        legal = reference_moves(board, side)
        record = {"empties": int(np.count_nonzero(board == 0)), "seed": seed, "engines": {}}
        # Alternate order to reduce first-run and CPU-temperature bias.
        order = list(engines) if index % 2 == 0 else list(reversed(engines))
        for name in order:
            engine = engines[name]
            bb = engine.board_from_array(board)
            stats = {}
            started = time.perf_counter()
            move = engine.best_move_iterative(bb, side, 10, engine._FIXED_WEIGHTS,
                                             node_limit=args.nodes, stats=stats)
            elapsed = time.perf_counter() - started
            assert move in legal or not legal and move is None
            fixed = {"seconds": elapsed, "move": move, **stats}
            stats = {}
            started = time.perf_counter()
            move = engine.best_move_iterative(bb, side, 10, engine._FIXED_WEIGHTS,
                                             time_budget=args.position_seconds, stats=stats)
            elapsed = time.perf_counter() - started
            assert move in legal or not legal and move is None
            timed = {"seconds": elapsed, "move": move, **stats}
            result = {"fixed_nodes": fixed, "equal_time": timed}
            if args.use_go_budget:
                ai = engine.AI(8, side, 5)
                _, elapsed = checked_go(ai, board, np)
                result["go_budget"] = {"seconds": elapsed, **ai.last_search_stats}
            record["engines"][name] = result
        report["positions"].append(record)
        save()
        print("POSITION", index + 1, json.dumps(record), flush=True)

    for pair in range(args.pairs):
        # Fixed random 8-ply openings; colors exchanged from identical positions.
        opening_seed = args.opening_seed + pair * 137
        board0, side0 = position(np, 52, opening_seed)
        for candidate_side in (-1, 1):
            board = board0.copy()
            side = side0
            names = {candidate_side: "candidate", -candidate_side: "baseline"}
            players = {side: engines[name].AI(8, side, args.game_time_out) for side, name in names.items()}
            totals = {name: 0.0 for name in engines}
            depths = {name: [] for name in engines}
            longest = 0.0
            started = time.perf_counter()
            for ply in range(125):
                moves = reference_moves(board, side)
                if not moves and not reference_moves(board, -side):
                    break
                name = names[side]
                ai = players[side]
                candidates, elapsed = checked_go(ai, board, np)
                totals[name] += elapsed
                longest = max(longest, elapsed)
                if candidates:
                    depths[name].append(ai.last_search_stats.get("completed_depth", 0))
                    apply_reference(board, side, candidates[-1], moves)
                side = -side
            else:
                raise AssertionError("game did not terminate")
            discs = {"candidate": int(np.count_nonzero(board == candidate_side)),
                     "baseline": int(np.count_nonzero(board == -candidate_side))}
            score = 1.0 if discs["candidate"] < discs["baseline"] else 0.0
            if discs["candidate"] == discs["baseline"]:
                score = 0.5
            result = {"pair": pair + 1, "opening_seed": opening_seed, "candidate_side": candidate_side,
                      "candidate_score": score, "discs": discs, "plies": ply,
                      "wall_seconds": time.perf_counter() - started,
                      "engine_seconds": totals, "max_go_seconds": longest,
                      "mean_completed_depth": {name: statistics.mean(values) if values else 0.0
                                               for name, values in depths.items()}}
            report["games"].append(result)
            save()
            print("GAME", json.dumps(result), flush=True)

    positions = report["positions"]
    speedups = [p["engines"]["baseline"]["fixed_nodes"]["seconds"] /
                p["engines"]["candidate"]["fixed_nodes"]["seconds"] for p in positions]
    report["summary"] = {
        "median_fixed_node_speedup": statistics.median(speedups),
        "equal_time_depth_mean": {
            name: statistics.mean(p["engines"][name]["equal_time"]["completed_depth"] for p in positions)
            for name in engines
        },
        "candidate_points": sum(g["candidate_score"] for g in report["games"]),
        "games": len(report["games"]),
        "memory": memory_snapshot(),
        "caution": "Small local sample; not a statistically established win-rate or Elo gain.",
    }
    save()
    print("SUMMARY", json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()