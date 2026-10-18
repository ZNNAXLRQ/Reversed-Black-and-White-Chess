"""Versioned independent-opening validation; never an optimizer objective.

Freeze before training; validate all finalists on shared conditions; select exactly
one finalist; use the disjoint holdout once. Failed/partial runs cannot be extended.
"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from reversi.board import board_from_array, board_to_array, initial_board
from reversi.rules import apply_move, is_terminal, legal_moves
from reversi.search import SEARCH_POLICY

PROTOCOL_VERSION = "independent_openings_v2"
VALIDATION_OPENINGS = 12
HOLDOUT_OPENINGS = 24
HISTORICAL_OPPONENTS = 3
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 2026101825


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def canonical(board):
    array = board_to_array(board)
    return hashlib.sha256(min(np.ascontiguousarray(x).tobytes()
                              for k in range(4) for x in
                              (np.rot90(array, k), np.fliplr(np.rot90(array, k))))).hexdigest()


def checked_weights(value):
    weights = np.asarray(value, dtype=np.float64)
    if weights.shape != (5,) or not np.isfinite(weights).all() or np.any(abs(weights) > 10):
        raise ValueError("validation weights must be five finite values within [-10,10]")
    return weights.tolist()


def select_opponents(champions, incumbent_weights):
    """Three distinct chronological anchors, independent of incomparable old costs."""
    incumbent = checked_weights(incumbent_weights)
    rows, seen, cycles = [], {tuple(incumbent)}, set()
    for champion in sorted(champions, key=lambda c: c["cycle_index"]):
        cycle = champion["cycle_index"]
        if type(cycle) is not int or cycle < 1 or cycle in cycles:
            raise ValueError("champion cycles must be unique positive integers")
        cycles.add(cycle)
        weights = checked_weights(champion["weights"])
        key = tuple(weights)
        if key not in seen:
            seen.add(key)
            rows.append({"id": f"cycle_{cycle}", "cycle_index": cycle, "weights": weights})
    if len(rows) < HISTORICAL_OPPONENTS:
        raise ValueError("independent validation requires three distinct historical opponents")
    return [rows[i] for i in (0, (len(rows) - 1) // 2, len(rows) - 1)]


def prior_positions(root):
    """Exclude all previously frozen samples and dataset boards, not action labels."""
    root, seen = Path(root), set()

    def collect(value):
        if isinstance(value, dict):
            if "canonical_sha256" in value:
                seen.add(value["canonical_sha256"])
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    artifacts = root / "artifacts"
    names = {"samples.json", "openings.json", "deployment_final_openings.json",
             "validation_protocol.json"}
    for path in artifacts.rglob("*.json"):
        if path.name in names:
            collect(json.loads(path.read_text(encoding="utf-8-sig")))
    for path in (root / "training" / "datasets").rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                row = json.loads(line)
                if "board_before" in row:
                    seen.add(canonical(board_from_array(np.asarray(row["board_before"], dtype=np.int8))))
    return seen


def generate_openings(number, rng, excluded):
    rows = []
    for _ in range(50000):
        if len(rows) == number:
            return rows
        board, color, prefix = initial_board(), -1, []
        plies = (6, 8, 10, 12, 14)[len(rows) % 5]
        for _ in range(plies):
            if is_terminal(board):
                break
            moves = legal_moves(board, color)
            move = moves[int(rng.integers(len(moves)))] if moves else None
            prefix.append([color, list(move) if move else None])
            if move is not None:
                board = apply_move(board, color, move)
            color = -color
        key = canonical(board)
        if is_terminal(board) or not legal_moves(board, color) or key in excluded:
            continue
        excluded.add(key)
        rows.append({"board": [int(x) for x in board], "color": color,
                     "prefix": prefix, "canonical_sha256": key})
    raise RuntimeError("could not generate enough disjoint openings")


def replay_opening(opening):
    board, color = initial_board(), -1
    for side, move in opening["prefix"]:
        if side != color or is_terminal(board):
            raise ValueError("invalid opening turn or premature terminal")
        moves = legal_moves(board, color)
        if move is None:
            if moves:
                raise ValueError("illegal opening pass")
        else:
            move = tuple(move)
            if move not in moves:
                raise ValueError("illegal opening move")
            board = apply_move(board, color, move)
        color = -color
    if ([int(x) for x in board] != opening["board"] or color != opening["color"]
            or canonical(board) != opening["canonical_sha256"] or is_terminal(board)):
        raise ValueError("opening prefix/board/hash mismatch")
    return board, color


def freeze_protocol(root, run_dir, run_id, incumbent_weights, champions, depth, node_limit):
    """One exclusive manifest per authorized cycle, before optimization starts."""
    if type(depth) is not int or not 1 <= depth <= 64 or type(node_limit) is not int or node_limit < 1:
        raise ValueError("formal validation requires bounded depth and positive node budget")
    manifest = dict(version=PROTOCOL_VERSION, run_id=run_id, search_policy=SEARCH_POLICY,
                    depth=depth, node_limit=node_limit, greedy_seed=999,
                    incumbent_weights=checked_weights(incumbent_weights),
                    opponents=select_opponents(champions, incumbent_weights),
                    bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
                    selection_gate="validation cost < incumbent AND self_play_mean_u > 0",
                    holdout_gate="paired panel delta lower95 > 0 AND H2H mean_u lower95 > 0",
                    stopping_rule="one finalist, one holdout; no extension or replacement")
    rng = np.random.default_rng(int(digest(manifest)[:16], 16))
    excluded = prior_positions(root)
    manifest["excluded_prior_positions"] = len(excluded)
    manifest["validation"] = generate_openings(VALIDATION_OPENINGS, rng, excluded)
    manifest["holdout"] = generate_openings(HOLDOUT_OPENINGS, rng, excluded)
    manifest["games_per_model"] = {"validation": VALIDATION_OPENINGS * 10,
                                    "holdout": HOLDOUT_OPENINGS * 10}
    manifest["sha256"] = digest(manifest)
    path = Path(run_dir) / "validation_protocol.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return manifest


def verify_protocol(protocol):
    content = {k: v for k, v in protocol.items() if k != "sha256"}
    if protocol.get("sha256") != digest(content) or protocol.get("version") != PROTOCOL_VERSION:
        raise ValueError("validation protocol version/hash mismatch")
    if protocol["search_policy"] != SEARCH_POLICY:
        raise ValueError("validation search policy mismatch")
    if (type(protocol["depth"]) is not int or not 1 <= protocol["depth"] <= 64
            or type(protocol["node_limit"]) is not int or protocol["node_limit"] < 1
            or protocol["bootstrap_replicates"] != BOOTSTRAP_REPLICATES
            or protocol["bootstrap_seed"] != BOOTSTRAP_SEED or protocol["greedy_seed"] != 999):
        raise ValueError("validation budget/statistics configuration mismatch")
    if len(protocol["opponents"]) != HISTORICAL_OPPONENTS:
        raise ValueError("validation requires exactly three historical opponents")
    weights = {tuple(checked_weights(protocol["incumbent_weights"]))}
    ids = {"incumbent", "greedy"}
    for opponent in protocol["opponents"]:
        key = tuple(checked_weights(opponent["weights"]))
        if key in weights or opponent["id"] in ids:
            raise ValueError("duplicate validation opponents")
        weights.add(key)
        ids.add(opponent["id"])
    seen = set()
    for phase in ("validation", "holdout"):
        expected = VALIDATION_OPENINGS if phase == "validation" else HOLDOUT_OPENINGS
        if len(protocol[phase]) != expected:
            raise ValueError("validation opening count mismatch")
        for opening in protocol[phase]:
            replay_opening(opening)
            key = opening["canonical_sha256"]
            if key in seen:
                raise ValueError("validation/holdout overlap")
            seen.add(key)


def cluster_interval(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty opening clusters required")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = values[rng.integers(len(values), size=(BOOTSTRAP_REPLICATES, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def evaluate_weights(weights, runner, protocol, phase="validation"):
    from .match_runner import GreedyAgent, WeightedAgent
    verify_protocol(protocol)
    if phase not in ("validation", "holdout"):
        raise ValueError("invalid validation phase")
    if runner.depth != protocol["depth"] or runner.node_limit != protocol["node_limit"]:
        raise ValueError("validation runner budget mismatch")
    weights = checked_weights(weights)
    opponents = [{"id": "incumbent", "weights": protocol["incumbent_weights"]},
                 *protocol["opponents"], {"id": "greedy", "weights": None}]
    games, us, self_us, rows, clusters, h2h_clusters = [], [], [], [], [], []
    for index, opening in enumerate(protocol[phase]):
        opening_us, opening_self = [], []
        for opponent in opponents:
            for color in (-1, 1):
                candidate = WeightedAgent(weights, runner.depth, runner.node_limit)
                other = (WeightedAgent(opponent["weights"], runner.depth, runner.node_limit)
                         if opponent["weights"] is not None else GreedyAgent())
                black, white = (candidate, other) if color == -1 else (other, candidate)
                game = runner.play_game(black, white, seed=protocol["greedy_seed"], opening=opening)
                u = (game.white_count - game.black_count) / 64.0 * (-color)
                games.append(asdict(game))
                rows.append(dict(opening_index=index, opening_sha256=opening["canonical_sha256"],
                                 opponent_id=opponent["id"], candidate_color=color, u=u))
                us.append(u)
                opening_us.append(u)
                if opponent["id"] == "incumbent":
                    self_us.append(u)
                    opening_self.append(u)
        clusters.append(float(np.mean(opening_us)))
        h2h_clusters.append(float(np.mean(opening_self)))
    mean = float(np.mean(us))
    return dict(protocol_version=PROTOCOL_VERSION, protocol_sha256=protocol["sha256"], phase=phase,
                search_policy=SEARCH_POLICY, weights=weights, wins=sum(u > 0 for u in us),
                draws=sum(u == 0 for u in us), losses=sum(u < 0 for u in us), mean_u=mean,
                cost=(1 - mean) / 2, u_values=us, self_play_u_values=self_us,
                self_play_mean_u=float(np.mean(self_us)), games=games, game_metadata=rows,
                opening_clusters=clusters, self_play_clusters=h2h_clusters,
                mean_u_cluster_95=cluster_interval(clusters),
                self_play_cluster_95=cluster_interval(h2h_clusters))


def evaluate_once(weights, runner, protocol, run_dir, label, phase="validation"):
    """Reserve before playing: failed/partial work stays consumed; no optional reruns."""
    verify_protocol(protocol)
    if label not in ("incumbent", "match_ncs", "preference", "finalist"):
        raise ValueError("invalid validation label")
    if phase not in ("validation", "holdout") or (phase == "holdout" and label not in ("incumbent", "finalist")):
        raise ValueError("invalid phase/label")
    path = Path(run_dir) / f"{phase}_{label}_usage.json"
    usage = dict(protocol_sha256=protocol["sha256"], weights=checked_weights(weights),
                 phase=phase, label=label, status="started")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(usage, stream, indent=2)
    result = evaluate_weights(weights, runner, protocol, phase)
    with (Path(run_dir) / f"{phase}_{label}_result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    usage["status"] = "completed"
    path.write_text(json.dumps(usage, indent=2), encoding="utf-8")
    return result


def _checked_holdout_clusters(result):
    """Reject truncated/duplicated or inconsistent results before applying a gate."""
    if (result["protocol_version"] != PROTOCOL_VERSION or result["search_policy"] != SEARCH_POLICY
            or result["phase"] != "holdout" or len(result["games"]) != HOLDOUT_OPENINGS * 10
            or len(result["game_metadata"]) != HOLDOUT_OPENINGS * 10
            or len(result["u_values"]) != HOLDOUT_OPENINGS * 10):
        raise ValueError("incomplete or incompatible holdout result")
    groups, self_groups = {}, {}
    cells = set()
    ids, hashes = {}, {}
    for row, u, game in zip(result["game_metadata"], result["u_values"], result["games"]):
        index, opponent_id, color = row["opening_index"], row["opponent_id"], row["candidate_color"]
        cell = (index, opponent_id, color)
        if (type(index) is not int or not 0 <= index < HOLDOUT_OPENINGS or color not in (-1, 1)
                or cell in cells or not np.isfinite(u) or not -1 <= u <= 1 or row["u"] != u):
            raise ValueError("invalid/duplicated holdout cell")
        black, white = game["black_count"], game["white_count"]
        if (type(black) is not int or type(white) is not int or min(black, white) < 0
                or black + white > 64 or (white - black) / 64.0 * (-color) != u):
            raise ValueError("holdout utility disagrees with final counts")
        cells.add(cell)
        groups.setdefault(index, []).append(u)
        ids.setdefault(index, set()).add(opponent_id)
        hashes.setdefault(index, set()).add(row["opening_sha256"])
        if opponent_id == "incumbent":
            self_groups.setdefault(index, []).append(u)
    if (len(groups) != HOLDOUT_OPENINGS or len(self_groups) != HOLDOUT_OPENINGS
            or any(len(us) != 10 for us in groups.values())
            or any(len(us) != 2 for us in self_groups.values())
            or any(len(x) != 1 for x in hashes.values())
            or len({next(iter(x)) for x in hashes.values()}) != HOLDOUT_OPENINGS
            or any(len(x) != 5 or "greedy" not in x or x != ids[0] for x in ids.values())):
        raise ValueError("incomplete holdout opening/color/opponent clusters")
    clusters = np.array([np.mean(groups[i]) for i in range(HOLDOUT_OPENINGS)])
    self_clusters = np.array([np.mean(self_groups[i]) for i in range(HOLDOUT_OPENINGS)])
    if (not np.array_equal(clusters, result["opening_clusters"])
            or not np.array_equal(self_clusters, result["self_play_clusters"])):
        raise ValueError("holdout cluster summaries disagree with games")
    return clusters, self_clusters


def compare_holdout(candidate, incumbent):
    if (candidate["phase"] != "holdout" or incumbent["phase"] != "holdout"
            or candidate["protocol_sha256"] != incumbent["protocol_sha256"]
            or candidate["game_metadata"] == []):
        raise ValueError("holdout requires common protocol and phase")
    keys = ("opening_index", "opening_sha256", "opponent_id", "candidate_color")
    if ([tuple(row[k] for k in keys) for row in candidate["game_metadata"]]
            != [tuple(row[k] for k in keys) for row in incumbent["game_metadata"]]):
        raise ValueError("unpaired holdout conditions")
    candidate_clusters, candidate_h2h = _checked_holdout_clusters(candidate)
    incumbent_clusters, _ = _checked_holdout_clusters(incumbent)
    deltas = candidate_clusters - incumbent_clusters
    panel_ci = cluster_interval(deltas)
    h2h_ci = cluster_interval(candidate_h2h)
    return dict(passed=panel_ci[0] > 0 and h2h_ci[0] > 0,
                mean_u_delta=float(deltas.mean()), delta_cluster_95=panel_ci,
                self_play_mean_u=float(np.mean(candidate["self_play_clusters"])),
                self_play_cluster_95=h2h_ci, opening_clusters=len(deltas),
                protocol_sha256=candidate["protocol_sha256"],
                note="single fixed-size holdout; no continuation after failure")