"""Freeze train-game positions, never recorded future actions, for fresh continuations."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from reversi.board import board_from_array, empty_count, initial_board
from reversi.rules import apply_move, is_terminal, legal_moves
from reversi.search import SEARCH_POLICY

from .validation import canonical, checked_weights, digest, replay_opening, select_opponents

VERSION = "recorded_position_continuation_v1"
DEFAULT_POSITIONS = 12
MAX_PER_GAME = 3
MIN_PLY_GAP = 8
STAGES = ("early", "middle", "late")
STAGE_RANGES = {"early": [40, 54], "middle": [20, 39], "late": [1, 19]}


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def stage_for(board):
    empties = empty_count(board)
    if 40 <= empties <= 54:
        return "early"
    if 20 <= empties < 40:
        return "middle"
    if 0 < empties < 20:
        return "late"
    return None


def checked_board(value):
    array = np.asarray(value)
    if array.shape != (8, 8) or not np.isin(array, (-1, 0, 1)).all():
        raise ValueError("recorded board must be 8x8 with values -1/0/1")
    return board_from_array(array)


def split_ids(manifest):
    """Use an existing entire-game split; never reshuffle it for this objective."""
    result, seen = {}, set()
    for phase in ("train", "validation", "test"):
        ids = manifest["split"][phase]["game_ids"]
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(g, str) or Path(g).name != g or g in (".", "..")
                           or "/" in g or "\\" in g for g in ids)
                or len(set(ids)) != len(ids) or seen.intersection(ids)):
            raise ValueError("dataset requires disjoint nonempty entire-game partitions")
        result[phase] = set(ids)
        seen.update(ids)
    if seen != set(manifest["included_game_ids"]):
        raise ValueError("dataset partitions disagree with included_game_ids")
    return result


def replay_positions(processed, game_id):
    """Validate full recorded provenance; prefixes only reconstruct the starting state."""
    directory = Path(processed) / game_id
    summary = read_json(directory / "summary.json")
    if (summary["game_id"] != game_id or summary["rule_validation_status"] != "ok"
            or not summary["terminal"] or not summary.get("source_sha256")):
        raise ValueError(f"invalid processed game: {game_id}")
    board, side, prefix, positions = initial_board(), -1, [], []
    rows = [json.loads(line) for line in (directory / "moves.jsonl").read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]
    for ply, row in enumerate(rows, 1):
        moves = legal_moves(board, side)
        if (row["game_id"] != game_id or row["source_sha256"] != summary["source_sha256"]
                or row["ply"] != ply or row["side_to_move"] != side or is_terminal(board)
                or not np.array_equal(checked_board(row["board_before"]), board)
                or sorted(map(tuple, row["legal_moves"])) != sorted(moves)
                or type(row["is_pass"]) is not bool or row["is_pass"] != (not moves)):
            raise ValueError(f"invalid recorded state: {game_id} ply={ply}")
        positions.append(dict(board=[int(x) for x in board], color=side,
                              prefix=[list(x) for x in prefix], canonical_sha256=canonical(board),
                              game_id=game_id, source_sha256=summary["source_sha256"],
                              ply=ply, stage=stage_for(board), split="train"))
        move = row["chosen_move"]
        if moves:
            if move is None or tuple(move) not in moves:
                raise ValueError(f"illegal recorded move: {game_id} ply={ply}")
            board = apply_move(board, side, tuple(move))
        elif move is not None:
            raise ValueError(f"illegal recorded pass: {game_id} ply={ply}")
        if not np.array_equal(checked_board(row["board_after"]), board):
            raise ValueError(f"recorded board_after mismatch: {game_id} ply={ply}")
        prefix.append([side, move])
        side = -side
    if not is_terminal(board) or len(rows) != summary["action_count"]:
        raise ValueError(f"incomplete recorded game: {game_id}")
    return positions, dict(summary_sha256=file_digest(directory / "summary.json"),
                           moves_sha256=file_digest(directory / "moves.jsonl"),
                           source_sha256=summary["source_sha256"])


def frozen_evaluation_boards(root):
    """Exclude old independent evaluation boards, without reading action labels."""
    seen = set()

    def collect(value):
        if isinstance(value, dict):
            if "canonical_sha256" in value:
                seen.add(value["canonical_sha256"])
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    names = {"samples.json", "openings.json", "deployment_final_openings.json",
             "validation_protocol.json"}
    for path in (Path(root) / "artifacts").rglob("*.json"):
        if path.name in names:
            collect(read_json(path))
    return seen


def build_position_bank(root, dataset_manifest, seed, number=DEFAULT_POSITIONS, excluded=()):
    if type(number) is not int or number < 3 or number % 3:
        raise ValueError("positions must be a positive multiple of three")
    root, path = Path(root).resolve(), Path(dataset_manifest).resolve()
    manifest = read_json(path)
    partitions = split_ids(manifest)
    processed = root / "training" / "processed"
    # Earlier dataset versions reshuffled games. Never reclaim any old held-out game.
    reserved = partitions["validation"] | partitions["test"]
    split_sources = {}
    for prior in sorted((root / "training" / "datasets").glob("*/dataset_manifest.json")):
        prior_split = split_ids(read_json(prior))
        reserved |= prior_split["validation"] | prior_split["test"]
        split_sources[str(prior.resolve())] = file_digest(prior)
    split_sources[str(path)] = file_digest(path)
    forbidden = set(excluded) | frozen_evaluation_boards(root)
    source_hashes, reserved_sha = {}, set()
    for game in sorted(reserved):
        rows, hashes = replay_positions(processed, game)
        source_hashes[game] = hashes
        reserved_sha.add(hashes["source_sha256"])
        forbidden.update(row["canonical_sha256"] for row in rows)
    eligible, seen_sources = [], {}
    for game in sorted(partitions["train"] - reserved):
        rows, hashes = replay_positions(processed, game)
        source_hashes[game] = hashes
        sha = hashes["source_sha256"]
        if sha in reserved_sha or sha in seen_sources:
            continue
        if sha not in manifest["source_sha256s"]:
            raise ValueError(f"source hash not present in dataset manifest: {game}")
        seen_sources[sha] = game
        eligible.extend(row for row in rows if row["stage"] is not None
                        and row["canonical_sha256"] not in forbidden)
    settings = dict(number=number, max_per_game=MAX_PER_GAME, min_ply_gap=MIN_PLY_GAP,
                    stages=STAGE_RANGES,
                    historical_holdout_exclusion=True)
    sampling_seed = int(digest(dict(seed=seed, dataset_sha256=file_digest(path),
                                    settings=settings))[:16], 16)
    rng = np.random.default_rng(sampling_seed)
    pools = {stage: [row for row in eligible if row["stage"] == stage] for stage in STAGES}
    for stage in STAGES:
        rng.shuffle(pools[stage])
    chosen, used, plies = [], set(), defaultdict(list)
    # Round-robin stages and prefer unused games: spread coverage before reusing a game.
    for _ in range(number // 3):
        for stage in STAGES:
            candidates = [row for row in pools[stage]
                          if row["canonical_sha256"] not in used
                          and len(plies[row["game_id"]]) < MAX_PER_GAME
                          and all(abs(row["ply"] - p) >= MIN_PLY_GAP
                                  for p in plies[row["game_id"]])]
            if not candidates:
                raise ValueError(f"insufficient isolated {stage} positions; no silent fallback")
            row = min(candidates, key=lambda x: len(plies[x["game_id"]]))
            chosen.append(row)
            used.add(row["canonical_sha256"])
            plies[row["game_id"]].append(row["ply"])
    return dict(dataset_id=manifest["dataset_id"], dataset_manifest=str(path),
                dataset_sha256=file_digest(path), split_manifests=split_sources,
                source_files=source_hashes, sampling_seed=sampling_seed, sampling=settings,
                reserved_games=len(reserved), eligible_games=len(seen_sources),
                eligible_game_ids=sorted(seen_sources.values()),
                excluded_boards=len(forbidden), positions=chosen)


def freeze_training(root, run_dir, dataset_manifest, seed, number, incumbent, champions,
                    depth, node_limit, validation_protocol, initialization):
    excluded = [row["canonical_sha256"] for phase in ("validation", "holdout")
                for row in validation_protocol[phase]]
    bank = build_position_bank(root, dataset_manifest, seed, number, excluded)
    opponents = [dict(id="incumbent", weights=checked_weights(incumbent), weight=.5)]
    opponents += [dict(row, weight=1 / 6) for row in select_opponents(champions, incumbent)]
    protocol = dict(version=VERSION, search_policy=SEARCH_POLICY, depth=depth,
                    node_limit=node_limit, initialization=initialization,
                    validation_protocol_sha256=validation_protocol["sha256"],
                    opponents=opponents, games_per_evaluation=number * 8,
                    utility="(opponent_discs-own_discs)/64; cost=(1-weighted_mean_u)/2",
                    **bank)
    protocol["sha256"] = digest(protocol)
    verify_training(protocol)
    with (Path(run_dir) / "continuation_protocol.json").open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return protocol


def verify_training(protocol):
    content = {k: v for k, v in protocol.items() if k != "sha256"}
    if (protocol.get("version") != VERSION or protocol.get("sha256") != digest(content)
            or protocol["search_policy"] != SEARCH_POLICY):
        raise ValueError("continuation protocol version/hash/search mismatch")
    if (type(protocol["depth"]) is not int or not 1 <= protocol["depth"] <= 64
            or type(protocol["node_limit"]) is not int or protocol["node_limit"] < 1
            or protocol["initialization"] not in ("champion_random", "random")):
        raise ValueError("invalid continuation budget or initialization")
    opponents = protocol["opponents"]
    if (len(opponents) != 4 or opponents[0]["id"] != "incumbent"
            or [row["weight"] for row in opponents] != [.5, 1 / 6, 1 / 6, 1 / 6]
            or len({row["id"] for row in opponents}) != 4
            or len({tuple(checked_weights(row["weights"])) for row in opponents}) != 4):
        raise ValueError("invalid frozen opponent pool")
    positions = protocol["positions"]
    number = protocol["sampling"]["number"]
    if (type(number) is not int or number < 3 or number % 3 or len(positions) != number
            or protocol["games_per_evaluation"] != number * 8
            or protocol["sampling"]["max_per_game"] != MAX_PER_GAME
            or protocol["sampling"]["min_ply_gap"] != MIN_PLY_GAP
            or protocol["sampling"]["stages"] != STAGE_RANGES
            or protocol["sampling"]["historical_holdout_exclusion"] is not True):
        raise ValueError("invalid continuation sample counts")
    seen, plies, stages = set(), defaultdict(list), Counter()
    for row in positions:
        board, _ = replay_opening(row)
        game = row["game_id"]
        if (row["split"] != "train" or row["stage"] not in STAGES
                or game not in protocol["eligible_game_ids"]
                or row["source_sha256"] != protocol["source_files"][game]["source_sha256"]
                or stage_for(board) != row["stage"] or row["ply"] != len(row["prefix"]) + 1
                or row["canonical_sha256"] in seen
                or len(plies[game]) >= MAX_PER_GAME
                or any(abs(row["ply"] - p) < MIN_PLY_GAP for p in plies[game])):
            raise ValueError("invalid/duplicated/adjacent continuation position")
        seen.add(row["canonical_sha256"])
        plies[game].append(row["ply"])
        stages[row["stage"]] += 1
    if any(stages[stage] != number // 3 for stage in STAGES):
        raise ValueError("unbalanced continuation stages")