"""Plan 11 强对手走法偏好数据集构建测试。"""
import csv
import hashlib
import json
import math
import os

import numpy as np

from experiments.preference_dataset import (
    FEATURE_NAMES, build_dataset, build_samples_for_game,
)
from reversi.board import COLOR_BLACK, COLOR_WHITE, initial_board
from reversi.evaluator import features as eval_features
from reversi.rules import apply_move as eval_apply, flips, is_terminal, legal_moves

MANIFEST_COLUMNS = [
    "source_relative_path", "game_id", "self_color", "opponent_color",
    "opponent_strength", "opponent_id", "metadata_source", "notes",
]


def _gen_game_moves(seed):
    """用真实规则生成一局随机对局，返回 Plan 10 moves.jsonl 格式的记录列表。"""
    board = initial_board()
    side = COLOR_BLACK
    rng = np.random.default_rng(seed)
    moves = []
    ply = 0
    while not is_terminal(board):
        lm = legal_moves(board, side)
        ply += 1
        if lm:
            m = tuple(lm[rng.integers(len(lm))])
            before = board.copy()
            flipped = flips(board, side, m[0], m[1])
            after = eval_apply(board, side, m)
            moves.append({
                "ply": ply,
                "side_to_move": int(side),
                "board_before": before.tolist(),
                "legal_moves": [[r, c] for r, c in lm],
                "chosen_move": [m[0], m[1]],
                "flipped_discs": [[r, c] for r, c in flipped],
                "board_after": after.tolist(),
                "is_pass": False,
            })
            board = after
        else:
            moves.append({
                "ply": ply,
                "side_to_move": int(side),
                "board_before": board.copy().tolist(),
                "legal_moves": [],
                "chosen_move": None,
                "flipped_discs": [],
                "board_after": board.copy().tolist(),
                "is_pass": True,
            })
        side = -side
    return moves


def _write_fixture(tmp, games):
    """构造 training 目录 fixture：processed/<game_id>/ + game_manifest.csv。"""
    processed_dir = os.path.join(str(tmp), "processed")
    os.makedirs(processed_dir, exist_ok=True)
    rows = []
    for g in games:
        game_id = g["game_id"]
        sha = g["sha"]
        opp_color = g.get("opponent_color", 1)
        strength = g.get("opponent_strength", "strong")
        opp_id = g.get("opponent_id", "unknown:" + game_id)
        game_dir = os.path.join(processed_dir, game_id)
        os.makedirs(game_dir, exist_ok=True)
        moves = _gen_game_moves(g.get("seed", 0))
        with open(os.path.join(game_dir, "moves.jsonl"), "w", encoding="utf-8") as f:
            for mv in moves:
                rec = dict(mv)
                rec["game_id"] = game_id
                rec["source_sha256"] = sha
                rec["source_snapshot_index"] = None
                f.write(json.dumps(rec) + "\n")
        summary = {
            "game_id": game_id,
            "source_relative_path": game_id + ".txt",
            "source_sha256": sha,
            "terminal": True,
            "rule_validation_status": "ok",
            "parser_version": "1.0.0",
            "winner": g.get("winner", opp_color),
        }
        with open(os.path.join(game_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f)
        rows.append({
            "source_relative_path": game_id + ".txt",
            "game_id": game_id,
            "self_color": COLOR_BLACK,
            "opponent_color": opp_color,
            "opponent_strength": strength,
            "opponent_id": opp_id,
            "metadata_source": "user_confirmed",
            "notes": "",
        })
    with open(os.path.join(str(tmp), "game_manifest.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _read_jsonl(path):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _make_game(game_id, sha, seed, strength="strong", opp_color=1, opp_id=None, winner=None):
    return {
        "game_id": game_id,
        "sha": sha,
        "seed": seed,
        "opponent_strength": strength,
        "opponent_color": opp_color,
        "opponent_id": opp_id or ("unknown:" + game_id),
        "winner": opp_color if winner is None else winner,
    }


def _make_sha(tag):
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


# ---------- 1. 仅强对手颜色生成样本 ----------

def test_only_strong_opponent_generates_samples(tmp_path):
    games = [
        _make_game("g_strong", _make_sha("s1"), 1, strength="strong", opp_color=1),
        _make_game("g_unknown", _make_sha("u1"), 2, strength="unknown", opp_color=1),
        _make_game("g_weak", _make_sha("w1"), 3, strength="weak", opp_color=1),
    ]
    _write_fixture(tmp_path, games)
    result = build_dataset(str(tmp_path), write_outputs=True)
    pool = _read_jsonl(os.path.join(result["output_dir"], "preference_pool.jsonl"))
    assert len(pool) > 0
    assert {s["game_id"] for s in pool} == {"g_strong"}
    assert all(s["mover_color"] == 1 for s in pool)
    assert all(s["label_type"] == "observed_move_preference" for s in pool)
    assert result["included_game_count"] == 1
    assert result["excluded_game_count"] == 2


def test_self_moves_not_in_pool(tmp_path):
    games = [_make_game("g_strong", _make_sha("s1"), 1, strength="strong", opp_color=1)]
    _write_fixture(tmp_path, games)
    result = build_dataset(str(tmp_path), write_outputs=True)
    pool = _read_jsonl(os.path.join(result["output_dir"], "preference_pool.jsonl"))
    # 对手白方（opponent_color=1），我方黑方（-1）；样本只应含白方落子
    assert all(s["mover_color"] == COLOR_WHITE for s in pool)


# ---------- 2. game_id 级隔离 ----------

def test_game_level_isolation(tmp_path):
    games = [_make_game(f"g{i}", _make_sha(f"g{i}"), i) for i in range(6)]
    _write_fixture(tmp_path, games)
    result = build_dataset(str(tmp_path), write_outputs=True)
    splits = {}
    for name in ("train", "validation", "test"):
        splits[name] = {s["game_id"] for s in _read_jsonl(os.path.join(result["output_dir"], f"{name}.jsonl"))}
    assert len(splits["train"] & splits["validation"]) == 0
    assert len(splits["train"] & splits["test"]) == 0
    assert len(splits["validation"] & splits["test"]) == 0
    assert splits["train"] | splits["validation"] | splits["test"] == {f"g{i}" for i in range(6)}


# ---------- 3. duplicate SHA 不重复计数 ----------

def test_duplicate_sha_not_double_counted(tmp_path):
    same = _make_sha("dup")
    games = [
        _make_game("g_a", same, 1),
        _make_game("g_b", same, 2),
    ]
    _write_fixture(tmp_path, games)
    result = build_dataset(str(tmp_path), write_outputs=True)
    assert result["included_game_count"] == 1
    assert result["excluded_game_count"] == 1
    assert result["manifest"]["excluded_games"][0]["reason"].startswith("duplicate_sha_of")



# ---------- 4/5. chosen 索引与特征一致性、特征顺序/长度/有限 ----------

def test_chosen_move_index_and_action_features(tmp_path):
    games = [_make_game("g1", _make_sha("g1"), 5)]
    _write_fixture(tmp_path, games)
    result = build_dataset(str(tmp_path), write_outputs=True)
    pool = _read_jsonl(os.path.join(result["output_dir"], "preference_pool.jsonl"))
    assert len(pool) > 0
    for s in pool:
        assert s["legal_moves"][s["chosen_move_index"]] == s["chosen_move"]
        board = np.array(s["board_before"], dtype=np.int8)
        mover = s["mover_color"]
        assert len(s["action_features"]) == len(s["legal_moves"])
        for i, m in enumerate(s["legal_moves"]):
            after = eval_apply(board, mover, (int(m[0]), int(m[1])))
            expected = eval_features(after, mover)
            got = np.array(s["action_features"][i], dtype=np.float64)
            assert np.allclose(got, expected)


def test_feature_order_length_finite(tmp_path):
    games = [_make_game("g1", _make_sha("g1"), 5)]
    _write_fixture(tmp_path, games)
    result = build_dataset(str(tmp_path), write_outputs=True)
    pool = _read_jsonl(os.path.join(result["output_dir"], "preference_pool.jsonl"))
    assert pool
    for s in pool:
        assert s["feature_names"] == FEATURE_NAMES
        for feat in s["action_features"]:
            assert len(feat) == 5
            assert all(math.isfinite(x) for x in feat)


# ---------- 6. 单合法走法样本被计数但不进入偏好损失样本 ----------

def test_forced_move_counted_not_in_pool():
    board = initial_board()
    moves = [
        {"is_pass": False, "side_to_move": 1, "legal_moves": [[2, 2]], "chosen_move": [2, 2],
         "board_before": board.tolist(), "ply": 1},
        {"is_pass": False, "side_to_move": 1, "legal_moves": [[2, 2], [2, 3]], "chosen_move": [2, 2],
         "board_before": board.tolist(), "ply": 2},
        {"is_pass": False, "side_to_move": -1, "legal_moves": [[2, 2], [2, 3]], "chosen_move": [2, 2],
         "board_before": board.tolist(), "ply": 3},
        {"is_pass": True, "side_to_move": 1, "legal_moves": [], "chosen_move": None,
         "board_before": board.tolist(), "ply": 4},
    ]
    samples, forced = build_samples_for_game(moves, "g1", "opp", 1, "sha")
    assert forced == 1
    assert len(samples) == 1
    assert samples[0]["ply"] == 2


# ---------- 7. 少于 5 盘不生成正式三分割 ----------

def test_insufficient_games_no_split(tmp_path):
    games = [_make_game(f"g{i}", _make_sha(f"g{i}"), i) for i in range(3)]
    _write_fixture(tmp_path, games)
    result = build_dataset(str(tmp_path), write_outputs=True)
    assert result["eligible_for_preference_training"] is False
    assert result["blocking_reason"] == "insufficient_distinct_games_for_group_split"
    assert result["manifest"]["test_split_must_not_be_used_for_selection"] is True
    out_dir = result["output_dir"]
    assert not os.path.exists(os.path.join(out_dir, "train.jsonl"))
    assert not os.path.exists(os.path.join(out_dir, "validation.jsonl"))
    assert not os.path.exists(os.path.join(out_dir, "test.jsonl"))
    assert os.path.exists(os.path.join(out_dir, "preference_pool.jsonl"))
    assert os.path.exists(os.path.join(out_dir, "dataset_manifest.json"))


# ---------- 8. 相同输入可复现相同 dataset_id 与划分 ----------

def test_reproducible_dataset_id_and_split(tmp_path):
    games = [_make_game(f"g{i}", _make_sha(f"g{i}"), i) for i in range(6)]
    _write_fixture(tmp_path, games)
    r1 = build_dataset(str(tmp_path), write_outputs=True)
    r2 = build_dataset(str(tmp_path), write_outputs=True)
    assert r1["dataset_id"] == r2["dataset_id"]
    assert r1["split"] == r2["split"]
    assert r1["manifest"]["split"] == r2["manifest"]["split"]


# ---------- 9. 只纳入「我方失败」（对手获胜）的对局 ----------

def test_winning_games_excluded(tmp_path):
    # 我方获胜（winner=-1=我方黑）的对局应被排除；我方失败（winner=1=对手白）应纳入
    games = [
        _make_game("g_loss", _make_sha("l1"), 1, opp_color=1, winner=1),
        _make_game("g_win", _make_sha("w1"), 2, opp_color=1, winner=-1),
    ]
    _write_fixture(tmp_path, games)
    result = build_dataset(str(tmp_path), write_outputs=True)
    assert result["included_game_count"] == 1
    pool = _read_jsonl(os.path.join(result["output_dir"], "preference_pool.jsonl"))
    assert {s["game_id"] for s in pool} == {"g_loss"}
    reasons = {e["game_id"]: e["reason"] for e in result["manifest"]["excluded_games"]}
    assert reasons["g_win"] == "self_won_or_draw_excluded"


def test_draw_excluded(tmp_path):
    games = [
        _make_game("g_loss", _make_sha("l1"), 1, opp_color=1, winner=1),
        _make_game("g_draw", _make_sha("d1"), 2, opp_color=1, winner=0),
    ]
    _write_fixture(tmp_path, games)
    result = build_dataset(str(tmp_path), write_outputs=True)
    assert result["included_game_count"] == 1
    reasons = {e["game_id"]: e["reason"] for e in result["manifest"]["excluded_games"]}
    assert reasons["g_draw"] == "self_won_or_draw_excluded"

