"""Plan 10 棋谱解析与重放验证测试。"""
import hashlib
import json
import os

import numpy as np
import pytest

import experiments.replay_parser as rp
from experiments.replay_parser import (
    PARSER_VERSION, ReplayError, TRAINING_DIR, determine_metadata,
    discover_source_files, parse_snapshots, process_file, reconstruct_game,
    validate_metadata,
)
from reversi.board import COLOR_BLACK, COLOR_WHITE, count, initial_board
from reversi.rules import apply_move, flips, legal_moves


def _training_files():
    files = []
    for name in sorted(os.listdir(TRAINING_DIR)):
        p = os.path.join(TRAINING_DIR, name)
        if os.path.isfile(p) and name.startswith("chess_log") and name.endswith(".txt"):
            files.append(name)
    return files


def _read_raw(rel_path):
    with open(os.path.join(TRAINING_DIR, rel_path), "rb") as f:
        return f.read()


# ---------- 正常解析 / 首手恢复 ----------

def test_parse_chess_log_1_valid():
    snaps = parse_snapshots(_read_raw("chess_log_1.txt").decode("utf-8"))
    assert len(snaps) == 60
    for s in snaps:
        assert s.shape == (8, 8)
        assert set(np.unique(s).tolist()) <= {-1, 0, 1}
    # 首帧 = 黑方 (2,3) 落子并翻转 (3,3) 之后的状态
    expected_first = apply_move(initial_board(), COLOR_BLACK, (2, 3))
    assert np.array_equal(snaps[0], expected_first)


def test_first_move_recovered_uniquely():
    # 首手必须由标准初始棋盘唯一恢复，不硬编码
    snaps = parse_snapshots(_read_raw("chess_log_1.txt").decode("utf-8"))
    board = initial_board()
    matches = [m for m in legal_moves(board, COLOR_BLACK)
               if np.array_equal(apply_move(board, COLOR_BLACK, m), snaps[0])]
    assert matches == [(2, 3)]


def test_reconstruct_chess_log_1_acceptance():
    snaps = parse_snapshots(_read_raw("chess_log_1.txt").decode("utf-8"))
    actions, terminal = reconstruct_game(snaps)
    assert terminal
    assert len(actions) == 60
    assert all(not a["is_pass"] for a in actions)  # 无 pass
    black_moves = [a for a in actions if a["side_to_move"] == COLOR_BLACK]
    white_moves = [a for a in actions if a["side_to_move"] == COLOR_WHITE]
    assert len(black_moves) == 30
    assert len(white_moves) == 30
    final_board = actions[-1]["board_after"]
    assert count(final_board, COLOR_BLACK) == 35
    assert count(final_board, COLOR_WHITE) == 29
    assert rp.compute_winner(final_board) == COLOR_WHITE  # 反转规则：白更少，白胜


def test_replay_recovers_all_snapshots():
    snaps = parse_snapshots(_read_raw("chess_log_1.txt").decode("utf-8"))
    actions, _ = reconstruct_game(snaps)
    move_boards = [a["board_after"] for a in actions if not a["is_pass"]]
    assert len(move_boards) == len(snaps)
    for got, want in zip(move_boards, snaps):
        assert np.array_equal(got, want)


def test_chosen_move_in_legal_and_flips_match():
    snaps = parse_snapshots(_read_raw("chess_log_1.txt").decode("utf-8"))
    actions, _ = reconstruct_game(snaps)
    for a in actions:
        if a["is_pass"]:
            assert a["chosen_move"] is None
            assert a["flipped_discs"] == []
            continue
        m = tuple(a["chosen_move"])
        assert m in {tuple(x) for x in a["legal_moves"]}
        expected_flips = [(r, c) for r, c in flips(a["board_before"], a["side_to_move"], m[0], m[1])]
        assert a["flipped_discs"] == [[r, c] for r, c in expected_flips]


def test_reconstruct_all_training_files_legal():
    files = _training_files()
    assert len(files) == 12
    for name in files:
        snaps = parse_snapshots(_read_raw(name).decode("utf-8"))
        actions, terminal = reconstruct_game(snaps)
        assert terminal, f"{name} 未到达终局"
        move_boards = [a["board_after"] for a in actions if not a["is_pass"]]
        assert len(move_boards) == len(snaps), f"{name} 快照数不一致"
        for got, want in zip(move_boards, snaps):
            assert np.array_equal(got, want), f"{name} 重放不一致"


# ---------- 元数据 ----------

def test_confirmed_file_metadata():
    meta = determine_metadata("chess_log_1.txt")
    assert meta["self_color"] == COLOR_BLACK
    assert meta["opponent_color"] == COLOR_WHITE
    assert meta["opponent_strength"] == "strong"
    assert meta["metadata_source"] == "user_confirmed"


def test_all_chess_log_n_get_black_white_strong():
    # 约束：所有 chess_log_n 文件我方黑、对方白、对手均视为强对手（我方失败对局），
    # 且对手不来自同一个强对手（每个文件 opponent_id 独立未知）。
    for n in range(1, 13):
        meta = determine_metadata(f"chess_log_{n}.txt")
        assert meta["self_color"] == COLOR_BLACK
        assert meta["opponent_color"] == COLOR_WHITE
        assert meta["opponent_strength"] == "strong"
        assert meta["metadata_source"] == "user_confirmed"
        assert meta["opponent_id"] == f"unknown:chess_log_{n}.txt"


def test_other_file_metadata_unknown():
    # 非 chess_log_n 命名的文件保持完全未确认
    meta = determine_metadata("other_file.txt")
    assert meta["self_color"] == ""
    assert meta["opponent_color"] == ""
    assert meta["opponent_strength"] == "unknown"
    assert meta["metadata_source"] == "unconfirmed"
    assert meta["opponent_id"] == "unknown:other_file.txt"


def test_valid_metadata_passes():
    validate_metadata({
        "self_color": COLOR_BLACK, "opponent_color": COLOR_WHITE,
        "opponent_strength": "strong", "opponent_id": "unknown:x",
        "metadata_source": "user_confirmed", "notes": "",
    })


# ---------- 明确失败 / 不伪造结果 ----------

def test_invalid_json():
    with pytest.raises(ReplayError) as exc:
        parse_snapshots("not a json")
    assert exc.value.kind == "invalid_json"


def test_unsupported_format():
    with pytest.raises(ReplayError) as exc:
        parse_snapshots('{"a": 1}')
    assert exc.value.kind == "unsupported_format"


def test_wrong_dimensions():
    bad = [[0] * 8 for _ in range(7)]  # 7 行
    with pytest.raises(ReplayError) as exc:
        parse_snapshots(json.dumps([bad]))
    assert exc.value.kind == "invalid_board"


def test_unknown_piece_value():
    board = [[0] * 8 for _ in range(8)]
    board[0][0] = 2
    with pytest.raises(ReplayError) as exc:
        parse_snapshots(json.dumps([board]))
    assert exc.value.kind == "invalid_board"


def test_illegal_transition_fails():
    bad = np.full((8, 8), 1, dtype=np.int8)  # 全白，不可能从初始局面一步到达
    with pytest.raises(ReplayError) as exc:
        reconstruct_game([bad])
    assert exc.value.kind == "illegal_transition"


def test_ambiguous_transition_fails(monkeypatch):
    target = np.zeros((8, 8), dtype=np.int8)
    monkeypatch.setattr("experiments.replay_parser.legal_moves", lambda b, c: [(2, 3), (3, 2)])
    monkeypatch.setattr("experiments.replay_parser.apply_move", lambda b, c, m: target)
    monkeypatch.setattr("experiments.replay_parser.flips", lambda b, c, r, col: [])
    with pytest.raises(ReplayError) as exc:
        rp.reconstruct_game([target])
    assert exc.value.kind == "ambiguous_transition"


def test_invalid_metadata_rejected():
    with pytest.raises(ReplayError) as exc:
        validate_metadata({
            "self_color": 2, "opponent_color": "",
            "opponent_strength": "unknown", "opponent_id": "x",
            "metadata_source": "unconfirmed", "notes": "",
        })
    assert exc.value.kind == "invalid_metadata"


def test_invalid_metadata_opposite_colors_rejected():
    with pytest.raises(ReplayError) as exc:
        validate_metadata({
            "self_color": COLOR_WHITE, "opponent_color": COLOR_WHITE,
            "opponent_strength": "strong", "opponent_id": "x",
            "metadata_source": "user_confirmed", "notes": "",
        })
    assert exc.value.kind == "invalid_metadata"


# ---------- 输入哈希不变 & 生成文件可重新解析 ----------

def test_input_hash_unchanged_and_outputs_reparseable(tmp_path):
    src = os.path.join(TRAINING_DIR, "chess_log_1.txt")
    with open(src, "rb") as f:
        raw1 = f.read()
    h1 = hashlib.sha256(raw1).hexdigest()

    out = str(tmp_path / "processed")
    info = process_file(src, "chess_log_1.txt", out, write_outputs=True)

    with open(src, "rb") as f:
        raw2 = f.read()
    h2 = hashlib.sha256(raw2).hexdigest()
    assert h1 == h2  # 解析不改动原始文件
    assert info["source_sha256"] == h1

    game_dir = os.path.join(out, info["game_id"])
    with open(os.path.join(game_dir, "moves.jsonl"), encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for key in ("game_id", "ply", "side_to_move", "board_before", "legal_moves",
                        "chosen_move", "flipped_discs", "board_after",
                        "source_snapshot_index", "source_sha256"):
                assert key in rec
            assert rec["source_sha256"] == h1
    with open(os.path.join(game_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["source_sha256"] == h1
    assert summary["snapshot_count"] == 60
    assert summary["parser_version"] == PARSER_VERSION


# ---------- 多棋谱扫描不把生成目录输出误读为输入 ----------

def test_scan_excludes_generated_dirs(tmp_path):
    training = str(tmp_path)
    os.makedirs(os.path.join(training, "raw"))
    os.makedirs(os.path.join(training, "processed"))
    os.makedirs(os.path.join(training, "datasets"))
    for rel in ("a.txt", "game_manifest.csv"):
        with open(os.path.join(training, rel), "w", encoding="utf-8") as f:
            f.write("[]")
    with open(os.path.join(training, "raw", "b.txt"), "w", encoding="utf-8") as f:
        f.write("[]")
    with open(os.path.join(training, "processed", "c.txt"), "w", encoding="utf-8") as f:
        f.write("[]")
    with open(os.path.join(training, "datasets", "d.txt"), "w", encoding="utf-8") as f:
        f.write("[]")

    sources = discover_source_files(training)
    assert set(sources.keys()) == {"a.txt", "raw/b.txt"}

