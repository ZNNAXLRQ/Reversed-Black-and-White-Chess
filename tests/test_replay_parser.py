"""Plan 10 棋谱解析与重放验证测试。"""
import hashlib
import json
import os

import numpy as np
import pytest

import experiments.replay_parser as rp
from experiments.replay_parser import (
    PARSER_VERSION, ReplayError, TRAINING_DIR, determine_metadata,
    discover_source_files, load_manifest, parse_snapshots, process_file,
    reconstruct_game, scan_and_process, validate_metadata,
)
from reversi.board import COLOR_BLACK, COLOR_WHITE, count, initial_board, opponent
from reversi.rules import apply_move, flips, is_terminal, legal_moves


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


def _synthetic_game(seed=12345):
    """用真实规则确定性地下一局到终局，返回 (snapshots, moves)。

    snapshots[i] 为第 i 手落子后的棋盘；moves[i] 为对应的 (row, col)。
    该对局完全由项目规则生成，不依赖任何磁盘上的棋谱文件。
    """
    board = initial_board()
    side = COLOR_BLACK
    snaps = []
    moves = []
    rng = np.random.default_rng(seed)
    while not is_terminal(board):
        lm = legal_moves(board, side)
        if not lm:
            side = opponent(side)
            continue
        move = lm[int(rng.integers(len(lm)))]
        board = apply_move(board, side, move)
        snaps.append(board.copy())
        moves.append((int(move[0]), int(move[1])))
        side = opponent(side)
    return snaps, moves


# ---------- 正常解析 / 首手恢复 ----------

def test_parse_snapshots_valid():
    snaps, _ = _synthetic_game()
    text = json.dumps([s.tolist() for s in snaps])
    parsed = parse_snapshots(text)
    assert len(parsed) == len(snaps)
    for s in parsed:
        assert s.shape == (8, 8)
        assert set(np.unique(s).tolist()) <= {-1, 0, 1}
    # 首帧 = 黑方首手落子后的状态
    assert np.array_equal(parsed[0], snaps[0])


def test_first_move_recovered_uniquely():
    # 首手必须由标准初始棋盘唯一恢复，不硬编码坐标
    snaps, moves = _synthetic_game()
    board = initial_board()
    matches = [m for m in legal_moves(board, COLOR_BLACK)
               if np.array_equal(apply_move(board, COLOR_BLACK, m), snaps[0])]
    assert len(matches) == 1
    assert matches[0] == moves[0]


def test_reconstruct_synthetic_game_acceptance():
    snaps, _ = _synthetic_game()
    actions, terminal = reconstruct_game(snaps)
    assert terminal
    move_actions = [a for a in actions if not a["is_pass"]]
    assert len(move_actions) == len(snaps)
    black_moves = [a for a in move_actions if a["side_to_move"] == COLOR_BLACK]
    white_moves = [a for a in move_actions if a["side_to_move"] == COLOR_WHITE]
    assert len(black_moves) + len(white_moves) == len(snaps)
    final_board = actions[-1]["board_after"]
    assert 0 <= count(final_board, COLOR_BLACK) <= 64
    assert 0 <= count(final_board, COLOR_WHITE) <= 64
    # 反转规则：己方棋子更少者胜，winner 只能是黑/白/平局
    assert rp.compute_winner(final_board) in (COLOR_BLACK, COLOR_WHITE, 0)


def test_replay_recovers_all_snapshots():
    snaps, _ = _synthetic_game()
    actions, _ = reconstruct_game(snaps)
    move_boards = [a["board_after"] for a in actions if not a["is_pass"]]
    assert len(move_boards) == len(snaps)
    for got, want in zip(move_boards, snaps):
        assert np.array_equal(got, want)


def test_chosen_move_in_legal_and_flips_match():
    snaps, _ = _synthetic_game()
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
    # 动态发现训练目录下所有现存 chess_log_*.txt，逐个验证能由项目真实规则合法重放；
    # 不硬编码数量（数据会随用户增删而变化）；无文件时跳过。
    files = _training_files()
    if not files:
        pytest.skip("训练目录下没有 chess_log_*.txt 文件")
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
    # 动态发现现存全部 chess_log_*.txt，而非固定数量。
    for name in _training_files():
        meta = determine_metadata(name)
        assert meta["self_color"] == COLOR_BLACK
        assert meta["opponent_color"] == COLOR_WHITE
        assert meta["opponent_strength"] == "strong"
        assert meta["metadata_source"] == "user_confirmed"
        assert meta["opponent_id"] == f"unknown:{name}"


def test_other_file_metadata_unknown():
    # 非 chess_log_n 命名的文件保持完全未确认
    meta = determine_metadata("other_file.txt")
    assert meta["self_color"] == ""
    assert meta["opponent_color"] == ""
    assert meta["opponent_strength"] == "unknown"
    assert meta["metadata_source"] == "unconfirmed"
    assert meta["opponent_id"] == "unknown:other_file.txt"


def test_white_data_file_metadata():
    # 白子数据 n.txt（无前缀纯数字）：我方白、对方黑、对手强
    meta = determine_metadata("1.txt")
    assert meta["self_color"] == COLOR_WHITE
    assert meta["opponent_color"] == COLOR_BLACK
    assert meta["opponent_strength"] == "strong"
    assert meta["metadata_source"] == "user_confirmed"
    assert meta["opponent_id"] == "unknown:1.txt"


def test_white_data_raw_path_metadata():
    # 白子数据也支持 raw/ 子目录
    meta = determine_metadata("raw/2.txt")
    assert meta["self_color"] == COLOR_WHITE
    assert meta["opponent_color"] == COLOR_BLACK
    assert meta["opponent_strength"] == "strong"


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
    snaps, _ = _synthetic_game()
    src = os.path.join(str(tmp_path), "chess_log_1.txt")
    with open(src, "w", encoding="utf-8") as f:
        json.dump([s.tolist() for s in snaps], f)

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
    assert summary["snapshot_count"] == len(snaps)
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


# ---------- 孤儿产物清理：源文件已删除时自动清理 ----------

def test_scan_and_process_removes_orphaned(tmp_path):
    training = str(tmp_path)
    for name, seed in (("chess_log_1.txt", 1), ("chess_log_2.txt", 2)):
        snaps, _ = _synthetic_game(seed)
        with open(os.path.join(training, name), "w", encoding="utf-8") as f:
            json.dump([s.tolist() for s in snaps], f)

    r1 = scan_and_process(training, write_outputs=True)
    assert r1["counts"]["parsed"] == 2
    assert os.path.isdir(os.path.join(training, "processed", "game_chess_log_1"))
    assert os.path.isdir(os.path.join(training, "processed", "game_chess_log_2"))

    os.remove(os.path.join(training, "chess_log_1.txt"))
    r2 = scan_and_process(training, write_outputs=True)
    assert r2["counts"]["removed_orphans"] == 1
    assert not os.path.isdir(os.path.join(training, "processed", "game_chess_log_1"))
    assert os.path.isdir(os.path.join(training, "processed", "game_chess_log_2"))

    manifest = load_manifest(os.path.join(training, "game_manifest.csv"))
    paths = {r["source_relative_path"] for r in manifest}
    assert paths == {"chess_log_2.txt"}


def test_scan_and_process_dry_run_does_not_remove(tmp_path):
    training = str(tmp_path)
    snaps, _ = _synthetic_game()
    with open(os.path.join(training, "chess_log_1.txt"), "w", encoding="utf-8") as f:
        json.dump([s.tolist() for s in snaps], f)
    scan_and_process(training, write_outputs=True)
    os.remove(os.path.join(training, "chess_log_1.txt"))
    r = scan_and_process(training, write_outputs=False)  # dry-run 只读
    assert r["counts"]["removed_orphans"] == 1
    assert os.path.isdir(os.path.join(training, "processed", "game_chess_log_1"))
