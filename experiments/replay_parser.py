"""Plan 10：从 training 目录解析、重建并验证原始 Reversed Reversi 棋谱。

本模块只负责把原始快照棋谱解析为可追溯的标准化逐局记录；不生成训练/验证/测试划分、
不运行权重训练、不修改 submission/agent.py。

输出：
- training/processed/<game_id>/moves.jsonl / summary.json / self_moves.jsonl / opponent_moves.jsonl
- training/game_manifest.csv（增量更新，不整体覆盖）
- training/processed/parse_manifest.json

规则引擎复用 reversi.rules（不另造一套生产规则）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from reversi.board import (
    BOARD_SIZE, COLOR_BLACK, COLOR_WHITE, count, initial_board, opponent,
)
from reversi.rules import apply_move, flips, is_terminal, legal_moves

PARSER_VERSION = "1.0.0"

# 命名约束（用户确认）：所有棋盘数据文件统一按 chess_log_n.txt 命名（n 为正整数），
# 可选位于 raw/ 子目录下。
_CHESS_LOG_RE = re.compile(r"^(?:raw/)?chess_log_\d+\.txt$")

# 颜色约束（用户确认，适用于所有 chess_log_n 文件）：我方执黑（self_color=-1）、
# 对方执白（opponent_color=1）。
# 对手强度约束（用户确认）：文件夹内棋谱均为“我方失败”的对局，对手均可视为强对手
# （opponent_strength=strong）；对手不来自同一个强对手（身份未知，opponent_id=unknown:<path>）。

# 生成目录（绝不作为原始输入递归扫描）。
_GENERATED_DIRS = {"processed", "datasets", "preference_runs", "candidates"}
_MANIFEST_CSV_NAME = "game_manifest.csv"
_PARSE_MANIFEST_NAME = "parse_manifest.json"

_PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.join(_PROJECT_ROOT, "training")
RAW_DIR = os.path.join(TRAINING_DIR, "raw")
PROCESSED_DIR = os.path.join(TRAINING_DIR, "processed")

_MANIFEST_CSV_COLUMNS = [
    "source_relative_path", "game_id", "self_color", "opponent_color",
    "opponent_strength", "opponent_id", "metadata_source", "notes",
]

_VALID_PIECE_VALUES = (-1, 0, 1)
_VALID_STRENGTHS = ("strong", "unknown", "weak")
_VALID_METADATA_SOURCES = ("user_confirmed", "platform", "unconfirmed")


class ReplayError(Exception):
    """棋谱解析/重建失败（明确失败，不伪造结果）。"""

    def __init__(self, kind: str, message: str):
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message


def make_game_id(source_relative_path: str) -> str:
    """生成稳定且唯一的 game_id，不因文件处理顺序变化而改变。"""
    base, _ext = os.path.splitext(source_relative_path.replace("\\", "/"))
    slug = re.sub(r"[^0-9A-Za-z]+", "_", base).strip("_")
    if not slug:
        slug = "unknown"
    return "game_" + slug


def board_to_list(board: np.ndarray) -> List[List[int]]:
    """把 int8 棋盘转为嵌套 list（用于 JSON 序列化）。"""
    return [[int(board[r, c]) for c in range(BOARD_SIZE)] for r in range(BOARD_SIZE)]


def compute_winner(board: np.ndarray) -> int:
    """反转规则胜负：己方棋子更少者胜。返回 COLOR_BLACK / COLOR_WHITE / 0(平局)。"""
    black = count(board, COLOR_BLACK)
    white = count(board, COLOR_WHITE)
    if black < white:
        return COLOR_BLACK
    if white < black:
        return COLOR_WHITE
    return 0


def parse_snapshots(text: str) -> List[np.ndarray]:
    """解析 JSON 快照数组并逐帧校验，返回 np.int8 棋盘列表。

    只接受一个 JSON 数组，内容为连续 8x8 棋盘快照，棋子值只允许 -1/0/1。
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReplayError("invalid_json", f"JSON 解析失败: {exc}") from exc
    if not isinstance(data, list):
        raise ReplayError("unsupported_format", "顶层不是 JSON 数组")
    if len(data) == 0:
        raise ReplayError("empty_snapshots", "快照数组为空")
    snaps: List[np.ndarray] = []
    for idx, raw in enumerate(data):
        snaps.append(_validate_snapshot(raw, idx))
    return snaps


def _validate_snapshot(raw, idx: int) -> np.ndarray:
    """校验单帧必须是 8x8 且取值在 {-1,0,1}。"""
    if not isinstance(raw, list) or len(raw) != BOARD_SIZE:
        raise ReplayError("invalid_board", f"快照[{idx}] 不是 {BOARD_SIZE} 行")
    rows: List[List[int]] = []
    for r, row in enumerate(raw):
        if not isinstance(row, list) or len(row) != BOARD_SIZE:
            raise ReplayError("invalid_board", f"快照[{idx}] 第 {r} 行不是 {BOARD_SIZE} 列")
        clean: List[int] = []
        for c, v in enumerate(row):
            if isinstance(v, bool) or not isinstance(v, int) or v not in _VALID_PIECE_VALUES:
                raise ReplayError("invalid_board", f"快照[{idx}] ({r},{c}) 取值非法: {v!r}")
            clean.append(v)
        rows.append(clean)
    return np.array(rows, dtype=np.int8)


def reconstruct_game(snapshots: List[np.ndarray]) -> Tuple[List[Dict], bool]:
    """从标准初始局面重放全部快照，返回 (动作列表, 是否终局)。

    动作列表按 ply（1-based）递增；每条动作 dict 字段：
    - ply, side_to_move, board_before(np), legal_moves(list of [r,c]),
      chosen_move([r,c] 或 None), flipped_discs(list of [r,c]),
      board_after(np), source_snapshot_index(int 或 None), is_pass(bool)

    首手按 2.3/3.5 规则：从标准初始棋盘穷举黑方合法首手并要求唯一匹配（不硬编码坐标）。
    之后每两帧之间按“实际轮到方”穷举合法落子并要求唯一匹配；若当前方无合法走法，
    仅当规则允许（非终局）时 pass 跳过该方，不因不匹配而静默改变颜色次序。
    """
    board = initial_board()
    side = COLOR_BLACK  # 黑先
    actions: List[Dict] = []
    ply = 0
    snap_idx = 0

    for target in snapshots:
        # 处理无合法走法的 pass（跳过该方，棋盘不变）
        while not legal_moves(board, side):
            if is_terminal(board):
                raise ReplayError(
                    "illegal_transition",
                    f"第 {snap_idx} 帧前双方均无合法走法（已终局），但仍有剩余快照",
                )
            ply += 1
            actions.append({
                "ply": ply,
                "side_to_move": int(side),
                "board_before": board.copy(),
                "legal_moves": [],
                "chosen_move": None,
                "flipped_discs": [],
                "board_after": board.copy(),
                "source_snapshot_index": None,
                "is_pass": True,
            })
            side = opponent(side)

        # 当前方有合法落子：穷举并要求唯一匹配下一帧
        lm = legal_moves(board, side)
        matches = [m for m in lm if np.array_equal(apply_move(board, side, m), target)]
        if not matches:
            raise ReplayError(
                "illegal_transition",
                f"第 {snap_idx} 帧无法由 {int(side)} 的任意合法落子得到（合法走法 {lm}）",
            )
        if len(matches) > 1:
            raise ReplayError(
                "ambiguous_transition",
                f"第 {snap_idx} 帧匹配 {len(matches)} 个合法落子: {matches}",
            )
        move = matches[0]
        flipped = flips(board, side, move[0], move[1])
        board_after = apply_move(board, side, move)
        ply += 1
        actions.append({
            "ply": ply,
            "side_to_move": int(side),
            "board_before": board.copy(),
            "legal_moves": [[int(r), int(c)] for r, c in lm],
            "chosen_move": [int(move[0]), int(move[1])],
            "flipped_discs": [[int(r), int(c)] for r, c in flipped],
            "board_after": board_after,
            "source_snapshot_index": snap_idx,
            "is_pass": False,
        })
        board = board_after
        side = opponent(side)
        snap_idx += 1

    terminal = bool(is_terminal(board))
    return actions, terminal


def _serialize_action(action: Dict, game_id: str, source_sha256: str) -> Dict:
    return {
        "game_id": game_id,
        "ply": action["ply"],
        "side_to_move": action["side_to_move"],
        "board_before": board_to_list(action["board_before"]),
        "legal_moves": action["legal_moves"],
        "chosen_move": action["chosen_move"],
        "flipped_discs": action["flipped_discs"],
        "board_after": board_to_list(action["board_after"]),
        "source_snapshot_index": action["source_snapshot_index"],
        "source_sha256": source_sha256,
        "is_pass": action["is_pass"],
    }


def _serialize_side_move(action: Dict, game_id: str, source_sha256: str, label: str) -> Dict:
    return {
        "game_id": game_id,
        "ply": action["ply"],
        "side": action["side_to_move"],
        "board_before": board_to_list(action["board_before"]),
        "legal_moves": action["legal_moves"],
        "chosen_move": action["chosen_move"],
        "flipped_discs": action["flipped_discs"],
        "source_sha256": source_sha256,
        "label": label,
    }



def build_summary(
    game_id: str,
    source_relative_path: str,
    source_sha256: str,
    snapshot_count: int,
    actions: List[Dict],
    terminal: bool,
    meta: Dict,
    final_board: np.ndarray,
) -> Dict:
    move_count = sum(1 for a in actions if not a["is_pass"])
    pass_count = sum(1 for a in actions if a["is_pass"])
    black = count(final_board, COLOR_BLACK)
    white = count(final_board, COLOR_WHITE)
    winner = compute_winner(final_board)
    self_color = meta["self_color"] if meta["self_color"] != "" else None
    opponent_color = meta["opponent_color"] if meta["opponent_color"] != "" else None
    return {
        "game_id": game_id,
        "source_relative_path": source_relative_path,
        "source_sha256": source_sha256,
        "snapshot_count": snapshot_count,
        "move_count": move_count,
        "pass_count": pass_count,
        "action_count": len(actions),
        "self_color": self_color,
        "opponent_color": opponent_color,
        "opponent_strength": meta["opponent_strength"],
        "opponent_id": meta["opponent_id"],
        "metadata_source": meta["metadata_source"],
        "metadata_status": "user_confirmed" if meta["metadata_source"] == "user_confirmed" else "unannotated",
        "black_count": black,
        "white_count": white,
        "winner": winner,
        "terminal": terminal,
        "parser_version": PARSER_VERSION,
        "rule_validation_status": "ok" if terminal else "replay_ok_not_terminal",
    }


def write_jsonl(path: str, records: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def process_bytes(
    raw: bytes,
    source_relative_path: str,
    output_dir: str,
    write_outputs: bool = True,
) -> Dict:
    """解析并验证一个棋谱文件的字节内容，可选择写出标准化产物。

    成功返回结果 dict（status="parsed"），失败抛出 ReplayError。
    """
    source_sha256 = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    snaps = parse_snapshots(text)
    actions, terminal = reconstruct_game(snaps)
    game_id = make_game_id(source_relative_path)
    meta = determine_metadata(source_relative_path)
    validate_metadata(meta)
    final_board = actions[-1]["board_after"]

    game_dir = os.path.join(output_dir, game_id)
    if write_outputs:
        os.makedirs(game_dir, exist_ok=True)
        write_jsonl(
            os.path.join(game_dir, "moves.jsonl"),
            [_serialize_action(a, game_id, source_sha256) for a in actions],
        )
        summary = build_summary(game_id, source_relative_path, source_sha256, len(snaps), actions, terminal, meta, final_board)
        write_json(os.path.join(game_dir, "summary.json"), summary)
        if meta["self_color"] != "":
            self_moves = [a for a in actions if not a["is_pass"] and a["side_to_move"] == meta["self_color"]]
            write_jsonl(
                os.path.join(game_dir, "self_moves.jsonl"),
                [_serialize_side_move(a, game_id, source_sha256, "self_diagnostic") for a in self_moves],
            )
        if meta["opponent_color"] != "":
            opp_moves = [a for a in actions if not a["is_pass"] and a["side_to_move"] == meta["opponent_color"]]
            label = "strong_opponent_training" if meta["opponent_strength"] == "strong" else "opponent_diagnostic"
            write_jsonl(
                os.path.join(game_dir, "opponent_moves.jsonl"),
                [_serialize_side_move(a, game_id, source_sha256, label) for a in opp_moves],
            )

    summary = build_summary(game_id, source_relative_path, source_sha256, len(snaps), actions, terminal, meta, final_board)
    return {
        "status": "parsed",
        "source_relative_path": source_relative_path,
        "source_sha256": source_sha256,
        "game_id": game_id,
        "snapshot_count": len(snaps),
        "move_count": summary["move_count"],
        "pass_count": summary["pass_count"],
        "action_count": summary["action_count"],
        "terminal": terminal,
        "winner": summary["winner"],
        "black_count": summary["black_count"],
        "white_count": summary["white_count"],
        "metadata": meta,
        "output_dir": game_dir,
    }


def process_file(source_abs_path: str, source_relative_path: str, output_dir: str, write_outputs: bool = True) -> Dict:
    with open(source_abs_path, "rb") as f:
        raw = f.read()
    return process_bytes(raw, source_relative_path, output_dir, write_outputs=write_outputs)



def determine_metadata(source_relative_path: str) -> Dict:
    """按用户确认的约束确定棋谱元数据。

    约束（已写入报告并持续遵守）：
    - 所有棋盘数据文件按 chess_log_n.txt 命名；
    - 所有棋盘中我方执黑（self_color=-1）、对方执白（opponent_color=1）；
    - 文件夹内均为“我方失败”对局，对手均可视为强对手（opponent_strength=strong）；
    - 对手不来自同一个强对手（身份未知，opponent_id=unknown:<path>）。
    """
    rel = source_relative_path.replace("\\", "/")
    if _CHESS_LOG_RE.match(rel):
        return {
            "self_color": COLOR_BLACK,
            "opponent_color": COLOR_WHITE,
            "opponent_strength": "strong",
            "opponent_id": f"unknown:{source_relative_path}",
            "metadata_source": "user_confirmed",
            "notes": "用户确认：我方黑/对方白；对手视为强对手（身份未知，非同一对手）；数据选择仅纳入对手获胜（我方失败）的对局",
        }
    return {
        "self_color": "",
        "opponent_color": "",
        "opponent_strength": "unknown",
        "opponent_id": f"unknown:{source_relative_path}",
        "metadata_source": "unconfirmed",
        "notes": "awaiting user metadata",
    }


def validate_metadata(meta: Dict) -> None:
    """校验元数据约定，非法则抛 ReplayError("invalid_metadata")。"""
    for key in ("self_color", "opponent_color"):
        v = meta.get(key)
        if v not in (COLOR_BLACK, COLOR_WHITE, ""):
            raise ReplayError("invalid_metadata", f"{key} 只能为 -1/1/空，得到 {v!r}")
    if meta["self_color"] != "" and meta["opponent_color"] != "":
        if int(meta["self_color"]) != -int(meta["opponent_color"]):
            raise ReplayError("invalid_metadata", "self_color 与 opponent_color 已知时必须互为相反数")
    if meta["opponent_strength"] not in _VALID_STRENGTHS:
        raise ReplayError("invalid_metadata", f"opponent_strength 非法: {meta['opponent_strength']!r}")
    if meta["metadata_source"] not in _VALID_METADATA_SOURCES:
        raise ReplayError("invalid_metadata", f"metadata_source 非法: {meta['metadata_source']!r}")


def manifest_row(source_relative_path: str, game_id: str, meta: Dict) -> Dict:
    return {
        "source_relative_path": source_relative_path,
        "game_id": game_id,
        "self_color": meta["self_color"],
        "opponent_color": meta["opponent_color"],
        "opponent_strength": meta["opponent_strength"],
        "opponent_id": meta["opponent_id"],
        "metadata_source": meta["metadata_source"],
        "notes": meta["notes"],
    }


def load_manifest(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def write_manifest(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_MANIFEST_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _MANIFEST_CSV_COLUMNS})



def discover_source_files(training_dir: str) -> Dict[str, str]:
    """返回 {source_relative_path: absolute_path}。

    首选读取 training/raw/ 下的直接子文件；也允许读取 training/ 根目录下的直接子级
    棋谱文件。绝不递归扫描 processed/、datasets/、preference_runs/、candidates/ 等生成目录。
    """
    sources: Dict[str, str] = {}
    raw_dir = os.path.join(training_dir, "raw")
    if os.path.isdir(raw_dir):
        for name in sorted(os.listdir(raw_dir)):
            p = os.path.join(raw_dir, name)
            if os.path.isfile(p):
                sources["raw/" + name] = p
    for name in sorted(os.listdir(training_dir)):
        p = os.path.join(training_dir, name)
        if not os.path.isfile(p):
            continue
        if name == _MANIFEST_CSV_NAME:
            continue
        sources[name] = p
    return sources


def scan_and_process(training_dir: Optional[str] = None, write_outputs: bool = True) -> Dict:
    """扫描全部原始棋谱、解析验证、按 SHA-256 去重、写产物并更新 manifest。

    返回 {"counts": {...}, "files": [...], "parse_manifest": {...}}。
    """
    training_dir = training_dir or TRAINING_DIR
    processed_dir = os.path.join(training_dir, "processed")
    manifest_path = os.path.join(training_dir, _MANIFEST_CSV_NAME)
    parse_manifest_path = os.path.join(processed_dir, _PARSE_MANIFEST_NAME)

    sources = discover_source_files(training_dir)

    results: List[Dict] = []
    seen_sha: Dict[str, str] = {}
    parsed_meta: List[Dict] = []  # 用于增量更新 manifest

    for rel in sorted(sources):
        abs_path = sources[rel]
        try:
            with open(abs_path, "rb") as f:
                raw = f.read()
        except OSError as exc:
            results.append({"source_relative_path": rel, "status": "failed", "reason": f"read_error: {exc}"})
            continue
        sha = hashlib.sha256(raw).hexdigest()
        if sha in seen_sha:
            results.append({
                "source_relative_path": rel,
                "source_sha256": sha,
                "status": "duplicate",
                "duplicate_of": seen_sha[sha],
            })
            continue
        seen_sha[sha] = rel

        try:
            info = process_bytes(raw, rel, processed_dir, write_outputs=write_outputs)
            info["status"] = "parsed"
            results.append(info)
            meta = info["metadata"]
            parsed_meta.append(manifest_row(rel, info["game_id"], meta))
        except ReplayError as exc:
            results.append({
                "source_relative_path": rel,
                "source_sha256": sha,
                "status": "failed",
                "reason": exc.kind,
                "detail": exc.message,
            })
        except Exception as exc:  # 意外错误：记录，不伪造结果
            results.append({
                "source_relative_path": rel,
                "source_sha256": sha,
                "status": "failed",
                "reason": "unexpected_error",
                "detail": str(exc),
            })

    # 清理孤儿产物：源文件已不在当前 sources 中的 processed/<game_id> 目录与 manifest 行。
    # 仅在 write_outputs=True 时执行删除；dry-run 只读不删（仍记录 removed_orphans 计数）。
    current_paths = set(sources.keys())
    existing_rows = load_manifest(manifest_path)
    orphan_rows = [r for r in existing_rows if r.get("source_relative_path") not in current_paths]
    removed_game_ids = [r.get("game_id") for r in orphan_rows if r.get("game_id")]
    if write_outputs:
        for gid in removed_game_ids:
            gdir = os.path.join(processed_dir, str(gid))
            if str(gid).startswith("game_") and os.path.isdir(gdir):
                shutil.rmtree(gdir)

    # 重写 manifest：只保留当前源文件对应的行（孤儿行删除），再增量新增/修正本次解析成功的行。
    # - 缺失行新增；
    # - 已存在行：仅当新元数据为 user_confirmed（权威用户确认）时才修正颜色/强度等字段，
    #   否则保留原行（不猜测玩家颜色和对手强度）。
    existing_rows = [r for r in existing_rows if r.get("source_relative_path") in current_paths]
    existing_by_path = {r["source_relative_path"]: r for r in existing_rows}
    for new_row in parsed_meta:
        rel = new_row["source_relative_path"]
        if rel not in existing_by_path:
            existing_rows.append(new_row)
        elif new_row["metadata_source"] == "user_confirmed":
            existing_by_path[rel].update(new_row)
    existing_rows.sort(key=lambda r: r["source_relative_path"])
    if write_outputs:
        write_manifest(manifest_path, existing_rows)

    counts = {
        "scanned": len(sources),
        "parsed": sum(1 for r in results if r["status"] == "parsed"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "duplicates": sum(1 for r in results if r["status"] == "duplicate"),
        "unannotated": sum(1 for r in parsed_meta if r["metadata_source"] == "unconfirmed"),
        "removed_orphans": len(removed_game_ids),
    }

    parse_manifest = {
        "parser_version": PARSER_VERSION,
        "generated_at": datetime.now().isoformat(),
        "training_dir": os.path.abspath(training_dir),
        "files": results,
        "counts": counts,
        "removed_orphans": removed_game_ids,
    }
    if write_outputs:
        write_json(parse_manifest_path, parse_manifest)

    return {"counts": counts, "files": results, "parse_manifest": parse_manifest}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plan 10 棋谱解析与验证")
    parser.add_argument("--training-dir", default=None, help="训练目录（默认 Project1/training）")
    parser.add_argument("--dry-run", action="store_true", help="只解析验证，不写任何输出文件")
    args = parser.parse_args(argv)

    result = scan_and_process(args.training_dir, write_outputs=not args.dry_run)
    counts = result["counts"]
    print(f"scanned={counts['scanned']} parsed={counts['parsed']} failed={counts['failed']} "
          f"duplicates={counts['duplicates']} unannotated={counts['unannotated']}")
    for f in result["files"]:
        if f["status"] == "parsed":
            print(f"  [parsed]   {f['source_relative_path']} -> {f['game_id']} "
                  f"(snapshots={f['snapshot_count']}, actions={f['action_count']}, winner={f['winner']})")
        elif f["status"] == "duplicate":
            print(f"  [duplicate] {f['source_relative_path']} (of {f['duplicate_of']})")
        else:
            print(f"  [failed]   {f['source_relative_path']}: {f.get('reason')} {f.get('detail', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

