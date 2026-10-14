"""Plan 11：构建无棋局泄漏的强对手走法偏好数据集。

从 Plan 10 的标准化输出（game_manifest.csv + processed/<game_id>/summary.json + moves.jsonl）
中选取 opponent_strength=strong 且「对手获胜（我方失败，winner==opponent_color）」的对局，
仅以强对手的实际落子构建单步静态评分偏好样本。我方获胜或平局的对局不纳入。

本阶段只生成数据集与特征，不运行 NCS、不修改提交权重。

关键约束：
- 仅纳入 opponent_strength=strong 且 winner==opponent_color（我方失败）的对局；
- 仅在 side_to_move == opponent_color 的决策状态生成示范样本；
- 只有多个合法走法的状态才进入偏好损失样本（单合法走法的 forced move 只计数不进入）；
- 以 game_id 为单位先打乱再切分（固定种子 42），绝不以单条 move 为单位切分；
- action_features = features(apply_move(s, mover, a), perspective=mover)，沿用 reversi.evaluator 的
  固定五项特征与位置表，不得另实现不同版本。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from reversi.evaluator import features
from reversi.rules import apply_move

FEATURE_NAMES = ["disc_diff", "position_diff", "mobility_diff", "corner_diff", "frontier_diff"]
FEATURE_VERSION = "1.0.0"
SPLIT_SEED = 42
LABEL_TYPE = "observed_move_preference"
MIN_GAMES_FOR_SPLIT = 5
INSUFFICIENT_REASON = "insufficient_distinct_games_for_group_split"

_PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.join(_PROJECT_ROOT, "training")
PROCESSED_DIR = os.path.join(TRAINING_DIR, "processed")
DATASETS_DIR = os.path.join(TRAINING_DIR, "datasets")
_MANIFEST_CSV_NAME = "game_manifest.csv"

# 训练/验证/测试划分的比例（固定规则）
_TRAIN_FRAC = 0.70
_VAL_FRAC = 0.15


class DatasetBuildError(Exception):
    """数据集构建失败（明确失败，不伪造结果）。"""


def board_from_list(lst) -> np.ndarray:
    """把嵌套 list 棋盘转成 int8 numpy 数组。"""
    return np.array(lst, dtype=np.int8)


def load_manifest(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_summary(processed_dir: str, game_id: str) -> Optional[Dict]:
    p = os.path.join(processed_dir, game_id, "summary.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_moves(processed_dir: str, game_id: str) -> List[Dict]:
    p = os.path.join(processed_dir, game_id, "moves.jsonl")
    if not os.path.exists(p):
        raise DatasetBuildError(f"缺少 moves.jsonl: {p}")
    moves: List[Dict] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                moves.append(json.loads(line))
    return moves


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


def discover_eligible_games(manifest_rows: List[Dict], processed_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """按资格检查筛选有效强对手棋局，并按源 SHA-256 去重。

    返回 (included, excluded)。
    included 元素含：game_id, source_sha256, opponent_color, opponent_strength,
    opponent_id, source_relative_path。
    """
    included: List[Dict] = []
    excluded: List[Dict] = []
    seen_sha: Dict[str, str] = {}

    for row in manifest_rows:
        game_id = row.get("game_id", "")
        strength = row.get("opponent_strength", "")
        opp_color = row.get("opponent_color", "")
        if strength != "strong" or opp_color not in ("-1", "1"):
            excluded.append({
                "game_id": game_id,
                "reason": "opponent_not_strong_or_color_unknown",
                "opponent_strength": strength,
                "opponent_color": opp_color,
            })
            continue

        summary = load_summary(processed_dir, game_id)
        if summary is None:
            excluded.append({"game_id": game_id, "reason": "missing_summary"})
            continue
        if summary.get("rule_validation_status") != "ok" or not summary.get("terminal"):
            excluded.append({
                "game_id": game_id,
                "reason": "not_replayed_or_not_terminal",
                "rule_validation_status": summary.get("rule_validation_status"),
                "terminal": summary.get("terminal"),
            })
            continue

        # 数据选择规则：只把「我方失败」（对手获胜）的对局当作强对手示范；我方获胜或平局排除。
        winner = summary.get("winner")
        if winner is None or int(winner) != int(opp_color):
            excluded.append({
                "game_id": game_id,
                "reason": "self_won_or_draw_excluded",
                "winner": winner,
                "opponent_color": opp_color,
            })
            continue

        sha = summary.get("source_sha256", "")
        if not sha:
            excluded.append({"game_id": game_id, "reason": "missing_source_sha256"})
            continue
        if sha in seen_sha:
            excluded.append({
                "game_id": game_id,
                "reason": f"duplicate_sha_of_{seen_sha[sha]}",
                "source_sha256": sha,
            })
            continue
        seen_sha[sha] = game_id

        included.append({
            "game_id": game_id,
            "source_sha256": sha,
            "opponent_color": int(opp_color),
            "opponent_strength": strength,
            "opponent_id": row.get("opponent_id", f"unknown:{row.get('source_relative_path', game_id)}"),
            "source_relative_path": row.get("source_relative_path", ""),
        })

    return included, excluded


def build_samples_for_game(
    moves: List[Dict],
    game_id: str,
    opponent_id: str,
    opponent_color: int,
    source_sha256: str,
) -> Tuple[List[Dict], int]:
    """从单局 moves 生成强对手示范样本，返回 (samples, forced_move_count)。"""
    samples: List[Dict] = []
    forced_count = 0
    for mv in moves:
        if mv.get("is_pass"):
            continue
        if mv.get("side_to_move") != opponent_color:
            continue
        legal_moves = mv.get("legal_moves", [])
        chosen = mv.get("chosen_move")
        if chosen is None:
            continue
        assert isinstance(chosen, (list, tuple))
        chosen_mv = [int(chosen[0]), int(chosen[1])]
        # 验证 chosen_move 在 legal_moves 中恰好出现一次
        chosen_idx_list = [i for i, m in enumerate(legal_moves) if [int(m[0]), int(m[1])] == chosen_mv]
        if len(chosen_idx_list) != 1:
            raise DatasetBuildError(
                f"{game_id} ply={mv.get('ply')} chosen_move={chosen} 在 legal_moves 中出现 {len(chosen_idx_list)} 次"
            )
        chosen_idx = chosen_idx_list[0]

        if len(legal_moves) < 2:
            forced_count += 1
            continue

        board = board_from_list(mv["board_before"])
        action_features: List[List[float]] = []
        for m in legal_moves:
            after = apply_move(board, opponent_color, (int(m[0]), int(m[1])))
            feat = features(after, opponent_color)
            action_features.append([float(x) for x in feat])

        samples.append({
            "game_id": game_id,
            "opponent_id": opponent_id,
            "ply": mv.get("ply"),
            "mover_color": opponent_color,
            "board_before": mv.get("board_before"),
            "legal_moves": legal_moves,
            "chosen_move": chosen,
            "chosen_move_index": chosen_idx,
            "action_features": action_features,
            "feature_names": list(FEATURE_NAMES),
            "source_sha256": source_sha256,
            "label_type": LABEL_TYPE,
        })
    return samples, forced_count


def split_game_ids(game_ids: List[str], seed: int = SPLIT_SEED) -> Optional[Dict[str, List[str]]]:
    """以 game_id 为单位确定性切分（固定种子），返回 {"train","validation","test"}。

    G < 5 时返回 None（不生成正式三分割）。
    """
    num_games = len(game_ids)
    if num_games < MIN_GAMES_FOR_SPLIT:
        return None
    sorted_ids = sorted(game_ids)
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(np.array(sorted_ids, dtype=object)))
    n_train = max(3, math.floor(_TRAIN_FRAC * num_games))
    n_val = max(1, math.floor(_VAL_FRAC * num_games))
    if n_train + n_val > num_games - 1:
        n_train = max(1, num_games - 1 - n_val)  # 保证 test 至少 1 个完整 game
    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def make_dataset_id(included_games: List[Dict], seed: int = SPLIT_SEED) -> str:
    """由纳入数据的规范化源 SHA-256、影响标签的字段、特征版本与切分种子生成确定性 id。"""
    sorted_shas = sorted(g["source_sha256"] for g in included_games)
    label_fields = sorted({(int(g["opponent_color"]), g["opponent_strength"]) for g in included_games})
    payload = {
        "source_sha256s": sorted_shas,
        "label_fields": [[c, s] for c, s in label_fields],
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "split_seed": seed,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return "ds_" + digest[:16]



def build_dataset(training_dir: Optional[str] = None, write_outputs: bool = True) -> Dict:
    """扫描、筛选、生成样本、切分并写数据集。返回构建结果摘要。"""
    training_dir = training_dir or TRAINING_DIR
    processed_dir = os.path.join(training_dir, "processed")
    datasets_dir = os.path.join(training_dir, "datasets")
    manifest_path = os.path.join(training_dir, _MANIFEST_CSV_NAME)

    manifest_rows = load_manifest(manifest_path)
    included, excluded = discover_eligible_games(manifest_rows, processed_dir)

    all_samples: List[Dict] = []
    forced_total = 0
    for g in included:
        moves = load_moves(processed_dir, g["game_id"])
        samples, forced = build_samples_for_game(
            moves, g["game_id"], g["opponent_id"], g["opponent_color"], g["source_sha256"],
        )
        all_samples.extend(samples)
        forced_total += forced

    # 保证输出顺序确定性（按 game_id、ply）
    all_samples.sort(key=lambda rec: (rec["game_id"], rec["ply"]))

    num_games = len(included)
    eligible = num_games >= MIN_GAMES_FOR_SPLIT
    dataset_id = make_dataset_id(included)

    split = split_game_ids([g["game_id"] for g in included]) if eligible else None

    samples_by_game: Dict[str, List[Dict]] = {}
    for s in all_samples:
        samples_by_game.setdefault(s["game_id"], []).append(s)

    manifest: Dict = {
        "dataset_id": dataset_id,
        "fingerprint": dataset_id,
        "scoring_model": "single_step_static_scoring_preference (not full-policy cloning)",
        "label_type": LABEL_TYPE,
        "feature_names": list(FEATURE_NAMES),
        "feature_version": FEATURE_VERSION,
        "split_seed": SPLIT_SEED,
        "eligible_for_preference_training": eligible,
        "blocking_reason": None if eligible else INSUFFICIENT_REASON,
        "test_split_must_not_be_used_for_selection": True,
        "opponent_colors": sorted({int(g["opponent_color"]) for g in included}),
        "opponent_strength": "strong",
        "opponent_ids": sorted({g["opponent_id"] for g in included}),
        "included_game_ids": sorted(g["game_id"] for g in included),
        "excluded_games": excluded,
        "source_sha256s": sorted(g["source_sha256"] for g in included),
        "opponent_isolation_note": "按棋局隔离、非按对手隔离（对手身份未知且非同一对手）",
        "excluded_forced_move_count": forced_total,
        "total_sample_count": len(all_samples),
        "generated_at": datetime.now().isoformat(),
    }

    if split is not None:
        split_info: Dict = {}
        for name in ("train", "validation", "test"):
            gids = split[name]
            split_info[name] = {
                "game_ids": gids,
                "game_count": len(gids),
                "sample_count": sum(len(samples_by_game.get(gid, [])) for gid in gids),
            }
        manifest["split"] = split_info
    else:
        manifest["split"] = None

    out_dir = os.path.join(datasets_dir, dataset_id)
    if write_outputs:
        # 防御：同一 dataset_id 目录已存在但 fingerprint 不同则拒绝覆盖（内容不同需新版本）
        existing_manifest_path = os.path.join(out_dir, "dataset_manifest.json")
        if os.path.exists(existing_manifest_path):
            with open(existing_manifest_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("fingerprint") != dataset_id:
                raise DatasetBuildError(
                    f"dataset_id {dataset_id} 已存在但 fingerprint 不同，拒绝静默覆盖，请使用新 dataset_id/version"
                )

        write_jsonl(os.path.join(out_dir, "preference_pool.jsonl"), all_samples)
        if eligible and split is not None:
            for name in ("train", "validation", "test"):
                recs: List[Dict] = []
                for gid in split[name]:
                    recs.extend(samples_by_game.get(gid, []))
                write_jsonl(os.path.join(out_dir, f"{name}.jsonl"), recs)
        write_json(os.path.join(out_dir, "dataset_manifest.json"), manifest)

    return {
        "dataset_id": dataset_id,
        "eligible_for_preference_training": eligible,
        "blocking_reason": None if eligible else INSUFFICIENT_REASON,
        "included_game_count": num_games,
        "excluded_game_count": len(excluded),
        "total_sample_count": len(all_samples),
        "forced_move_count": forced_total,
        "split": split,
        "output_dir": out_dir,
        "manifest": manifest,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plan 11 强对手走法偏好数据集构建")
    parser.add_argument("--training-dir", default=None, help="训练目录（默认 Project1/training）")
    parser.add_argument("--dry-run", action="store_true", help="只构建验证，不写输出文件")
    args = parser.parse_args(argv)

    result = build_dataset(args.training_dir, write_outputs=not args.dry_run)
    print(f"dataset_id={result['dataset_id']}")
    print(f"eligible_for_preference_training={result['eligible_for_preference_training']}")
    print(f"included_games={result['included_game_count']} excluded_games={result['excluded_game_count']}")
    print(f"total_samples={result['total_sample_count']} forced_moves={result['forced_move_count']}")
    if result["split"] is not None:
        for name in ("train", "validation", "test"):
            gids = result["split"][name]
            print(f"  {name}: games={len(gids)} {gids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

