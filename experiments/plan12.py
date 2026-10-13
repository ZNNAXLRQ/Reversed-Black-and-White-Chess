"""Plan 12：使用 NCS-C 优化强对手走法偏好目标。

只读取 Plan 11 数据集的预计算特征，用通用 NCSOptimizer 在归一化空间 z∈[0,1]^5 搜索，
最小化 train split 的按 game 等权偏好损失；validation 只用于候选 eligibility 判定（严格更低才合格），
test 只用于最终报告。不调用 MatchRunner，不修改正式 ReversiObjective / NCS 源码 / submission。

用法：
    python -m experiments.plan12 --dataset-id ds_351633bed8b8b15b --seed 1001
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from optimization.config import OptimizationConfig
from optimization.ncs.optimizer import NCSConfig, NCSOptimizer
from optimization.parameter_space import ParameterSpace
from reversi.agent import BASELINE_WEIGHTS

from .move_preference_objective import (
    TEMPERATURE, ReplayPreferenceObjective, compute_preference_loss,
    is_eligible, load_samples_by_game,
)
from .preference_dataset import FEATURE_NAMES, FEATURE_VERSION, TRAINING_DIR

# 显式指定的数据集（Plan 11 生成的唯一 eligible 数据集）
DEFAULT_DATASET_ID = "ds_351633bed8b8b15b"
# 第一次偏好优化使用的 seed（后续重跑/新数据集用下一个未使用 seed）
DEFAULT_SEED = 1001

# NCS-C 参数（Plan 12 固定）
POPULATION_SIZE = 10
T_MAX = 50
R = 0.99
EPOCH = 10
SIGMA0 = 0.1

_BASELINE_LIST = [float(x) for x in BASELINE_WEIGHTS]


def _atomic_write_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        raise ValueError(f"目标文件已存在，拒绝覆盖: {path}")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def _write_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _append_registry(path: str, key: str, entry: Dict, dedup_key: str) -> None:
    """只追加、不重写历史值的 registry。按 dedup_key 去重。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data: Dict = {key: []}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    records = data.get(key, [])
    if not any(r.get(dedup_key) == entry.get(dedup_key) for r in records):
        records.append(entry)
    data[key] = records
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def check_admission(manifest: Dict, train_sg: Dict, val_sg: Dict, test_sg: Dict) -> Dict:
    """按 Plan 12 第二节做数据准入检查，不满足则抛 ValueError。"""
    if not manifest.get("eligible_for_preference_training"):
        raise ValueError("dataset_manifest.eligible_for_preference_training != true")
    if list(manifest.get("feature_names", [])) != list(FEATURE_NAMES):
        raise ValueError("feature_names 与固定版本不一致")
    if manifest.get("feature_version") != FEATURE_VERSION:
        raise ValueError("feature_version 与固定版本不一致")

    train_games = set(train_sg)
    val_games = set(val_sg)
    test_games = set(test_sg)
    if (train_games & val_games) or (train_games & test_games) or (val_games & test_games):
        raise ValueError("split 之间 game_id 有交集，违反棋局隔离")

    all_games = train_games | val_games | test_games
    train_samples = sum(len(v) for v in train_sg.values())
    val_samples = sum(len(v) for v in val_sg.values())
    test_samples = sum(len(v) for v in test_sg.values())

    problems: List[str] = []
    if len(all_games) < 5:
        problems.append(f"unique games {len(all_games)} < 5")
    if len(train_games) < 3:
        problems.append(f"train games {len(train_games)} < 3")
    if len(val_games) < 1:
        problems.append("validation games < 1")
    if len(test_games) < 1:
        problems.append("test games < 1")
    if train_samples < 30:
        problems.append(f"train samples {train_samples} < 30")
    if val_samples < 5:
        problems.append(f"validation samples {val_samples} < 5")
    if test_samples < 5:
        problems.append(f"test samples {test_samples} < 5")
    if problems:
        raise ValueError("数据准入不满足: " + "; ".join(problems))

    return {
        "train_games": len(train_games),
        "val_games": len(val_games),
        "test_games": len(test_games),
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
    }


def run_preference_training(
    dataset_id: str,
    seed: int,
    training_dir: Optional[str] = None,
    write_outputs: bool = True,
) -> Dict:
    """执行一次偏好优化运行，返回结果摘要。"""
    training_dir = training_dir or TRAINING_DIR
    dataset_dir = os.path.join(training_dir, "datasets", dataset_id)
    manifest_path = os.path.join(dataset_dir, "dataset_manifest.json")
    if not os.path.exists(manifest_path):
        raise ValueError(f"dataset_manifest.json 不存在: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_sg = load_samples_by_game(os.path.join(dataset_dir, "train.jsonl"))
    val_sg = load_samples_by_game(os.path.join(dataset_dir, "validation.jsonl"))
    test_sg = load_samples_by_game(os.path.join(dataset_dir, "test.jsonl"))

    admission = check_admission(manifest, train_sg, val_sg, test_sg)

    preference_run_id = f"pref_dataset_{dataset_id}_seed_{seed}"
    run_dir = os.path.join(training_dir, "preference_runs", preference_run_id)

    # 1. baseline 对照 loss
    baseline_weights = np.asarray(BASELINE_WEIGHTS, dtype=np.float64)
    baseline_train = compute_preference_loss(train_sg, baseline_weights)
    baseline_val = compute_preference_loss(val_sg, baseline_weights)
    baseline_test = compute_preference_loss(test_sg, baseline_weights)

    # 2. NCS-C 仅在 train split 上优化
    objective = ReplayPreferenceObjective(train_sg, temperature=TEMPERATURE)
    space = ParameterSpace([0.0] * 5, [1.0] * 5)
    optimizer = NCSOptimizer(
        NCSConfig(T_max=T_MAX, population_size=POPULATION_SIZE, sigma0=SIGMA0, r=R, epoch=EPOCH)
    )
    config = OptimizationConfig(seed=seed)

    t0 = time.time()
    result = optimizer.optimize(objective, space, config)
    elapsed = time.time() - t0

    best_z = np.asarray(result.best_candidate, dtype=np.float64)
    candidate_weights = objective.decode(best_z)

    # 3. 候选只计算一次 train/val/test loss
    candidate_train = compute_preference_loss(train_sg, candidate_weights)
    candidate_val = compute_preference_loss(val_sg, candidate_weights)
    candidate_test = compute_preference_loss(test_sg, candidate_weights)

    eligible = is_eligible(candidate_val, baseline_val)

    candidate_id = f"preference_candidate_{preference_run_id}"
    config_obj: Dict = {
        "dataset_id": dataset_id,
        "source_summary": {
            "included_game_ids": manifest.get("included_game_ids"),
            "source_sha256s": manifest.get("source_sha256s"),
            "feature_names": list(FEATURE_NAMES),
            "feature_version": FEATURE_VERSION,
        },
        "ncs_params": {
            "population_size": POPULATION_SIZE,
            "T_max": T_MAX,
            "r": R,
            "epoch": EPOCH,
            "sigma0": SIGMA0,
        },
        "seed": seed,
        "temperature": TEMPERATURE,
        "decode": "w = -10 + 20*z",
        "objective_formula": "loss(s)=logsumexp(logits)-logit_chosen; logit=dot(w,phi(s,a))/T; cost=mean over games of mean over samples",
    }

    training_summary: Dict = {
        "preference_run_id": preference_run_id,
        "dataset_id": dataset_id,
        "seed": seed,
        "best_cost_train": result.best_cost,
        "best_weights": [float(x) for x in candidate_weights],
        "best_z": [float(x) for x in best_z],
        "num_evaluations": result.num_evaluations,
        "num_iterations": result.num_iterations,
        "stop_reason": result.stop_reason,
        "elapsed_s": elapsed,
        "history": result.history,
    }

    baseline_metrics: Dict = {
        "weights": _BASELINE_LIST,
        "train_loss": baseline_train,
        "validation_loss": baseline_val,
        "test_loss": baseline_test,
    }

    candidate_metrics: Dict = {
        "candidate_id": candidate_id,
        "weights": [float(x) for x in candidate_weights],
        "train_loss": candidate_train,
        "validation_loss": candidate_val,
        "test_loss": candidate_test,
        "baseline_validation_loss": baseline_val,
        "eligible_for_match_evaluation": eligible,
        "eligibility_rule": "candidate_validation_loss < baseline_validation_loss",
    }


    if write_outputs:
        os.makedirs(run_dir, exist_ok=True)
        _write_json(os.path.join(run_dir, "config.json"), config_obj)
        _write_json(os.path.join(run_dir, "training_summary.json"), training_summary)
        _write_json(os.path.join(run_dir, "baseline_metrics.json"), baseline_metrics)
        _write_json(os.path.join(run_dir, "candidate_metrics.json"), candidate_metrics)

        log_lines = [
            f"[{datetime.now().isoformat()}] preference_run_id={preference_run_id}",
            f"dataset_id={dataset_id} seed={seed}",
            f"admission={json.dumps(admission)}",
            f"baseline train={baseline_train:.8f} val={baseline_val:.8f} test={baseline_test:.8f}",
            f"ncs best_cost(train)={result.best_cost:.8f} evals={result.num_evaluations} "
            f"stop={result.stop_reason} elapsed={elapsed:.2f}s",
            f"candidate train={candidate_train:.8f} val={candidate_val:.8f} test={candidate_test:.8f}",
            f"eligible_for_match_evaluation={eligible}",
        ]
        with open(os.path.join(run_dir, "run.log"), "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")

        candidate_path = os.path.join(training_dir, "candidates", f"{candidate_id}.json")
        candidate_obj: Dict = {
            "candidate_id": candidate_id,
            "weights": [float(x) for x in candidate_weights],
            "eligible_for_match_evaluation": eligible,
            "dataset_id": dataset_id,
            "preference_run_id": preference_run_id,
            "train_loss": candidate_train,
            "validation_loss": candidate_val,
            "baseline_validation_loss": baseline_val,
            "test_loss": candidate_test,
            "seed": seed,
            "config": config_obj,
            "feature_names": list(FEATURE_NAMES),
            "created_at": datetime.now().isoformat(),
            "status": "preference_candidate_not_champion",
        }
        _atomic_write_json(candidate_path, candidate_obj)

        _append_registry(
            os.path.join(training_dir, "preference_runs", "seed_registry.json"),
            "seeds",
            {"seed": seed, "preference_run_id": preference_run_id, "dataset_id": dataset_id,
             "used_at": datetime.now().isoformat()},
            dedup_key="seed",
        )
        _append_registry(
            os.path.join(training_dir, "candidates", "candidate_registry.json"),
            "candidates",
            {"candidate_id": candidate_id, "preference_run_id": preference_run_id,
             "dataset_id": dataset_id, "seed": seed,
             "eligible_for_match_evaluation": eligible,
             "status": "preference_candidate_not_champion",
             "created_at": datetime.now().isoformat()},
            dedup_key="candidate_id",
        )

    return {
        "preference_run_id": preference_run_id,
        "dataset_id": dataset_id,
        "seed": seed,
        "admission": admission,
        "baseline": {"train_loss": baseline_train, "validation_loss": baseline_val, "test_loss": baseline_test},
        "candidate": {
            "weights": [float(x) for x in candidate_weights],
            "train_loss": candidate_train,
            "validation_loss": candidate_val,
            "test_loss": candidate_test,
        },
        "eligible_for_match_evaluation": eligible,
        "ncs": {
            "best_cost": result.best_cost,
            "num_evaluations": result.num_evaluations,
            "num_iterations": result.num_iterations,
            "stop_reason": result.stop_reason,
            "elapsed_s": elapsed,
        },
        "run_dir": run_dir,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plan 12 强对手走法偏好目标 NCS-C 优化")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="显式指定的 dataset_id")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="偏好训练 RNG seed")
    parser.add_argument("--training-dir", default=None)
    parser.add_argument("--dry-run", action="store_true", help="只运行不写输出")
    args = parser.parse_args(argv)

    result = run_preference_training(
        args.dataset_id, args.seed, training_dir=args.training_dir, write_outputs=not args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

