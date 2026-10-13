"""Plan 9：重复 NCS 训练、候选晋级与提交代码更新（单周期执行入口）。

严格按 plans/Plan_9_Task_9.md 完成一个训练周期：

1. 建立可追溯运行记录（run_id 目录 + config + 源码摘要 + 提交版本备份）；
2. 用固定扩展预算运行 NCS-C（N=10, T_max=20, sigma0=0.1, r=0.99, epoch=10），
   Objective 始终只使用固定 baseline AI 与反转贪心 AI（种子 101/202），每候选 6 局；
3. 用当周期独立验证种子（601/602）对候选 BestFound 与当前 incumbent 做同协议 6 局验证；
4. 仅当 challenger_validation_cost < incumbent_validation_cost 时执行 H2H（种子 701/702）；
5. 仅当 H2H mean_u > 0 才把候选写入 submission/agent.py 的最终权重标记块；
6. 保存本周期全部产物，绝不覆盖既有运行目录。

本模块复用已通过验收的 NCSOptimizer / ParameterSpace / ReversiObjective / MatchRunner，
只负责编排，不重写算法、特征、Objective 或晋级规则。
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import time
from typing import Dict, Optional

import numpy as np

from optimization.config import OptimizationConfig
from optimization.ncs.optimizer import NCSConfig, NCSOptimizer
from optimization.parameter_space import ParameterSpace
from reversi.agent import BASELINE_WEIGHTS

from .match_runner import MatchRunner
from .reversi_objective import ReversiObjective

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "artifacts")
_TRAINING_RUNS_DIR = os.path.join(_ARTIFACTS_DIR, "training_runs")
_SUBMISSION_VERSIONS_DIR = os.path.join(_ARTIFACTS_DIR, "submission_versions")
_SUBMISSION_PATH = os.path.join(_PROJECT_ROOT, "submission", "agent.py")

FEATURE_NAMES = ["disc_diff", "position_diff", "mobility_diff", "corner_diff", "frontier_diff"]
WEIGHT_LOWER = -10.0
WEIGHT_UPPER = 10.0

# NCS 参数基线（Plan 9 固定扩展预算）
POPULATION_SIZE = 10
T_MAX = 20
SIGMA0 = 0.1
R = 0.99
EPOCH = 10

# 搜索深度与对局种子（自本任务起统一为 4，与正式 Agent _SEARCH_DEPTH 一致）
DEPTH = 4
TRAINING_GREEDY_SEEDS = (101, 202)

BASELINE_LIST = [float(x) for x in BASELINE_WEIGHTS]

# 用于记录源码版本标识的文件（相对项目根目录）
_SOURCE_FILES = [
    "submission/agent.py",
    "reversi/agent.py",
    "reversi/evaluator.py",
    "reversi/rules.py",
    "reversi/search.py",
    "experiments/match_runner.py",
    "experiments/reversi_objective.py",
    "optimization/ncs/optimizer.py",
]


class Logger:
    """同时写 stdout 与 run.log，逐行 flush，便于长任务监控。"""

    def __init__(self, path: str):
        self.f = open(path, "w", encoding="utf-8")
        self._t0 = time.time()

    def log(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')} +{time.time() - self._t0:6.1f}s] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


class LoggingObjective:
    """包装真实 Objective，逐候选打印进度，不改变 cost 语义。"""

    def __init__(self, objective, total: int, logger: Logger):
        self.objective = objective
        self.total = total
        self.logger = logger
        self.count = 0

    def __call__(self, candidate):
        result = self.objective(candidate)
        self.count += 1
        self.logger.log(f"[ncs] eval {self.count}/{self.total} cost={result.cost:.9f}")
        return result


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def decode(z) -> np.ndarray:
    """归一化 z ∈ [0,1]^5 -> 权重 w = -10 + 20*z ∈ [-10,10]^5。"""
    return WEIGHT_LOWER + (WEIGHT_UPPER - WEIGHT_LOWER) * np.asarray(z, dtype=np.float64)


def load_submission_weights(path: str) -> np.ndarray:
    """读取当前提交版内嵌权重常量（不依赖项目模块，隔离加载）。"""
    spec = importlib.util.spec_from_file_location("submission_agent_snapshot", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("FINAL_WEIGHTS", "_FIXED_WEIGHTS"):
        if hasattr(mod, name):
            return np.asarray(getattr(mod, name), dtype=np.float64).copy()
    raise RuntimeError(f"no FINAL_WEIGHTS / _FIXED_WEIGHTS found in {path}")


def next_version_number() -> int:
    """根据 submission_versions/ 中已有备份确定下一个版本号（从 1 开始）。"""
    os.makedirs(_SUBMISSION_VERSIONS_DIR, exist_ok=True)
    existing = [f for f in os.listdir(_SUBMISSION_VERSIONS_DIR) if f.startswith("agent_v") and f.endswith(".py")]
    return len(existing) + 1


def backup_submission(logger: Logger, version: int) -> Dict:
    """备份当前 submission/agent.py 到 artifacts/submission_versions/，返回元数据。"""
    os.makedirs(_SUBMISSION_VERSIONS_DIR, exist_ok=True)
    digest = sha256_file(_SUBMISSION_PATH)
    short = digest[:12]
    py_path = os.path.join(_SUBMISSION_VERSIONS_DIR, f"agent_v{version}_{short}.py")
    meta_path = os.path.join(_SUBMISSION_VERSIONS_DIR, f"agent_v{version}_{short}.json")
    with open(_SUBMISSION_PATH, "rb") as src:
        data = src.read()
    with open(py_path, "wb") as dst:
        dst.write(data)
    meta = {
        "version": version,
        "sha256": digest,
        "source": _SUBMISSION_PATH,
        "backup_py": py_path,
        "weights": [float(x) for x in load_submission_weights(_SUBMISSION_PATH)],
        "backup_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Plan 9 周期开始前的当前提交版本备份",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.log(f"[backup] submission/agent.py -> {os.path.basename(py_path)} (sha256={short}...)")
    return meta


def update_submission_weights(new_weights, logger: Logger) -> Dict:
    """受保护地把 winner 权重写回 submission/agent.py 的标记块内。

    约束（严格按 Plan 9 第 5.7 节）：
    - BEGIN/END 标记各恰好出现一次且顺序正确，否则报错停止；
    - 权重长度 5、全部有限且位于 [-10,10]；
    - 先备份旧文件并记录 SHA-256；
    - 只替换标记块内的权重赋值（临时文件 + 原子替换）；
    - 写回后验证标记仍唯一、文件语法可编译、标记外内容未变、实际权重对应；
    - 任一检查失败，自动恢复备份并抛错。
    """
    begin = "# BEGIN AUTO-UPDATED FINAL WEIGHTS"
    end = "# END AUTO-UPDATED FINAL WEIGHTS"

    w = np.asarray(new_weights, dtype=np.float64)
    if w.shape != (5,):
        raise RuntimeError(f"weights must have shape (5,), got {w.shape}")
    if not np.all(np.isfinite(w)):
        raise RuntimeError("weights must be finite")
    if np.any(w < WEIGHT_LOWER) or np.any(w > WEIGHT_UPPER):
        raise RuntimeError(f"weights must be within [{WEIGHT_LOWER}, {WEIGHT_UPPER}]")

    with open(_SUBMISSION_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    if content.count(begin) != 1 or content.count(end) != 1:
        raise RuntimeError(
            f"marker block must appear exactly once each "
            f"(BEGIN={content.count(begin)}, END={content.count(end)})"
        )
    b_idx = content.index(begin)
    e_idx = content.index(end)
    if b_idx >= e_idx:
        raise RuntimeError("marker order invalid: BEGIN must precede END")

    old_sha = sha256_file(_SUBMISSION_PATH)
    os.makedirs(_SUBMISSION_VERSIONS_DIR, exist_ok=True)
    backup_path = os.path.join(
        _SUBMISSION_VERSIONS_DIR, f"agent_pre_writeback_{old_sha[:12]}.py"
    )
    with open(_SUBMISSION_PATH, "rb") as src, open(backup_path, "wb") as dst:
        dst.write(src.read())

    end_idx = e_idx + len(end)
    before = content[:b_idx]
    after = content[end_idx:]
    weights_repr = ", ".join(repr(float(x)) for x in w)
    new_block = (
        begin + "\n"
        "_FIXED_WEIGHTS = np.array(\n"
        "    [" + weights_repr + "],\n"
        "    dtype=np.float64,\n"
        ")\n"
        + end
    )
    new_content = before + new_block + after

    try:
        ast.parse(new_content)
    except SyntaxError as exc:
        raise RuntimeError(f"new submission content does not compile: {exc}") from exc

    tmp_path = _SUBMISSION_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    try:
        os.replace(tmp_path, _SUBMISSION_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    try:
        with open(_SUBMISSION_PATH, "r", encoding="utf-8") as f:
            written = f.read()
        if written.count(begin) != 1 or written.count(end) != 1:
            raise RuntimeError("marker block not unique after writeback")
        wb = written.index(begin)
        we = written.index(end) + len(end)
        if written[:wb] != before or written[we:] != after:
            raise RuntimeError("content outside marker block changed")
        loaded = load_submission_weights(_SUBMISSION_PATH)
        if not np.allclose(loaded, w):
            raise RuntimeError(f"written weights {loaded.tolist()} != expected {w.tolist()}")
    except Exception:
        with open(backup_path, "rb") as src, open(_SUBMISSION_PATH, "wb") as dst:
            dst.write(src.read())
        raise

    new_sha = sha256_file(_SUBMISSION_PATH)
    logger.log(f"[writeback] submission weights updated: {old_sha[:12]} -> {new_sha[:12]}")
    return {"updated": True, "old_sha256": old_sha, "new_sha256": new_sha, "backup_path": backup_path}


def validate_weights(weights, runner: MatchRunner, greedy_seeds) -> Dict:
    """6 局独立验证（固定基准 AI 两色各 1 局 + 反转贪心两种子各两色各 1 局）。"""
    weights = np.asarray(weights, dtype=np.float64)
    summary = runner.evaluate_candidate(weights, BASELINE_WEIGHTS, greedy_seeds=greedy_seeds)
    cost = (1.0 - summary.mean_u) / 2.0
    return {
        "weights": weights.tolist(),
        "wins": summary.wins,
        "draws": summary.draws,
        "losses": summary.losses,
        "mean_u": summary.mean_u,
        "cost": cost,
        "u_values": list(summary.u_values),
        "games": [
            {
                "black_count": g.black_count,
                "white_count": g.white_count,
                "steps": g.steps,
                "seed": g.seed,
            }
            for g in summary.games
        ],
    }


def run_plan9_cycle(
    cycle_index: int,
    training_seed: int,
    run_id: str,
    logger: Logger,
) -> Dict:
    """执行一个完整 Plan 9 周期，返回 promotion 报告。"""
    validation_seed_1 = 601 + 2 * (cycle_index - 1)
    validation_seed_2 = 602 + 2 * (cycle_index - 1)
    validation_seeds = (validation_seed_1, validation_seed_2)
    h2h_seed_1 = 701 + 2 * (cycle_index - 1)
    h2h_seed_2 = 702 + 2 * (cycle_index - 1)
    h2h_seeds = (h2h_seed_1, h2h_seed_2)

    run_dir = os.path.join(_TRAINING_RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Step 1：可追溯记录。
    incumbent_weights = load_submission_weights(_SUBMISSION_PATH)
    version = next_version_number()
    backup_meta = backup_submission(logger, version)
    source_hashes = {p: sha256_file(os.path.join(_PROJECT_ROOT, p)) for p in _SOURCE_FILES}

    config = {
        "cycle_index": cycle_index,
        "run_id": run_id,
        "training_seed": training_seed,
        "version_number": version,
        "incumbent_weights": incumbent_weights.tolist(),
        "baseline_weights": BASELINE_LIST,
        "feature_names": FEATURE_NAMES,
        "weight_bounds": [WEIGHT_LOWER, WEIGHT_UPPER],
        "decode": "w = -10 + 20*z, z in [0,1]^5",
        "objective_version": "cost=(1-mean_u)/2; u_g=(opp_count-own_count)/64 (candidate perspective)",
        "ncs_params": {
            "population_size": POPULATION_SIZE,
            "T_max": T_MAX,
            "sigma0": SIGMA0,
            "r": R,
            "epoch": EPOCH,
        },
        "candidate_evaluations": POPULATION_SIZE * (T_MAX + 1),
        "objective_games": POPULATION_SIZE * (T_MAX + 1) * 6,
        "training_greedy_seeds": list(TRAINING_GREEDY_SEEDS),
        "validation_greedy_seeds": list(validation_seeds),
        "h2h_seeds": list(h2h_seeds),
        "search_depth": DEPTH,
        "source_hashes": source_hashes,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backup": backup_meta,
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.log(
        f"[config] cycle_index={cycle_index} run_id={run_id} training_seed={training_seed} "
        f"validation_seeds={validation_seeds} h2h_seeds={h2h_seeds}"
    )
    logger.log(f"[config] incumbent_weights={np.round(incumbent_weights, 6).tolist()}")
    logger.log(f"[config] baseline_weights={BASELINE_LIST}")

    # Step 2：NCS 训练。
    space = ParameterSpace([0.0] * 5, [1.0] * 5)
    objective = ReversiObjective(
        BASELINE_WEIGHTS, depth=DEPTH, greedy_seeds=TRAINING_GREEDY_SEEDS
    )
    total_evals = POPULATION_SIZE * (T_MAX + 1)
    logging_objective = LoggingObjective(objective, total_evals, logger)
    optimizer = NCSOptimizer(
        NCSConfig(
            T_max=T_MAX,
            population_size=POPULATION_SIZE,
            sigma0=SIGMA0,
            r=R,
            epoch=EPOCH,
        )
    )
    opt_config = OptimizationConfig(seed=training_seed)

    logger.log(f"[train] start NCS-C N={POPULATION_SIZE} T_max={T_MAX} seed={training_seed} ...")
    t0 = time.time()
    result = optimizer.optimize(logging_objective, space, opt_config)
    train_elapsed = time.time() - t0

    best_z = np.asarray(result.best_candidate, dtype=np.float64)
    challenger_weights = decode(best_z)
    initial_z = np.asarray(result.metrics.get("initial_best_z", best_z), dtype=np.float64)

    training_summary = {
        "seed": training_seed,
        "best_weights": challenger_weights.tolist(),
        "best_z": best_z.tolist(),
        "initial_weights": decode(initial_z).tolist(),
        "best_cost": result.best_cost,
        "num_evaluations": result.num_evaluations,
        "num_iterations": result.num_iterations,
        "stop_reason": result.stop_reason,
        "objective_games": result.num_evaluations * 6,
        "elapsed_s": train_elapsed,
        "history": list(result.history),
        "sigma_mean_history": list(result.metrics.get("sigma_mean_history", [])),
        "acceptance_rate_history": list(result.metrics.get("acceptance_rate_history", [])),
    }
    with open(os.path.join(run_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2, ensure_ascii=False)
    logger.log(
        f"[train] done: best_cost={result.best_cost:.9f} "
        f"evals={result.num_evaluations} games={result.num_evaluations * 6} "
        f"elapsed={train_elapsed:.1f}s"
    )
    logger.log(f"[train] BestFound weights={np.round(challenger_weights, 9).tolist()}")

    # Step 3：候选与 incumbent 的同协议独立验证。
    logger.log("[validation] challenger vs incumbent (same protocol) ...")
    runner = MatchRunner(depth=DEPTH)
    candidate_validation = validate_weights(challenger_weights, runner, validation_seeds)
    incumbent_validation = validate_weights(incumbent_weights, runner, validation_seeds)
    with open(os.path.join(run_dir, "candidate_validation.json"), "w", encoding="utf-8") as f:
        json.dump(candidate_validation, f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "incumbent_validation.json"), "w", encoding="utf-8") as f:
        json.dump(incumbent_validation, f, indent=2, ensure_ascii=False)
    logger.log(
        f"[validation] challenger cost={candidate_validation['cost']:.9f} "
        f"mean_u={candidate_validation['mean_u']:.6f} "
        f"W/D/L={candidate_validation['wins']}/{candidate_validation['draws']}/{candidate_validation['losses']}"
    )
    logger.log(
        f"[validation] incumbent  cost={incumbent_validation['cost']:.9f} "
        f"mean_u={incumbent_validation['mean_u']:.6f} "
        f"W/D/L={incumbent_validation['wins']}/{incumbent_validation['draws']}/{incumbent_validation['losses']}"
    )

    challenger_cost = candidate_validation["cost"]
    incumbent_cost = incumbent_validation["cost"]
    do_h2h = challenger_cost < incumbent_cost

    # Step 4：H2H（仅当验证门槛通过）。
    h2h: Optional[Dict] = None
    promoted = False
    res = None
    if do_h2h:
        logger.log(
            f"[h2h] validation gate passed ({challenger_cost:.9f} < {incumbent_cost:.9f}), "
            f"running H2H seeds={h2h_seeds} ..."
        )
        res = runner.head_to_head(challenger_weights, incumbent_weights, seeds=h2h_seeds)
        h2h = {
            "u_values": list(res.u_values),
            "mean_u": res.mean_u,
            "challenger_wins": res.challenger_wins,
            "games": [
                {
                    "black_count": g.black_count,
                    "white_count": g.white_count,
                    "steps": g.steps,
                    "seed": g.seed,
                }
                for g in res.games
            ],
        }
        with open(os.path.join(run_dir, "h2h.json"), "w", encoding="utf-8") as f:
            json.dump(h2h, f, indent=2, ensure_ascii=False)
        logger.log(
            f"[h2h] u_values={h2h['u_values']} mean_u={res.mean_u:.6f} "
            f"challenger_wins={res.challenger_wins}"
        )
        promoted = res.mean_u > 0.0
    else:
        logger.log(
            f"[h2h] skipped: challenger validation cost ({challenger_cost:.9f}) "
            f">= incumbent ({incumbent_cost:.9f})"
        )

    # Step 5：晋级判断与（若晋级）更新提交权重。
    update_applied = False
    old_version = version
    new_version = version
    writeback_info = None
    if do_h2h and promoted:
        logger.log(f"[promote] H2H mean_u={res.mean_u:.6f} > 0 -> candidate qualified for promotion")
        try:
            writeback_info = update_submission_weights(challenger_weights, logger)
            update_applied = bool(writeback_info["updated"])
            new_version = version + 1
        except Exception as exc:
            logger.log(f"[promote] writeback FAILED: {exc}")
    elif do_h2h:
        logger.log(f"[promote] H2H mean_u={res.mean_u:.6f} <= 0 -> keep incumbent (no update)")
    else:
        logger.log("[promote] validation gate not passed -> keep incumbent (no update)")

    if do_h2h and promoted and update_applied:
        reason = None
    elif do_h2h and promoted:
        reason = "promotion qualified but submission writeback failed"
    elif do_h2h:
        reason = "H2H mean_u <= 0 (no promotion)"
    else:
        reason = "challenger validation cost >= incumbent validation cost"

    promotion = {
        "promoted": promoted,
        "update_applied": update_applied,
        "old_version": old_version,
        "new_version": new_version,
        "run_id": run_id,
        "challenger": {
            "weights": challenger_weights.tolist(),
            "validation_cost": challenger_cost,
            "validation_mean_u": candidate_validation["mean_u"],
        },
        "incumbent": {
            "weights": incumbent_weights.tolist(),
            "validation_cost": incumbent_cost,
            "validation_mean_u": incumbent_validation["mean_u"],
        },
        "gate_validation": {
            "passed": do_h2h,
            "evidence": (
                f"challenger_cost={challenger_cost:.9f} < incumbent_cost={incumbent_cost:.9f}"
                if do_h2h
                else f"challenger_cost={challenger_cost:.9f} >= incumbent_cost={incumbent_cost:.9f}"
            ),
        },
        "gate_h2h": {
            "triggered": do_h2h,
            "mean_u": h2h["mean_u"] if h2h else None,
            "passed": promoted,
        },
        "submission_updated": update_applied,
        "reason_not_updated": reason,
        "writeback": writeback_info,
    }
    with open(os.path.join(run_dir, "promotion.json"), "w", encoding="utf-8") as f:
        json.dump(promotion, f, indent=2, ensure_ascii=False)

    logger.log(f"[done] promotion={promoted} submission_updated={update_applied}")
    logger.log(f"[done] artifacts in {run_dir}")
    return promotion


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan 9 单周期执行入口")
    parser.add_argument("--cycle-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    cycle_index = args.cycle_index
    training_seed = args.seed
    run_id = args.run_id or f"ncs_cycle_{cycle_index:03d}_seed_{training_seed}"

    os.makedirs(_TRAINING_RUNS_DIR, exist_ok=True)
    run_dir = os.path.join(_TRAINING_RUNS_DIR, run_id)
    if os.path.exists(run_dir):
        raise RuntimeError(f"run_id directory already exists: {run_dir} (禁止覆盖)")
    os.makedirs(run_dir, exist_ok=True)
    logger = Logger(os.path.join(run_dir, "run.log"))
    try:
        promotion = run_plan9_cycle(cycle_index, training_seed, run_id, logger)
        logger.log("[summary] " + json.dumps(promotion, ensure_ascii=False))
    finally:
        logger.close()


if __name__ == "__main__":
    main()
