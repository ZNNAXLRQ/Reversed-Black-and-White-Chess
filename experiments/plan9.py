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
from typing import Dict, List, Optional

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
_CANDIDATES_DIR = os.path.join(_PROJECT_ROOT, "training", "candidates")
_CANDIDATE_REGISTRY_PATH = os.path.join(_CANDIDATES_DIR, "candidate_registry.json")
_CHAMPION_POOL_PATH = os.path.join(_PROJECT_ROOT, "training", "champion_pool.json")

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
# 验证协议中固定池的反转贪心对手种子（方向二：固定 seed 999）。
POOL_GREEDY_SEED = 999

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

    # 保留 BOM 和所有原始换行；只有权重标记块允许变化。
    with open(_SUBMISSION_PATH, "rb") as f:
        original_bytes = f.read()
    content = original_bytes.decode("utf-8")

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
    if os.path.exists(backup_path):
        with open(backup_path, "rb") as src:
            if src.read() != original_bytes:
                raise RuntimeError(f"existing backup does not match submission: {backup_path}")
    else:
        with open(backup_path, "xb") as dst:
            dst.write(original_bytes)

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
    newline = "\r\n" if content[b_idx + len(begin):].startswith("\r\n") else "\n"
    new_block = new_block.replace("\n", newline)
    new_content = before + new_block + after
    new_bytes = new_content.encode("utf-8")

    try:
        # Python 的 bytes 源码编译会正确识别 UTF-8 BOM；不要剥离部署文件的 BOM。
        ast.parse(new_bytes)
        compile(new_bytes, _SUBMISSION_PATH, "exec")
    except SyntaxError as exc:
        raise RuntimeError(f"new submission content does not compile: {exc}") from exc

    tmp_path = _SUBMISSION_PATH + ".tmp"
    with open(tmp_path, "xb") as f:
        f.write(new_bytes)

    try:
        os.replace(tmp_path, _SUBMISSION_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    try:
        with open(_SUBMISSION_PATH, "rb") as f:
            written_bytes = f.read()
        written = written_bytes.decode("utf-8")
        if written.count(begin) != 1 or written.count(end) != 1:
            raise RuntimeError("marker block not unique after writeback")
        wb = written.index(begin)
        we = written.index(end) + len(end)
        if written[:wb] != before or written[we:] != after:
            raise RuntimeError("content outside marker block changed")
        if written_bytes != new_bytes:
            raise RuntimeError("written submission bytes differ from expected content")
        compile(written_bytes, _SUBMISSION_PATH, "exec")
        loaded = load_submission_weights(_SUBMISSION_PATH)
        if not np.array_equal(loaded, w):
            raise RuntimeError(f"written weights {loaded.tolist()} != expected {w.tolist()}")
    except Exception:
        with open(backup_path, "rb") as src, open(_SUBMISSION_PATH, "wb") as dst:
            dst.write(src.read())
        raise

    new_sha = sha256_file(_SUBMISSION_PATH)
    logger.log(f"[writeback] submission weights updated: {old_sha[:12]} -> {new_sha[:12]}")
    return {"updated": True, "old_sha256": old_sha, "new_sha256": new_sha, "backup_path": backup_path}


def validate_weights(weights, runner: MatchRunner, incumbent_weights,
                     opponent_weights_list, greedy_seed=POOL_GREEDY_SEED) -> Dict:
    """24 局验证协议（方向一）：8 局自对弈（vs 现任冠军）+ 16 局固定池（3 冠军 + 反转贪心）。

    mean_u = 24 局 u 平均；cost = (1 - mean_u)/2。
    self_play_mean_u = 8 局自对弈 u 平均（晋级门槛条件②用）。
    """
    weights = np.asarray(weights, dtype=np.float64)
    incumbent = np.asarray(incumbent_weights, dtype=np.float64)
    result = runner.validate_protocol(weights, incumbent, opponent_weights_list, greedy_seed=greedy_seed)
    u_arr = np.asarray(result.u_values, dtype=np.float64)
    self_play_arr = np.asarray(result.self_play_u_values, dtype=np.float64)
    mean_u = float(u_arr.mean())
    self_play_mean_u = float(self_play_arr.mean())
    cost = (1.0 - mean_u) / 2.0
    wins = int(np.count_nonzero(u_arr > 0))
    losses = int(np.count_nonzero(u_arr < 0))
    draws = int(np.count_nonzero(u_arr == 0))
    return {
        "weights": weights.tolist(),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "mean_u": mean_u,
        "cost": cost,
        "u_values": list(result.u_values),
        "self_play_mean_u": self_play_mean_u,
        "self_play_u_values": list(result.self_play_u_values),
        "games": [
            {
                "black_count": g.black_count,
                "white_count": g.white_count,
                "steps": g.steps,
                "seed": g.seed,
            }
            for g in result.games
        ],
    }


def load_champion_pool() -> List[Dict]:
    """读取冠军池，返回按 cycle_index 升序的条目列表。"""
    if not os.path.exists(_CHAMPION_POOL_PATH):
        return []
    try:
        with open(_CHAMPION_POOL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    champions = data.get("champions", [])
    return sorted(champions, key=lambda c: c.get("cycle_index", 0))


def select_fixed_opponents(champions: List[Dict], incumbent_weights, logger: Logger) -> List[list]:
    """从冠军池选择固定对手：不含现任冠军，按 validation_cost 升序取前 3。"""
    incumbent_arr = np.asarray(incumbent_weights, dtype=np.float64)
    non_current = []
    for c in champions:
        try:
            w = np.asarray(c.get("weights"), dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if w.shape == (5,) and not np.allclose(w, incumbent_arr):
            non_current.append(c)
    non_current.sort(key=lambda c: c.get("validation_cost", 1.0))
    top3 = non_current[:3]
    logger.log(
        f"[pool] fixed opponents (top3 by cost) = {[c.get('cycle_index') for c in top3]}"
    )
    return [c["weights"] for c in top3]


def append_champion_to_pool(champions: List[Dict], cycle_index: int, winner_weights,
                            winner_validation_cost: float, logger: Logger) -> None:
    """把新冠军追加进池（validation_cost = 其 24 局验证 cost）。"""
    entry = {
        "cycle_index": cycle_index,
        "weights": [float(x) for x in np.asarray(winner_weights, dtype=np.float64)],
        "validation_cost": float(winner_validation_cost),
        "promoted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    champions.append(entry)
    champions.sort(key=lambda c: c.get("cycle_index", 0))
    os.makedirs(os.path.dirname(_CHAMPION_POOL_PATH), exist_ok=True)
    with open(_CHAMPION_POOL_PATH, "w", encoding="utf-8") as f:
        json.dump({"champions": champions}, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.log(f"[pool] appended champion cycle_index={cycle_index}")


_TERMINAL_CANDIDATE_STATUSES = ("promoted", "evaluated", "rejected")


def _load_evaluated_candidate_ids() -> set:
    """从 candidate_registry.json 读取已被评估/终态的 candidate_id 集合。"""
    if not os.path.exists(_CANDIDATE_REGISTRY_PATH):
        return set()
    try:
        with open(_CANDIDATE_REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set()
    return {
        rec.get("candidate_id")
        for rec in data.get("candidates", [])
        if rec.get("evaluated_in_cycle") or rec.get("status") in _TERMINAL_CANDIDATE_STATUSES
    }


def select_preference_candidate(logger: Logger) -> Optional[Dict]:
    """按 Plan 9 第 4 节选择第一个符合条件的未评估偏好候选。

    条件：eligible_for_match_evaluation=true；特征顺序与当前代码一致；五个权重有限且
    都在 [-10,10]；存在 dataset_id/preference_run_id/validation_loss 可追溯；且未被评估。
    按 candidate_id 字典序升序取第一个；不合格候选跳过并记录原因。
    """
    os.makedirs(_CANDIDATES_DIR, exist_ok=True)
    evaluated = _load_evaluated_candidate_ids()
    files = sorted(
        f for f in os.listdir(_CANDIDATES_DIR)
        if f.startswith("preference_candidate_") and f.endswith(".json")
    )
    for fname in files:
        path = os.path.join(_CANDIDATES_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                cand = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.log(f"[pref] skip {fname}: read error {exc}")
            continue
        cid = cand.get("candidate_id", fname)
        if not cand.get("eligible_for_match_evaluation"):
            logger.log(f"[pref] skip {cid}: eligible_for_match_evaluation != true")
            continue
        if list(cand.get("feature_names", [])) != list(FEATURE_NAMES):
            logger.log(f"[pref] skip {cid}: feature_names mismatch")
            continue
        try:
            w = np.asarray(cand.get("weights"), dtype=np.float64)
        except (TypeError, ValueError):
            logger.log(f"[pref] skip {cid}: weights unreadable")
            continue
        if w.shape != (5,) or not np.all(np.isfinite(w)) or np.any(w < WEIGHT_LOWER) or np.any(w > WEIGHT_UPPER):
            logger.log(f"[pref] skip {cid}: weights invalid (shape/NaN/out-of-bounds)")
            continue
        if not cand.get("dataset_id") or not cand.get("preference_run_id") or "validation_loss" not in cand:
            logger.log(f"[pref] skip {cid}: missing traceability fields")
            continue
        if cid in evaluated or cand.get("status") in _TERMINAL_CANDIDATE_STATUSES:
            logger.log(f"[pref] skip {cid}: already evaluated / terminal status")
            continue
        logger.log(f"[pref] selected candidate_id={cid}")
        return cand
    return None


def record_preference_candidate_evaluated(candidate_id: str, outcome: str, logger: Logger) -> None:
    """在 candidate_registry.json 中追加该候选已评估标记（不篡改候选原始文件）。"""
    os.makedirs(_CANDIDATES_DIR, exist_ok=True)
    data: Dict = {"candidates": []}
    if os.path.exists(_CANDIDATE_REGISTRY_PATH):
        try:
            with open(_CANDIDATE_REGISTRY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = {"candidates": []}
    found = False
    for rec in data.get("candidates", []):
        if rec.get("candidate_id") == candidate_id:
            rec["evaluated_in_cycle"] = True
            rec["evaluation_outcome"] = outcome
            rec["evaluated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            found = True
            break
    if not found:
        data.setdefault("candidates", []).append({
            "candidate_id": candidate_id,
            "evaluated_in_cycle": True,
            "evaluation_outcome": outcome,
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    with open(_CANDIDATE_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.log(f"[pref] recorded evaluated candidate_id={candidate_id} outcome={outcome}")


def run_plan9_cycle(
    cycle_index: int,
    training_seed: int,
    run_id: str,
    logger: Logger,
    selfplay: bool = False,
) -> Dict:
    """执行一个完整 Plan 9 周期，返回 promotion 报告。

    selfplay=True 时，训练 objective 的固定陪练从 baseline 权重换成当前 incumbent
    冠军权重（self-play 自我迭代训练目标），验证/H2H/晋级规则不变。
    """
    run_dir = os.path.join(_TRAINING_RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Step 1：可追溯记录。
    incumbent_weights = load_submission_weights(_SUBMISSION_PATH)
    version = next_version_number()
    backup_meta = backup_submission(logger, version)
    source_hashes = {p: sha256_file(os.path.join(_PROJECT_ROOT, p)) for p in _SOURCE_FILES}

    # 冠军池：读取历史冠军，选择固定对手（不含现任冠军，按 validation_cost 升序取前 3）。
    champions = load_champion_pool()
    fixed_opponent_weights = select_fixed_opponents(champions, incumbent_weights, logger)

    # 训练 objective 配置（selfplay 4 冠军 + 2 贪心，match 2 baseline + 4 贪心）。
    if selfplay:
        training_greedy_seeds = (TRAINING_GREEDY_SEEDS[0],)
        baseline_weight = 2.0
        greedy_weight = 1.0
        games_per_eval = 4
    else:
        training_greedy_seeds = TRAINING_GREEDY_SEEDS
        baseline_weight = 1.0
        greedy_weight = 1.0
        games_per_eval = 6

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
        "objective_mode": "selfplay" if selfplay else "match",
        "training_opponent_weights": incumbent_weights.tolist() if selfplay else BASELINE_LIST,
        "ncs_params": {
            "population_size": POPULATION_SIZE,
            "T_max": T_MAX,
            "sigma0": SIGMA0,
            "r": R,
            "epoch": EPOCH,
        },
        "candidate_evaluations": POPULATION_SIZE * (T_MAX + 1),
        "objective_games": POPULATION_SIZE * (T_MAX + 1) * games_per_eval,
        "training_greedy_seeds": list(training_greedy_seeds),
        "baseline_weight": baseline_weight,
        "greedy_weight": greedy_weight,
        "validation_protocol": "24 games: 8 selfplay vs incumbent (4W+4B) + 16 fixed pool (3 champions + greedy, 2W+2B each)",
        "champion_pool": [
            {"cycle_index": c["cycle_index"], "validation_cost": c["validation_cost"]}
            for c in champions
        ],
        "fixed_opponent_weights": [list(w) for w in fixed_opponent_weights],
        "pool_greedy_seed": POOL_GREEDY_SEED,
        "search_depth": DEPTH,
        "source_hashes": source_hashes,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backup": backup_meta,
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.log(
        f"[config] cycle_index={cycle_index} run_id={run_id} training_seed={training_seed} "
        f"fixed_opponents={len(fixed_opponent_weights)} pool_greedy_seed={POOL_GREEDY_SEED}"
    )
    logger.log(f"[config] incumbent_weights={np.round(incumbent_weights, 6).tolist()}")
    logger.log(f"[config] baseline_weights={BASELINE_LIST}")
    if selfplay:
        logger.log(f"[config] selfplay opponent=incumbent weights={np.round(incumbent_weights, 6).tolist()}")

    # Step 2：NCS 训练。self-play 模式用当前冠军做固定陪练（4 冠军 + 2 贪心），
    # 否则用固定 baseline（2 baseline + 4 贪心）。
    space = ParameterSpace([0.0] * 5, [1.0] * 5)
    if selfplay:
        objective = ReversiObjective(
            incumbent_weights, depth=DEPTH, greedy_seeds=training_greedy_seeds,
            baseline_weight=baseline_weight, greedy_weight=greedy_weight,
        )
    else:
        objective = ReversiObjective(
            BASELINE_WEIGHTS, depth=DEPTH, greedy_seeds=training_greedy_seeds,
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
        "objective_games": result.num_evaluations * games_per_eval,
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

    # Step 3：构建 challenger 列表（总是 match_ncs_candidate；若存在合格偏好候选，另纳入）。
    preference_candidate = select_preference_candidate(logger)
    challengers = [{
        "kind": "match_ncs",
        "candidate_id": None,
        "weights": challenger_weights,
        "pref_meta": None,
    }]
    if preference_candidate is None:
        logger.log("[pref] preference_candidate_available=false")
    else:
        logger.log(
            f"[pref] preference_candidate_available=true "
            f"candidate_id={preference_candidate['candidate_id']}"
        )
        challengers.append({
            "kind": "preference",
            "candidate_id": preference_candidate["candidate_id"],
            "weights": np.asarray(preference_candidate["weights"], dtype=np.float64),
            "pref_meta": {
                "candidate_id": preference_candidate["candidate_id"],
                "dataset_id": preference_candidate.get("dataset_id"),
                "preference_run_id": preference_candidate.get("preference_run_id"),
                "validation_loss": preference_candidate.get("validation_loss"),
                "baseline_validation_loss": preference_candidate.get("baseline_validation_loss"),
                "train_loss": preference_candidate.get("train_loss"),
                "test_loss": preference_candidate.get("test_loss"),
            },
        })

    # Step 4：24 局验证协议（incumbent + 每个 challenger）。
    logger.log("[validation] 24-game protocol (incumbent + challengers) ...")
    runner = MatchRunner(depth=DEPTH)
    incumbent_validation = validate_weights(
        incumbent_weights, runner, incumbent_weights, fixed_opponent_weights, POOL_GREEDY_SEED
    )
    incumbent_cost = incumbent_validation["cost"]
    with open(os.path.join(run_dir, "incumbent_validation.json"), "w", encoding="utf-8") as f:
        json.dump(incumbent_validation, f, indent=2, ensure_ascii=False)
    logger.log(
        f"[validation] incumbent cost={incumbent_cost:.9f} "
        f"mean_u={incumbent_validation['mean_u']:.6f} "
        f"self_play_mean_u={incumbent_validation['self_play_mean_u']:.6f} "
        f"W/D/L={incumbent_validation['wins']}/{incumbent_validation['draws']}/{incumbent_validation['losses']}"
    )

    for ch in challengers:
        ch["validation"] = validate_weights(
            ch["weights"], runner, incumbent_weights, fixed_opponent_weights, POOL_GREEDY_SEED
        )
        ch["validation_cost"] = ch["validation"]["cost"]
        ch["self_play_mean_u"] = ch["validation"]["self_play_mean_u"]
        label = ch["candidate_id"] or "match_ncs_candidate"
        logger.log(
            f"[validation] {label} cost={ch['validation_cost']:.9f} "
            f"mean_u={ch['validation']['mean_u']:.6f} "
            f"self_play_mean_u={ch['self_play_mean_u']:.6f} "
            f"W/D/L={ch['validation']['wins']}/{ch['validation']['draws']}/{ch['validation']['losses']}"
        )
        # 晋级门槛：① 24 局 cost 严格低于 incumbent；② 对现任冠军 8 局 mean_u > 0。
        ch["qualified"] = (
            ch["validation_cost"] < incumbent_cost
            and ch["self_play_mean_u"] > 0.0
        )

    match_val = next(c for c in challengers if c["kind"] == "match_ncs")["validation"]
    with open(os.path.join(run_dir, "candidate_validation.json"), "w", encoding="utf-8") as f:
        json.dump(match_val, f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "match_ncs_candidate_validation.json"), "w", encoding="utf-8") as f:
        json.dump(match_val, f, indent=2, ensure_ascii=False)
    if preference_candidate is not None:
        pref = next(c for c in challengers if c["kind"] == "preference")
        pref_val = dict(pref["validation"])
        pref_val["candidate_id"] = pref["candidate_id"]
        pref_val["preference_meta"] = pref["pref_meta"]
        with open(os.path.join(run_dir, "preference_candidate_validation.json"), "w", encoding="utf-8") as f:
            json.dump(pref_val, f, indent=2, ensure_ascii=False)

    # Step 5：晋级门槛已在 Step 4 判定（24 局 cost + 8 局自对弈 mean_u）。
    for ch in challengers:
        label = ch["candidate_id"] or "match_ncs_candidate"
        if not ch["qualified"]:
            logger.log(
                f"[gate] {label} NOT qualified: "
                f"cost={ch['validation_cost']:.9f} (incumbent {incumbent_cost:.9f}), "
                f"self_play_mean_u={ch['self_play_mean_u']:.6f}"
            )
        else:
            logger.log(
                f"[gate] {label} qualified: "
                f"cost={ch['validation_cost']:.9f} < {incumbent_cost:.9f} AND "
                f"self_play_mean_u={ch['self_play_mean_u']:.6f} > 0"
            )

    # Step 6：当多个候选通过时的唯一选择规则（self_play_mean_u 即原 H2H mean_u）。
    qualified = [c for c in challengers if c["qualified"]]
    winner = None
    if len(qualified) == 1:
        winner = qualified[0]
    elif len(qualified) > 1:
        qualified.sort(key=lambda c: (
            -c["self_play_mean_u"],
            c["validation_cost"],
            0 if c["kind"] == "match_ncs" else 1,
        ))
        winner = qualified[0]

    # Step 7：更新 submission（仅 winner）。
    update_applied = False
    old_version = version
    new_version = version
    writeback_info = None
    reason = None
    if winner is not None:
        winner_label = winner["candidate_id"] or "match_ncs_candidate"
        logger.log(
            f"[promote] winner={winner_label} self_play_mean_u={winner['self_play_mean_u']:.6f} "
            f"validation_cost={winner['validation_cost']:.9f}"
        )
        try:
            writeback_info = update_submission_weights(winner["weights"], logger)
            update_applied = bool(writeback_info["updated"])
            new_version = version + 1
        except Exception as exc:
            logger.log(f"[promote] writeback FAILED: {exc}")
            reason = "promotion qualified but submission writeback failed"
    else:
        reason = "no challenger passed validation gate (24-game cost + self-play mean_u)"
        logger.log("[promote] no winner -> keep incumbent (no update)")
    if winner is not None and update_applied:
        reason = None
        # 把新冠军追加进冠军池（validation_cost = 其 24 局验证 cost）。
        append_champion_to_pool(
            champions, cycle_index, winner["weights"], winner["validation_cost"], logger
        )

    # Step 8：记录偏好候选消费状态（无论通过还是拒绝）。
    if preference_candidate is not None:
        pref = next(c for c in challengers if c["kind"] == "preference")
        outcome = (
            "promoted" if (winner is pref and update_applied)
            else ("qualified_not_promoted" if pref["qualified"] else "not_qualified")
        )
        record_preference_candidate_evaluated(pref["candidate_id"], outcome, logger)

    promotion = {
        "promoted": winner is not None and update_applied,
        "update_applied": update_applied,
        "old_version": old_version,
        "new_version": new_version,
        "run_id": run_id,
        "preference_candidate_available": preference_candidate is not None,
        "winner": (
            {
                "kind": winner["kind"],
                "candidate_id": winner["candidate_id"],
                "weights": winner["weights"].tolist(),
                "validation_cost": winner["validation_cost"],
                "self_play_mean_u": winner["self_play_mean_u"],
            }
            if winner is not None else None
        ),
        "challengers": [
            {
                "kind": c["kind"],
                "candidate_id": c["candidate_id"],
                "weights": c["weights"].tolist(),
                "validation_cost": c["validation_cost"],
                "validation_mean_u": c["validation"]["mean_u"],
                "self_play_mean_u": c["self_play_mean_u"],
                "qualified": c["qualified"],
            }
            for c in challengers
        ],
        "incumbent": {
            "weights": incumbent_weights.tolist(),
            "validation_cost": incumbent_cost,
            "validation_mean_u": incumbent_validation["mean_u"],
            "self_play_mean_u": incumbent_validation["self_play_mean_u"],
        },
        "submission_updated": update_applied,
        "reason_not_updated": reason,
        "writeback": writeback_info,
    }
    with open(os.path.join(run_dir, "promotion.json"), "w", encoding="utf-8") as f:
        json.dump(promotion, f, indent=2, ensure_ascii=False)

    logger.log(f"[done] promoted={promotion['promoted']} submission_updated={update_applied}")
    logger.log(f"[done] artifacts in {run_dir}")
    return promotion


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan 9 单周期执行入口")
    parser.add_argument("--cycle-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--selfplay", action="store_true", help="self-play 训练目标（陪练=现任冠军）")
    args = parser.parse_args()

    cycle_index = args.cycle_index
    training_seed = args.seed
    prefix = "selfplay_cycle" if args.selfplay else "ncs_cycle"
    run_id = args.run_id or f"{prefix}_{cycle_index:03d}_seed_{training_seed}"

    os.makedirs(_TRAINING_RUNS_DIR, exist_ok=True)
    run_dir = os.path.join(_TRAINING_RUNS_DIR, run_id)
    if os.path.exists(run_dir):
        raise RuntimeError(f"run_id directory already exists: {run_dir} (禁止覆盖)")
    os.makedirs(run_dir, exist_ok=True)
    logger = Logger(os.path.join(run_dir, "run.log"))
    try:
        promotion = run_plan9_cycle(cycle_index, training_seed, run_id, logger, selfplay=args.selfplay)
        logger.log("[summary] " + json.dumps(promotion, ensure_ascii=False))
    finally:
        logger.close()


if __name__ == "__main__":
    main()
