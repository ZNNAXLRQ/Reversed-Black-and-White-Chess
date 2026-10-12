"""Plan 7：对照实验、泛化验证与候选版本定稿。

流程（严格按 Plan 7 任务书）：
1. 复用种子 42 的 Plan 6 训练结果（experiments/outputs/full_result.json）；
2. 追加种子 43、44 两次 NCS 训练（N=10, T_max=3，每候选 6 局，共 240 局/次）；
3. 对 baseline 与 3 个训练最佳权重做 6 局独立验证（贪心种子 303/404）；
4. 选独立验证 cost 最低者为挑战者（并列按 42 < 43 < 44）；
5. 仅当挑战者验证 cost 严格低于 baseline 验证 cost 时才 H2H（种子 501/502）；
6. mean_u > 0 才晋级为新冠军，否则保留 baseline。

本模块是离线实验入口，不进入正式 Agent 的 go() 调用路径。
"""
import argparse
import json
import os
import time
from typing import Dict, Optional, Sequence

import numpy as np

from optimization.config import OptimizationConfig
from optimization.ncs.optimizer import NCSConfig, NCSOptimizer
from optimization.parameter_space import ParameterSpace
from reversi.agent import BASELINE_WEIGHTS

from .match_runner import MatchRunner
from .reversi_objective import ReversiObjective

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(_PROJECT_ROOT, "experiments", "outputs")
DEFAULT_SEED42_RESULT = os.path.join(DEFAULT_OUT_DIR, "full_result.json")

FEATURE_NAMES = ["disc_diff", "position_diff", "mobility_diff", "corner_diff", "frontier_diff"]

WEIGHT_LOWER = -10.0
WEIGHT_UPPER = 10.0

# NCS 参数基线（项目唯一基线）
POPULATION_SIZE = 10
T_MAX = 3
SIGMA0 = 0.1
R = 0.99
EPOCH = 10

# 搜索深度与对局种子
DEPTH = 3
TRAINING_GREEDY_SEEDS = (101, 202)
VALIDATION_GREEDY_SEEDS = (303, 404)
H2H_SEEDS = (501, 502)
SEEDS = (42, 43, 44)

BASELINE_LIST = [float(x) for x in BASELINE_WEIGHTS]


def decode(z) -> np.ndarray:
    """归一化 z ∈ [0,1]^5 -> 权重 w = -10 + 20*z ∈ [-10,10]^5。"""
    return WEIGHT_LOWER + (WEIGHT_UPPER - WEIGHT_LOWER) * np.asarray(z, dtype=np.float64)


def train_once(seed: int) -> Dict:
    """一次完整 NCS 训练（N=10, T_max=3），返回训练统计 + 最佳权重。"""
    space = ParameterSpace([0.0] * 5, [1.0] * 5)
    objective = ReversiObjective(
        BASELINE_WEIGHTS, depth=DEPTH, greedy_seeds=TRAINING_GREEDY_SEEDS
    )
    optimizer = NCSOptimizer(
        NCSConfig(
            T_max=T_MAX,
            population_size=POPULATION_SIZE,
            sigma0=SIGMA0,
            r=R,
            epoch=EPOCH,
        )
    )
    config = OptimizationConfig(seed=seed)

    t0 = time.time()
    result = optimizer.optimize(objective, space, config)
    elapsed = time.time() - t0

    best_z = np.asarray(result.best_candidate, dtype=np.float64)
    best_w = decode(best_z)
    initial_z = np.asarray(result.metrics.get("initial_best_z", best_z), dtype=np.float64)

    return {
        "seed": seed,
        "best_weights": best_w.tolist(),
        "best_z": best_z.tolist(),
        "initial_weights": decode(initial_z).tolist(),
        "best_cost": result.best_cost,
        "num_evaluations": result.num_evaluations,
        "num_iterations": result.num_iterations,
        "stop_reason": result.stop_reason,
        "objective_games": result.num_evaluations * 6,
        "elapsed_s": elapsed,
        "history": list(result.history),
        "sigma_mean_history": list(result.metrics.get("sigma_mean_history", [])),
        "acceptance_rate_history": list(result.metrics.get("acceptance_rate_history", [])),
    }



def validate_weights(weights, runner: MatchRunner) -> Dict:
    """6 局独立验证（贪心种子 303/404），返回胜平负、mean_u 与 cost。

    cost 复用 ReversiObjective 的固定定义：cost = (1 - mean_u) / 2。
    """
    weights = np.asarray(weights, dtype=np.float64)
    summary = runner.evaluate_candidate(
        weights, BASELINE_WEIGHTS, greedy_seeds=VALIDATION_GREEDY_SEEDS
    )
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


def select_challenger(validation_costs: Dict[int, float]) -> int:
    """从候选验证 cost 中选最低者；cost 完全相同时按种子 42 < 43 < 44 优先。"""
    if not validation_costs:
        raise ValueError("validation_costs must not be empty")
    return min(validation_costs, key=lambda s: (validation_costs[s], s))


def should_challenge(challenger_cost: float, baseline_cost: float) -> bool:
    """仅当挑战者验证 cost 严格低于 baseline 验证 cost 时才发起 H2H。"""
    return challenger_cost < baseline_cost


def decide_champion(h2h_mean_u: float) -> bool:
    """mean_u > 0 才判挑战者胜并晋级；mean_u <= 0 保留原冠军。"""
    return h2h_mean_u > 0.0


def load_seed42(seed42_result: str) -> Dict:
    """加载 Plan 6 已完成、保存到 full_result.json 的种子 42 训练结果。"""
    with open(seed42_result, "r", encoding="utf-8") as f:
        s = json.load(f)
    return {
        "seed": 42,
        "best_weights": s["best_weights"],
        "best_z": s.get("best_z"),
        "initial_weights": s.get("initial_weights"),
        "best_cost": s["best_cost"],
        "num_evaluations": s["num_evaluations"],
        "num_iterations": s["num_iterations"],
        "stop_reason": s["stop_reason"],
        "objective_games": s.get("objective_games", s["num_evaluations"] * 6),
        "elapsed_s": s.get("elapsed_s"),
        "history": s.get("history"),
    }



def run_plan7(
    out_dir: str = DEFAULT_OUT_DIR,
    seed42_result: str = DEFAULT_SEED42_RESULT,
    seeds: Sequence[int] = SEEDS,
) -> Dict:
    """执行 Plan 7 完整流程并保存结果，返回最终报告字典。"""
    os.makedirs(out_dir, exist_ok=True)
    runner = MatchRunner(depth=DEPTH)

    # 1. 组装三个种子的训练结果：42 复用 Plan 6，43/44 现跑（若已保存则复用）。
    training: Dict[int, Dict] = {}
    training[42] = load_seed42(seed42_result)

    for seed in seeds:
        if seed == 42:
            continue
        cache_path = os.path.join(out_dir, f"plan7_seed{seed}_result.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                training[seed] = json.load(f)
            print(f"[plan7] reuse training seed={seed} from {cache_path}")
        else:
            print(f"[plan7] training seed={seed} ...")
            training[seed] = train_once(seed)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(training[seed], f, indent=2, ensure_ascii=False)
            np.save(
                os.path.join(out_dir, f"plan7_seed{seed}_best_weights.npy"),
                np.asarray(training[seed]["best_weights"], dtype=np.float64),
            )
            print(
                f"[plan7] training seed={seed} done: best_cost={training[seed]['best_cost']:.6f} "
                f"in {training[seed]['elapsed_s']:.1f}s"
            )

    # 2. 独立验证：baseline + 三个候选。
    print("[plan7] validation ...")
    validation = {"baseline": validate_weights(BASELINE_WEIGHTS, runner)}
    for seed in seeds:
        validation[str(seed)] = validate_weights(training[seed]["best_weights"], runner)

    # 3. 选挑战者。
    candidate_costs = {seed: validation[str(seed)]["cost"] for seed in seeds}
    challenger_seed = select_challenger(candidate_costs)
    challenger_weights = np.asarray(training[challenger_seed]["best_weights"], dtype=np.float64)
    challenger_cost = candidate_costs[challenger_seed]
    baseline_cost = validation["baseline"]["cost"]

    # 4. 条件 H2H。
    do_h2h = should_challenge(challenger_cost, baseline_cost)
    h2h: Optional[Dict] = None
    promoted = False
    if do_h2h:
        print(f"[plan7] H2H challenger seed={challenger_seed} vs baseline ...")
        res = runner.head_to_head(challenger_weights, BASELINE_WEIGHTS, seeds=H2H_SEEDS)
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
        promoted = decide_champion(res.mean_u)

    # 5. 冠军定稿。
    if do_h2h and promoted:
        champion_weights = challenger_weights
        champion_source = f"ncs_seed_{challenger_seed}"
    else:
        champion_weights = BASELINE_WEIGHTS.copy()
        champion_source = "baseline"


    report = {
        "plan": 7,
        "feature_names": FEATURE_NAMES,
        "weight_bounds": [WEIGHT_LOWER, WEIGHT_UPPER],
        "objective_version": "cost=(1-mean_u)/2; u_g=(opp_count-own_count)/64 (candidate perspective)",
        "ncs_params": {
            "population_size": POPULATION_SIZE,
            "T_max": T_MAX,
            "sigma0": SIGMA0,
            "r": R,
            "epoch": EPOCH,
        },
        "baseline_weights": BASELINE_LIST,
        "training_opponents": {
            "baseline_depth": DEPTH,
            "greedy_seeds": list(TRAINING_GREEDY_SEEDS),
        },
        "validation_opponents": {
            "baseline_depth": DEPTH,
            "greedy_seeds": list(VALIDATION_GREEDY_SEEDS),
        },
        "h2h_seeds": list(H2H_SEEDS),
        "search_depth": DEPTH,
        "training": {str(s): training[s] for s in seeds},
        "validation": validation,
        "challenger": {
            "seed": challenger_seed,
            "weights": challenger_weights.tolist(),
            "validation_cost": challenger_cost,
            "baseline_validation_cost": baseline_cost,
            "h2h_triggered": do_h2h,
        },
        "h2h": h2h,
        "final_champion": {
            "source": champion_source,
            "weights": champion_weights.tolist(),
            "promoted": promoted,
        },
    }

    final_path = os.path.join(out_dir, "plan7_final.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    np.save(
        os.path.join(out_dir, "plan7_champion_weights.npy"),
        np.asarray(champion_weights, dtype=np.float64),
    )
    print(f"[plan7] saved final report to {final_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan 7 对照实验、泛化验证与候选版本定稿")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed42", default=DEFAULT_SEED42_RESULT)
    args = parser.parse_args()
    report = run_plan7(out_dir=args.out, seed42_result=args.seed42)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

