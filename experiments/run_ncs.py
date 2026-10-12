"""离线 NCS 权重优化入口（不进入正式 Agent 的 go() 调用路径）。

用法：
    python -m experiments.run_ncs --mode smoke   # 端到端烟雾测试 N=3, T_max=1（36 局）
    python -m experiments.run_ncs --mode full    # 小预算实验 N=10, T_max=3（240 局）+ 训练后 H2H（2 局）
"""
import argparse
import json
import os
import time

import numpy as np

from optimization.config import OptimizationConfig
from optimization.ncs.optimizer import NCSConfig, NCSOptimizer
from optimization.parameter_space import ParameterSpace
from reversi.agent import BASELINE_WEIGHTS

from .match_runner import MatchRunner
from .reversi_objective import ReversiObjective

# 特征顺序（固定，与评估器一致）
FEATURE_NAMES = ["disc_diff", "position_diff", "mobility_diff", "corner_diff", "frontier_diff"]

# NCS 参数基线（项目唯一基线）
R = 0.99
EPOCH = 10
SIGMA0 = 0.1

# 实验配置：population_size / T_max
SMOKE = {"population_size": 3, "T_max": 1}
FULL = {"population_size": 10, "T_max": 3}

BASELINE_WEIGHTS_LIST = [float(x) for x in BASELINE_WEIGHTS]


def decode(z):
    """归一化 z ∈ [0,1]^5 -> 权重 w = -10 + 20*z ∈ [-10,10]^5。"""
    return -10.0 + 20.0 * np.asarray(z, dtype=np.float64)


def run_experiment(name, population_size, T_max, seed, out_dir):
    """运行一次 NCS 权重优化，保存结果并返回摘要字典。"""
    space = ParameterSpace([0.0] * 5, [1.0] * 5)
    objective = ReversiObjective(BASELINE_WEIGHTS, depth=3)
    optimizer = NCSOptimizer(
        NCSConfig(
            T_max=T_max, population_size=population_size,
            sigma0=SIGMA0, r=R, epoch=EPOCH,
        )
    )
    config = OptimizationConfig(seed=seed)

    t0 = time.time()
    result = optimizer.optimize(objective, space, config)
    elapsed = time.time() - t0

    best_z = np.asarray(result.best_candidate, dtype=np.float64)
    best_w = decode(best_z)
    initial_z = np.asarray(result.metrics.get("initial_best_z", best_z), dtype=np.float64)

    # 训练后 H2H 诊断（最佳权重 vs baseline，2 局，不晋级）
    h2h = MatchRunner(depth=3).head_to_head(best_w, BASELINE_WEIGHTS, seeds=(501, 502))

    summary = {
        "name": name,
        "seed": seed,
        "population_size": population_size,
        "T_max": T_max,
        "sigma0": SIGMA0,
        "r": R,
        "epoch": EPOCH,
        "feature_names": FEATURE_NAMES,
        "baseline_weights": BASELINE_WEIGHTS_LIST,
        "initial_weights": decode(initial_z).tolist(),
        "best_weights": best_w.tolist(),
        "best_z": best_z.tolist(),
        "best_cost": result.best_cost,
        "num_evaluations": result.num_evaluations,
        "num_iterations": result.num_iterations,
        "stop_reason": result.stop_reason,
        "objective_games": result.num_evaluations * 6,
        "h2h_games": 2,
        "h2h": {
            "u_values": h2h.u_values,
            "mean_u": h2h.mean_u,
            "challenger_wins": h2h.challenger_wins,
        },
        "elapsed_s": elapsed,
        "history": result.history,
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{name}_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    np.save(os.path.join(out_dir, f"{name}_best_weights.npy"), best_w)

    return summary


def main():
    parser = argparse.ArgumentParser(description="离线 NCS 权重优化")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="experiments/outputs")
    args = parser.parse_args()

    cfg = SMOKE if args.mode == "smoke" else FULL
    summary = run_experiment(args.mode, cfg["population_size"], cfg["T_max"], args.seed, args.out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

