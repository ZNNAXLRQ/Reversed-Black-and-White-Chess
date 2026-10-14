"""Cycle 1 winner 在 depth=4 下的晋级资格重新验证。

历史事实：Cycle 1 的 210 候选评价、1260 局训练以及既有验证/H2H 均在 depth=3 下完成，
其结果保留为历史记录（State9-Task4），不得改写。本脚本在统一后的 depth=4 下，对
Cycle 1 winner 与原 incumbent 重新做成对验证与 H2H，作为本次是否晋级的唯一依据。

验证协议沿用 Cycle 1 原约定：
- 独立验证：与固定 baseline 黑白各一局 + 与 Reversed Greedy（种子 601/602）各黑白互换，共 6 局；
- H2H：winner 先黑后白，共 2 局，mean_u > 0 才通过；
- 晋级条件：winner 的 depth=4 validation cost 严格小于 incumbent 的 depth=4 validation cost，
  且 H2H mean_u > 0。
"""
import json
import os
import time

import numpy as np

from experiments.match_runner import MatchRunner
from reversi.agent import BASELINE_WEIGHTS

WINNER_WEIGHTS = np.array(
    [-9.79726314898437, -4.302298754340315, 1.685966608200344, -8.507084729322496, 5.230027020280524],
    dtype=np.float64,
)
INCUMBENT_WEIGHTS = np.array(
    [-3.0861110771421476, -10.0, -0.3857156832111528, -7.651743487832784, 0.48587137914320344],
    dtype=np.float64,
)

DEPTH = 4
VALIDATION_GREEDY_SEEDS = (601, 602)
H2H_SEEDS = (701, 702)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT_PATH = os.path.join(
    _PROJECT_ROOT,
    "artifacts",
    "training_runs",
    "ncs_cycle_001_seed_45",
    "depth4_winner_verification.json",
)


def validate(runner, weights):
    summary = runner.evaluate_candidate(weights, BASELINE_WEIGHTS, greedy_seeds=VALIDATION_GREEDY_SEEDS)
    cost = (1.0 - summary.mean_u) / 2.0
    return {
        "weights": weights.tolist(),
        "cost": cost,
        "mean_u": summary.mean_u,
        "wins": summary.wins,
        "draws": summary.draws,
        "losses": summary.losses,
        "u_values": list(summary.u_values),
        "games": [
            {"black_count": g.black_count, "white_count": g.white_count, "steps": g.steps, "seed": g.seed}
            for g in summary.games
        ],
    }


def main():
    runner = MatchRunner(depth=DEPTH)
    t0 = time.time()
    winner_val = validate(runner, WINNER_WEIGHTS)
    incumbent_val = validate(runner, INCUMBENT_WEIGHTS)
    validation_passed = winner_val["cost"] < incumbent_val["cost"]

    h2h = None
    h2h_passed = False
    if validation_passed:
        res = runner.head_to_head(WINNER_WEIGHTS, INCUMBENT_WEIGHTS, seeds=H2H_SEEDS)
        h2h = {
            "u_values": list(res.u_values),
            "mean_u": res.mean_u,
            "challenger_wins": res.challenger_wins,
            "games": [
                {"black_count": g.black_count, "white_count": g.white_count, "steps": g.steps, "seed": g.seed}
                for g in res.games
            ],
        }
        h2h_passed = res.mean_u > 0.0

    promoted = validation_passed and h2h_passed
    report = {
        "depth": DEPTH,
        "winner": winner_val,
        "incumbent": incumbent_val,
        "validation_passed": validation_passed,
        "h2h": h2h,
        "h2h_passed": h2h_passed,
        "promoted": promoted,
        "elapsed_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
