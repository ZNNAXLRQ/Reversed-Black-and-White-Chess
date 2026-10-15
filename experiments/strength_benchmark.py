"""strength_benchmark.py：可复现棋力评测基准（只读，不修改正式文件）。

用途：在搜索/评估改动前后，用固定对手或自对弈 H2H 量化棋力变化。

- challenger/opponent 取值：submission | baseline | greedy | <逗号分隔权重>。
- 两方各用独立搜索深度（--challenger-depth / --opponent-depth）。
- 黑/白对半分（challenger 黑 ceil(n/2) 局，challenger 白 floor(n/2) 局），
  以 challenger 视角计算 u=(opp_count-own_count)/64，cost=(1-mean_u)/2。
- 结果写入 --out 指定的 JSON 文件，同时打印到 stdout。
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from experiments.match_runner import GreedyAgent, MatchRunner, WeightedAgent
from reversi.agent import BASELINE_WEIGHTS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SUBMISSION_PATH = _PROJECT_ROOT / "submission" / "agent.py"


def load_submission_weights():
    spec = importlib.util.spec_from_file_location("_bench_submission", _SUBMISSION_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("FINAL_WEIGHTS", "_FIXED_WEIGHTS"):
        if hasattr(mod, name):
            return np.asarray(getattr(mod, name), dtype=np.float64).copy()
    raise RuntimeError("no FINAL_WEIGHTS / _FIXED_WEIGHTS found in submission/agent.py")


def parse_weights(spec: str) -> np.ndarray:
    if spec == "submission":
        return load_submission_weights()
    if spec == "baseline":
        return np.asarray(BASELINE_WEIGHTS, dtype=np.float64).copy()
    return np.array([float(x) for x in spec.split(",")], dtype=np.float64)


def make_agent(spec: str, depth: int):
    if spec == "greedy":
        return GreedyAgent()
    return WeightedAgent(parse_weights(spec), depth)


def run_h2h(chall_agent, opp_agent, games: int, seed: int, runner: MatchRunner):
    u_values = []
    records = []
    black_games = (games + 1) // 2
    white_games = games // 2
    for i in range(black_games):
        r = runner.play_game(chall_agent, opp_agent, seed=seed)
        seed += 1
        u = (r.white_count - r.black_count) / 64.0  # challenger 黑视角
        u_values.append(u)
        records.append({"side": "black", "u": u, "black_count": r.black_count,
                        "white_count": r.white_count, "steps": r.steps})
    for i in range(white_games):
        r = runner.play_game(opp_agent, chall_agent, seed=seed)
        seed += 1
        u = (r.black_count - r.white_count) / 64.0  # challenger 白视角
        u_values.append(u)
        records.append({"side": "white", "u": u, "black_count": r.black_count,
                        "white_count": r.white_count, "steps": r.steps})
    u = np.asarray(u_values, dtype=np.float64)
    wins = int(np.sum(u > 0))
    draws = int(np.sum(u == 0))
    losses = int(np.sum(u < 0))
    mean_u = float(np.mean(u))
    cost = (1.0 - mean_u) / 2.0
    n = len(u)
    std = float(np.std(u, ddof=1)) if n > 1 else 0.0
    ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return {
        "games": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "mean_u": mean_u,
        "cost": cost,
        "std_u": std,
        "ci95_half": float(ci95),
        "u_values": [float(x) for x in u_values],
        "records": records,
    }


def main():
    ap = argparse.ArgumentParser(description="可复现棋力评测基准")
    ap.add_argument("--challenger", default="submission")
    ap.add_argument("--opponent", default="baseline")
    ap.add_argument("--challenger-depth", type=int, default=4)
    ap.add_argument("--opponent-depth", type=int, default=4)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    runner = MatchRunner(depth=4)
    chall_agent = make_agent(args.challenger, args.challenger_depth)
    opp_agent = make_agent(args.opponent, args.opponent_depth)

    result = run_h2h(chall_agent, opp_agent, args.games, args.seed, runner)
    result.update({
        "challenger": args.challenger,
        "opponent": args.opponent,
        "challenger_depth": args.challenger_depth,
        "opponent_depth": args.opponent_depth,
        "seed": args.seed,
    })
    text = json.dumps(result, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
