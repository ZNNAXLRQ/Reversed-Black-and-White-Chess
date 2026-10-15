"""ReversiObjective：固定训练目标（最小化 cost）。

候选在归一化空间 z ∈ [0,1]^5 中搜索，解码为权重 w = -10 + 20*z ∈ [-10,10]^5。
cost = (1 - mean(u_g)) / 2，越小越好；u_g = (opp_count - own_count) / 64，
以候选自身为视角。训练只用固定基准 AI 与反转贪心 AI，禁止动态加入冠军。
"""
import numpy as np

from optimization.objective import EvaluationResult

from .match_runner import MatchRunner

DECODE_SCALE = 20.0
DECODE_OFFSET = -10.0


class ReversiObjective:
    """候选权重（归一化 z ∈ [0,1]^5）的固定训练目标。"""

    def __init__(self, baseline_weights, depth=4, greedy_seeds=(101, 202),
                 baseline_weight=1.0, greedy_weight=1.0):
        self.baseline_weights = np.asarray(baseline_weights, dtype=np.float64)
        if self.baseline_weights.shape != (5,):
            raise ValueError(
                f"baseline_weights must have shape (5,), got {self.baseline_weights.shape}"
            )
        self.depth = depth
        self.greedy_seeds = tuple(greedy_seeds)
        self.baseline_weight = float(baseline_weight)
        self.greedy_weight = float(greedy_weight)
        self.runner = MatchRunner(depth=depth)
        self._cache = {}

    def decode(self, z) -> np.ndarray:
        """归一化 z ∈ [0,1]^5 -> 权重 w = -10 + 20*z ∈ [-10,10]^5。"""
        z = np.asarray(z, dtype=np.float64)
        if z.shape != (5,):
            raise ValueError(f"candidate must have shape (5,), got {z.shape}")
        if not np.all(np.isfinite(z)):
            raise ValueError("candidate must be finite")
        if np.any(z < 0.0) or np.any(z > 1.0):
            raise ValueError("candidate must be within [0, 1]^5")
        return DECODE_OFFSET + DECODE_SCALE * z

    def __call__(self, candidate) -> EvaluationResult:
        weights = self.decode(candidate)
        key = tuple(weights)
        if key in self._cache:
            return self._cache[key]

        summary = self.runner.evaluate_candidate(
            weights, self.baseline_weights, greedy_seeds=self.greedy_seeds,
            baseline_weight=self.baseline_weight, greedy_weight=self.greedy_weight,
        )
        cost = (1.0 - summary.mean_u) / 2.0
        metrics = {
            "wins": summary.wins,
            "draws": summary.draws,
            "losses": summary.losses,
            "mean_u": summary.mean_u,
            "weights": weights.copy(),
            "per_game": [
                {"seed": g.seed, "steps": g.steps, "elapsed": g.elapsed, "u": u}
                for g, u in zip(summary.games, summary.u_values)
            ],
        }
        result = EvaluationResult(cost=cost, metrics=metrics)
        self._cache[key] = result
        return result
