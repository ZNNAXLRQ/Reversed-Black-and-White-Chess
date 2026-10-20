"""NCS-C 优化器（论文 Tang/Yang/Yao 2016，第 IV 节）。

步长更新方向：采用论文式（6）与 Algorithm 1 的方向（p > 0.2 时 sigma /= r，
p < 0.2 时 sigma *= r，p == 0.2 不变）；论文正文文字解释与公式方向相反，
本项目已裁决按公式执行，不提供可切换方向参数。

边界处理为项目工程适配：候选在归一化空间高斯变异后越界坐标用 np.clip 投影，
Bhattacharyya 距离按投影后中心与原 sigma 对应的名义高斯分布计算。
"""
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..base import Optimizer
from ..config import OptimizationConfig
from ..parameter_space import ParameterSpace
from ..result import OptimizationResult
from .bhattacharyya import correlation
from .process import SearchProcess

EPS = 1e-12
SIGMA_MIN = 1e-8
SIGMA_MAX = 1.0


@dataclass
class NCSConfig:
    """NCS-C 专属配置。T_max 必须由调用方显式传入。"""

    T_max: int
    population_size: int = 10
    sigma0: float = 0.1
    r: float = 0.99
    epoch: int = 10
    initial_candidates: Optional[Sequence] = None

    def __post_init__(self):
        if self.T_max < 0:
            raise ValueError("T_max must be non-negative")
        if self.population_size <= 1:
            raise ValueError("population_size must be > 1 (N > 1 required)")
        if not (0 < self.sigma0 <= SIGMA_MAX):
            raise ValueError(f"sigma0 must be in (0, {SIGMA_MAX}]")
        if not (0 < self.r < 1):
            raise ValueError("r must be in (0, 1)")
        if self.epoch <= 0:
            raise ValueError("epoch must be positive")


def selection_decision(g_old, g_new, corr_old, corr_new, lambda_t, eps=EPS) -> bool:
    """Algorithm 1 的选择规则（Eq.(4) 方向），返回是否接受候选。

    质量越好 cost 越低、相关性越大越好；比较归一化候选目标份额 q_new 与
    候选相关性份额 c_new 的比值 ratio = q_new / c_new，当 ratio < lambda_t 接受。

    退化处理（EPS=1e-12）：
    - g_old + g_new <= EPS -> q_new = 0.5
    - corr_old + corr_new <= EPS -> c_new = 0.5
    - corr 总和 > EPS 但 c_new <= EPS -> 拒绝（ratio = +inf）
    """
    if not np.isfinite(g_old) or not np.isfinite(g_new):
        raise ValueError("g_old and g_new must be finite")
    if not np.isfinite(corr_old) or not np.isfinite(corr_new):
        raise ValueError("corr_old and corr_new must be finite")
    if not np.isfinite(lambda_t):
        raise ValueError("lambda_t must be finite")

    if g_old + g_new <= eps:
        q_new = 0.5
    else:
        q_new = g_new / (g_old + g_new)

    if corr_old + corr_new <= eps:
        c_new = 0.5
    else:
        c_new = corr_new / (corr_old + corr_new)

    if corr_old + corr_new > eps and c_new <= eps:
        return False

    ratio = q_new / c_new
    if not np.isfinite(ratio):
        raise ValueError("ratio is not finite")
    return ratio < lambda_t


class NCSOptimizer(Optimizer):
    """NCS-C 优化器：在连续参数空间上做负相关搜索。"""

    def __init__(self, ncs_config: NCSConfig):
        self.ncs_config = ncs_config

    @staticmethod
    def _sample_lambda(rng, t, T_max) -> float:
        """lambda_t ~ Normal(1.0, 0.1 * (1 - t/T_max))，标准差非方差；<=0 重采样。"""
        std = 0.1 * (1.0 - t / T_max)
        while True:
            val = rng.normal(1.0, std)
            if val > 0:
                return float(val)

    def optimize(
        self,
        objective,
        parameter_space: ParameterSpace,
        config: OptimizationConfig,
    ) -> OptimizationResult:
        nc = self.ncs_config
        rng = np.random.default_rng(config.seed)
        dim = parameter_space.dimension
        n = nc.population_size

        init_evals = n
        max_evals = config.max_evaluations
        if max_evals is not None and max_evals < init_evals:
            raise ValueError(
                f"max_evaluations={max_evals} insufficient for initialization ({init_evals})"
            )

        # 保留全部随机抽样的 RNG 次序，再替换显式种子槽位，便于同种子消融。
        initial = [parameter_space.sample(rng) for _ in range(n)]
        seeds = [] if nc.initial_candidates is None else list(nc.initial_candidates)
        if len(seeds) > n:
            raise ValueError("initial_candidates exceeds population_size")
        for i, candidate in enumerate(seeds):
            if not parameter_space.validate(candidate):
                raise ValueError("initial_candidates must be finite and within parameter bounds")
            initial[i] = np.asarray(candidate, dtype=np.float64).copy()

        # 所有初始中心均按本轮 objective 重新计分，不导入冠军的历史 cost。
        processes = []
        for x in initial:
            cost = float(objective(x).cost)
            processes.append(SearchProcess(x=x, cost=cost, sigma=nc.sigma0))
        num_evals = init_evals

        best_idx = int(np.argmin([p.cost for p in processes]))
        best_x = processes[best_idx].x.copy()
        best_cost = processes[best_idx].cost
        initial_best_z = best_x.copy()

        history = [best_cost]
        sigma_mean_history = []
        acceptance_rate_history = []

        actual_iterations = 0
        stop_reason = "T_max"

        for t in range(nc.T_max):
            if max_evals is not None and num_evals + n > max_evals:
                stop_reason = "max_evaluations"
                break

            lambda_t = self._sample_lambda(rng, t, nc.T_max)

            # 生成候选：高斯变异 + clip 投影（工程边界适配）
            candidates = []
            for p in processes:
                raw = p.x + p.sigma * rng.standard_normal(dim)
                candidates.append(parameter_space.repair(raw))

            candidate_costs = [float(objective(c).cost) for c in candidates]
            num_evals += n

            # 分别计算旧分布集与候选分布集的 Corr（投影后中心 + 原 sigma）
            sigmas = np.array([p.sigma for p in processes], dtype=np.float64)
            old_centers = np.stack([p.x for p in processes])
            new_centers = np.stack(candidates)
            corr_old = correlation(old_centers, sigmas)
            corr_new = correlation(new_centers, sigmas)

            # 按索引顺序执行 Algorithm 1 选择
            accepted = 0
            for i in range(n):
                if candidate_costs[i] < best_cost:
                    best_cost = candidate_costs[i]
                    best_x = candidates[i].copy()
                g_old = max(processes[i].cost - best_cost, 0.0)
                g_new = max(candidate_costs[i] - best_cost, 0.0)
                if selection_decision(g_old, g_new, corr_old[i], corr_new[i], lambda_t):
                    processes[i].x = candidates[i].copy()
                    processes[i].cost = candidate_costs[i]
                    processes[i].accepted += 1
                    accepted += 1

            # sigma 更新：每满 epoch 轮触发；不足部分不触发
            if (t + 1) % nc.epoch == 0:
                for p in processes:
                    rate = p.accepted / nc.epoch
                    if rate > 0.2:
                        p.sigma = p.sigma / nc.r
                    elif rate < 0.2:
                        p.sigma = p.sigma * nc.r
                    p.sigma = float(np.clip(p.sigma, SIGMA_MIN, SIGMA_MAX))
                    p.accepted = 0

            history.append(best_cost)
            sigma_mean_history.append(float(np.mean([p.sigma for p in processes])))
            acceptance_rate_history.append(accepted / n)
            actual_iterations += 1

        return OptimizationResult(
            best_candidate=best_x,
            best_cost=best_cost,
            num_evaluations=num_evals,
            num_iterations=actual_iterations,
            stop_reason=stop_reason,
            history=history,
            metrics={
                "sigma_mean_history": sigma_mean_history,
                "acceptance_rate_history": acceptance_rate_history,
                "initial_best_z": initial_best_z.tolist(),
                "initial_population": [x.tolist() for x in initial],
                "seeded_processes": len(seeds),
            },
        )


