"""接口级测试：极简临时目标 + 证明通用优化接口可调用。

注意：这里仅用随机采样证明接口可组合，不是 NCS 实现。
"""
import numpy as np
import pytest

from optimization.base import Optimizer
from optimization.config import OptimizationConfig
from optimization.objective import EvaluationResult
from optimization.parameter_space import ParameterSpace
from optimization.result import OptimizationResult


class SphereObjective:
    """5 维 Sphere 目标：cost = sum(x^2)，x in [-1,1]^5，全局最优 0。"""

    def __call__(self, candidate: np.ndarray) -> EvaluationResult:
        x = np.asarray(candidate, dtype=np.float64)
        cost = float(np.sum(x * x))
        return EvaluationResult(cost=cost, metrics={"squared_norm": cost})


class RandomSampler(Optimizer):
    """极简随机采样器：仅用于接口验证，不是 NCS。"""

    def optimize(self, objective, parameter_space, config):
        rng = np.random.default_rng(config.seed)
        best = None
        best_cost = np.inf
        history = []
        num_evaluations = 0

        for _ in range(config.max_iterations):
            candidate = parameter_space.sample(rng)
            result = objective(candidate)
            num_evaluations += 1
            if result.cost < best_cost:
                best = candidate
                best_cost = result.cost
            history.append(best_cost)

        return OptimizationResult(
            best_candidate=best,
            best_cost=best_cost,
            num_evaluations=num_evaluations,
            num_iterations=config.max_iterations,
            stop_reason="max_iterations",
            history=history,
        )


def test_interface_is_callable():
    space = ParameterSpace([-1.0] * 5, [1.0] * 5)
    objective = SphereObjective()
    config = OptimizationConfig(seed=42, max_iterations=100)
    result = RandomSampler().optimize(objective, space, config)
    assert isinstance(result, OptimizationResult)
    assert result.num_evaluations == 100
    assert result.num_iterations == 100
    assert result.stop_reason == "max_iterations"
    assert result.best_cost >= 0.0
    assert result.best_candidate.shape == (5,)


def test_reproducible_with_same_seed():
    space = ParameterSpace([-1.0] * 5, [1.0] * 5)
    objective = SphereObjective()
    r1 = RandomSampler().optimize(objective, space, OptimizationConfig(seed=7, max_iterations=50))
    r2 = RandomSampler().optimize(objective, space, OptimizationConfig(seed=7, max_iterations=50))
    assert np.array_equal(r1.best_candidate, r2.best_candidate)
    assert r1.best_cost == r2.best_cost


def test_evaluation_result_rejects_nonfinite_cost():
    with pytest.raises(ValueError):
        EvaluationResult(cost=float("nan"))
    with pytest.raises(ValueError):
        EvaluationResult(cost=float("inf"))
