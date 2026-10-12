"""NCSOptimizer 冒烟测试：预算、复现、边界、停止、sigma 界。"""
import numpy as np
import pytest

from optimization.config import OptimizationConfig
from optimization.ncs.optimizer import NCSConfig, NCSOptimizer
from optimization.objective import EvaluationResult
from optimization.parameter_space import ParameterSpace
from optimization.result import OptimizationResult


class Sphere:
    def __call__(self, candidate):
        x = np.asarray(candidate, dtype=np.float64)
        return EvaluationResult(cost=float(np.sum(x * x)))


def make_optimizer(T_max=2, population_size=3, sigma0=0.1):
    return NCSOptimizer(NCSConfig(T_max=T_max, population_size=population_size, sigma0=sigma0))


def test_smoke_returns_result():
    space = ParameterSpace([-1.0] * 5, [1.0] * 5)
    result = make_optimizer(T_max=2, population_size=3).optimize(
        Sphere(), space, OptimizationConfig(seed=42)
    )
    assert isinstance(result, OptimizationResult)
    assert result.best_cost >= 0.0
    assert result.best_candidate.shape == (5,)
    assert result.num_evaluations == 3 * (2 + 1)
    assert result.num_iterations == 2
    assert result.stop_reason == "T_max"


def test_evaluation_budget_respected():
    space = ParameterSpace([-1.0] * 5, [1.0] * 5)
    result = make_optimizer(T_max=5, population_size=3).optimize(
        Sphere(), space, OptimizationConfig(seed=42, max_evaluations=6)
    )
    assert result.num_evaluations <= 6
    assert result.stop_reason == "max_evaluations"


def test_insufficient_budget_raises():
    space = ParameterSpace([-1.0] * 5, [1.0] * 5)
    with pytest.raises(ValueError):
        make_optimizer(T_max=2, population_size=3).optimize(
            Sphere(), space, OptimizationConfig(seed=42, max_evaluations=2)
        )


def test_reproducible_with_same_seed():
    space = ParameterSpace([-1.0] * 5, [1.0] * 5)
    r1 = make_optimizer(T_max=3, population_size=4).optimize(
        Sphere(), space, OptimizationConfig(seed=7)
    )
    r2 = make_optimizer(T_max=3, population_size=4).optimize(
        Sphere(), space, OptimizationConfig(seed=7)
    )
    assert np.array_equal(r1.best_candidate, r2.best_candidate)
    assert r1.best_cost == r2.best_cost


def test_best_candidate_within_bounds():
    space = ParameterSpace([-1.0] * 5, [1.0] * 5)
    result = make_optimizer(T_max=5, population_size=4, sigma0=0.9).optimize(
        Sphere(), space, OptimizationConfig(seed=1)
    )
    assert space.validate(result.best_candidate)


def test_sigma_within_bounds_after_epoch_update():
    space = ParameterSpace([-1.0] * 5, [1.0] * 5)
    result = make_optimizer(T_max=12, population_size=4).optimize(
        Sphere(), space, OptimizationConfig(seed=3)
    )
    for s in result.metrics.get("sigma_mean_history", []):
        assert 1e-8 <= s <= 1.0

