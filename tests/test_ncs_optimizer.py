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


def test_sphere_5d_seed_42_smoke():
    """5 维 Sphere：f(x)=sum(x^2)，x in [0,1]^5，已知最优 x*=[0]*5，f*=0。"""
    space = ParameterSpace([0.0] * 5, [1.0] * 5)
    opt = NCSOptimizer(NCSConfig(T_max=5, population_size=10, sigma0=0.1))
    result = opt.optimize(Sphere(), space, OptimizationConfig(seed=42))
    assert np.isfinite(result.best_cost)
    assert result.best_cost >= 0.0
    assert result.best_candidate.shape == (5,)
    assert space.validate(result.best_candidate)
    assert result.num_evaluations == 10 * (5 + 1)


def test_multiple_seeds_stable_and_finite():
    space = ParameterSpace([0.0] * 5, [1.0] * 5)
    costs = []
    for seed in range(6):
        opt = NCSOptimizer(NCSConfig(T_max=5, population_size=10, sigma0=0.1))
        result = opt.optimize(Sphere(), space, OptimizationConfig(seed=seed))
        assert np.isfinite(result.best_cost)
        assert np.all(np.isfinite(result.best_candidate))
        assert result.best_cost >= 0.0
        costs.append(result.best_cost)
    # Sphere 在 [0,1]^5 上 f(x) <= 5，cost 有界；不硬编码"必须找到全局最优"
    assert all(0.0 <= c <= 5.0 for c in costs)


def test_best_cost_monotonic_non_increasing():
    space = ParameterSpace([0.0] * 5, [1.0] * 5)
    result = make_optimizer(T_max=6, population_size=5).optimize(
        Sphere(), space, OptimizationConfig(seed=42)
    )
    hist = result.history
    assert len(hist) == result.num_iterations + 1
    for i in range(1, len(hist)):
        assert hist[i] <= hist[i - 1] + 1e-12


def test_best_candidate_not_shared_across_runs():
    space = ParameterSpace([0.0] * 5, [1.0] * 5)
    opt = make_optimizer(T_max=3, population_size=4)
    r1 = opt.optimize(Sphere(), space, OptimizationConfig(seed=42))
    r2 = opt.optimize(Sphere(), space, OptimizationConfig(seed=42))
    assert r1.best_candidate is not r2.best_candidate
    r1.best_candidate[0] = 999.0
    assert r2.best_candidate[0] != 999.0


def test_champion_initialization_rescored_and_preserves_random_slots():
    space = ParameterSpace([0.] * 5, [1.] * 5)
    seed = np.full(5, .3)
    seen = []

    def objective(x):
        seen.append(x.copy())
        return EvaluationResult(cost=float(np.sum((x - .3) ** 2)))

    optimizer = NCSOptimizer(NCSConfig(T_max=1, population_size=10, initial_candidates=[seed]))
    result = optimizer.optimize(objective, space, OptimizationConfig(seed=900025))
    random = make_optimizer(T_max=0, population_size=10).optimize(
        Sphere(), space, OptimizationConfig(seed=900025))
    assert np.array_equal(seen[0], seed)
    assert result.metrics["initial_population"][1:] == random.metrics["initial_population"][1:]
    assert result.metrics["seeded_processes"] == 1
    assert result.num_evaluations == len(seen) == 20
    assert result.best_cost == 0
    assert not np.array_equal(seen[10], seen[0])  # Seed participates in ordinary Gaussian mutation.
    assert len(result.metrics["acceptance_rate_history"]) == 1
    seed[:] = 999
    assert result.metrics["initial_population"][0] == [.3] * 5


@pytest.mark.parametrize("seeds", [[[2.] * 5], [[float("nan")] * 5], [[0.] * 4],
                                 [[0.] * 5] * 4])
def test_invalid_initial_candidates_fail_before_any_evaluation(seeds):
    called = []
    opt = NCSOptimizer(NCSConfig(T_max=0, population_size=3, initial_candidates=seeds))
    with pytest.raises(ValueError, match="initial_candidates"):
        opt.optimize(lambda x: called.append(x), ParameterSpace([0.] * 5, [1.] * 5),
                     OptimizationConfig(seed=900025))
    assert called == []


