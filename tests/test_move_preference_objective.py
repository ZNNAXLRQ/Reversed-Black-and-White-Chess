"""Plan 12 强对手走法偏好目标单元测试。"""
import math

import numpy as np

from experiments.move_preference_objective import (
    ReplayPreferenceObjective, compute_preference_loss, is_eligible, logsumexp,
)
from optimization.config import OptimizationConfig
from optimization.ncs.optimizer import NCSConfig, NCSOptimizer
from optimization.parameter_space import ParameterSpace


def test_manual_nll_and_loss():
    # 1 game, 1 sample, 2 actions, weights=[1,0,0,0,0], temperature=1
    sg = {"g1": [{"action_features": np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=np.float64),
                  "chosen_idx": 1}]}
    w = np.array([1, 0, 0, 0, 0], dtype=np.float64)
    loss = compute_preference_loss(sg, w, temperature=1.0)
    # logits=[0,1]; loss = logsumexp([0,1]) - 1 = log(1 + e^-1)
    expected = math.log(1 + math.exp(-1))
    assert math.isclose(loss, expected, rel_tol=1e-9)


def test_logsumexp_stable():
    assert math.isclose(logsumexp([1000, 1000, 1000]), 1000 + math.log(3), rel_tol=1e-9)
    assert math.isclose(logsumexp([-1000, -1000]), -1000 + math.log(2), rel_tol=1e-9)
    assert math.isfinite(logsumexp([1e6, 1e6]))
    assert math.isfinite(logsumexp([-1e6, -1e6]))


def test_loss_invariant_to_action_order():
    f0 = [1, 0, 0, 0, 0]
    f1 = [0, 1, 0, 0, 0]
    f2 = [0, 0, 1, 0, 0]
    w = np.array([0.5, -0.3, 0.2, 0.1, -0.4])
    s1 = [{"action_features": np.array([f0, f1, f2], dtype=np.float64), "chosen_idx": 1}]
    s2 = [{"action_features": np.array([f2, f0, f1], dtype=np.float64), "chosen_idx": 2}]
    l1 = compute_preference_loss({"g": s1}, w)
    l2 = compute_preference_loss({"g": s2}, w)
    assert math.isclose(l1, l2, rel_tol=1e-12)


def test_decode_and_clip():
    obj = ReplayPreferenceObjective({})
    w = obj.decode(np.array([0, 0.5, 1, 0.2, 0.8]))
    assert np.allclose(w, [-10, 0, 10, -6, 6])
    space = ParameterSpace([0.0] * 5, [1.0] * 5)
    assert np.allclose(space.repair([-0.5, 1.5, 0, 0, 0]), [0, 1, 0, 0, 0])


def test_loss_nonnegative_finite_reproducible():
    rng = np.random.default_rng(0)
    sg = {}
    for g in range(3):
        samples = []
        for _ in range(5):
            feats = rng.normal(size=(3, 5))
            samples.append({"action_features": feats, "chosen_idx": int(rng.integers(3))})
        sg[f"g{g}"] = samples
    w = np.array([0.1, -0.2, 0.3, 0.4, -0.5])
    l1 = compute_preference_loss(sg, w)
    l2 = compute_preference_loss(sg, w)
    assert math.isclose(l1, l2, rel_tol=1e-12)
    assert l1 >= 0.0 and math.isfinite(l1)


def test_objective_only_uses_train():
    train_sg = {"g1": [{"action_features": np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float64),
                        "chosen_idx": 0}]}
    obj = ReplayPreferenceObjective(train_sg)
    z = np.array([0.5] * 5)
    res = obj(z)
    assert math.isclose(res.cost, compute_preference_loss(train_sg, obj.decode(z)))


def test_eligibility_strict():
    assert is_eligible(0.5, 0.6) is True
    assert is_eligible(0.6, 0.6) is False
    assert is_eligible(0.7, 0.6) is False


def test_objective_module_does_not_import_match_runner():
    import experiments.move_preference_objective as mpo
    src = open(mpo.__file__, encoding="utf-8").read()
    # 不得 import 对局运行器或搜索（只依赖预计算特征）
    assert "import match_runner" not in src
    assert "from .match_runner" not in src
    assert "from experiments.match_runner" not in src
    assert "reversi.search" not in src
    assert "best_move" not in src
    assert "from reversi" not in src


def test_ncs_reproducible_with_objective():
    rng = np.random.default_rng(0)
    sg = {}
    for g in range(3):
        samples = []
        for _ in range(4):
            feats = rng.normal(size=(3, 5))
            samples.append({"action_features": feats, "chosen_idx": 0})
        sg[f"g{g}"] = samples
    obj = ReplayPreferenceObjective(sg)
    space = ParameterSpace([0.0] * 5, [1.0] * 5)

    def run_once():
        opt = NCSOptimizer(NCSConfig(T_max=2, population_size=5, sigma0=0.1, r=0.99, epoch=10))
        return opt.optimize(obj, space, OptimizationConfig(seed=123))

    r1 = run_once()
    r2 = run_once()
    assert math.isclose(r1.best_cost, r2.best_cost, rel_tol=1e-12)
