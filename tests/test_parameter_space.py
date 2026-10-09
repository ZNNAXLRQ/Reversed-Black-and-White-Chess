"""参数空间测试：边界、维度、非法数值、随机初始化、复制行为。"""
import numpy as np
import pytest

from optimization.parameter_space import ParameterSpace


def make_space():
    return ParameterSpace([-1.0, -2.0, 0.0], [1.0, 2.0, 1.0])


def test_dimension_and_bounds():
    space = make_space()
    assert space.dimension == 3
    assert np.allclose(space.lower, [-1.0, -2.0, 0.0])
    assert np.allclose(space.upper, [1.0, 2.0, 1.0])


def test_lower_gt_upper_raises():
    with pytest.raises(ValueError):
        ParameterSpace([1.0, 0.0], [0.0, 0.0])


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        ParameterSpace([0.0, 0.0], [1.0])


def test_nonfinite_bounds_raises():
    with pytest.raises(ValueError):
        ParameterSpace([0.0, np.nan], [1.0, 1.0])


def test_sample_within_bounds_and_shape():
    space = make_space()
    rng = np.random.default_rng(42)
    x = space.sample(rng)
    assert x.shape == (3,)
    assert np.all(x >= space.lower) and np.all(x <= space.upper)


def test_sample_reproducible_with_same_seed():
    space = make_space()
    x1 = space.sample(np.random.default_rng(0))
    x2 = space.sample(np.random.default_rng(0))
    assert np.array_equal(x1, x2)


def test_validate_legal():
    space = make_space()
    assert space.validate(np.array([0.0, 0.0, 0.5]))


def test_validate_wrong_shape():
    space = make_space()
    assert not space.validate(np.array([0.0, 0.0]))


def test_validate_nonfinite():
    space = make_space()
    assert not space.validate(np.array([0.0, np.nan, 0.5]))
    assert not space.validate(np.array([0.0, np.inf, 0.5]))


def test_validate_out_of_bounds():
    space = make_space()
    assert not space.validate(np.array([2.0, 0.0, 0.5]))


def test_repair_clips_to_bounds():
    space = make_space()
    x = np.array([5.0, -5.0, 0.5])
    repaired = space.repair(x)
    assert np.allclose(repaired, [1.0, -2.0, 0.5])


def test_repair_returns_new_array_and_does_not_modify_input():
    space = make_space()
    x = np.array([5.0, -5.0, 0.5])
    repaired = space.repair(x)
    assert repaired is not x
    assert np.array_equal(x, [5.0, -5.0, 0.5])


def test_constructor_copies_input():
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    space = ParameterSpace(lower, upper)
    lower[0] = 99.0
    upper[0] = 99.0
    assert np.allclose(space.lower, [-1.0, -1.0])
    assert np.allclose(space.upper, [1.0, 1.0])


def test_sample_returns_independent_arrays():
    space = make_space()
    rng = np.random.default_rng(7)
    a = space.sample(rng)
    b = space.sample(rng)
    assert a is not b
    a[0] = 999.0
    assert not space.validate(a)
    assert np.all(b >= space.lower) and np.all(b <= space.upper)

