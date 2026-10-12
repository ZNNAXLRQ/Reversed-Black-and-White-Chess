"""Bhattacharyya 距离与 Corr 测试。"""
import numpy as np
import pytest

from optimization.ncs.bhattacharyya import bhattacharyya_distance, correlation


def test_symmetry():
    d1 = bhattacharyya_distance([0.0, 0.0], 0.1, [1.0, 1.0], 0.2)
    d2 = bhattacharyya_distance([1.0, 1.0], 0.2, [0.0, 0.0], 0.1)
    assert d1 == pytest.approx(d2)


def test_nonnegative():
    assert bhattacharyya_distance([0.0, 0.0], 0.1, [1.0, 1.0], 0.2) >= 0.0


def test_zero_for_identical_distribution():
    assert bhattacharyya_distance([0.5, 0.5], 0.1, [0.5, 0.5], 0.1) == pytest.approx(0.0)


def test_simplified_formula_equal_sigma():
    x_i = np.array([1.0, 2.0, 3.0])
    x_j = np.array([0.0, 0.0, 0.0])
    sigma = 0.2
    expected = np.dot(x_i - x_j, x_i - x_j) / (8.0 * sigma * sigma)
    assert bhattacharyya_distance(x_i, sigma, x_j, sigma) == pytest.approx(expected)


def test_nonpositive_sigma_raises():
    with pytest.raises(ValueError):
        bhattacharyya_distance([0.0], 0.0, [1.0], 0.1)
    with pytest.raises(ValueError):
        bhattacharyya_distance([0.0], -0.1, [1.0], 0.1)


def test_nonfinite_input_raises():
    with pytest.raises(ValueError):
        bhattacharyya_distance([0.0, np.nan], 0.1, [1.0, 1.0], 0.1)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        bhattacharyya_distance([0.0, 0.0], 0.1, [1.0], 0.1)


def test_correlation_takes_minimum():
    centers = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]])
    sigmas = np.array([0.1, 0.1, 0.1])
    corr = correlation(centers, sigmas)
    assert corr.shape == (3,)
    for i in range(3):
        expected = min(
            bhattacharyya_distance(centers[i], sigmas[i], centers[j], sigmas[j])
            for j in range(3)
            if j != i
        )
        assert corr[i] == pytest.approx(expected)


def test_correlation_requires_n_gt_1():
    with pytest.raises(ValueError):
        correlation(np.array([[0.0, 0.0]]), np.array([0.1]))

