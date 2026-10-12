"""选择规则测试（退化规则，EPS=1e-12）。"""
import numpy as np
import pytest

from optimization.ncs.optimizer import selection_decision


def test_accept_when_ratio_lt_lambda():
    # g_old=1, g_new=1 -> q_new=0.5；corr_old=1, corr_new=3 -> c_new=0.75
    # ratio = 0.5/0.75 = 2/3 < 0.8 -> accept
    assert selection_decision(1.0, 1.0, 1.0, 3.0, 0.8) is True


def test_reject_when_ratio_ge_lambda():
    # ratio = 2/3 >= 0.5 -> reject
    assert selection_decision(1.0, 1.0, 1.0, 3.0, 0.5) is False


def test_g_degenerate_sets_q_half():
    # g_old+g_new <= EPS -> q_new=0.5；c_new=0.5；ratio=1.0
    assert selection_decision(0.0, 0.0, 1.0, 1.0, 2.0) is True
    assert selection_decision(0.0, 0.0, 1.0, 1.0, 0.5) is False


def test_corr_degenerate_sets_c_half():
    # corr_old+corr_new <= EPS -> c_new=0.5；q_new=0.5；ratio=1.0
    assert selection_decision(1.0, 1.0, 0.0, 0.0, 2.0) is True
    assert selection_decision(1.0, 1.0, 0.0, 0.0, 0.5) is False


def test_c_new_le_eps_rejects():
    # corr 总和 > EPS 但 c_new <= EPS -> 拒绝（ratio=inf）
    assert selection_decision(1.0, 1.0, 1.0, 0.0, 100.0) is False


def test_hand_computed_ratio():
    # g_old=3, g_new=1 -> q_new=0.25；corr_old=1, corr_new=3 -> c_new=0.75
    # ratio = 1/3
    assert selection_decision(3.0, 1.0, 1.0, 3.0, 0.34) is True
    assert selection_decision(3.0, 1.0, 1.0, 3.0, 0.33) is False


def test_nonfinite_raises():
    with pytest.raises(ValueError):
        selection_decision(np.nan, 1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        selection_decision(1.0, 1.0, np.inf, 1.0, 1.0)

