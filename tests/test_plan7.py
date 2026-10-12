"""Plan 7 纯逻辑测试：挑战者选择、挑战门槛与晋级判定。"""
import pytest

from experiments.plan7 import decide_champion, select_challenger, should_challenge


def test_select_challenger_lowest_cost():
    costs = {42: 0.55, 43: 0.31, 44: 0.42}
    assert select_challenger(costs) == 43


def test_select_challenger_tie_prefers_lower_seed():
    costs = {42: 0.31, 43: 0.31, 44: 0.31}
    assert select_challenger(costs) == 42


def test_select_challenger_tie_prefers_43_over_44():
    costs = {42: 0.60, 43: 0.40, 44: 0.40}
    assert select_challenger(costs) == 43


def test_select_challenger_empty_raises():
    with pytest.raises(ValueError):
        select_challenger({})


def test_should_challenge_strictly_less():
    assert should_challenge(0.20, 0.30) is True
    assert should_challenge(0.30, 0.30) is False
    assert should_challenge(0.31, 0.30) is False


def test_decide_champion_mean_u_positive():
    assert decide_champion(0.001) is True


def test_decide_champion_mean_u_nonpositive_keeps_baseline():
    assert decide_champion(0.0) is False
    assert decide_champion(-0.1) is False
