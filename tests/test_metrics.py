"""Offline tests for metrics: Wilson CI, accuracy/AUC, cost math, equity curves."""

import numpy as np
import pytest

from dl_financial_time_series.metrics import (
    auc_wilson_ci,
    buy_hold_returns,
    classification_metrics,
    equity_curve,
    max_drawdown,
    positions_from_scores,
    sharpe_ratio,
    strategy_net_returns,
    wilson_interval,
)


def test_wilson_interval_hand_computed():
    lo, hi = wilson_interval(30, 50)  # p=0.6, n=50
    assert lo == pytest.approx(0.4618, abs=1e-3)
    assert hi == pytest.approx(0.7239, abs=1e-3)
    lo0, hi0 = wilson_interval(0, 25)
    assert lo0 == pytest.approx(0.0, abs=1e-12) and 0.0 < hi0 < 0.2
    lo1, hi1 = wilson_interval(25, 25)
    assert hi1 == pytest.approx(1.0, abs=1e-12) and lo1 > 0.8
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_auc_wilson_approximation_bounds():
    lo, hi = auc_wilson_ci(0.5, 10, 10)
    assert lo < 0.5 < hi


def test_accuracy_and_auc_hand_computed():
    y = [0, 0, 1, 1]
    s = [0.1, 0.4, 0.35, 0.8]
    m = classification_metrics(y, s, threshold=0.5)
    assert m["accuracy"] == pytest.approx(0.75)  # preds [0,0,0,1] vs y
    assert m["roc_auc"] == pytest.approx(0.75)  # 3 of 4 pos/neg pairs concordant
    assert m["n"] == 4 and m["n_pos"] == 2
    assert m["accuracy_ci95"][0] < 0.75 < m["accuracy_ci95"][1]


def test_constant_scores_give_null_auc():
    m = classification_metrics([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
    assert m["roc_auc"] is None
    assert m["roc_auc_ci95"] is None


def test_strategy_cost_math_ten_bps_per_side():
    positions = np.array([0.0, 1.0, 1.0, 0.0])
    returns = np.array([0.01, 0.02, -0.01, 0.005])
    net = strategy_net_returns(positions, returns, cost_per_side=0.001)
    # day 0: flat, day 1: enter (10bps), day 2: hold, day 3: exit (10bps)
    np.testing.assert_allclose(net, [0.0, 0.019, -0.01, -0.001])


def test_buy_hold_single_entry_cost():
    net = buy_hold_returns(np.array([0.01, 0.02]), cost_per_side=0.001)
    np.testing.assert_allclose(net, [0.009, 0.02])


def test_equity_curve_and_max_drawdown():
    eq = equity_curve([0.10, -0.10])
    assert eq[-1] == pytest.approx(0.99)
    assert max_drawdown(eq) == pytest.approx(-0.10)  # peak 1.1 -> trough 0.99
    assert max_drawdown(np.array([])) == 0.0


def test_sharpe_and_positions():
    assert sharpe_ratio(np.array([0.001, 0.002, 0.0015, 0.0025])) > 0
    assert sharpe_ratio(np.array([0.0, 0.0, 0.0])) is None
    np.testing.assert_array_equal(
        positions_from_scores([0.4, 0.6, 0.5], threshold=0.5), [0.0, 1.0, 1.0]
    )
