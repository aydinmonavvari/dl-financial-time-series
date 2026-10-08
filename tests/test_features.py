"""Offline tests for causal feature engineering and window construction."""

import numpy as np
import pandas as pd
import pytest

from dl_financial_time_series.features import (
    FEATURES,
    build_study_frame,
    build_windows,
    compute_features,
    direction_label,
    rsi_wilder,
)


def _synthetic_ohlcv(n: int = 320, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.005, n)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.005, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def test_feature_causality_no_lookahead():
    """Perturbing the close from day j onward must leave features of days < j unchanged."""
    df = _synthetic_ohlcv()
    f1 = compute_features(df)
    j = 200
    df2 = df.copy()
    df2.loc[df2.index[j]:, "close"] *= 1.5
    f2 = compute_features(df2)
    pd.testing.assert_frame_equal(f1.iloc[:j], f2.iloc[:j])
    assert f1.iloc[j]["log_ret"] != pytest.approx(f2.iloc[j]["log_ret"])


def test_direction_label_alignment():
    df = _synthetic_ohlcv()
    close = df["close"]
    lab = direction_label(close)
    assert np.isnan(lab.iloc[-1])
    for i in range(len(close) - 1):
        expected = 1.0 if close.iloc[i + 1] > close.iloc[i] else 0.0
        assert lab.iloc[i] == expected


def test_build_study_frame_warmup_and_columns():
    df = _synthetic_ohlcv(n=300)
    frame = build_study_frame(df)
    assert list(frame.columns) == list(FEATURES) + ["next_ret", "label"]
    assert frame.notna().all().all()
    feats = compute_features(df)
    positions = feats.index.get_indexer(frame.index)
    assert positions[0] > 0  # warmup rows (200d MA) dropped
    assert np.all(np.diff(positions) == 1)  # usable rows are contiguous


def test_window_builder_shapes_and_alignment():
    frame = build_study_frame(_synthetic_ohlcv(n=300))
    matrix = frame[list(FEATURES)].to_numpy()
    x, targets = build_windows(matrix, 30)
    assert x.shape == (len(matrix) - 29, 30, len(FEATURES))
    assert targets[0] == 29 and targets[-1] == len(matrix) - 1
    np.testing.assert_allclose(x[0, -1], matrix[29])  # last window row = target day features
    np.testing.assert_allclose(x[5, 0], matrix[5])  # first window row = t-29
    np.testing.assert_allclose(x[-1, -1], matrix[-1])
    with pytest.raises(ValueError):
        build_windows(matrix[:10], 30)


def test_rsi_wilder_extremes_and_warmup():
    up = pd.Series(np.linspace(1.0, 100.0, 60))
    r_up = rsi_wilder(up)
    assert r_up.iloc[:14].isna().all()  # Wilder warmup
    assert (r_up.iloc[14:] == 100.0).all()
    r_down = rsi_wilder(pd.Series(np.linspace(100.0, 1.0, 60)))
    assert (r_down.iloc[14:] == 0.0).all()
    assert rsi_wilder(pd.Series(np.full(60, 5.0))).isna().all()  # flat -> undefined
