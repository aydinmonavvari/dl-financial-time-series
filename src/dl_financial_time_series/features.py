"""Causal feature engineering and sequence window construction.

Every feature at day ``t`` uses data up to and including day ``t``; the target is
the *next* day's direction (``close[t+1] > close[t]``). No look-ahead anywhere:
rolling windows end at ``t``, lags are shifted, and the label is shifted forward.

Feature set (12 features, 4 groups):
- momentum:   log_ret, ret_lag1..ret_lag5
- volatility: realized_vol_10, realized_vol_21, hl_range_10
- volume:     volume_z_21
- technical:  rsi_14, ma50_dist, ma200_dist
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = (
    "log_ret",
    "ret_lag1",
    "ret_lag2",
    "ret_lag3",
    "ret_lag4",
    "ret_lag5",
    "realized_vol_10",
    "realized_vol_21",
    "hl_range_10",
    "volume_z_21",
    "rsi_14",
    "ma50_dist",
    "ma200_dist",
)

FEATURE_GROUPS = {
    "momentum": ["log_ret", "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_lag5"],
    "volatility": ["realized_vol_10", "realized_vol_21", "hl_range_10"],
    "volume": ["volume_z_21"],
    "technical": ["rsi_14", "ma50_dist", "ma200_dist"],
}


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI. NaN during warmup; NaN on fully flat windows."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss.replace(0.0, np.nan))
    rsi = rsi.where(avg_loss > 0.0)  # avg_loss == 0 -> NaN, then masked below
    return rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 12 strictly-causal features from daily OHLCV data."""
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    log_ret = np.log(close / close.shift(1))
    out = pd.DataFrame(index=df.index)
    out["log_ret"] = log_ret
    for k in range(1, 6):
        out[f"ret_lag{k}"] = log_ret.shift(k)
    out["realized_vol_10"] = log_ret.rolling(10).std()
    out["realized_vol_21"] = log_ret.rolling(21).std()
    out["hl_range_10"] = ((high - low) / close).rolling(10).mean()
    vol_mean = volume.rolling(21).mean()
    vol_std = volume.rolling(21).std()
    out["volume_z_21"] = (volume - vol_mean) / vol_std
    out["rsi_14"] = rsi_wilder(close, 14)
    out["ma50_dist"] = close / close.rolling(50).mean() - 1.0
    out["ma200_dist"] = close / close.rolling(200).mean() - 1.0
    return out[list(FEATURES)]


def next_day_return(close: pd.Series) -> pd.Series:
    """Log return from close[t] to close[t+1] (the return a day-t decision earns)."""
    return np.log(close.shift(-1) / close)


def direction_label(close: pd.Series) -> pd.Series:
    """1.0 if close[t+1] > close[t] else 0.0; NaN on the final row."""
    nxt = close.shift(-1)
    lab = (nxt > close).astype(float)
    return lab.mask(nxt.isna())


def build_study_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Features + next_ret + label, dropping the warmup rows; asserts contiguity."""
    feats = compute_features(df)
    frame = feats.copy()
    frame["next_ret"] = next_day_return(df["close"])
    frame["label"] = direction_label(df["close"])
    keep = frame.dropna()
    if keep.empty:
        raise ValueError("no usable rows after warmup")
    positions = feats.index.get_indexer(keep.index)
    if len(positions) > 1 and not np.all(np.diff(positions) == 1):
        raise ValueError("usable rows are not contiguous — inspect the data for interior NaNs")
    return keep


def build_windows(matrix: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows over a 2-D feature matrix.

    Returns ``(x_seq, targets)`` where ``x_seq[i]`` spans rows
    ``targets[i] - seq_len + 1 .. targets[i]`` (inclusive) — the last row of each
    window is the feature vector of the target day itself, so a prediction for
    day ``t+1`` uses only information available at the end of day ``t``.
    """
    matrix = np.asarray(matrix, dtype=float)
    n, f = matrix.shape
    if n < seq_len:
        raise ValueError(f"need at least seq_len={seq_len} rows, got {n}")
    windows = np.lib.stride_tricks.sliding_window_view(matrix, (seq_len, f))[:, 0, :, :]
    targets = np.arange(seq_len - 1, n)
    return np.ascontiguousarray(windows), targets


def windows_at_targets(x_seq: np.ndarray, targets: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Select pre-built windows for the given target rows (must be a subset, in order)."""
    wanted = np.asarray(wanted)
    if wanted.size == 0:
        return x_seq[:0]
    pos = np.searchsorted(targets, wanted)
    if pos.max() >= len(targets) or not np.array_equal(targets[pos], wanted):
        raise ValueError("requested targets outside the window range")
    return x_seq[pos]
