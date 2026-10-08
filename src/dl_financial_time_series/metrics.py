"""Evaluation metrics: accuracy, ROC-AUC, Wilson CIs, cost-adjusted strategy math."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score

Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def wilson_interval(successes: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def auc_wilson_ci(auc: float, n_pos: int, n_neg: int, z: float = Z_95) -> tuple[float, float]:
    """Approximate Wilson interval treating AUC as the concordant-pair proportion.

    ROC-AUC estimates P(score_pos > score_neg); the Wilson interval is exact for
    binomial proportions, so this applies it to the implied count of concordant
    pairs (auc * n_pos * n_neg). Pairs are not independent, so this is a
    documented approximation, not a rigorous AUC interval (Hanley-McNeil /
    DeLong would be the standard choices).
    """
    n_pairs = n_pos * n_neg
    if n_pairs <= 0:
        raise ValueError("both classes must be present")
    return wilson_interval(auc * n_pairs, n_pairs, z)


def classification_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> dict:
    """Accuracy (+ Wilson CI) and ROC-AUC (+ approximate Wilson CI) for a score vector."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    if len(y) != len(s):
        raise ValueError("y_true and scores must have the same length")
    n = len(y)
    pred = (s >= threshold).astype(int)
    correct = int((pred == y).sum())
    acc = correct / n
    acc_ci = wilson_interval(correct, n)
    n_pos = int((y == 1).sum())
    n_neg = n - n_pos
    auc = None
    auc_ci = None
    if n_pos > 0 and n_neg > 0 and len(np.unique(s)) > 1:
        auc = float(roc_auc_score(y, s))
        auc_ci = auc_wilson_ci(auc, n_pos, n_neg)
    return {
        "n": n,
        "n_pos": n_pos,
        "accuracy": acc,
        "accuracy_ci95": [float(acc_ci[0]), float(acc_ci[1])],
        "roc_auc": auc,
        "roc_auc_ci95": None if auc_ci is None else [float(auc_ci[0]), float(auc_ci[1])],
    }


def positions_from_scores(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Long/flat positions in {0,1} from a score vector (ridge uses threshold 0.0)."""
    return (np.asarray(scores, dtype=float) >= threshold).astype(float)


def strategy_net_returns(
    positions: np.ndarray, next_returns: np.ndarray, cost_per_side: float = 0.001
) -> np.ndarray:
    """Daily strategy returns net of ``cost_per_side`` paid on every position change.

    A position change of ±1 is one side (buy or sell) costing ``cost_per_side``
    of notional; entering from flat counts as one side.
    """
    pos = np.asarray(positions, dtype=float)
    ret = np.asarray(next_returns, dtype=float)
    if len(pos) != len(ret):
        raise ValueError("positions and returns must have the same length")
    turnover = np.abs(np.diff(pos, prepend=0.0))
    return pos * ret - cost_per_side * turnover


def buy_hold_returns(next_returns: np.ndarray, cost_per_side: float = 0.001) -> np.ndarray:
    """Buy-and-hold daily returns with a single entry cost on the first day."""
    ret = np.asarray(next_returns, dtype=float).copy()
    if len(ret):
        ret[0] -= cost_per_side
    return ret


def equity_curve(net_returns: np.ndarray) -> np.ndarray:
    """Compounded equity path from net daily returns (starts at 1.0 growth factor)."""
    return np.cumprod(1.0 + np.asarray(net_returns, dtype=float))


def total_return(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=float)
    return float(eq[-1] - 1.0) if len(eq) else 0.0


def max_drawdown(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=float)
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min())


def sharpe_ratio(net_returns: np.ndarray, periods: int = 252) -> float | None:
    """Annualised Sharpe of daily net returns; None when returns are constant."""
    r = np.asarray(net_returns, dtype=float)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    if sd == 0.0:
        return None
    return float(r.mean() / sd * math.sqrt(periods))
