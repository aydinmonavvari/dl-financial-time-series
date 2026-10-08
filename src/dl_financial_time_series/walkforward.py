"""Walk-forward fold construction with embargo and train-only scaling.

Design (strict — no leakage):
- chronological, expanding-window folds; later folds train on earlier folds' test
  periods (standard walk-forward: once a period has been "lived through", it may
  be used for training subsequent folds);
- ``embargo`` rows are skipped at every segment boundary because the label of the
  last row of a segment uses the *next* day's close — the skipped rows guarantee
  train labels never reach into validation/test rows;
- the ``StandardScaler`` is fitted on the training rows of each fold only and then
  applied to validation/test (fit-on-train-only, per fold);
- there is no shuffling across time anywhere; minibatch shuffling is applied
  *inside* the training window only (documented in the pipeline/README).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from dl_financial_time_series import config


@dataclass(frozen=True)
class Fold:
    """Contiguous index segments into the usable (post-warmup) study frame."""

    name: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def make_folds(
    n: int,
    n_folds: int = config.N_FOLDS,
    train_end_fracs: tuple[float, ...] = config.TRAIN_END_FRACS,
    val_frac: float = config.VAL_FRAC,
    embargo: int = config.EMBARGO_DAYS,
) -> list[Fold]:
    """Build expanding-window walk-forward folds with an embargo at each boundary."""
    if len(train_end_fracs) != n_folds:
        raise ValueError("train_end_fracs must contain exactly n_folds entries")
    ends = [int(round(f * n)) for f in train_end_fracs]
    next_ends = ends[1:] + [n]
    v = max(int(round(val_frac * n)), 2)
    folds: list[Fold] = []
    for k, (end, next_end) in enumerate(zip(ends, next_ends, strict=True), start=1):
        train = np.arange(0, end)
        val = np.arange(end + embargo, end + embargo + v)
        test = np.arange(val[-1] + 1 + embargo, next_end)
        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            raise ValueError(f"fold {k} is empty (n={n} too small for this design)")
        folds.append(Fold(f"fold{k}", train, val, test))
    for prev, cur in zip(folds[:-1], folds[1:], strict=True):
        # expanding: the next fold's train window (arange excludes its end) must reach
        # at least through the previous fold's last test row
        if cur.train_idx[-1] < prev.test_idx[-1]:
            raise ValueError("folds must be expanding (each train includes the previous test)")
        if np.intersect1d(prev.test_idx, cur.val_idx).size:
            raise ValueError("fold test and next validation windows overlap")
    return folds


def fit_transform_scaler(matrix: np.ndarray, fold: Fold) -> np.ndarray:
    """StandardScaler fitted on the fold's training rows only, applied to all rows."""
    matrix = np.asarray(matrix, dtype=float)
    scaler = StandardScaler().fit(matrix[fold.train_idx])
    return scaler.transform(matrix)
