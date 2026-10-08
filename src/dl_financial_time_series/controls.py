"""Leakage controls: no-signal noise features and feature-group ablation designs."""

from __future__ import annotations

import numpy as np


def noise_feature_matrix(n_rows: int, n_features: int, seed: int) -> np.ndarray:
    """Pure Gaussian noise with the same shape as the real feature matrix.

    Used by the no-signal control: any model trained on these features can only
    be at chance level on out-of-sample data. If it is not, the evaluation
    pipeline leaks information somewhere.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_rows, n_features))


def ablation_designs(
    feature_names: list[str] | tuple[str, ...], groups: dict[str, list[str]]
) -> dict[str, list[str]]:
    """For each group, the features kept when that group is dropped.

    Raises ValueError unless the groups partition the feature set exactly once.
    """
    flat = [f for names in groups.values() for f in names]
    if len(flat) != len(set(flat)):
        raise ValueError("feature groups overlap")
    if sorted(flat) != sorted(feature_names):
        raise ValueError("feature groups must cover the feature set exactly")
    return {
        group: [f for f in feature_names if f not in names] for group, names in groups.items()
    }
