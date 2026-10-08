"""Offline tests for leakage controls (no-signal control, ablation design)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sklearn.metrics import roc_auc_score  # noqa: E402

from dl_financial_time_series.controls import ablation_designs, noise_feature_matrix  # noqa: E402
from dl_financial_time_series.features import (  # noqa: E402
    FEATURE_GROUPS,
    FEATURES,
    build_windows,  # noqa: E402
)
from dl_financial_time_series.models import build_model, predict_proba, train_model  # noqa: E402


def test_ablation_groups_partition_features():
    designs = ablation_designs(list(FEATURES), FEATURE_GROUPS)
    assert set(designs) == set(FEATURE_GROUPS)
    for group, kept in designs.items():
        assert len(kept) == len(FEATURES) - len(FEATURE_GROUPS[group])
        assert not set(kept) & set(FEATURE_GROUPS[group])
    with pytest.raises(ValueError):
        ablation_designs(["a", "b"], {"g1": ["a"], "g2": ["a"]})  # overlap
    with pytest.raises(ValueError):
        ablation_designs(["a", "b"], {"g1": ["a"]})  # incomplete cover


def test_noise_features_shape_and_reproducibility():
    n1 = noise_feature_matrix(50, 12, seed=3)
    n2 = noise_feature_matrix(50, 12, seed=3)
    assert n1.shape == (50, 12)
    np.testing.assert_array_equal(n1, n2)


def test_no_signal_control_is_chance_level():
    """A tiny GRU trained on pure noise must land near chance ROC-AUC out-of-sample."""
    rng = np.random.default_rng(11)
    seq_len = 30
    n_rows = 900
    x2d = noise_feature_matrix(n_rows, 12, seed=11)
    y_all = (rng.random(n_rows) > 0.5).astype(np.float32)  # independent of features
    x_seq, targets = build_windows(x2d, seq_len)  # targets: seq_len-1 .. n_rows-1
    x_tr, y_tr = x_seq[:600], y_all[targets[:600]]
    x_va, y_va = x_seq[600:750], y_all[targets[600:750]]
    x_te, y_te = x_seq[750:], y_all[targets[750:]]
    np.random.seed(5)
    torch.manual_seed(5)
    model = build_model("gru", 12)
    model, _ = train_model(model, x_tr, y_tr, x_va, y_va, epochs=2, seed=5)
    proba = predict_proba(model, x_te)
    auc = roc_auc_score(y_te, proba)
    assert 0.30 < auc < 0.70, f"model extracted signal from noise: AUC={auc:.3f}"
