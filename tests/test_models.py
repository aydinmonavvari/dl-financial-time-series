"""Offline tests for the four tiny architectures (torch required — skipped in CI)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dl_financial_time_series import config  # noqa: E402
from dl_financial_time_series.models import (  # noqa: E402
    build_model,
    count_parameters,
    predict_proba,
    train_model,
)


def _rand_batch(seed: int = 0, b: int = 4, t: int = 30, f: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(b, t, f)).astype(np.float32)


@pytest.mark.parametrize("name", ["lstm", "gru", "cnn", "transformer"])
def test_forward_output_shape(name):
    model = build_model(name, 12)
    out = model(torch.tensor(_rand_batch()))
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", ["lstm", "gru", "cnn", "transformer"])
def test_param_count_below_cap(name):
    assert count_parameters(build_model(name, 12)) < config.PARAM_BUDGET


def test_predict_proba_range():
    proba = predict_proba(build_model("gru", 12), _rand_batch(b=8))
    assert proba.shape == (8,)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_train_determinism_same_seed():
    """Same seed -> functionally identical results.

    Rationale: BLAS/thread-scheduled float reductions may reorder at the ~1e-16
    level across runs, so we assert functional determinism with
    allclose(atol=1e-10) rather than bitwise equality (same protocol as the
    Project 03 study).
    """
    rng = np.random.default_rng(3)
    x = rng.normal(size=(256, 30, 12)).astype(np.float32)
    y = (rng.random(256) > 0.5).astype(np.float32)
    outs = []
    for _ in range(2):
        np.random.seed(7)
        torch.manual_seed(7)
        model = build_model("gru", 12)
        model, _ = train_model(model, x[:192], y[:192], x[192:], y[192:], epochs=2, seed=7)
        outs.append(predict_proba(model, x[192:]))
    assert np.allclose(outs[0], outs[1], atol=1e-10)


def test_transformer_causal_mask():
    """Position i's encoder output must not depend on inputs after i."""
    model = build_model("transformer", 12)
    model.eval()
    x = torch.tensor(_rand_batch(5, 30, 12))
    with torch.no_grad():
        h1 = model.encode(x)
        x2 = x.clone()
        x2[:, 10:, :] = torch.randn_like(x2[:, 10:, :]) * 5.0
        h2 = model.encode(x2)
    assert torch.allclose(h1[:, :10], h2[:, :10], atol=1e-6)
    assert not torch.allclose(h1[:, -1], h2[:, -1], atol=1e-4)
