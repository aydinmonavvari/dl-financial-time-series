"""Offline tests for walk-forward folds, embargo and train-only scaling."""

import numpy as np

from dl_financial_time_series.walkforward import fit_transform_scaler, make_folds


def test_fold_structure_and_ordering():
    folds = make_folds(1000, n_folds=3)
    assert len(folds) == 3
    for f in folds:
        assert f.train_idx[0] == 0  # expanding windows anchored at the start
        assert len(f.train_idx) and len(f.val_idx) and len(f.test_idx)
        assert f.train_idx[-1] < f.val_idx[0] < f.val_idx[-1] < f.test_idx[0]
        assert np.all(np.diff(f.train_idx) == 1)
    f1, f2, f3 = folds
    assert f1.train_idx[-1] < f2.train_idx[-1] < f3.train_idx[-1]  # expanding
    assert f1.test_idx[-1] < f2.val_idx[0]  # no overlap between fold windows
    assert f2.test_idx[-1] < f3.val_idx[0]


def test_embargo_gap_of_one_row():
    folds = make_folds(1000, n_folds=3, embargo=1)
    f = folds[0]
    assert f.val_idx[0] - f.train_idx[-1] == 2  # exactly 1 row skipped
    assert f.test_idx[0] - f.val_idx[-1] == 2


def test_test_segments_are_disjoint_and_chronological():
    folds = make_folds(1000, n_folds=3)
    all_test = np.concatenate([f.test_idx for f in folds])
    assert np.all(np.diff(all_test) > 0)  # strictly increasing across folds


def test_scaler_fit_on_train_only():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 2.0, (200, 4))
    folds = make_folds(200, n_folds=3)
    fold = folds[0]
    scaled = fit_transform_scaler(x, fold)
    tr, va = fold.train_idx, fold.val_idx
    # training rows are standardised exactly
    np.testing.assert_allclose(scaled[tr].mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(scaled[tr].std(axis=0), 1.0, atol=1e-9)
    # validation rows are transformed with TRAIN statistics, not their own
    manual = (x[va] - x[tr].mean(axis=0)) / x[tr].std(axis=0)
    np.testing.assert_allclose(scaled[va], manual, atol=1e-12)


def test_make_folds_rejects_bad_design():
    try:
        make_folds(10, n_folds=3)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for n too small")
