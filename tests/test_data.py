"""Offline tests for the data layer (no network)."""

import numpy as np
import pandas as pd
import pytest

from dl_financial_time_series.data import load_spy_csv


def _make_csv(tmp_path, rows, columns=("Date", "open", "high", "low", "close", "volume")):
    df = pd.DataFrame(rows, columns=list(columns))
    path = tmp_path / "spy.csv"
    df.to_csv(path, index=False)
    return path


def test_load_sorts_by_date_and_validates(tmp_path):
    rng = np.random.default_rng(0)
    dates = ["2020-01-05", "2020-01-02", "2020-01-03"]
    rows = [
        [d, 100.0, 101.0, 99.0, 100.5, float(rng.integers(1e6, 5e6))] for d in dates
    ]
    path = _make_csv(tmp_path, rows)
    df = load_spy_csv(path)
    assert df.index.is_monotonic_increasing
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 3


def _write_text_csv(path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_load_rejects_bad_input(tmp_path):
    good = (
        "Date,open,high,low,close,volume\n"
        "2020-01-02,100.0,101.0,99.0,100.5,1000000.0\n"
        "2020-01-03,100.5,102.0,100.0,101.0,1100000.0\n"
    )
    # missing column
    sub = tmp_path / "missing_col"
    sub.mkdir()
    path = sub / "spy.csv"
    _write_text_csv(
        path,
        "Date,open,high,low,volume\n"
        "2020-01-02,100.0,101.0,99.0,1000000.0\n"
        "2020-01-03,100.5,102.0,100.0,1100000.0\n",
    )
    with pytest.raises(ValueError, match="missing required columns"):
        load_spy_csv(path)
    # duplicate dates
    sub = tmp_path / "dup"
    sub.mkdir()
    _write_text_csv(sub / "spy.csv", good + "2020-01-02,100.0,101.0,99.0,100.5,1000000.0\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_spy_csv(sub / "spy.csv")
    # non-positive prices
    sub = tmp_path / "badprice"
    sub.mkdir()
    _write_text_csv(
        sub / "spy.csv",
        "Date,open,high,low,close,volume\n"
        "2020-01-02,100.0,101.0,99.0,-100.5,1000000.0\n",
    )
    with pytest.raises(ValueError, match="non-positive"):
        load_spy_csv(sub / "spy.csv")
    # missing file
    with pytest.raises(FileNotFoundError):
        load_spy_csv(tmp_path / "nope.csv")
