"""Data layer: SPY daily OHLCV, cache-first with offline fallbacks.

Resolution order (offline-safe by design):
1. local cache ``data/raw/spy_daily.csv`` (git-ignored),
2. the Project 03 cache (same schema) copied into the local cache,
3. a Yahoo Finance download via ``yfinance`` (network), written to the cache.

The raw CSV schema is ``Date,open,high,low,close,volume`` — the same schema
Project 03 (ml-market-prediction-study) cached for SPY 2010→present.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

# This repo's root (src/dl_financial_time_series/data.py -> parents[2]).
REPO_ROOT = Path(__file__).resolve().parents[2]


def p03_spy_cache() -> Path:
    """Optional sibling-repo fallback: Project 03's cached SPY daily CSV.

    Resolution order:
    1. the ``P03_SPY_CACHE_PATH`` environment variable, if set (explicit override);
    2. ``../ml-market-prediction-study/data/raw/spy_daily.csv`` relative to this
       repo's root (the sibling portfolio repo that already cached SPY 2010→present).

    The sibling cache is a purely optional fallback: when it does not exist,
    ``ensure_spy_data`` simply falls through to a fresh ``yfinance`` download.
    """
    override = os.environ.get("P03_SPY_CACHE_PATH")
    if override:
        return Path(override).expanduser()
    return REPO_ROOT.parent / "ml-market-prediction-study" / "data" / "raw" / "spy_daily.csv"


def load_spy_csv(path: Path | str) -> pd.DataFrame:
    """Load and validate a SPY daily CSV; return a date-indexed frame sorted by date."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    df = df.sort_index()
    if df.index.duplicated().any():
        raise ValueError("duplicate dates in data")
    if not df.index.is_monotonic_increasing:
        raise ValueError("dates not sorted after sort_index()")
    if (df[list(REQUIRED_COLUMNS[:4])] <= 0).any().any():
        raise ValueError("non-positive prices found")
    return df[list(REQUIRED_COLUMNS)].astype(float)


def fetch_yfinance_spy(start: str = "2010-01-01") -> pd.DataFrame:
    """Download SPY daily bars from Yahoo Finance (network required)."""
    import yfinance as yf  # imported lazily: network dependency

    raw = yf.download("SPY", start=start, auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned an empty frame for SPY")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    raw.index.name = "Date"
    return raw[list(REQUIRED_COLUMNS)].astype(float)


def ensure_spy_data(raw_dir: Path | str, start: str = "2010-01-01") -> Path:
    """Return a path to a cached SPY daily CSV, creating the cache if needed."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / "spy_daily.csv"
    if cache.exists():
        logger.info("using cached SPY data: %s", cache)
        return cache
    p03_cache = p03_spy_cache()
    if p03_cache.exists():
        shutil.copyfile(p03_cache, cache)
        logger.info("copied Project 03 SPY cache into %s", cache)
        return cache
    logger.info("fetching SPY from Yahoo Finance (start=%s)", start)
    fetch_yfinance_spy(start).to_csv(cache)
    return cache
