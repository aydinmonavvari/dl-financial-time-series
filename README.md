# dl-financial-time-series

![CI](https://github.com/aydinmonavvari/dl-financial-time-series/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 1 · Short description

Do *tiny* deep-learning architectures help forecast the **next-day direction of the
S&P 500 ETF (SPY)**? This repo trains four small sequence models — LSTM, GRU,
temporal 1D-CNN, and a tiny Transformer encoder (all < 10k parameters, hard cap
100k) — against classical baselines under **strict walk-forward evaluation** with an
embargo, train-only scaling, a no-signal leakage control, and a cost-adjusted
backtest.

*Educational research project. Not a trading system and not investment advice. The
headline finding is an honestly-reported null result: none of the models shows a
reliable out-of-sample edge.*

## 2 · Research question

> Evaluating whether small deep learning architectures (LSTM, GRU, temporal 1D-CNN,
> a tiny Transformer encoder) improve next-day S&P 500 ETF (SPY) direction
> forecasting over simple baselines, under strict walk-forward evaluation.

The question is deliberately framed so that "no" is an acceptable, publishable
answer — and, on this data with this protocol, that is what the study finds.

## 3 · Motivation

Deep learning papers applied to finance routinely report impressive accuracies that
evaporate under honest evaluation: shuffled splits leak the future into training,
scalers are fitted on the whole series, and transaction costs are ignored. This
project inverts that workflow. It starts from the strictest reasonable protocol
(walk-forward, embargo, train-only scaling, fixed seeds) and asks whether the models
can beat *any* baseline at all. A sibling project (P03, `ml-market-prediction-study`)
already found that gradient boosting could not beat the majority class on the same
instrument; here the question is whether tiny sequence architectures do any better.

## 4 · Why this matters

- **For finance:** the efficient-market null (Fama 1970) is the bar every
  forecasting claim must clear; demonstrating a disciplined pipeline that *fails* to
  clear it is more informative than a leaderboard score.
- **For machine learning:** sequence models, early stopping, leakage controls and
  the no-signal test generalize to any temporal prediction problem (churn, demand,
  energy).
- **For research integrity:** the no-signal control (best architecture on pure-noise
  features must land at chance) is a cheap, decisive test that the evaluation
  machinery itself does not leak. Breiman's (2001) two cultures both get scrutinized:
  the DL models *and* the linear baselines.

## 5 · Methodology

1. **Task** — binary classification: is `close[t+1] > close[t]` (UP) or not (DOWN)?
   Every feature at day `t` uses data up to and including day `t` only.
2. **Features (13, strictly causal)** — log return + 5 lags; realized volatility
   (10d, 21d rolling std of log returns); mean 10d high-low range; 21d volume
   z-score; Wilder RSI(14); distance from the 50d and 200d moving averages.
3. **Walk-forward design** — 3 chronological **expanding-window** folds. Inside each
   fold: train → 1-row **embargo** → validation (201 days) → 1-row embargo → test.
   The embargo exists because the label at day `t` is resolved by the close of
   `t+1`; the skipped rows guarantee train labels never reach into val/test rows.
4. **Scaling** — `StandardScaler` fitted on each fold's **training rows only**.
5. **No shuffling across time** — minibatch shuffling happens *inside* the training
   window only (documented and seeded); validation/test are scored chronologically.
6. **Training** — Adam (lr 1e-3, weight decay 1e-4), batch 64, max 20 epochs, early
   stopping with patience 5 on validation loss, best weights restored, dropout 0.2,
   sequence length 30 trading days, fixed seed 42 everywhere.
7. **Controls** — (a) *no-signal control*: each architecture retrained on pure
   Gaussian-noise features under the identical protocol must be chance-level;
   (b) *feature-group ablation*: retrain the GRU with momentum / volatility /
   volume / technical groups dropped one at a time.

## 6 · Dataset

| Property | Value |
| --- | --- |
| Instrument | SPY (SPDR S&P 500 ETF), daily OHLCV |
| Raw span | 2010-01-04 → 2026-10-06 · **4,216 rows** |
| Usable span | 2010-10-18 → 2026-10-06 · **4,016 rows** (after 200d-MA warmup) |
| Pooled OOS test rows | **1,399** (440 + 439 + 520 across the 3 folds) |
| Positive (UP) base rate | ≈ 54% — mildly imbalanced toward "UP" |

## 7 · Data sources

Yahoo Finance daily bars via `yfinance` (start 2010-01-01). The pipeline is
**cache-first and offline-safe**:

1. it reuses the local cache `data/raw/spy_daily.csv` (git-ignored; this run copied
   the cache already produced by the sibling Project 03 repo);
2. if that is missing, it copies the **optional** sibling Project 03 cache
   (`../ml-market-prediction-study/data/raw/spy_daily.csv`, resolved relative to this
   repo's root — never hardcoded). Set the environment variable `P03_SPY_CACHE_PATH`
   to point somewhere else. This fallback is purely optional: if the sibling cache is
   absent, the pipeline simply falls through to step 3;
3. otherwise it fetches fresh bars with `yfinance` (network required).

All tests run offline on synthetic fixtures. The repository never commits datasets.

## 8 · Architecture

```
src/dl_financial_time_series/
├── config.py        # every hyperparameter + fold design in one place
├── data.py          # cache-first SPY loader (local cache → P03 cache → yfinance)
├── features.py      # causal features, labels, sliding-window builder
├── walkforward.py   # expanding folds, embargo, train-only StandardScaler
├── models.py        # LSTM / GRU / temporal CNN / tiny Transformer + training loop
├── metrics.py       # accuracy, ROC-AUC, Wilson CIs, cost math, equity curves
├── controls.py      # no-signal noise features, feature-group ablation design
└── pipeline.py      # end-to-end study → reports/ + figures/
scripts/run_study.py # CLI: --seed --epochs --folds --data
tests/               # 34 offline test cases (28 test functions; torch-dependent
                     #   ones skip when torch is absent)
```

Fold geometry (from the actual run, `reports/metrics.json`):

| fold | train (days) | val (days) | test (days) | n_train | n_val | n_test |
| --- | --- | --- | --- | --- | --- | --- |
| fold1 | 2010-11-29 → 2018-10-08 | 2018-10-10 → 2019-07-30 | 2019-08-01 → 2021-04-29 | 1,979 | 201 | 440 |
| fold2 | 2010-11-29 → 2021-04-29 | 2021-05-03 → 2022-02-15 | 2022-02-17 → 2023-11-15 | 2,622 | 201 | 439 |
| fold3 | 2010-11-29 → 2023-11-15 | 2023-11-17 → 2024-09-06 | 2024-09-10 → 2026-10-06 | 3,264 | 201 | 520 |

Later folds train on strictly more history (expanding windows): once a period has
been "lived through", it may be used for training subsequent folds — standard
walk-forward practice, with the embargo preventing any label overlap at boundaries.

## 9 · Experimental design

- **Task:** binary next-day SPY direction; sequence length 30 days of scaled causal
  features; the last row of each window is the decision day itself, so the
  prediction for `t+1` never sees `t+1` information.
- **Folds:** 3 expanding walk-forward folds, 1-row embargo at every boundary;
  validation only used for early stopping / best-weight selection.
- **Seeds:** everything pinned (seed 42); a determinism test asserts same-seed runs
  agree to `allclose(atol=1e-10)` (functional determinism — BLAS float reductions
  may reorder at ~1e-16).
- **Models (parameter counts from the actual run, all < 100k cap):**

| model | params | description |
| --- | --- | --- |
| GRU | 4,545 | 1 layer, hidden 32, dropout head (Cho et al. 2014) |
| LSTM | 6,049 | 1 layer, hidden 32, dropout head (Hochreiter & Schmidhuber 1997) |
| temporal CNN | 7,521 | 3 causal (left-padded) conv blocks, kernel 3, global avg pooling |
| tiny Transformer | 9,025 | 1 encoder layer, d_model 32, 2 heads, FF 64, causal mask (Vaswani et al. 2017) |

- **Baselines:** majority class (train positive rate); logistic regression on the
  13 scaled features; ridge regression predicting the *next return* (direction =
  sign of prediction).
- **Costs:** long/flat positions pay 10 bps per side on every position change;
  buy&hold pays one entry side.

## 10 · Models

All four architectures consume `(batch, 30, 13)` tensors of standardized causal
features and emit a single logit, trained with `BCEWithLogitsLoss`. The Transformer
uses a causal attention mask (unit-tested: perturbing positions `i+1:` leaves
position `i`'s encoding unchanged). The CNN is strictly causal via left-padding.
Training uses early stopping on validation loss and restores the best weights —
selected *only* on validation, never on test.

## 11 · Evaluation metrics

- **Accuracy** with an exact **Wilson 95% CI** (Wilson 1927) — appropriate for the
  modest per-fold sample sizes.
- **ROC-AUC** with an *approximate* Wilson interval (AUC treated as the
  concordant-pair proportion — a documented approximation; DeLong/Hanley-McNeil
  would be the rigorous choices).
- **Economics:** long/flat directional strategy net of 10 bps per-side costs —
  total return, annualized Sharpe, max drawdown — versus buy&hold SPY net of its
  single entry cost.
- **Per-fold AND pooled** out-of-sample reporting; the pooled numbers below are the
  headline.

## 12 · Results

All numbers below are actual outputs of the committed pipeline run
(seed 42, torch 2.14.1+cpu, runtime **75.4 s** CPU; `reports/metrics.json`).

**Pooled out-of-sample (n = 1,399 test days, 2019-08 → 2026-10):**

| method | params | accuracy | acc CI (Wilson) | ROC-AUC | AUC CI (approx) |
| --- | --- | --- | --- | --- | --- |
| lstm | 6,049 | 0.5468 | [0.521, 0.573] | **0.4569** | [0.456, 0.458] |
| gru | 4,545 | 0.5375 | [0.511, 0.564] | 0.4704 | [0.469, 0.472] |
| cnn | 7,521 | 0.5432 | [0.517, 0.569] | 0.4807 | [0.479, 0.482] |
| transformer | 9,025 | 0.5368 | [0.511, 0.563] | 0.4807 | [0.479, 0.482] |
| logistic | — | 0.5375 | [0.511, 0.564] | 0.4802 | [0.479, 0.482] |
| ridge | — | 0.4968 | [0.471, 0.523] | 0.4969 | [0.495, 0.498] |
| majority | — | 0.5432 | [0.517, 0.569] | (n/a per fold) | — |

Per-fold ROC-AUC (no method is consistent across folds):

| method | fold1 | fold2 | fold3 |
| --- | --- | --- | --- |
| lstm | 0.5355 | 0.4874 | 0.4431 |
| gru | 0.5085 | 0.5005 | 0.4555 |
| cnn | 0.4829 | 0.4820 | 0.4895 |
| transformer | 0.4758 | 0.5029 | 0.5134 |
| logistic | 0.5228 | 0.4671 | 0.4999 |
| ridge | 0.5626 | 0.4669 | 0.5132 |

**Directional strategy net of 10 bps/side (pooled OOS, long/flat):**

| strategy | total return | Sharpe (ann.) | max drawdown |
| --- | --- | --- | --- |
| transformer | **+101.54%** | 0.71 | −35.75% |
| lstm | +98.55% | 0.69 | −35.75% |
| cnn / majority | +97.77% | 0.68 | −35.75% |
| buy&hold SPY (net) | **+97.77%** | 0.68 | −35.75% |
| gru | +75.54% | 0.59 | −35.75% |
| logistic | +62.15% | 0.52 | −35.61% |
| ridge | −16.56% | −0.09 | −35.89% |

**No-signal control (pure-noise features, identical protocol) — all chance-level:**

| model | pooled AUC on noise | pooled AUC on real features |
| --- | --- | --- |
| lstm | 0.4782 | 0.4569 |
| gru | 0.4739 | 0.4704 |
| cnn | 0.4699 | 0.4807 |
| transformer | 0.5075 | 0.4807 |

**Feature-group ablation (GRU, pooled AUC):** full features 0.4704; drop momentum
0.4961; drop volatility 0.4667; drop volume 0.4727; drop technical 0.4750.

## 13 · Interpretation

- **No model beats the majority baseline.** The best accuracy (LSTM 0.5468) sits
  inside the majority baseline's Wilson interval (0.517–0.569); every ROC-AUC is at
  or *below* 0.5, i.e. the models' probability rankings carry **no** out-of-sample
  ordering information. This replicates — with a different model family — the null
  result of Project 03 on the same instrument.
- **Per-fold AUCs flip sign across folds** (LSTM: 0.5355 → 0.4431; ridge: 0.5626 →
  0.4669), the classic signature of noise rather than a decaying edge.
- **The equity-curve "winner" is not evidence.** The Transformer strategy's
  +101.54% vs buy&hold's +97.77% (+3.8 pp over ~7 years, with a near-identical
  Sharpe and identical drawdown) is well within noise. Because the UP base rate is
  ≈ 54%, the majority constant prediction is "always long" — so the majority
  baseline *is* buy&hold, and any model that stays long most days tracks it closely.
  The max drawdowns are identical (−35.75%) for the same reason.
- **The controls pass.** On pure-noise features every architecture lands at chance
  (AUC 0.470–0.508), which rules out leakage in the evaluation machinery; the
  ablation shows no feature group carries signal (removing *momentum* actually
  nudges AUC up to 0.4961, still below chance).
- Honest conclusion: under this protocol, **tiny deep sequence models add nothing**
  to next-day SPY direction forecasting over a one-line baseline.

## 14 · Limitations

- **Single asset, daily frequency.** SPY only; intraday or cross-sectional settings
  are untested.
- **Small models, CPU budget.** All architectures are < 10k params trained ≤ 20
  epochs on a CPU (75 s total) — deliberately tiny; larger models, longer training,
  or richer inputs (order flow, options, cross-asset) are out of scope and might
  behave differently (though the burden of proof stays with the claim).
- **One seed headline.** Results use a single fixed seed (42) plus a determinism
  guarantee; a multi-seed ensemble of runs would tighten the intervals.
- **Non-stationarity.** One expanding-window protocol over 2010–2026 includes
  radically different regimes; no regime conditioning is attempted.
- **Long/flat strategy space.** Only threshold-0.5 long/flat positions are
  backtested; other position sizing or thresholds are not searched (avoiding
  test-set tuning).
- **AUC interval approximation.** The Wilson-on-concordant-pairs AUC interval is a
  documented approximation, not DeLong.
- **Pooled majority "AUC" is an artifact** (fold-specific constants); per-fold
  majority AUC is genuinely undefined (constant scores).

## 15 · Reproducibility

- Fixed seed 42 for NumPy/PyTorch; seeded minibatch shuffling; single-threaded
  torch (`torch.set_num_threads(1)`) for deterministic CPU reductions.
- Determinism contract: same seed → identical predictions up to
  `allclose(atol=1e-10)` (asserted in the test suite; bitwise equality is not
  required because BLAS may reorder float reductions at ~1e-16).
- Full run: `python scripts/run_study.py` → rewrites `reports/metrics.json`,
  `reports/summary.md` and all six figures in ~75 s on CPU.
- `torch` is an optional extra (`pip install -e .[torch]`); the test suite skips
  torch-dependent modules when torch is absent.
- **CI (`.github/workflows/ci.yml`) runs two jobs, and only these:**
  - **Job A (`test`, light deps):** installs `-e .[dev]` only — no torch — so the
    torch-dependent modules skip themselves via `pytest.importorskip`: **20 of the 34
    test cases run** (plus ruff).
  - **Job B (`torch`, full suite):** installs the CPU-only torch wheel first, then
    `-e .[dev,torch]`, and runs the **entire suite: 34 of 34 test cases, no skips**
    (plus ruff), with `PYTHONHASHSEED` pinned.
  - Neither job runs the study pipeline, fetches network data, or uses a GPU.
- Data: cache-first (local CSV → optional sibling P03 cache (env override
  `P03_SPY_CACHE_PATH`) → yfinance). No dataset is
  committed; `data/raw/*` and `data/processed/*` are git-ignored except `.gitkeep`.

## 16 · Installation

```bash
git clone https://github.com/aydinmonavvari/dl-financial-time-series.git
cd dl-financial-time-series
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev,torch]     # torch needed for the models; [dev] suffices for light tests
```

Python ≥ 3.11. `requirements.txt` pins the full validated local environment
(Python 3.12.14, torch 2.14.1+cpu via the CPU wheel index, numpy 2.1.3, pandas 2.2.3,
scikit-learn 1.5.2, matplotlib 3.9.2, yfinance 1.7.0).

## 17 · Usage

```bash
# full study (data -> features -> folds -> training -> controls -> reports + figures)
python scripts/run_study.py

# variant knobs
python scripts/run_study.py --seed 42 --epochs 20 --folds 3

# quality gates
pytest tests/ -q
ruff check .
```

## 18 · Example

```python
from dl_financial_time_series.data import ensure_spy_data, load_spy_csv
from dl_financial_time_series.features import build_study_frame, build_windows, FEATURES

csv = ensure_spy_data("data/raw")           # cached / copied / fetched — offline-safe
frame = build_study_frame(load_spy_csv(csv))  # 4,016 usable rows, features + label
x_seq, targets = build_windows(frame[list(FEATURES)].to_numpy(float), seq_len=30)
print(x_seq.shape)   # (4016, 30, 13) — windows ending at each decision day
```

## 19 · Project structure

```
├── data/{raw,processed}/       # git-ignored caches (.gitkeep only)
├── docs/research_report.md     # 13-section research report
├── figures/                    # 6 PNGs generated by the actual run
├── reports/metrics.json        # every number in this README, machine-readable
├── reports/summary.md          # human-readable run summary
├── scripts/run_study.py        # CLI entry point
├── src/dl_financial_time_series/
└── tests/                      # 34 offline test cases (28 test functions)
```

## 20 · Future work

- Multi-seed repetition (e.g. 10 seeds) with seed-level uncertainty on every metric.
- Regime-conditioned evaluation (pre/post-2022 rate regime) and longer embargo
  sensitivity analysis (1 vs 5 vs 10 rows).
- Sample-weighted / triple-barrier labeling (Lopez de Prado 2018) and probabilistic
  position sizing instead of the 0.5-threshold long/flat rule.
- Richer inputs (cross-asset features, options-implied vol) and a proper
  hyperparameter search *inside* the validation window only.
- DeLong intervals for AUC and deflated Sharpe ratios for the strategy table.

## 21 · Citation

```bibtex
@software{monavvari2026dlfts,
  author  = {Monavvari, Aydin},
  title   = {dl-financial-time-series: tiny deep-learning models for next-day SPY
             direction forecasting under strict walk-forward evaluation},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/aydinmonavvari/dl-financial-time-series}
}
```

See `CITATION.cff`.

## 22 · License

MIT — see `LICENSE`.

## 23 · Acknowledgments

Built as part of a 10-repo quantitative-finance research portfolio. Thanks to the
maintainers of PyTorch, scikit-learn, pandas and yfinance, and to the open
literature on honest backtesting — especially López de Prado (2018), whose
leakage warnings shaped the no-signal control and embargo design in this repo.
