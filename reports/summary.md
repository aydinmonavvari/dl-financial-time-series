# dl-financial-time-series — study summary

Run date: 2026-10-08 · seed 42 · torch 2.14.1+cpu · runtime 75.4s

Data: SPY daily 2010-10-18 → 2026-10-06 · 4216 raw rows → 4016 usable (post-warmup) · seq_len 30 · 3 expanding folds · embargo 1 row(s) · costs 10 bps/side

## Folds

| fold | train (days) | val (days) | test (days) | n_train | n_val | n_test |
| --- | --- | --- | --- | --- | --- | --- |
| fold1 | 2010-11-29→2018-10-08 | 2018-10-10→2019-07-30 | 2019-08-01→2021-04-29 | 1979 | 201 | 440 |
| fold2 | 2010-11-29→2021-04-29 | 2021-05-03→2022-02-15 | 2022-02-17→2023-11-15 | 2622 | 201 | 439 |
| fold3 | 2010-11-29→2023-11-15 | 2023-11-17→2024-09-06 | 2024-09-10→2026-10-06 | 3264 | 201 | 520 |

## Pooled out-of-sample results

| method | params | accuracy | acc CI (Wilson) | ROC-AUC | AUC CI (Wilson, approx) |
| --- | --- | --- | --- | --- | --- |
| lstm | 6049 | 0.5468 | [0.521, 0.573] | 0.4569 | [0.456, 0.458] |
| gru | 4545 | 0.5375 | [0.511, 0.564] | 0.4704 | [0.469, 0.472] |
| cnn | 7521 | 0.5432 | [0.517, 0.569] | 0.4807 | [0.479, 0.482] |
| transformer | 9025 | 0.5368 | [0.511, 0.563] | 0.4807 | [0.479, 0.482] |
| logistic | — | 0.5375 | [0.511, 0.564] | 0.4802 | [0.479, 0.482] |
| ridge | — | 0.4968 | [0.471, 0.523] | 0.4969 | [0.495, 0.498] |
| majority | — | 0.5432 | [0.517, 0.569] | 0.4723 | [0.471, 0.474] |

## Per-fold OOS ROC-AUC

| method | fold1 | fold2 | fold3 |
| --- | --- | --- | --- |
| lstm | 0.5355 | 0.4874 | 0.4431 |
| gru | 0.5085 | 0.5005 | 0.4555 |
| cnn | 0.4829 | 0.4820 | 0.4895 |
| transformer | 0.4758 | 0.5029 | 0.5134 |
| logistic | 0.5228 | 0.4671 | 0.4999 |
| ridge | 0.5626 | 0.4669 | 0.5132 |
| majority | n/a | n/a | n/a |

## Directional strategy net of costs (pooled OOS, long/flat)

| method | total return | Sharpe (ann.) | max drawdown |
| --- | --- | --- | --- |
| lstm | +98.55% | 0.69 | -35.75% |
| gru | +75.54% | 0.59 | -35.75% |
| cnn | +97.77% | 0.68 | -35.75% |
| transformer | +101.54% | 0.71 | -35.75% |
| logistic | +62.15% | 0.52 | -35.61% |
| ridge | -16.56% | -0.09 | -35.89% |
| majority | +97.77% | 0.68 | -35.75% |
| buy_hold | +97.77% | 0.68 | -35.75% |

Buy&hold total return (net): +97.77%; every active strategy is compared against this.

## No-signal control (pure-noise features, same protocol)

| model | pooled ROC-AUC on noise | pooled ROC-AUC on real features |
| --- | --- | --- |
| lstm | 0.4782 | 0.4569 |
| gru | 0.4739 | 0.4704 |
| cnn | 0.4699 | 0.4807 |
| transformer | 0.5075 | 0.4807 |

## Feature-group ablation (GRU)

| design | pooled ROC-AUC |
| --- | --- |
| full features | 0.4704 |
| drop momentum | 0.4961 |
| drop volatility | 0.4667 |
| drop volume | 0.4727 |
| drop technical | 0.4750 |

All numbers are actual outputs of the committed pipeline run (see `reports/metrics.json`).
