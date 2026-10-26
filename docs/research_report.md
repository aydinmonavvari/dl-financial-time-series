# dl-financial-time-series — Research Report

*Tiny deep-learning architectures for next-day SPY direction forecasting under
strict walk-forward evaluation*

Author: Aydin Monavvari · Date: 2026-10-08 · Version: 1.0.0

## Abstract

We evaluate whether four small deep sequence models — an LSTM, a GRU, a temporal
1D-CNN, and a one-layer Transformer encoder, each under 10k trainable parameters —
improve next-day direction forecasting of the S&P 500 ETF (SPY) over simple
baselines. The protocol is deliberately strict: three expanding-window walk-forward
folds with a one-row embargo at every segment boundary, feature scaling fitted on
training rows only, no shuffling across time, fixed seeds, early stopping on a
validation window, and two leakage controls (a no-signal control on pure-noise
features and a feature-group ablation). On 1,399 pooled out-of-sample test days
(2019–2026), no model exceeds the majority baseline in a statistically meaningful
way: accuracies fall in a narrow 0.50–0.55 band with heavily overlapping Wilson
intervals, and every pooled ROC-AUC is at or below 0.5 (best deep model: 0.4807).
The no-signal control confirms the machinery is leak-free (all noise-feature AUCs
0.470–0.508). A long/flat strategy net of 10 bps per-side costs roughly matches
buy&hold (+101.5% vs +97.8% for the best model, with identical drawdowns) — a
difference within noise. We conclude that, under an honest protocol, tiny deep
sequence models add nothing to next-day SPY direction prediction on this data, and
we report this null result as the study's primary finding.

## Introduction

Next-day equity direction forecasting is the canonical "hard" problem of applied
machine learning in finance. The signal-to-noise ratio of daily returns is
extremely low, the label is only resolved one day ahead (creating subtle leakage
paths at every segment boundary), and non-stationarity means that any estimated
regularity may decay before it is exploited. The deep-learning literature applied
to markets frequently reports accuracies of 55–65% that later prove to be artifacts
of shuffled cross-validation, whole-series normalization, or cost-free trading
assumptions.

This study takes the opposite approach. It fixes the evaluation first —
walk-forward folds, embargo, train-only scaling, seeded determinism, leakage
controls, transaction costs — and only then asks whether the models beat trivial
baselines. The design deliberately uses *tiny* models (a hard 100k-parameter cap;
actual sizes 4.5k–9k parameters) so that the question "do sequence architectures
extract directional signal from 13 causal daily features?" is answered without
confounding the answer with model scale or compute.

## Research Question

> Evaluating whether small deep learning architectures (LSTM, GRU, temporal 1D-CNN,
> a tiny Transformer encoder) improve next-day S&P 500 ETF (SPY) direction
> forecasting over simple baselines, under strict walk-forward evaluation.

Operationalized as: does any architecture achieve pooled out-of-sample ROC-AUC or
accuracy (Wilson 95% CI) significantly above the majority-class baseline, and does
a long/flat directional strategy net of 10 bps per-side costs outperform buy&hold?

## Related Work

- **Recurrent architectures.** LSTM (Hochreiter & Schmidhuber 1997) introduced
  gated memory cells to stabilize long-range credit assignment; GRU (Cho et al.
  2014) simplified the gating at lower parameter cost. Both are the default first
  choices for small sequence classifiers.
- **Attention.** The Transformer (Vaswani et al. 2017) replaced recurrence with
  self-attention; here we use a single encoder layer with a causal mask, the
  smallest faithful instantiation of that design.
- **Honest backtesting.** López de Prado (2018) catalogues the leakage and
  multiple-testing failure modes of financial ML; the embargo, walk-forward
  protocol, and skepticism about backtest winners here follow that tradition.
- **Optimization.** Adam (Kingma & Ba 2015) with weight decay is used for all
  training runs.
- **The null hypothesis.** Fama's (1970) efficient-market framing sets the
  benchmark: under weak-form efficiency, public price/volume history should not
  yield exploitable next-day direction. Breiman (2001) reminds us that both
  algorithmic and data-modeling cultures deserve equal, skeptical scrutiny — we
  apply it to the deep models *and* the linear baselines alike.

## Data

- **Instrument:** SPY daily OHLCV, Yahoo Finance via `yfinance`, 2010-01-04 →
  2026-10-06 (4,216 raw rows). Data acquisition is cache-first: the pipeline reuses
  a local CSV cache (this run copied the cache produced by the sibling Project 03
  repository) and only downloads when no cache exists; tests never touch the
  network.
- **Features (13, strictly causal at day t):** log return and 5 lagged returns;
  realized volatility (10d and 21d rolling std of log returns); mean 10d
  high-low range; 21d volume z-score; Wilder-smoothed RSI(14); distance from the
  50d and 200d simple moving averages.
- **Target:** binary next-day direction, `close[t+1] > close[t]`; the final row is
  unlabeled.
- **Usable sample:** 4,016 contiguous rows (2010-10-18 → 2026-10-06) after the
  200-day moving-average warmup; the pipeline asserts contiguity.
- **Class balance:** ≈ 54% UP days — a mild imbalance that already makes "always
  long" a strong baseline.

## Methodology

1. **Causal feature engineering.** Every rolling window ends at day `t`; labels
   are shifted forward by one day; a unit test perturbs the close from day `j`
   onward and asserts features before `j` are unchanged.
2. **Sequence windows.** Length-30 sliding windows over the scaled feature matrix;
   the last row of each window is the decision day itself, so a prediction for
   `t+1` uses only information available at the end of day `t`.
3. **Models.** LSTM (1×32), GRU (1×32), temporal CNN (3 causal conv blocks, kernel
   3, global average pooling), tiny Transformer (1 encoder layer, d_model 32, 2
   heads, FF 64, causal mask). All emit one logit; BCEWithLogitsLoss; dropout 0.2.
   Parameter counts (actual): GRU 4,545; LSTM 6,049; CNN 7,521; Transformer 9,025.
4. **Baselines.** Majority class (train positive rate); logistic regression on the
   13 scaled features; ridge regression predicting the next return (direction =
   sign of the prediction).
5. **Metrics.** Accuracy (Wilson 1927 exact 95% CI), ROC-AUC (approximate Wilson
   interval over concordant pairs — documented as an approximation), long/flat
   strategy returns net of 10 bps per side, Sharpe, max drawdown, versus buy&hold
   net of one entry cost.
6. **Controls.** No-signal control: each architecture retrained on pure Gaussian
   noise features under the identical protocol must be chance-level. Feature-group
   ablation: GRU retrained with each of the four feature groups (momentum,
   volatility, volume, technical) dropped one at a time.

## Experimental Design

- **Walk-forward, chronological, expanding:** three folds; later folds train on
  strictly longer history (standard walk-forward: lived-through periods may be
  used for training subsequent folds).
- **Embargo:** one row is skipped at every train/val and val/test boundary because
  the label of the last row of a segment is resolved by the next day's close;
  without the skip, train labels would reach into evaluation rows.
- **Scaling:** `StandardScaler` fitted on each fold's training rows only, applied
  unchanged to validation/test (asserted by a test that compares against
  hand-computed train statistics).
- **No cross-time shuffling.** Minibatch shuffling is applied *inside* the training
  window only, through a seeded generator; validation and test are scored
  chronologically.
- **Training budget.** Adam lr 1e-3, weight decay 1e-4, batch 64, max 20 epochs,
  early stopping patience 5 on validation loss, best weights restored; seed 42;
  single-threaded CPU torch. Total pipeline runtime: **75.4 s**.
- **Fold geometry (actual):** fold1 train 2010-11-29→2018-10-08, val
  2018-10-10→2019-07-30, test 2019-08-01→2021-04-29 (n=440); fold2 test
  2022-02-17→2023-11-15 (n=439); fold3 test 2024-09-10→2026-10-06 (n=520).
  Validation windows are 201 days each.

## Results

**Pooled out-of-sample, n = 1,399 test days (accuracy; ROC-AUC):**

| method | params | accuracy | acc CI95 (Wilson) | ROC-AUC | AUC CI95 (approx) |
| --- | --- | --- | --- | --- | --- |
| lstm | 6,049 | 0.5468 | [0.521, 0.573] | 0.4569 | [0.456, 0.458] |
| gru | 4,545 | 0.5375 | [0.511, 0.564] | 0.4704 | [0.469, 0.472] |
| cnn | 7,521 | 0.5432 | [0.517, 0.569] | 0.4807 | [0.479, 0.482] |
| transformer | 9,025 | 0.5368 | [0.511, 0.563] | 0.4807 | [0.479, 0.482] |
| logistic | — | 0.5375 | [0.511, 0.564] | 0.4802 | [0.479, 0.482] |
| ridge | — | 0.4968 | [0.471, 0.523] | 0.4969 | [0.495, 0.498] |
| majority | — | 0.5432 | [0.517, 0.569] | n/a (per fold) | — |

Per-fold ROC-AUC: LSTM 0.5355 / 0.4874 / 0.4431; GRU 0.5085 / 0.5005 / 0.4555; CNN
0.4829 / 0.4820 / 0.4895; Transformer 0.4758 / 0.5029 / 0.5134; logistic 0.5228 /
0.4671 / 0.4999; ridge 0.5626 / 0.4669 / 0.5132 (fold1 / fold2 / fold3).

**Directional strategy net of 10 bps/side (pooled OOS, long/flat at the 0.5
threshold; ridge at the 0.0 sign threshold):** transformer +101.54% (Sharpe 0.71);
lstm +98.55% (0.69); cnn/majority +97.77% (0.68); buy&hold +97.77% (0.68); gru
+75.54% (0.59); logistic +62.15% (0.52); ridge −16.56% (−0.09). Max drawdowns are
≈ −35.6% to −35.9% for all long/flat variants, identical to buy&hold's −35.75%.

**No-signal control (pure-noise features):** lstm 0.4782, gru 0.4739, cnn 0.4699,
transformer 0.5075 pooled AUC — all consistent with chance; the evaluation
machinery does not leak.

**Feature-group ablation (GRU, pooled AUC):** full 0.4704; drop momentum 0.4961;
drop volatility 0.4667; drop volume 0.4727; drop technical 0.4750 — no group
carries out-of-sample signal.

## Discussion

The null result is coherent across every lens the study provides. Ranking skill:
all pooled AUCs are ≤ 0.4969 — the models' probabilities do not order outcomes
better than chance, and several are significantly *below* 0.5 (which itself is
consistent with noise plus a mild overfitting-to-train effect). Classification
skill: the best accuracy (LSTM 0.5468) is inside the majority baseline's Wilson
interval [0.517, 0.569]; no model separates from the trivial rule. Consistency:
per-fold AUCs flip direction between folds (LSTM 0.5355 → 0.4431; ridge 0.5626 →
0.4669), the signature of estimation noise rather than a decaying edge. Economics:
because UP days are ≈ 54% of the sample, the majority constant prediction is
"always long", so the majority baseline *is* buy&hold (+97.77% net); the
Transformer's +101.54% amounts to +3.8 percentage points over roughly seven years
with an identical drawdown and near-identical Sharpe — indistinguishable from luck,
especially given its AUC is 0.48. The ridge baseline actively loses money (−16.56%)
after costs, a useful reminder that sign-of-prediction strategies are fragile to
costs even before considering signal.

The controls strengthen the interpretation: the no-signal control shows the
pipeline cannot manufacture AUC from noise, so the real-feature AUCs near 0.5 are
not an artifact of the harness; the ablation shows no single feature group is
carrying (or masking) signal. These results replicate, with a different model
family, the finding of the sibling Project 03 study on the same instrument:
under strict walk-forward evaluation, public daily price/volume features do not
yield a next-day directional edge over trivial baselines.

## Limitations

- **Scale and budget.** Models are deliberately tiny (< 10k params, ≤ 20 epochs,
  75 s CPU total). Larger architectures or longer training could in principle do
  better, though the burden of proof lies with the claim, and the AUC ≈ 0.5
  ceiling suggests the input features, not capacity, are the binding constraint.
- **Single asset and feature universe.** SPY only; 13 daily price/volume features;
  no cross-asset, options, or order-flow information.
- **Single-seed headline.** All numbers use seed 42 with a determinism guarantee;
  seed-level uncertainty is not quantified.
- **Strategy space.** One long/flat threshold rule at 0.5 is backtested; thresholds
  and sizing are not searched (deliberately, to avoid test-set tuning).
- **Approximate AUC intervals.** The Wilson-on-concordant-pairs interval is a
  documented approximation (DeLong would be rigorous).
- **Non-stationarity.** The 2010–2026 span mixes regimes; results are reported
  pooled and per fold, but no regime model is attempted.
- **Pooled majority AUC artifact.** The majority baseline predicts a fold-specific
  constant; its pooled "AUC" (0.4723) is an artifact of concatenating three
  constants and is reported as undefined per fold.

## Conclusion

Under a strictly leak-free walk-forward protocol with embargo, train-only scaling,
seeded training, and cost-adjusted accounting, four tiny deep sequence models
(LSTM, GRU, temporal CNN, tiny Transformer) do **not** improve next-day SPY
direction forecasting over simple baselines. Pooled out-of-sample ROC-AUCs are at
or below 0.5, accuracies are statistically indistinguishable from the majority
class, and the best backtest exceeds buy&hold by a noise-level margin with
identical drawdowns. Leakage controls pass, so the null is credible. The study's
contribution is the protocol itself — a reusable, tested template for honest
sequence-model evaluation on financial data — and a documented negative result that
sets a realistic expectation for future, larger-scale attempts.

## Future Research

- Multi-seed repetition to put seed-level error bars on every metric.
- Richer inputs (cross-asset features, implied volatility, order imbalance) where
  the marginal information content is higher than in daily OHLCV.
- Label engineering from the financial-ML literature: triple-barrier labeling and
  sample weighting (López de Prado 2018), with meta-labeling for position sizing.
- Regime conditioning and longer embargo/sequence-length sensitivity analyses.
- Rigorous AUC inference (DeLong) and deflated Sharpe ratios for the backtest
  table, plus deflation for the multiple architectures compared.

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural
   Computation*, 9(8), 1735–1780.
2. Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk,
   H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder–
   Decoder for Statistical Machine Translation. *EMNLP 2014*, arXiv:1406.1078.
3. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
   Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 30*.
4. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
5. Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization.
   *ICLR 2015*, arXiv:1412.6980.
6. Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical
   Work. *The Journal of Finance*, 25(2), 383–417.
7. Breiman, L. (2001). Statistical Modeling: The Two Cultures. *Statistical
   Science*, 16(3), 199–231.
8. Wilson, E. B. (1927). Probable Inference, the Law of Succession, and Statistical
   Inference. *Journal of the American Statistical Association*, 22(158), 209–212.
