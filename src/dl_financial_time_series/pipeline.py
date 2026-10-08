"""End-to-end walk-forward study: data -> features -> folds -> models -> reports/figures.

Protocol (strict, no leakage):
- binary next-day SPY direction; features at day t use data <= t only;
- 3 expanding-window walk-forward folds, 1-row embargo at every segment boundary;
- StandardScaler fitted on each fold's training rows only;
- minibatch shuffling inside the training window only (never across time);
- early stopping (patience 5) on validation loss; best weights restored;
- per-fold AND pooled out-of-sample metrics; directional strategy net of 10 bps
  per-side costs vs buy&hold; no-signal control on pure-noise features;
  feature-group ablation.
"""

from __future__ import annotations

import json
import logging
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from dl_financial_time_series import config  # noqa: E402
from dl_financial_time_series.controls import ablation_designs, noise_feature_matrix  # noqa: E402
from dl_financial_time_series.data import ensure_spy_data, load_spy_csv  # noqa: E402
from dl_financial_time_series.features import (  # noqa: E402
    FEATURES,
    build_study_frame,
    build_windows,
    windows_at_targets,
)
from dl_financial_time_series.metrics import (  # noqa: E402
    buy_hold_returns,
    classification_metrics,
    equity_curve,
    max_drawdown,
    positions_from_scores,
    sharpe_ratio,
    strategy_net_returns,
    total_return,
)
from dl_financial_time_series.models import (  # noqa: E402
    build_model,
    count_parameters,
    predict_proba,
    train_model,
)
from dl_financial_time_series.walkforward import fit_transform_scaler, make_folds  # noqa: E402

logger = logging.getLogger(__name__)

RIDGE_THRESHOLD = 0.0  # ridge predicts the next return; direction = sign of prediction


# --------------------------------------------------------------------------- folds
def _fold_arrays(matrix: np.ndarray, fold, seq_len: int):
    """Scaled matrix + sequence arrays for one fold (scaler fitted on train only)."""
    scaled = fit_transform_scaler(matrix, fold)
    x_seq, targets = build_windows(scaled, seq_len)

    def seg(idx: np.ndarray):
        keep = idx[idx >= targets[0]]  # first seq_len-1 rows cannot form full windows
        return windows_at_targets(x_seq, targets, keep), keep

    x_tr, tr_t = seg(fold.train_idx)
    x_va, va_t = seg(fold.val_idx)
    x_te, te_t = seg(fold.test_idx)
    return x_tr, x_va, x_te, tr_t, va_t, te_t, scaled


def _train_dl_on_fold(name: str, fd: dict, seed: int, epochs: int):
    model = build_model(name, fd["x_tr"].shape[2])
    model, history = train_model(
        model,
        fd["x_tr"],
        fd["y_tr"],
        fd["x_va"],
        fd["y_va"],
        epochs=epochs,
        seed=seed,
    )
    return predict_proba(model, fd["x_te"]), history


def _baseline_probas(name: str, fd: dict, seed: int, scaled: np.ndarray) -> np.ndarray:
    if name == "majority":
        p = float(np.mean(fd["y_tr"]))
        return np.full(len(fd["y_te"]), p)
    if name == "logistic":
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(scaled[fd["tr_t"]], fd["y_tr"])
        return clf.predict_proba(scaled[fd["te_t"]])[:, 1]
    if name == "ridge":
        reg = Ridge(alpha=1.0)
        reg.fit(scaled[fd["tr_t"]], fd["next_ret_tr"])
        return reg.predict(scaled[fd["te_t"]])
    raise ValueError(f"unknown baseline: {name}")


# --------------------------------------------------------------------------- study
def run_study(
    seed: int = config.SEED,
    epochs: int = config.MAX_EPOCHS,
    n_folds: int = config.N_FOLDS,
    data_path: str | None = None,
) -> dict:
    """Run the full study; write reports/metrics.json, reports/summary.md, figures/."""
    t0 = time.time()
    torch.set_num_threads(1)  # tiny models: 1 thread is faster and deterministic
    np.random.seed(seed)
    torch.manual_seed(seed)

    csv = data_path or str(ensure_spy_data(config.DATA_RAW_DIR))
    raw = load_spy_csv(csv)
    frame = build_study_frame(raw)
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.DATA_PROCESSED_DIR / "study_frame.csv")

    matrix = frame[list(FEATURES)].to_numpy(float)
    y_all = frame["label"].to_numpy(float)
    next_ret_all = frame["next_ret"].to_numpy(float)
    n = len(frame)
    folds = make_folds(n, n_folds=n_folds)
    cost = config.COST_BPS_PER_SIDE / 1e4

    prepared = []
    for fold in folds:
        x_tr, x_va, x_te, tr_t, va_t, te_t, scaled = _fold_arrays(matrix, fold, config.SEQ_LEN)
        prepared.append(
            {
                "fold": fold,
                "x_tr": x_tr,
                "x_va": x_va,
                "x_te": x_te,
                "y_tr": y_all[tr_t],
                "y_va": y_all[va_t],
                "y_te": y_all[te_t],
                "tr_t": tr_t,
                "te_t": te_t,
                "next_ret_tr": next_ret_all[tr_t],
                "next_ret_te": next_ret_all[te_t],
                "scaled": scaled,
            }
        )
        logger.info(
            "%s: train=%d val=%d test=%d", fold.name, len(tr_t), len(va_t), len(te_t)
        )

    # --- deep learning models (per fold) --------------------------------------
    results: dict[str, list[np.ndarray]] = {}
    histories: dict[str, dict] = {}
    for name in config.MODELS:
        results[name] = []
        histories[name] = {}
        for i, fd in enumerate(prepared, start=1):
            proba, hist = _train_dl_on_fold(name, fd, seed, epochs)
            results[name].append(proba)
            histories[name][f"fold{i}"] = hist
            best_val_loss = min(h["val_loss"] for h in hist)
            print(f"  trained {name} {fd['fold'].name}: best val loss {best_val_loss:.4f}")

    # --- classical baselines (per fold) ---------------------------------------
    for name in ("logistic", "ridge", "majority"):
        results[name] = [_baseline_probas(name, fd, seed, fd["scaled"]) for fd in prepared]

    # --- pooled out-of-sample arrays ------------------------------------------
    pooled_y = np.concatenate([fd["y_te"] for fd in prepared])
    pooled_ret = np.concatenate([fd["next_ret_te"] for fd in prepared])
    pooled_dates = frame.index[np.concatenate([fd["te_t"] for fd in prepared])]

    per_fold: dict[str, list[dict]] = {}
    pooled: dict[str, dict] = {}
    for method, probas in results.items():
        thr = RIDGE_THRESHOLD if method == "ridge" else 0.5
        per_fold[method] = [
            {"fold": fd["fold"].name, **classification_metrics(fd["y_te"], pr, threshold=thr)}
            for fd, pr in zip(prepared, probas, strict=True)
        ]
        pooled[method] = classification_metrics(
            pooled_y, np.concatenate(probas), threshold=thr
        )

    # --- directional strategy net of costs vs buy&hold ------------------------
    strategy: dict[str, dict] = {}
    equity_paths: dict[str, np.ndarray] = {}
    for method, probas in results.items():
        thr = RIDGE_THRESHOLD if method == "ridge" else 0.5
        pos = positions_from_scores(np.concatenate(probas), threshold=thr)
        net = strategy_net_returns(pos, pooled_ret, cost)
        eq = equity_curve(net)
        equity_paths[method] = eq
        strategy[method] = {
            "total_return": total_return(eq),
            "sharpe": sharpe_ratio(net),
            "max_drawdown": max_drawdown(eq),
        }
    bh_net = buy_hold_returns(pooled_ret, cost)
    bh_eq = equity_curve(bh_net)
    equity_paths["buy_hold"] = bh_eq
    strategy["buy_hold"] = {
        "total_return": total_return(bh_eq),
        "sharpe": sharpe_ratio(bh_net),
        "max_drawdown": max_drawdown(bh_eq),
    }

    # --- control: no-signal (pure-noise features) ------------------------------
    print("  running no-signal control (noise features)...")
    noise = noise_feature_matrix(n, matrix.shape[1], seed + 1)
    control: dict[str, dict] = {}
    for name in config.MODELS:
        probas = []
        for fold in folds:
            x_tr, x_va, x_te, tr_t, va_t, te_t, _ = _fold_arrays(noise, fold, config.SEQ_LEN)
            fd = {
                "x_tr": x_tr, "x_va": x_va, "x_te": x_te,
                "y_tr": y_all[tr_t], "y_va": y_all[va_t], "y_te": y_all[te_t],
            }
            proba, _ = _train_dl_on_fold(name, fd, seed, epochs)
            probas.append(proba)
        control[name] = classification_metrics(pooled_y, np.concatenate(probas))

    # --- control: feature-group ablation (GRU as the fixed vehicle) ------------
    print("  running feature-group ablation (GRU)...")
    ablation: dict[str, dict] = {}
    designs = ablation_designs(list(FEATURES), _groups())
    for group, kept in designs.items():
        cols = [list(FEATURES).index(f) for f in kept]
        sub = matrix[:, cols]
        probas = []
        for fold in folds:
            x_tr, x_va, x_te, tr_t, va_t, te_t, _ = _fold_arrays(sub, fold, config.SEQ_LEN)
            fd = {
                "x_tr": x_tr, "x_va": x_va, "x_te": x_te,
                "y_tr": y_all[tr_t], "y_va": y_all[va_t], "y_te": y_all[te_t],
            }
            proba, _ = _train_dl_on_fold("gru", fd, seed, epochs)
            probas.append(proba)
        ablation[group] = classification_metrics(pooled_y, np.concatenate(probas))

    metrics = {
        "meta": {
            "run_date": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
            "seed": seed,
            "epochs": epochs,
            "seq_len": config.SEQ_LEN,
            "n_folds": n_folds,
            "embargo_rows": config.EMBARGO_DAYS,
            "cost_bps_per_side": config.COST_BPS_PER_SIDE,
            "rows_raw": len(raw),
            "rows_usable": n,
            "date_start": str(frame.index[0].date()),
            "date_end": str(frame.index[-1].date()),
            "torch_version": torch.__version__,
            "runtime_seconds": None,  # filled below
        },
        "folds": [_fold_meta(fd, frame) for fd in prepared],
        "model_params": {
            name: count_parameters(build_model(name, matrix.shape[1])) for name in config.MODELS
        },
        "per_fold": per_fold,
        "pooled": pooled,
        "strategy": strategy,
        "no_signal_control": control,
        "ablation": ablation,
        "ablation_full_gru": {"pooled": pooled["gru"]},
        "training_history": histories,
    }
    metrics["meta"]["runtime_seconds"] = round(time.time() - t0, 1)

    # --- outputs ----------------------------------------------------------------
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _write_figures(
        results, prepared, pooled_y, pooled_ret, pooled_dates,
        histories, control, ablation, equity_paths,
    )
    _write_summary(metrics, config.REPORTS_DIR / "summary.md")
    with open(config.REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\nstudy done in {metrics['meta']['runtime_seconds']}s")
    print(f"reports: {config.REPORTS_DIR}")
    return metrics


def _groups() -> dict[str, list[str]]:
    """Feature groups (kept here to avoid a circular import at module scope)."""
    from dl_financial_time_series.features import FEATURE_GROUPS

    return FEATURE_GROUPS


# --------------------------------------------------------------------------- figures
def _roc_axis(ax, y: np.ndarray, probas: dict[str, np.ndarray], title: str) -> None:
    for label, p in probas.items():
        if p is None or len(np.unique(p)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=1.2, label=f"{label} (AUC={roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="lower right")


def _write_figures(
    results,
    prepared,
    pooled_y,
    pooled_ret,
    pooled_dates,
    histories,
    control,
    ablation,
    equity_paths,
) -> None:
    out = config.FIGURES_DIR

    # 1. training curves (fold 1, all four architectures)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, name in zip(axes.ravel(), config.MODELS, strict=True):
        hist = histories[name]["fold1"]
        ax.plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist], label="train")
        ax.plot([h["epoch"] for h in hist], [h["val_loss"] for h in hist], label="val")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel("BCE loss")
        ax.legend(fontsize=8)
    fig.suptitle("Training curves — fold 1 (early stopping on validation loss)")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)

    # 2. per-fold ROC
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (i, fd) in zip(axes, enumerate(prepared, start=1), strict=True):
        probas = {m: results[m][i - 1] for m in (*config.MODELS, "logistic")}
        _roc_axis(ax, fd["y_te"], probas, f"{fd['fold'].name} OOS")
    fig.suptitle("Per-fold out-of-sample ROC")
    fig.tight_layout()
    fig.savefig(out / "per_fold_roc.png", dpi=150)
    plt.close(fig)

    # 3. pooled ROC
    fig, ax = plt.subplots(figsize=(7, 5.5))
    probas = {m: np.concatenate(results[m]) for m in (*config.MODELS, "logistic", "ridge")}
    _roc_axis(ax, pooled_y, probas, f"Pooled out-of-sample ROC (n={len(pooled_y)})")
    fig.tight_layout()
    fig.savefig(out / "pooled_roc.png", dpi=150)
    plt.close(fig)

    # 4. equity curves net of costs vs buy&hold
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in (*config.MODELS, "logistic"):
        ax.plot(pooled_dates, equity_paths[method], lw=1.0, label=f"{method} (net)")
    ax.plot(pooled_dates, equity_paths["buy_hold"], "k--", lw=1.2, label="buy&hold SPY (net)")
    title = f"Directional strategy vs buy&hold — net of {config.COST_BPS_PER_SIDE:.0f} bps/side"
    ax.set_title(title)
    ax.set_ylabel("growth of 1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "equity_curves.png", dpi=150)
    plt.close(fig)

    # 5. no-signal control
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(config.MODELS))
    width = 0.35
    real = [pooled_auc(results, pooled_y, m) for m in config.MODELS]
    noise = [control[m]["roc_auc"] for m in config.MODELS]
    ax.bar(x - width / 2, real, width, label="real features")
    ax.bar(x + width / 2, noise, width, label="pure-noise features")
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.text(len(config.MODELS) - 0.5, 0.505, "chance", fontsize=8)
    ax.set_xticks(x, config.MODELS)
    ax.set_ylabel("pooled OOS ROC-AUC")
    ax.set_title("No-signal control: models on noise features must be chance-level")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "nosignal_control.png", dpi=150)
    plt.close(fig)

    # 6. feature-group ablation
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["full features"] + [f"drop {g}" for g in ablation]
    values = [pooled_auc(results, pooled_y, "gru")] + [ablation[g]["roc_auc"] for g in ablation]
    ax.barh(labels, values, color="#4C72B0")
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlabel("pooled OOS ROC-AUC (GRU)")
    ax.set_title("Feature-group ablation (GRU as fixed vehicle)")
    for i, v in enumerate(values):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "feature_ablation.png", dpi=150)
    plt.close(fig)


def pooled_auc(results: dict, pooled_y: np.ndarray, method: str) -> float:
    """Pooled OOS ROC-AUC for a method (used by figures)."""
    return float(roc_auc_score(pooled_y, np.concatenate(results[method])))


# --------------------------------------------------------------------------- summary
def _fmt_ci(ci: list[float] | None) -> str:
    return "n/a" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _fmt_auc(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def _fold_meta(fd: dict, frame: pd.DataFrame) -> dict:
    """Human-readable date ranges and sample counts for one prepared fold."""

    def rng(idx: np.ndarray) -> list[str]:
        return [str(frame.index[idx[0]].date()), str(frame.index[idx[-1]].date())]

    return {
        "name": fd["fold"].name,
        "train_days": rng(fd["tr_t"]),
        "val_days": rng(fd["fold"].val_idx),
        "test_days": rng(fd["te_t"]),
        "n_train": int(len(fd["y_tr"])),
        "n_val": int(len(fd["y_va"])),
        "n_test": int(len(fd["y_te"])),
    }


def _write_summary(metrics: dict, path) -> None:
    meta, pooled, strategy = metrics["meta"], metrics["pooled"], metrics["strategy"]
    lines: list[str] = []
    lines.append("# dl-financial-time-series — study summary\n")
    lines.append(
        f"Run date: {meta['run_date']} · seed {meta['seed']} · torch {meta['torch_version']} · "
        f"runtime {meta['runtime_seconds']}s\n"
    )
    lines.append(
        f"Data: SPY daily {meta['date_start']} → {meta['date_end']} · "
        f"{meta['rows_raw']} raw rows → {meta['rows_usable']} usable (post-warmup) · "
        f"seq_len {meta['seq_len']} · {meta['n_folds']} expanding folds · "
        f"embargo {meta['embargo_rows']} row(s) · costs {meta['cost_bps_per_side']:.0f} bps/side\n"
    )
    lines.append("## Folds\n")
    lines.append("| fold | train (days) | val (days) | test (days) | n_train | n_val | n_test |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for f in metrics["folds"]:
        cells = (
            f"{f['train_days'][0]}→{f['train_days'][1]}",
            f"{f['val_days'][0]}→{f['val_days'][1]}",
            f"{f['test_days'][0]}→{f['test_days'][1]}",
        )
        lines.append(
            f"| {f['name']} | {cells[0]} | {cells[1]} | {cells[2]} "
            f"| {f['n_train']} | {f['n_val']} | {f['n_test']} |"
        )
    lines.append("\n## Pooled out-of-sample results\n")
    lines.append(
        "| method | params | accuracy | acc CI (Wilson) | ROC-AUC | AUC CI (Wilson, approx) |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    params = metrics["model_params"]
    for method in (*config.MODELS, "logistic", "ridge", "majority"):
        p = pooled[method]
        ptxt = str(params[method]) if method in params else "—"
        lines.append(
            f"| {method} | {ptxt} | {p['accuracy']:.4f} | {_fmt_ci(p['accuracy_ci95'])} "
            f"| {_fmt_auc(p['roc_auc'])} | {_fmt_ci(p.get('roc_auc_ci95'))} |"
        )
    lines.append("\n## Per-fold OOS ROC-AUC\n")
    lines.append("| method | " + " | ".join(f["name"] for f in metrics["folds"]) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in metrics["folds"]) + " |")
    for method, rows in metrics["per_fold"].items():
        lines.append(f"| {method} | " + " | ".join(_fmt_auc(r["roc_auc"]) for r in rows) + " |")
    lines.append("\n## Directional strategy net of costs (pooled OOS, long/flat)\n")
    lines.append("| method | total return | Sharpe (ann.) | max drawdown |")
    lines.append("| --- | --- | --- | --- |")
    for method, s in strategy.items():
        sh = "n/a" if s["sharpe"] is None else f"{s['sharpe']:.2f}"
        lines.append(f"| {method} | {s['total_return']:+.2%} | {sh} | {s['max_drawdown']:+.2%} |")
    bh = strategy["buy_hold"]["total_return"]
    lines.append(
        f"\nBuy&hold total return (net): {bh:+.2%}; "
        "every active strategy is compared against this.\n"
    )
    lines.append("## No-signal control (pure-noise features, same protocol)\n")
    lines.append("| model | pooled ROC-AUC on noise | pooled ROC-AUC on real features |")
    lines.append("| --- | --- | --- |")
    for m in config.MODELS:
        noise_auc = _fmt_auc(metrics["no_signal_control"][m]["roc_auc"])
        real_auc = _fmt_auc(pooled[m]["roc_auc"])
        lines.append(f"| {m} | {noise_auc} | {real_auc} |")
    lines.append("\n## Feature-group ablation (GRU)\n")
    lines.append("| design | pooled ROC-AUC |")
    lines.append("| --- | --- |")
    lines.append(f"| full features | {_fmt_auc(pooled['gru']['roc_auc'])} |")
    for g, p in metrics["ablation"].items():
        lines.append(f"| drop {g} | {_fmt_auc(p['roc_auc'])} |")
    lines.append("\nAll numbers are actual outputs of the committed pipeline run "
                 "(see `reports/metrics.json`).\n")
    path.write_text("\n".join(lines), encoding="utf-8")
