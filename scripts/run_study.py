"""CLI entry point for the full walk-forward deep-learning study."""

from __future__ import annotations

import argparse

from dl_financial_time_series import config
from dl_financial_time_series.pipeline import run_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=config.SEED, help="global seed")
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS, help="max training epochs")
    parser.add_argument("--folds", type=int, default=config.N_FOLDS, help="walk-forward folds")
    parser.add_argument("--data", type=str, default=None, help="path to an existing SPY daily CSV")
    args = parser.parse_args()

    metrics = run_study(seed=args.seed, epochs=args.epochs, n_folds=args.folds, data_path=args.data)

    print("\nPooled out-of-sample results:")
    for method, p in metrics["pooled"].items():
        auc = "n/a" if p["roc_auc"] is None else f"{p['roc_auc']:.4f}"
        print(f"  {method:12s} accuracy={p['accuracy']:.4f}  roc_auc={auc}")
    print(f"\nReports written to {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()
