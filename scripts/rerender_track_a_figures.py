"""Re-render Track A figures from on-disk artifacts.

Reads experiments/track_a_headline/per_fold_metrics.csv and
experiments/track_a_headline/predictions/h10_d05.parquet, then calls
the three updated plot functions, writing PDFs over the existing
files in outputs/track_a/. Also prints a STATUS block with PR-AUC
values, the empirical base rate, and the realised reliability bin
counts so they can be cited without re-deriving.

This script does NOT re-fit any model. The per-fold CSVs and prediction
parquets it reads are exactly the ones produced by the most recent
`python src/seet/run_track_a.py --full` run.

Usage:
    python scripts/rerender_track_a_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from seet.run_track_a import (  # noqa: E402
    plot_lift_with_ci,
    plot_pr_curves,
    plot_reliability,
)


def main() -> int:
    exp_dir = REPO_ROOT / "experiments" / "track_a_headline"
    out_dir = REPO_ROOT / "outputs" / "track_a"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_fold_path = exp_dir / "per_fold_metrics.csv"
    pred_path = exp_dir / "predictions" / "h10_d05.parquet"
    if not per_fold_path.exists():
        sys.stderr.write(f"Missing: {per_fold_path}. Run --full first.\n")
        return 1
    if not pred_path.exists():
        sys.stderr.write(f"Missing: {pred_path}. Run --full first.\n")
        return 1

    per_fold = pd.read_csv(per_fold_path)
    predictions_h10 = pd.read_parquet(pred_path)

    plot_lift_with_ci(per_fold, out_dir / "fig_lift_with_ci.pdf")
    pr_info = plot_pr_curves(predictions_h10, out_dir / "fig_pr_curves.pdf")
    rel_info = plot_reliability(predictions_h10, out_dir / "fig_reliability.pdf")

    print()
    print("===== FIGURE RE-RENDER STATUS =====")
    print()
    print("PR-AUC (average precision) values for h=10, d=5%:")
    for model, ap in pr_info["pr_aucs"].items():
        if pd.isna(ap):
            print(f"  {model:<26}  AP = NaN")
        else:
            print(f"  {model:<26}  AP = {ap:.4f}")

    base = pr_info["base_rate"]
    base_str = "NaN" if pd.isna(base) else f"{100.0 * base:.3f}%"
    print()
    print(f"Empirical base rate (h=10, d=5% pooled OOS): {base_str}")

    print()
    print("Reliability bin counts (equal-frequency, h=10, d=5%):")
    for model, counts in rel_info["bin_counts"].items():
        if not counts:
            print(f"  {model:<26}  (degenerate; no bins rendered)")
        else:
            counts_str = ", ".join(str(n) for n in counts)
            print(f"  {model:<26}  n = [{counts_str}]")

    print()
    print("Saved PDFs (over existing files):")
    for name in ("fig_lift_with_ci.pdf", "fig_pr_curves.pdf", "fig_reliability.pdf"):
        print(f"  {out_dir / name}")
    print()
    print("===================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
