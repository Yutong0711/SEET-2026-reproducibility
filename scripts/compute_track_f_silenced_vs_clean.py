"""Compute delta_vs_clean for Track F's validation-gap corruptions.

For C3 (stale_quotes), C4 (extreme_jumps), C5 (calendar_gaps) — the
three corruption types the v1 architecture's V1+V2 validators
cannot detect — the SILENCED branch carries the corruption straight
through. Track F's standard ENABLED-vs-SILENCED contrast yields
delta = 0 for these by construction (no validator action ⇒ both
branches are identical), which the paper-relevant interpretation
notes as "validation gap". But the paper also needs to quantify
what the *operational cost* of the uncaught corruption is — i.e.,
how much the SILENCED-corrupted-panel run differs from a clean
Track A run on the same (fold, model, seed) cell.

This script computes:

    delta_vs_clean = metric(SILENCED, corrupted) - metric(Track A, clean)

per (corruption_type in {C3, C4, C5}, rate, model, fold_id,
lgbm_seed, injection_seed) cell, paired against
experiments/track_a_headline/per_fold_metrics.csv (h10_d05). The
script aggregates per (corruption, rate, model, metric) using the
paired bootstrap from seet.stats with 10k resamples (fold-level,
preserving pairing) and paired Wilcoxon p-values.

Outputs:
    outputs/track_f/table_silenced_vs_clean.csv
        cols: corruption_type, rate, model, metric, delta_mean,
              delta_ci_low, delta_ci_high, wilcoxon_pvalue, n_pairs

A summary is printed to stdout. Idempotent — overwrites the
table; doesn't modify silenced_impact.csv or the workdir.

Usage:
    python scripts/compute_track_f_silenced_vs_clean.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from seet.stats import paired_bootstrap_ci, paired_wilcoxon  # noqa: E402

WORKDIR = REPO_ROOT / "experiments" / "track_f_injection" / "_workdir"
TRACK_A_METRICS = REPO_ROOT / "experiments" / "track_a_headline" / "per_fold_metrics.csv"
OUT_DIR = REPO_ROOT / "outputs" / "track_f"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GAP_CORRUPTIONS = ("stale_quotes", "extreme_jumps", "calendar_gaps")
RATES = (0.01, 0.05, 0.10)
INJECTION_SEEDS = (42, 43, 44, 45, 46)
MODELS = ("LogisticRegressionL2", "LightGbmTuned")
METRICS = ("auc", "pr_auc", "brier", "drawdown_lift", "alarm_rate")


def main() -> int:
    if not TRACK_A_METRICS.exists():
        sys.stderr.write(
            f"ERROR: Track A baseline not found at {TRACK_A_METRICS}.\n"
        )
        return 1
    track_a = pd.read_csv(TRACK_A_METRICS)
    track_a = track_a[track_a["stress_def"] == "h10_d05"].copy()
    track_a_idx = track_a.set_index(["model", "fold_id", "seed"])

    rows: list[dict] = []
    for cid in GAP_CORRUPTIONS:
        for rate in RATES:
            # Per-cell paired (silenced, track_a) arrays per (model, metric).
            paired_per_model_metric: dict[
                tuple[str, str], dict[str, list[float]]
            ] = {(m, k): {"a": [], "b": []} for m in MODELS for k in METRICS}
            for seed in INJECTION_SEEDS:
                cell = WORKDIR / f"{cid}_{rate:.2f}_{seed}" / "silenced" / "per_fold_metrics.csv"
                if not cell.exists():
                    sys.stderr.write(
                        f"WARN: missing {cell.relative_to(REPO_ROOT)}; skipping cell\n"
                    )
                    continue
                si = pd.read_csv(cell)
                si = si[si["stress_def"] == "h10_d05"]
                si_idx = si.set_index(["model", "fold_id", "seed"])
                # Align with track_a on (model, fold_id, seed).
                common = si_idx.index.intersection(track_a_idx.index)
                si_aligned = si_idx.loc[common]
                ta_aligned = track_a_idx.loc[common]
                for model in MODELS:
                    if model not in si_aligned.index.get_level_values("model"):
                        continue
                    si_m = si_aligned.xs(model, level="model").sort_index()
                    ta_m = ta_aligned.xs(model, level="model").sort_index()
                    for metric in METRICS:
                        if metric not in si_m.columns or metric not in ta_m.columns:
                            continue
                        a = pd.to_numeric(si_m[metric], errors="coerce").to_numpy(dtype=float)
                        b = pd.to_numeric(ta_m[metric], errors="coerce").to_numpy(dtype=float)
                        if a.size == 0:
                            continue
                        paired_per_model_metric[(model, metric)]["a"].extend(a.tolist())
                        paired_per_model_metric[(model, metric)]["b"].extend(b.tolist())

            for (model, metric), arr_dict in paired_per_model_metric.items():
                a = np.array(arr_dict["a"], dtype=float)
                b = np.array(arr_dict["b"], dtype=float)
                if a.size == 0:
                    continue
                ci = paired_bootstrap_ci(a, b, seed=42, n_boot=10000)
                wlx = paired_wilcoxon(a, b)
                rows.append({
                    "corruption_type": cid,
                    "rate": float(rate),
                    "model": model,
                    "metric": metric,
                    "delta_mean": ci["delta_mean"],
                    "delta_ci_low": ci["ci_low"],
                    "delta_ci_high": ci["ci_high"],
                    "wilcoxon_pvalue": wlx["p_value"],
                    "n_pairs": ci["n_pairs"],
                })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "table_silenced_vs_clean.csv", index=False)
    print()
    print("=" * 78)
    print("Track F — silenced vs clean: operational cost of uncaught corruption")
    print("=" * 78)
    print(f"\nWritten: outputs/track_f/table_silenced_vs_clean.csv  ({len(out)} rows)")

    print("\nFor C3 (stale_quotes), C4 (extreme_jumps), C5 (calendar_gaps): "
          "by how much\ndo the SILENCED metrics on the corrupted panel "
          "differ from the Track A clean\nbaseline on the same "
          "(fold, model, seed) cell?\n")
    print("Showing AUC and drawdown_lift cells where |delta| > 0.005 OR "
          "p < 0.05:")
    show = out[out["metric"].isin(("auc", "drawdown_lift"))].copy()
    show = show[
        (show["delta_mean"].abs() > 0.005)
        | (show["wilcoxon_pvalue"] < 0.05)
    ].sort_values(["corruption_type", "rate", "metric", "model"])
    if show.empty:
        print("  none — all uncaught corruptions produce CONSISTENT operational metrics\n"
              "  on the matrix-feature models (the corrupted panel runs "
              "indistinguishably\n  from the clean panel within paired-bootstrap noise).")
    else:
        print(f"  {'corruption':<18} {'rate':>5} {'model':<26} "
              f"{'metric':<14} {'delta':>10} {'CI':>22} {'p':>8}")
        print("  " + "-" * 100)
        for _, r in show.iterrows():
            d = r["delta_mean"]
            sig = "*" if (
                not pd.isna(r["wilcoxon_pvalue"])
                and r["wilcoxon_pvalue"] < 0.05
            ) else " "
            print(
                f"  {r['corruption_type']:<18} {r['rate']:>5.0%} "
                f"{r['model']:<26} {r['metric']:<14} {d:+10.4f} "
                f"[{r['delta_ci_low']:+.4f},{r['delta_ci_high']:+.4f}] "
                f"{r['wilcoxon_pvalue']:>7.3f}{sig}"
            )

    print("\nFull table for the gap corruptions × {AUC, drawdown_lift} "
          "(NaN-safe):")
    pretty = out[out["metric"].isin(("auc", "drawdown_lift"))].sort_values(
        ["corruption_type", "rate", "metric", "model"]
    )
    print(f"\n  {'corruption':<18} {'rate':>5} {'model':<26} "
          f"{'metric':<14} {'delta':>10} {'p':>8} {'n':>5}")
    print("  " + "-" * 90)
    for _, r in pretty.iterrows():
        d = r["delta_mean"]
        d_str = f"{d:+10.4f}" if not pd.isna(d) else "       n/a"
        p_str = (
            f"{r['wilcoxon_pvalue']:>7.3f}"
            if not pd.isna(r["wilcoxon_pvalue"]) else "    n/a"
        )
        print(
            f"  {r['corruption_type']:<18} {r['rate']:>5.0%} "
            f"{r['model']:<26} {r['metric']:<14} {d_str} {p_str} "
            f"{int(r['n_pairs']):>5}"
        )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
