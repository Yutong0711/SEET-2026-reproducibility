"""Re-process Track F per-fold CSVs already on disk to surface the
branch-failure finding for missing_values.

Track F's main runner reports `delta_mean = NaN, n_pairs = 0` for
the C2 (missing_values) cells because the SILENCED branch produces
all-NaN metrics (rolling-window NaN propagation from corrupted SPX
cells covers ~every panel row at rate >= 1%, so `_fit_predict_one`
sees fewer than 10 valid training rows and the matrix-feature
models can't fit). The standard paired-bootstrap output collapses
this to "n/a", which understates a much stronger finding: silencing
the validator for C2 turns a working model into total failure.

This script reads the per-cell per-fold CSVs from
experiments/track_f_injection/_workdir/<cid>_<rate>_<seed>/{enabled,
silenced}/per_fold_metrics.csv, computes a `branch_failure_rate` per
(cid, rate, model) cell — the fraction of (fold_id, lgbm_seed) pairs
where ENABLED has a valid metric value but SILENCED is NaN — and
writes outputs/track_f/table_branch_failure.csv plus a small text
summary printed to stdout.

Idempotent: writes to a new file, doesn't modify silenced_impact.csv
or the workdir.

Usage:
    python scripts/rerun_track_f_postprocess.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKDIR = REPO_ROOT / "experiments" / "track_f_injection" / "_workdir"
OUT_DIR = REPO_ROOT / "outputs" / "track_f"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CORRUPTION_IDS = (
    "duplicate_dates",
    "missing_values",
    "stale_quotes",
    "extreme_jumps",
    "calendar_gaps",
)
RATES = (0.01, 0.05, 0.10)
SEEDS = (42, 43, 44, 45, 46)
MODELS = ("LogisticRegressionL2", "LightGbmTuned")
METRICS = ("auc", "pr_auc", "brier", "drawdown_lift", "alarm_rate")


def _load_pair(cid: str, rate: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    cell = WORKDIR / f"{cid}_{rate:.2f}_{seed}"
    e = cell / "enabled" / "per_fold_metrics.csv"
    s = cell / "silenced" / "per_fold_metrics.csv"
    if not e.exists() or not s.exists():
        return None
    return pd.read_csv(e), pd.read_csv(s)


def main() -> int:
    rows: list[dict] = []
    for cid in CORRUPTION_IDS:
        for rate in RATES:
            for seed in SEEDS:
                pair = _load_pair(cid, rate, seed)
                if pair is None:
                    continue
                e_df, s_df = pair
                e_df = e_df.set_index(["model", "fold_id", "seed"])
                s_df = s_df.set_index(["model", "fold_id", "seed"])
                common = e_df.index.intersection(s_df.index)
                e_df = e_df.loc[common]
                s_df = s_df.loc[common]
                for model in MODELS:
                    if model not in e_df.index.get_level_values("model"):
                        continue
                    e_m = e_df.xs(model, level="model")
                    s_m = s_df.xs(model, level="model")
                    for metric in METRICS:
                        if metric not in e_m.columns or metric not in s_m.columns:
                            continue
                        e_vals = e_m[metric].to_numpy(dtype=float)
                        s_vals = s_m[metric].to_numpy(dtype=float)
                        e_valid = ~np.isnan(e_vals)
                        s_valid = ~np.isnan(s_vals)
                        n = e_vals.size
                        rows.append({
                            "corruption_type": cid,
                            "rate": float(rate),
                            "injection_seed": int(seed),
                            "model": model,
                            "metric": metric,
                            "n_obs": int(n),
                            "n_both_valid": int((e_valid & s_valid).sum()),
                            "n_enabled_only_valid": int((e_valid & ~s_valid).sum()),
                            "n_silenced_only_valid": int((~e_valid & s_valid).sum()),
                            "n_both_nan": int((~e_valid & ~s_valid).sum()),
                        })
    if not rows:
        print("No per-fold CSVs found in", WORKDIR)
        return 1

    per_seed = pd.DataFrame(rows)

    # Aggregate across the 5 injection seeds per (cid, rate, model, metric).
    agg = (
        per_seed.groupby(["corruption_type", "rate", "model", "metric"])
        .agg(
            n_obs=("n_obs", "sum"),
            n_both_valid=("n_both_valid", "sum"),
            n_enabled_only_valid=("n_enabled_only_valid", "sum"),
            n_silenced_only_valid=("n_silenced_only_valid", "sum"),
            n_both_nan=("n_both_nan", "sum"),
        )
        .reset_index()
    )
    agg["enabled_valid_rate"] = (
        agg["n_both_valid"] + agg["n_enabled_only_valid"]
    ) / agg["n_obs"]
    agg["silenced_valid_rate"] = (
        agg["n_both_valid"] + agg["n_silenced_only_valid"]
    ) / agg["n_obs"]
    agg["branch_failure_rate"] = (
        agg["n_enabled_only_valid"] / agg["n_obs"]
    )
    agg.to_csv(OUT_DIR / "table_branch_failure.csv", index=False)

    # Print a focused summary.
    print()
    print("=" * 78)
    print("Track F branch-failure post-process")
    print("=" * 78)
    print(f"\nWritten: outputs/track_f/table_branch_failure.csv  "
          f"({len(agg)} rows)")

    print("\nCells where SILENCED branch fails (silenced_valid_rate < 0.5) "
          "but ENABLED branch produces valid metrics (enabled_valid_rate > 0.5):")
    print(f"  {'corruption':<18} {'rate':>5} {'model':<26} {'metric':<14} "
          f"{'enabled_valid':>14} {'silenced_valid':>15} {'br_failure':>12}")
    print("  " + "-" * 110)
    flagged = agg[
        (agg["silenced_valid_rate"] < 0.5)
        & (agg["enabled_valid_rate"] > 0.5)
    ].sort_values(
        ["corruption_type", "rate", "model", "metric"]
    )
    for _, r in flagged.iterrows():
        print(
            f"  {r['corruption_type']:<18} {r['rate']:>5.0%} {r['model']:<26} "
            f"{r['metric']:<14} {r['enabled_valid_rate']:>14.3f} "
            f"{r['silenced_valid_rate']:>15.3f} {r['branch_failure_rate']:>12.3f}"
        )

    if flagged.empty:
        print("  none")

    print("\nFor reference, the same metric in the canonical (no-corruption) "
          "Track A baseline\nproduces enabled_valid_rate ~= 0.5 for AUC (calm "
          "folds have no events, so AUC is\nundefined by construction) and "
          "~= 1.0 for drawdown_lift on event folds.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
