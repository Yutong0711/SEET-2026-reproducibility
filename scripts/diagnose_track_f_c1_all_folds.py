"""All-fold leakage scan for C1 (duplicate_dates).

Extends scripts/diagnose_track_f_c1.py: instead of inspecting a single
fold, this scans every (fold_id, lgbm_seed) cell of every C1 injection
seed, prints the SILENCED-branch raw drawdown_lift, and flags any
fold where SILENCED >> Track-A-baseline (the leakage signature).

Helps decide whether the aggregate delta_lift = -4.17 (rate=5%,
LightGbmTuned) is driven by real leakage in a small number of folds
or by paired-bootstrap aggregation noise.

Usage:
    python scripts/diagnose_track_f_c1_all_folds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKDIR = REPO_ROOT / "experiments" / "track_f_injection" / "_workdir"
TRACK_A = REPO_ROOT / "experiments" / "track_a_headline" / "per_fold_metrics.csv"

LIFT_THRESHOLD = 4.0  # flag SILENCED lift >= this as suspicious
INJECTION_SEEDS = (42, 43, 44, 45, 46)


def main() -> int:
    ta = pd.read_csv(TRACK_A)
    ta = ta[(ta["stress_def"] == "h10_d05") & (ta["model"] == "LightGbmTuned")]
    ta_baseline = (
        ta.groupby("fold_id")["drawdown_lift"].mean().to_dict()
    )

    rate = 0.05
    print(f"\n{'='*80}")
    print(f"C1 all-fold scan — rate={rate}, LightGbmTuned (5 lgbm seeds × 5 injection seeds)")
    print(f"{'='*80}\n")

    rows: list[dict] = []
    for inj_seed in INJECTION_SEEDS:
        cell = WORKDIR / f"duplicate_dates_{rate:.2f}_{inj_seed}" / "silenced" / "per_fold_metrics.csv"
        if not cell.exists():
            print(f"  [missing] {cell.relative_to(REPO_ROOT)}")
            continue
        si = pd.read_csv(cell)
        si = si[(si["stress_def"] == "h10_d05") & (si["model"] == "LightGbmTuned")]
        for _, r in si.iterrows():
            rows.append({
                "injection_seed": int(inj_seed),
                "fold_id": int(r["fold_id"]),
                "lgbm_seed": int(r["seed"]),
                "silenced_lift": float(r["drawdown_lift"]) if not pd.isna(r["drawdown_lift"]) else float("nan"),
                "silenced_auc": float(r["auc"]) if not pd.isna(r["auc"]) else float("nan"),
                "n_alarms": int(r["n_alarms"]),
                "n_events": int(r["n_events"]),
                "n_test": int(r["n_test"]),
            })
    df = pd.DataFrame(rows)

    # Per-(fold, inj_seed) SILENCED lift averaged over lgbm seeds (LightGBM is
    # deterministic given non-bagging HPs at this scale; the 5 seeds give 5
    # identical values per fold-cell in Track A and Track F).
    cell_means = (
        df.groupby(["injection_seed", "fold_id"])
        .agg(silenced_lift=("silenced_lift", "mean"),
             silenced_auc=("silenced_auc", "mean"),
             n_alarms=("n_alarms", "first"),
             n_events=("n_events", "first"),
             n_test=("n_test", "first"))
        .reset_index()
    )

    print("Per-fold SILENCED drawdown_lift, all injection seeds + Track-A baseline:")
    print(f"{'fold':>4} {'inj_seed':>9} {'silenced_lift':>14} {'TA_baseline':>13} "
          f"{'gap':>8} {'silenced_auc':>13} {'n_alarms':>9} {'n_events':>9}")
    print("  " + "-" * 86)
    flagged: list[dict] = []
    for _, r in cell_means.sort_values(["fold_id", "injection_seed"]).iterrows():
        fid = int(r["fold_id"])
        ta_base = ta_baseline.get(fid, float("nan"))
        gap = r["silenced_lift"] - ta_base if not pd.isna(r["silenced_lift"]) and not pd.isna(ta_base) else float("nan")
        flag = ""
        if not pd.isna(r["silenced_lift"]) and r["silenced_lift"] >= LIFT_THRESHOLD:
            flag = " ***"
            flagged.append({
                **r.to_dict(), "ta_baseline": ta_base, "gap": gap,
            })
        lift_s = f"{r['silenced_lift']:14.3f}" if not pd.isna(r["silenced_lift"]) else f"{'n/a':>14}"
        ta_s = f"{ta_base:13.3f}" if not pd.isna(ta_base) else f"{'n/a':>13}"
        gap_s = f"{gap:+8.3f}" if not pd.isna(gap) else f"{'n/a':>8}"
        auc_s = f"{r['silenced_auc']:13.3f}" if not pd.isna(r["silenced_auc"]) else f"{'n/a':>13}"
        print(f"{fid:>4} {int(r['injection_seed']):>9} {lift_s} {ta_s} {gap_s} "
              f"{auc_s} {int(r['n_alarms']):>9} {int(r['n_events']):>9}{flag}")

    print(f"\nFlagged cells (SILENCED lift >= {LIFT_THRESHOLD}): {len(flagged)}")
    if flagged:
        print("\n  These cells show SILENCED lift far above Track A baseline.")
        print("  Most likely mechanism: a duplicate row inserted at a TEST date carries")
        print("  SOURCE-row values with a future-stress-event SPX level. The dup row's")
        print("  forward-looking drawdown is then computed against the panel's REAL SPX")
        print("  at later positions -> artificial stress label, artificial alarm hit,")
        print("  artificially boosted lift.")
    else:
        print("\n  No flagged cells. The aggregate delta of -4.17 must be driven by")
        print("  smaller per-cell variations adding up across n_pairs=105 (paired-")
        print("  bootstrap noise rather than systematic leakage).")
    print(f"{'='*80}\n")

    # Show the percentile distribution of SILENCED lift across all valid cells.
    valid_lifts = cell_means["silenced_lift"].dropna().to_numpy()
    if valid_lifts.size:
        pcts = [10, 25, 50, 75, 90, 95, 99, 100]
        ps = np.percentile(valid_lifts, pcts)
        print("SILENCED drawdown_lift distribution across valid (fold, inj_seed) cells:")
        for p, v in zip(pcts, ps):
            print(f"  P{p:>3} = {v:>10.3f}")
        print(f"  mean  = {valid_lifts.mean():>10.3f}  (Track A mean ≈ 1.3)")
        print(f"  count = {valid_lifts.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
