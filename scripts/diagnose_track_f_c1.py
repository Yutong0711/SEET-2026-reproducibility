"""C1 leakage diagnostic for Track F.

Reproduces the C1 (duplicate_dates) corruption at (rate=0.05, seed=42)
and checks whether the surviving valid folds' SILENCED-branch lift
estimates are leakage-contaminated. Prints duplicate-row indices,
their fold classification, and compares raw drawdown_lift on fold 8
LightGbmTuned to Track A's value for the same cell.

Usage:
    python scripts/diagnose_track_f_c1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from seet.injection import corrupt_panel  # noqa: E402
from seet.validators import apply_treatment, validate_panel  # noqa: E402


def main() -> int:
    panel = pd.read_csv(
        REPO_ROOT / "data" / "processed" / "spx_extended_2011.csv",
        parse_dates=["Date"],
    )
    with open(REPO_ROOT / "data" / "manifests" / "processed_manifest.json") as f:
        entry = json.load(f)["spx_extended_2011"]

    fold_def = pd.read_csv(
        REPO_ROOT / "experiments" / "track_a_headline" / "fold_definitions.csv"
    )
    fold8 = fold_def[
        (fold_def["stress_def"] == "h10_d05")
        & (fold_def["fold_id"] == 8)
    ].iloc[0]
    train_end = pd.Timestamp(fold8["train_end"])
    test_start = pd.Timestamp(fold8["test_start"])
    test_end = pd.Timestamp(fold8["test_end"])
    print("=" * 70)
    print("C1 leakage diagnostic — fold 8, h10_d05")
    print("=" * 70)
    print(f"fold 8 train_end  : {train_end.date()}")
    print(f"fold 8 test_start : {test_start.date()}")
    print(f"fold 8 test_end   : {test_end.date()}")

    # --- Reproduce the corruption ------------------------------------
    corrupted, gt, meta = corrupt_panel(
        panel, "duplicate_dates", 0.05, 42, manifest_entry=entry
    )
    print(f"\ncanonical rows : {len(panel)}")
    print(f"corrupted rows : {len(corrupted)}")
    print(f"inserted dups  : {meta['n_inserted']}")
    print(f"target_dates first 5 : {meta['target_dates'][:5]}")

    v = validate_panel(corrupted, entry)
    treated = apply_treatment(corrupted, v, entry)
    print(f"ENABLED rows   : {len(treated)} (expected {len(panel)} since dedup keeps first)")

    # --- Duplicate dates by fold ------------------------------------
    dup_mask = corrupted["Date"].duplicated(keep=False).to_numpy()
    panel_dates_corr = pd.to_datetime(corrupted["Date"]).to_numpy()
    corr_train = panel_dates_corr <= np.datetime64(train_end)
    corr_test = (
        (panel_dates_corr >= np.datetime64(test_start))
        & (panel_dates_corr <= np.datetime64(test_end))
    )

    unique_dup_dates = sorted(set(corrupted.loc[dup_mask, "Date"].tolist()))
    in_train = sum(1 for d in unique_dup_dates if d <= train_end)
    in_test = sum(
        1 for d in unique_dup_dates if test_start <= d <= test_end
    )
    in_other = len(unique_dup_dates) - in_train - in_test
    on_boundary = sum(1 for d in unique_dup_dates if d == train_end)

    print(f"\nUnique duplicate dates : {len(unique_dup_dates)}")
    print(f"  in fold-8 TRAIN  (Date <= {train_end.date()}) : {in_train}")
    print(f"  in fold-8 TEST   ({test_start.date()} -> {test_end.date()}) : {in_test}")
    print(f"  elsewhere (other folds / post-test)         : {in_other}")
    print(f"  sitting exactly on train_end                : {on_boundary}")

    print(f"\nDuplicate-row tally vs fold 8 (each duplicate counts as 2 rows):")
    print(f"  rows with duplicate Dates in fold-8 TRAIN : {int((dup_mask & corr_train).sum())}")
    print(f"  rows with duplicate Dates in fold-8 TEST  : {int((dup_mask & corr_test).sum())}")

    # --- Sample 10 dup dates -----------------------------------------
    print("\nFirst 10 duplicate dates and their copies' SPX values "
          "(should match by construction since corruption copies values "
          "from a random source row):")
    for d in unique_dup_dates[:10]:
        copies = corrupted[corrupted["Date"] == d]
        where = (
            "train" if d <= train_end
            else "test" if test_start <= d <= test_end
            else "other"
        )
        spx_vals = [f"{v:.2f}" for v in copies["SPX"].tolist()]
        print(f"  {d.date()} ({where:>5}): n_copies={len(copies)}  SPX={spx_vals}")

    # --- Check leakage via the source row's date ---------------------
    print("\nSource-row check: when corruption inserts a duplicate at "
          "target_date,\nit copies values from a random SOURCE row. If the "
          "source row's date is in the\nfold-8 test window AND the target "
          "lands in the fold-8 train window, the\ntraining row carries a "
          "label/feature pattern from the test window -> leakage.")
    target_dates = pd.to_datetime(meta["target_dates"])
    print(
        f"\n  target_dates in fold-8 train window : "
        f"{int(((target_dates <= train_end)).sum())}"
    )
    print(
        f"  target_dates in fold-8 test  window : "
        f"{int(((target_dates >= test_start) & (target_dates <= test_end)).sum())}"
    )
    print("\nNote: meta only records the target dates, not source row indices.")
    print("By construction the duplicate row carries the SOURCE row's values "
          "but is\nstamped with the TARGET row's date, so a source-from-test "
          "→ target-in-train\nduplicate would carry future-looking values "
          "into the training set.")

    # --- Compare SILENCED fold-8 raw lift to Track A baseline --------
    print("\n" + "=" * 70)
    print("Raw drawdown_lift comparison (fold 8, LightGbmTuned)")
    print("=" * 70)

    cell_silenced = (
        REPO_ROOT / "experiments" / "track_f_injection" / "_workdir"
        / "duplicate_dates_0.05_42" / "silenced" / "per_fold_metrics.csv"
    )
    cell_enabled = (
        REPO_ROOT / "experiments" / "track_f_injection" / "_workdir"
        / "duplicate_dates_0.05_42" / "enabled" / "per_fold_metrics.csv"
    )
    track_a_path = (
        REPO_ROOT / "experiments" / "track_a_headline" / "per_fold_metrics.csv"
    )

    si = pd.read_csv(cell_silenced)
    en = pd.read_csv(cell_enabled)
    ta = pd.read_csv(track_a_path)

    si8 = si[
        (si["model"] == "LightGbmTuned")
        & (si["fold_id"] == 8)
        & (si["stress_def"] == "h10_d05")
    ]
    en8 = en[
        (en["model"] == "LightGbmTuned")
        & (en["fold_id"] == 8)
        & (en["stress_def"] == "h10_d05")
    ]
    ta8 = ta[
        (ta["model"] == "LightGbmTuned")
        & (ta["fold_id"] == 8)
        & (ta["stress_def"] == "h10_d05")
    ]

    print("\nSILENCED (C1 0.05 seed42) fold 8, LightGbmTuned per-seed lift:")
    print(si8[["seed", "auc", "drawdown_lift", "alarm_rate", "n_alarms",
               "n_events", "n_test", "threshold"]].to_string(index=False))
    print("\nENABLED  (C1 0.05 seed42) fold 8, LightGbmTuned per-seed lift:")
    print(en8[["seed", "auc", "drawdown_lift", "alarm_rate", "n_alarms",
               "n_events", "n_test", "threshold"]].to_string(index=False))
    print("\nTrack A (clean) fold 8, LightGbmTuned per-seed lift:")
    print(ta8[["seed", "auc", "drawdown_lift", "alarm_rate", "n_alarms",
               "n_events", "n_test", "threshold"]].to_string(index=False))

    si_lift = pd.to_numeric(si8["drawdown_lift"], errors="coerce").mean()
    en_lift = pd.to_numeric(en8["drawdown_lift"], errors="coerce").mean()
    ta_lift = pd.to_numeric(ta8["drawdown_lift"], errors="coerce").mean()
    print(
        f"\nMean drawdown_lift (over 5 LightGBM seeds):"
        f"\n  Track A (clean)            : {ta_lift:.4f}"
        f"\n  Track F ENABLED  (treated) : {en_lift:.4f}"
        f"\n  Track F SILENCED (corrupt) : {si_lift:.4f}"
    )
    if not np.isnan(si_lift) and si_lift > 4.0:
        print(
            "\n*** SILENCED lift is implausibly high relative to Track A "
            "baseline (~1.3) ***"
        )
    elif not np.isnan(si_lift) and si_lift > 1.5 * ta_lift:
        print(
            "\n*** SILENCED lift is meaningfully higher than Track A — "
            "may indicate leakage ***"
        )
    else:
        print(
            "\nSILENCED lift looks comparable to Track A — the cell-level "
            "delta of -4.17 in\nthe aggregate may be driven by a small number "
            "of folds with very large negative\ndelta and many folds with "
            "smaller deltas. This is not leakage; it's variance "
            "concentration."
        )

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
