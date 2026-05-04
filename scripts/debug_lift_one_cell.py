"""Diagnostic for one (model, stress_def, fold) cell.

Reads on-disk artifacts only — does NOT re-run the model.

Usage:
    python scripts/debug_lift_one_cell.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# Pinned for the cell the user asked about
MODEL = "LightGbmTuned"
SD = "h5_d03"
H = 5
D = 0.03
FOLD_ID = 1
SEED = 42  # only the seed=42 row matters; LightGbmTuned is deterministic
           # under Track A's placeholder HPs so all 5 seeds agree


def main() -> int:
    # ---- artifacts on disk ----
    per_fold = pd.read_csv(REPO_ROOT / "experiments/track_a_headline/per_fold_metrics.csv")
    fold_def = pd.read_csv(REPO_ROOT / "experiments/track_a_headline/fold_definitions.csv")
    panel = pd.read_csv(REPO_ROOT / "data/processed/spx_extended_2011.csv", parse_dates=["Date"])
    pred = pd.read_parquet(REPO_ROOT / "experiments/track_a_headline/predictions" / f"{SD}.parquet")

    # ---- locate the cell in per_fold_metrics ----
    cell = per_fold[
        (per_fold["model"] == MODEL)
        & (per_fold["stress_def"] == SD)
        & (per_fold["fold_id"] == FOLD_ID)
        & (per_fold["seed"] == SEED)
    ]
    if cell.empty:
        sys.stderr.write(f"No cell found for {MODEL=}, {SD=}, fold {FOLD_ID}, seed {SEED}\n")
        return 1
    cell_row = cell.iloc[0]

    # ---- locate the fold definition ----
    fd_row = fold_def[(fold_def["stress_def"] == SD) & (fold_def["fold_id"] == FOLD_ID)].iloc[0]
    train_end = pd.Timestamp(fd_row["train_end"])
    test_start = pd.Timestamp(fd_row["test_start"])
    test_end = pd.Timestamp(fd_row["test_end"])

    # ---- recompute future_drawdown on the full panel for this stress_def ----
    prices = panel["SPX"].to_numpy(dtype=float)
    n = prices.size
    fd_full = np.full(n, np.nan)
    for t in range(n - H):
        fd_full[t] = float((prices[t + 1: t + H + 1] / prices[t] - 1.0).min())

    # ---- pull this fold's test scores from predictions parquet ----
    fold_pred = pred[(pred["model"] == MODEL) & (pred["fold_id"] == FOLD_ID) & (pred["seed"] == SEED)]
    fold_pred = fold_pred.sort_values("Date").reset_index(drop=True)
    fold_pred["Date"] = pd.to_datetime(fold_pred["Date"])

    # Align test dates to panel rows so we can look up future_drawdown
    panel_by_date = panel.set_index("Date")
    test_dates = fold_pred["Date"]
    test_scores = fold_pred["score"].to_numpy(dtype=float)
    test_labels = fold_pred["label"].to_numpy(dtype=float)
    test_fd = np.array(
        [fd_full[panel_by_date.index.get_loc(d)] for d in test_dates],
        dtype=float,
    )

    # ---- training scores: we need the model's training predictions for this fold.
    # Those weren't materialized to disk (predictions/ only has OOS test rows).
    # The 95th-percentile threshold is, however, persisted in per_fold_metrics.csv
    # under the 'threshold' column. We use the persisted threshold and report the
    # current-implementation alarm count that goes with it.
    threshold = float(cell_row["threshold"])

    # Apply the *implementation* convention: strict > (matches compute_layer2).
    valid = ~(np.isnan(test_scores) | np.isnan(test_labels))
    s = test_scores[valid]
    y = test_labels[valid].astype(int)
    fd_t = test_fd[valid]

    alarms_strict = s > threshold
    n_oos = int(s.size)
    n_alarms = int(alarms_strict.sum())
    n_events = int((y == 1).sum())

    if n_alarms == 0:
        mean_fd_alarm = float("nan")
        signed_lift = float("nan")
        abs_lift = float("nan")
    else:
        mean_fd_alarm = float(np.nanmean(fd_t[alarms_strict]))
        mean_fd_oos = float(np.nanmean(fd_t))
        signed_lift = float(mean_fd_alarm / mean_fd_oos) if mean_fd_oos != 0 else float("nan")
        abs_lift = (
            float(abs(mean_fd_alarm) / abs(mean_fd_oos))
            if mean_fd_oos != 0
            else float("nan")
        )

    mean_fd_oos_all = float(np.nanmean(fd_t))

    # ---- print diagnostic ----
    print()
    print("===== LIFT DIAGNOSTIC =====")
    print(f"cell:             model={MODEL}  stress_def={SD}  fold_id={FOLD_ID}  seed={SEED}")
    print(f"fold window:      train_end={train_end.date()}  "
          f"test={test_start.date()} -> {test_end.date()}")
    print(f"threshold (95p of training scores, from per_fold_metrics): {threshold:.6f}")
    print(f"n_oos in fold:                                              {n_oos}")
    print(f"n_alarms (test_scores > threshold, strict):                 {n_alarms}")
    print(f"n_events (label == 1):                                      {n_events}")
    print()
    print(f"mean future_drawdown on ALARM days (signed):                {mean_fd_alarm:+.6f}")
    print(f"mean future_drawdown on ALL OOS days (signed):              {mean_fd_oos_all:+.6f}")
    print()
    print(f"lift as currently computed (signed/signed):                 {signed_lift:+.6f}")
    print(f"lift as |mean_alarm| / |mean_oos|:                          {abs_lift:+.6f}")
    print(f"drawdown_lift in per_fold_metrics.csv (sanity):             "
          f"{cell_row['drawdown_lift']:+.6f}")
    print("===========================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
