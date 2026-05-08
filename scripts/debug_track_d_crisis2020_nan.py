"""Track D Crisis-2020 asymmetric (spx_full) diagnostic.

Confirms why LR and LightGBM return NaN across all metrics on the
35-feature configuration of Crisis-2020 while panel-DataFrame baselines
work cleanly on the same configuration.

Reads on-disk artifacts only — does not re-run any track.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from seet.baselines import LightGbmTuned, LogisticRegressionL2  # noqa: E402
from seet.features import build_features  # noqa: E402
from seet.pipeline import compute_stress_labels  # noqa: E402


PANEL_NAME = "spx_extended_2011"
TRAIN_END = pd.Timestamp("2019-12-31")
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2020-12-31")

STRESS_DEFS = [
    {"name": "h5_d03",  "h": 5,  "d": 0.03},
    {"name": "h10_d05", "h": 10, "d": 0.05},
    {"name": "h20_d07", "h": 20, "d": 0.07},
]


def _summary(name: str, mask: np.ndarray, feat_df: pd.DataFrame, feat_cols: list[str]) -> tuple[int, int]:
    n_total = int(mask.sum())
    feat_nan_any = feat_df.loc[mask, feat_cols].isna().any(axis=1).to_numpy()
    n_after = int((~feat_nan_any).sum())
    print(f"  {name}: {n_total} rows total, {n_after} after dropping any-NaN-feature rows ({n_total - n_after} dropped)")
    return n_total, n_after


def main() -> int:
    panel = pd.read_csv(
        REPO_ROOT / "data" / "processed" / f"{PANEL_NAME}.csv",
        parse_dates=["Date"],
    )
    feat = pd.read_csv(
        REPO_ROOT / "data" / "processed" / "features"
        / f"{PANEL_NAME}_features.csv",
        parse_dates=["Date"],
    )
    feat_cols = [c for c in feat.columns if c != "Date"]
    panel_dates = panel["Date"]

    train_mask = (panel_dates <= TRAIN_END).to_numpy()
    test_mask = (
        (panel_dates >= TEST_START) & (panel_dates <= TEST_END)
    ).to_numpy()

    print()
    print("=" * 72)
    print("Track D Crisis-2020 ASYMMETRIC (spx_full) diagnostic")
    print("=" * 72)

    # ---- A: train-window row counts ----
    print(f"\n[A] Panel rows through train_end={TRAIN_END.date()}:")
    n_train_total, n_train_after = _summary("train", train_mask, feat, feat_cols)

    # ---- B: per-feature NaN count in training window ----
    feat_train = feat.loc[train_mask, feat_cols]
    feat_nan_train = feat_train.isna().sum().sort_values(ascending=False)
    nan_train = feat_nan_train[feat_nan_train > 0]
    print(f"\n[B] Features with any NaN in training window ({len(nan_train)} of {len(feat_cols)}):")
    if nan_train.empty:
        print("    (none)")
    else:
        for col, n in nan_train.items():
            pct = 100.0 * n / n_train_total
            print(f"    {col:<26}  {n:>5} NaN ({pct:.1f}% of {n_train_total} train rows)")

    # ---- C: test-window row counts ----
    print(f"\n[C] Test rows {TEST_START.date()} -> {TEST_END.date()}:")
    n_test_total, n_test_after = _summary("test", test_mask, feat, feat_cols)

    # ---- D: per-feature NaN count in TEST window ----
    feat_test = feat.loc[test_mask, feat_cols]
    feat_nan_test = feat_test.isna().sum().sort_values(ascending=False)
    nan_test = feat_nan_test[feat_nan_test > 0]
    print(f"\n[D] Features with any NaN in TEST window ({len(nan_test)} of {len(feat_cols)}):")
    if nan_test.empty:
        print("    (none)")
    else:
        always_nan = feat_test.columns[feat_test.isna().all(axis=0)].tolist()
        for col, n in nan_test.items():
            pct = 100.0 * n / n_test_total
            tag = "  <-- ALWAYS NaN" if col in always_nan else ""
            print(f"    {col:<26}  {n:>5} NaN ({pct:.1f}% of {n_test_total} test rows){tag}")
        print(f"\n    Features NaN on EVERY test row: {always_nan}")

    # ---- E: per-stress_def label distribution ----
    print(f"\n[E] Per-stress_def positive label counts in training (after dropna):")
    feat_nan_any_train = feat.loc[train_mask, feat_cols].isna().any(axis=1).to_numpy()
    prices = panel["SPX"].to_numpy(dtype=float)
    for sd in STRESS_DEFS:
        _, labels = compute_stress_labels(prices, sd["h"], sd["d"])
        labels_train = labels[train_mask]
        n_pos_total = int(np.nansum(labels_train == 1))
        n_pos_after = int(
            np.nansum(labels_train[~feat_nan_any_train] == 1)
        )
        print(
            f"    {sd['name']}: positives in train_total = {n_pos_total}, "
            f"in train_after_dropna = {n_pos_after}"
        )

    # ---- F: try LR + LightGBM fits on h10_d05 ----
    sd = next(s for s in STRESS_DEFS if s["name"] == "h10_d05")
    _, labels = compute_stress_labels(prices, sd["h"], sd["d"])
    label_nan = np.isnan(labels)
    train_keep_label = train_mask & ~label_nan
    feat_nan_any_train_label = (
        feat.loc[train_keep_label, feat_cols].isna().any(axis=1).to_numpy()
    )
    train_keep = train_keep_label.copy()
    idx_train_label = np.where(train_keep_label)[0]
    train_keep[idx_train_label[feat_nan_any_train_label]] = False
    X_tr = feat.loc[train_keep, feat_cols]
    y_tr = labels[train_keep]

    test_keep_label = test_mask & ~label_nan
    feat_nan_any_test_label = (
        feat.loc[test_keep_label, feat_cols].isna().any(axis=1).to_numpy()
    )
    test_keep = test_keep_label.copy()
    idx_test_label = np.where(test_keep_label)[0]
    test_keep[idx_test_label[feat_nan_any_test_label]] = False
    X_te = feat.loc[test_keep, feat_cols]

    print(f"\n[F] Fit attempt on h10_d05:")
    print(f"    X_tr shape: {X_tr.shape}, y_tr unique: {np.unique(y_tr.astype(int))}")
    print(f"    X_te shape: {X_te.shape}")

    if X_te.shape[0] == 0:
        print("    *** Test set is EMPTY after dropping NaN-feature rows. ***")
        print("        Every test prediction by LR / LightGBM will be NaN.")
        print("        AUC / PR-AUC / Brier / ECE all undefined.")
    if X_tr.shape[0] >= 10 and np.unique(y_tr.astype(int)).size >= 2:
        try:
            lr = LogisticRegressionL2(seed=42).fit(X_tr, y_tr)
            print(f"    LR fit:       OK")
            if X_te.shape[0] > 0:
                lr_p = lr.predict_proba(X_te)[:, 1]
                print(f"    LR first 5 predictions: {np.round(lr_p[:5], 4)}")
                print(f"    LR predictions all identical? {bool(np.allclose(lr_p, lr_p[0]))}")
        except Exception as e:
            print(f"    LR fit:       EXCEPTION  {type(e).__name__}: {e}")
        try:
            lgb = LightGbmTuned(seed=42).fit(X_tr, y_tr)
            n_trees = (
                int(lgb.model_.booster_.num_trees())
                if (lgb.model_ is not None) else 0
            )
            print(f"    LightGBM fit: OK  (n_trees_built={n_trees})")
            if X_te.shape[0] > 0:
                lgb_p = lgb.predict_proba(X_te)[:, 1]
                print(f"    LightGBM first 5 predictions: {np.round(lgb_p[:5], 4)}")
                print(f"    LightGBM predictions all identical? {bool(np.allclose(lgb_p, lgb_p[0]))}")
        except Exception as e:
            print(f"    LightGBM fit: EXCEPTION  {type(e).__name__}: {e}")
    else:
        print(f"    Skipping LR/LightGBM fits: X_tr too small or single-class.")

    # ---- G: Crisis-2008 (the working configuration) for comparison ----
    print(f"\n[G] Crisis-2008 comparison (spx_core_2007 panel + spx_crisis_2008 features):")
    panel_2008 = pd.read_csv(
        REPO_ROOT / "data" / "processed" / "spx_core_2007.csv",
        parse_dates=["Date"],
    )
    feat_2008, _ = build_features(panel_2008, "spx_crisis_2008")
    feat_2008_cols = [c for c in feat_2008.columns if c != "Date"]
    train_2008_end = pd.Timestamp("2007-12-31")
    test_2008_start = pd.Timestamp("2008-01-01")
    test_2008_end = pd.Timestamp("2009-12-31")
    panel_2008_dates = panel_2008["Date"]
    train_mask_2008 = (panel_2008_dates <= train_2008_end).to_numpy()
    test_mask_2008 = (
        (panel_2008_dates >= test_2008_start)
        & (panel_2008_dates <= test_2008_end)
    ).to_numpy()
    print(f"    n_features: {len(feat_2008_cols)} (vs. {len(feat_cols)} for spx_full)")
    _summary("crisis_2008 train", train_mask_2008, feat_2008, feat_2008_cols)
    _summary("crisis_2008 test ", test_mask_2008,  feat_2008, feat_2008_cols)

    print()
    print("=" * 72)
    print("If [F] X_te shape is (0, 35) and the listed always-NaN features")
    print("include vvix_pctile_252d, the hypothesis is confirmed:")
    print("  - 2019-07-05 and 2020-06-11 are documented VVIX gap dates,")
    print("  - vvix_pctile_252d uses a 252-day rolling window,")
    print("  - so every 2020 test row has NaN vvix_pctile_252d,")
    print("  - so LR / LightGBM cannot predict on any test row.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
