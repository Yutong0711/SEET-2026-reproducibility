"""Apply Track B tuned hyperparameters to Track A's LightGbmTuned rows.

Re-fits only LightGbmTuned per (stress_def, outer_fold) using the
selected hyperparameters in experiments/track_b_tuning/selected_hp.csv.
Replaces the LightGbmTuned rows in per_fold_metrics.csv (other models'
rows are preserved byte-for-byte). Re-aggregates Layer 1-4 tables,
rebuilds pairwise Wilcoxon matrices for AUC and lift, re-renders the
three Track A figures. Builds outputs/track_b/table_selected_hp_summary.csv.

Usage:
    python scripts/apply_track_b_to_track_a.py            # dry-run (default)
    python scripts/apply_track_b_to_track_a.py --apply    # actually overwrite

Default mode is dry-run: re-fits LightGbmTuned, computes the diff vs.
the existing Track A LightGbmTuned cells, prints the diff, and exits
WITHOUT writing any files. Re-run with --apply to commit the changes
to disk.

Track A's deterministic baselines (NaiveBaseRate, VIXPercentileRaw,
VIXPercentileCalibrated, HarRvThreshold, LogisticRegressionL2) are not
re-fit; their rows in per_fold_metrics.csv are preserved verbatim.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from seet.baselines import LightGbmTuned  # noqa: E402
from seet.run_track_a import (  # noqa: E402
    ALL_MODELS,
    INITIAL_TRAIN_END,
    L1_METRICS,
    L2_METRICS,
    PANEL_NAME,
    PRICE_COL,
    PRIMARY_SEED,
    SEEDS,
    STRESS_DEFS,
    build_folds_with_init,
    build_layer3_table,
    build_layer4_table,
    build_pairwise_files,
    build_table_with_ci,
    compute_layer1,
    compute_layer2,
    compute_stress_labels,
    failure_probe_f1,
    failure_probe_f2,
    failure_probe_f3,
    plot_lift_with_ci,
    plot_pr_curves,
    plot_reliability,
)


HP_CSV_PATH = REPO_ROOT / "experiments" / "track_b_tuning" / "selected_hp.csv"
TRACK_A_METRICS = REPO_ROOT / "experiments" / "track_a_headline" / "per_fold_metrics.csv"
TRACK_A_PRED_DIR = REPO_ROOT / "experiments" / "track_a_headline" / "predictions"
TRACK_A_OUT_DIR = REPO_ROOT / "outputs" / "track_a"
TRACK_B_OUT_DIR = REPO_ROOT / "outputs" / "track_b"


def _load_panel_and_features() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    panel_path = REPO_ROOT / "data" / "processed" / f"{PANEL_NAME}.csv"
    feat_path = REPO_ROOT / "data" / "processed" / "features" / f"{PANEL_NAME}_features.csv"
    panel = pd.read_csv(panel_path, parse_dates=["Date"])
    feat = pd.read_csv(feat_path, parse_dates=["Date"])
    if not panel["Date"].equals(feat["Date"]):
        raise ValueError("panel and features Date columns are not aligned")
    return panel, feat, [c for c in feat.columns if c != "Date"]


def _fit_lightgbm_tuned(
    panel_df: pd.DataFrame,
    features_df: pd.DataFrame,
    feat_cols: list[str],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    labels: np.ndarray,
    seed: int,
    sd_name: str,
    fold_id: int,
    hp_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Identical row-mask + NaN-feature logic to Track A's _fit_predict_one
    for LightGbmTuned, but with tuned HPs loaded via the constructor's
    (stress_def, outer_fold, hp_path) lookup branch."""
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    train_scores = np.full(train_idx.size, np.nan)
    test_scores = np.full(test_idx.size, np.nan)

    train_feat_nan = features_df.iloc[train_idx][feat_cols].isna().any(axis=1).to_numpy()
    test_feat_nan = features_df.iloc[test_idx][feat_cols].isna().any(axis=1).to_numpy()
    train_keep = ~train_feat_nan
    if train_keep.sum() < 10:
        return train_scores, test_scores

    X_tr = features_df.iloc[train_idx[train_keep]][feat_cols]
    y_tr = labels[train_idx[train_keep]]

    model = LightGbmTuned(
        seed=seed,
        stress_def=sd_name,
        outer_fold=int(fold_id),
        hp_path=str(hp_path),
    )
    model.fit(X_tr, y_tr)
    train_scores[train_keep] = model.predict_proba(X_tr)[:, 1]

    test_keep = ~test_feat_nan
    if test_keep.any():
        X_te = features_df.iloc[test_idx[test_keep]][feat_cols]
        test_scores[test_keep] = model.predict_proba(X_te)[:, 1]
    return train_scores, test_scores


def recompute_lightgbm_rows(
    panel: pd.DataFrame, feat: pd.DataFrame, feat_cols: list[str]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Returns (new_lgbm_per_fold_rows_df, new_lgbm_predictions_by_def)."""
    panel_dates = pd.to_datetime(panel["Date"])
    folds = build_folds_with_init(panel_dates, INITIAL_TRAIN_END)
    prices = panel[PRICE_COL].to_numpy(dtype=float)

    new_rows: list[dict] = []
    new_predictions: dict[str, list[dict]] = {sd["name"]: [] for sd in STRESS_DEFS}

    for sd in STRESS_DEFS:
        sd_name = sd["name"]
        future_drawdown, labels = compute_stress_labels(prices, sd["h"], sd["d"])
        label_nan = np.isnan(labels)

        for fold in folds:
            train_mask_dates = (panel_dates <= fold["train_end"]).to_numpy()
            test_mask_dates = (
                (panel_dates >= fold["test_start"])
                & (panel_dates <= fold["test_end"])
            ).to_numpy()
            train_mask = train_mask_dates & ~label_nan
            test_mask = test_mask_dates & ~label_nan

            if train_mask.sum() < 30 or test_mask.sum() < 5:
                continue

            test_dates = panel_dates[test_mask].reset_index(drop=True)
            test_fd = future_drawdown[test_mask]
            test_labels = labels[test_mask]

            # LightGBM with the placeholder HPs is deterministic in the
            # absence of bagging/feature_fraction; the tuned HPs grid
            # also has no bagging, so all 5 seeds reproduce identically.
            # Fit once with PRIMARY_SEED, replicate row across SEEDS.
            train_scores, test_scores = _fit_lightgbm_tuned(
                panel, feat, feat_cols,
                train_mask, test_mask, labels,
                seed=PRIMARY_SEED,
                sd_name=sd_name,
                fold_id=fold["fold_id"],
                hp_path=HP_CSV_PATH,
            )
            l1 = compute_layer1(test_scores, test_labels)
            l2 = compute_layer2(test_scores, test_labels, train_scores, test_fd)
            base = {
                "stress_def": sd_name,
                "model": "LightGbmTuned",
                "fold_id": int(fold["fold_id"]),
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                **l1,
                **l2,
            }
            for s in SEEDS:
                new_rows.append({**base, "seed": int(s)})
                for date_, score, lab in zip(test_dates, test_scores, test_labels):
                    new_predictions[sd_name].append(
                        {
                            "Date": date_,
                            "model": "LightGbmTuned",
                            "seed": int(s),
                            "fold_id": int(fold["fold_id"]),
                            "score": float(score) if not np.isnan(score) else np.nan,
                            "label": float(lab) if not np.isnan(lab) else np.nan,
                        }
                    )

    return pd.DataFrame(new_rows), {k: pd.DataFrame(v) for k, v in new_predictions.items()}


def _diff_summary(old_full: pd.DataFrame, new_lgbm: pd.DataFrame) -> None:
    """Per-stress_def diff summary (mean across folds, after collapsing seeds)."""
    metrics = [
        "auc", "pr_auc", "brier", "ece",
        "alarm_rate", "alarm_precision", "drawdown_lift", "severity_lift",
    ]
    keys = ["stress_def", "fold_id"]
    old_lgbm = old_full[old_full["model"] == "LightGbmTuned"]
    old_avg = old_lgbm.groupby(keys)[metrics].mean().reset_index()
    new_avg = new_lgbm.groupby(keys)[metrics].mean().reset_index()
    merged = old_avg.merge(new_avg, on=keys, suffixes=("_old", "_new"))

    print()
    print("===== TRACK A LightGbmTuned DIFF (placeholder -> tuned) =====")
    for sd_name, grp in merged.groupby("stress_def"):
        print(f"\n[{sd_name}]  ({len(grp)} folds)")
        for m in metrics:
            old_mean = float(grp[f"{m}_old"].mean())
            new_mean = float(grp[f"{m}_new"].mean())
            delta = new_mean - old_mean
            sign = "+" if delta >= 0 else ""
            print(
                f"  {m:<20}  old={old_mean: .4f}  new={new_mean: .4f}  "
                f"delta={sign}{delta: .4f}"
            )

    # Headline: mean AUC and mean lift change.
    print()
    print("Headline shifts (mean across folds × stress_defs):")
    for m in ("auc", "drawdown_lift", "pr_auc"):
        old_mean = float(old_avg[m].mean())
        new_mean = float(new_avg[m].mean())
        delta = new_mean - old_mean
        sign = "+" if delta >= 0 else ""
        print(f"  {m:<20}  old={old_mean:.4f}  new={new_mean:.4f}  delta={sign}{delta:.4f}")
    print("=============================================================")


def _f_probes_diff(old_full: pd.DataFrame, new_full: pd.DataFrame) -> None:
    print()
    print("Failure-mode probes (placeholder -> tuned):")
    print("  F1 OLD:", failure_probe_f1(old_full))
    print("  F1 NEW:", failure_probe_f1(new_full))
    print(f"  F2 OLD: {failure_probe_f2(old_full)}")
    print(f"  F2 NEW: {failure_probe_f2(new_full)}")
    f3_old = failure_probe_f3(old_full)
    f3_new = failure_probe_f3(new_full)
    f3_old_str = ", ".join(f"{k}: {v:.3f}" for k, v in f3_old.items())
    f3_new_str = ", ".join(f"{k}: {v:.3f}" for k, v in f3_new.items())
    print(f"  F3 OLD: {f3_old_str}")
    print(f"  F3 NEW: {f3_new_str}")


def _rebuild_per_fold_metrics(
    old_full: pd.DataFrame, new_lgbm: pd.DataFrame
) -> pd.DataFrame:
    """Replace LightGbmTuned rows; preserve original (sd, fold, model, seed) order."""
    other = old_full[old_full["model"] != "LightGbmTuned"]
    combined = pd.concat([other, new_lgbm], ignore_index=True)

    sd_order = {sd["name"]: i for i, sd in enumerate(STRESS_DEFS)}
    model_order = {m: i for i, m in enumerate(ALL_MODELS)}
    combined["_sd_ord"] = combined["stress_def"].map(sd_order)
    combined["_m_ord"] = combined["model"].map(model_order)
    combined = combined.sort_values(
        ["_sd_ord", "fold_id", "_m_ord", "seed"], kind="mergesort"
    ).reset_index(drop=True)
    combined = combined.drop(columns=["_sd_ord", "_m_ord"])
    # Match original column order
    combined = combined[list(old_full.columns)]
    return combined


def _update_predictions_parquets(
    new_predictions: dict[str, pd.DataFrame],
) -> None:
    for sd_name, new_df in new_predictions.items():
        if new_df.empty:
            continue
        path = TRACK_A_PRED_DIR / f"{sd_name}.parquet"
        if not path.exists():
            print(f"  [warn] {path.name} missing; skipping update.")
            continue
        existing = pd.read_parquet(path)
        other = existing[existing["model"] != "LightGbmTuned"]
        merged = pd.concat([other, new_df], ignore_index=True)
        merged.to_parquet(path, index=False)
        print(f"  wrote {path.name} ({len(merged)} rows)")


def _write_aggregated_tables(updated_full: pd.DataFrame) -> None:
    TRACK_A_OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_table_with_ci(updated_full, L1_METRICS).to_csv(
        TRACK_A_OUT_DIR / "table_layer1.csv", index=False
    )
    build_table_with_ci(updated_full, L2_METRICS).to_csv(
        TRACK_A_OUT_DIR / "table_layer2.csv", index=False
    )
    pred_dfs: dict[str, pd.DataFrame] = {}
    for sd in STRESS_DEFS:
        path = TRACK_A_PRED_DIR / f"{sd['name']}.parquet"
        if path.exists():
            pred_dfs[sd["name"]] = pd.read_parquet(path)
    build_layer3_table(updated_full, pred_dfs).to_csv(
        TRACK_A_OUT_DIR / "table_layer3.csv", index=False
    )
    build_layer4_table(updated_full).to_csv(
        TRACK_A_OUT_DIR / "table_layer4.csv", index=False
    )
    build_pairwise_files(updated_full, TRACK_A_OUT_DIR)
    print(f"  wrote table_layer{{1,2,3,4}}.csv and pairwise_pvalues_*.csv")
    return pred_dfs


def _rerender_figures(updated_full: pd.DataFrame, pred_dfs: dict[str, pd.DataFrame]) -> None:
    plot_lift_with_ci(updated_full, TRACK_A_OUT_DIR / "fig_lift_with_ci.pdf")
    if "h10_d05" in pred_dfs:
        plot_pr_curves(pred_dfs["h10_d05"], TRACK_A_OUT_DIR / "fig_pr_curves.pdf")
        plot_reliability(pred_dfs["h10_d05"], TRACK_A_OUT_DIR / "fig_reliability.pdf")
    print("  wrote fig_lift_with_ci.pdf, fig_pr_curves.pdf, fig_reliability.pdf")


def _build_track_b_summary() -> Path:
    sel = pd.read_csv(HP_CSV_PATH)
    rows: list[dict] = []
    for sd_name, grp in sel.groupby("stress_def"):
        param_dicts = [json.loads(p) for p in grp["best_params"]]
        modal: dict = {}
        for k in sorted(param_dicts[0].keys()):
            series = pd.Series([d[k] for d in param_dicts])
            modal[k] = series.mode().iloc[0]
        row = {
            "stress_def": sd_name,
            "median_inner_pr_auc": float(grp["inner_pr_auc"].median(skipna=True)),
            "n_folds_with_valid_inner_pr_auc": int(grp["inner_pr_auc"].notna().sum()),
            "n_folds_total": int(len(grp)),
        }
        row.update(modal)
        rows.append(row)
    out = pd.DataFrame(rows)
    TRACK_B_OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACK_B_OUT_DIR / "table_selected_hp_summary.csv"
    out.to_csv(path, index=False)
    return path


def _flag_nan_inner_cells() -> list[tuple[str, int]]:
    sel = pd.read_csv(HP_CSV_PATH)
    nan = sel[sel["inner_pr_auc"].isna()]
    return [(r["stress_def"], int(r["outer_fold"])) for _, r in nan.iterrows()]


def main(apply: bool) -> int:
    if not HP_CSV_PATH.exists():
        sys.stderr.write(f"Missing {HP_CSV_PATH}. Run track B tuning first.\n")
        return 1
    if not TRACK_A_METRICS.exists():
        sys.stderr.write(f"Missing {TRACK_A_METRICS}. Run Track A first.\n")
        return 1

    nan_cells = _flag_nan_inner_cells()
    if nan_cells:
        print(
            f"[warn] {len(nan_cells)} cell(s) in selected_hp.csv have "
            f"NaN inner_pr_auc; their best_params is the iteration-order "
            f"fallback (no data-driven selection):"
        )
        for sd_name, fold_id in nan_cells:
            print(f"   - {sd_name}  fold {fold_id}")

    panel, feat, feat_cols = _load_panel_and_features()
    print(f"\n[Re-fit] LightGbmTuned with tuned HPs from {HP_CSV_PATH.name}...")
    t0 = time.time()
    new_lgbm, new_predictions = recompute_lightgbm_rows(panel, feat, feat_cols)
    print(
        f"[Re-fit] done in {time.time() - t0:.1f}s; "
        f"{len(new_lgbm)} new rows across "
        f"{new_lgbm[['stress_def', 'fold_id']].drop_duplicates().shape[0]} cells."
    )

    old_full = pd.read_csv(TRACK_A_METRICS)
    _diff_summary(old_full, new_lgbm)

    updated_full = _rebuild_per_fold_metrics(old_full, new_lgbm)
    _f_probes_diff(old_full, updated_full)

    if not apply:
        print()
        print("[DRY RUN] No files written. Re-run with --apply to commit changes.")
        return 0

    print()
    print("[Apply] Writing updated artifacts...")
    updated_full.to_csv(TRACK_A_METRICS, index=False)
    print(f"  wrote {TRACK_A_METRICS.name} ({len(updated_full)} rows)")
    _update_predictions_parquets(new_predictions)
    pred_dfs = _write_aggregated_tables(updated_full)
    _rerender_figures(updated_full, pred_dfs)

    summary_path = _build_track_b_summary()
    print(f"  wrote {summary_path.relative_to(REPO_ROOT)}")

    print()
    print("===== TRACK B FINAL STATUS =====")
    print(f"selected_hp.csv applied to {len(new_lgbm)} LightGbmTuned rows.")
    print("Re-aggregated: table_layer{1,2,3,4}.csv, pairwise_pvalues_*.csv.")
    print("Re-rendered: fig_lift_with_ci.pdf, fig_pr_curves.pdf, fig_reliability.pdf.")
    print(f"Track B summary: {summary_path.relative_to(REPO_ROOT)}.")
    if nan_cells:
        print(
            f"[note] {len(nan_cells)} cell(s) used iteration-order fallback HPs: "
            + ", ".join(f"{sd}-{f}" for sd, f in nan_cells)
        )
    print("================================")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually overwrite Track A artifacts (default is dry-run).",
    )
    args = parser.parse_args()
    sys.exit(main(apply=args.apply))
