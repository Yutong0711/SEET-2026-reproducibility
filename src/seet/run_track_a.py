"""Track A SPX runner — thin wrapper around seet.pipeline.

This module supplies the SPX-specific configuration (panel name, feature
set, price column, stress definitions, fold scheme, seed list) and
delegates the actual grid execution, four-layer evaluation, aggregation,
plotting, and failure-mode probes to seet.pipeline. Track C lives in
scripts/run_track_c.py and uses the same pipeline with NDX/RUT-specific
configuration.

Public CLI:
    python src/seet/run_track_a.py --smoke
    python src/seet/run_track_a.py --full

Backward compatibility: the helpers and constants previously defined
inline in this file (compute_stress_labels, build_folds_with_init,
compute_layer1, compute_layer2, build_table_with_ci, build_layer3_table,
build_layer4_table, build_pairwise_files, plot_lift_with_ci,
plot_pr_curves, plot_reliability, failure_probe_f1, failure_probe_f2,
failure_probe_f3, L1_METRICS, L2_METRICS, ECE_BINS, THRESHOLD_PERCENTILE,
N_BOOT, ALPHA, BOOTSTRAP_SEED) are re-exported below so that
scripts/apply_track_b_to_track_a.py and tests/test_track_a_smoke.py
continue to import from seet.run_track_a unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------
# Re-exports from seet.pipeline (the canonical implementation)
# ---------------------------------------------------------------------
from seet.baselines import (  # noqa: E402  (re-exported for back-compat)
    ALL_MODELS,
    DETERMINISTIC_MODELS,
    STOCHASTIC_MODELS,
    make_model,
)
from seet.pipeline import (  # noqa: E402
    ALPHA,
    BOOTSTRAP_SEED,
    ECE_BINS,
    L1_METRICS,
    L2_METRICS,
    N_BOOT,
    THRESHOLD_PERCENTILE,
    bootstrap_std_ci,
    build_folds_with_init,
    build_layer3_table,
    build_layer4_table,
    build_pairwise_files,
    build_table_with_ci,
    compute_ece,
    compute_layer1,
    compute_layer2,
    compute_stress_labels,
    failure_probe_f1,
    failure_probe_f2,
    failure_probe_f3,
    plot_lift_with_ci,
    plot_pr_curves,
    plot_reliability,
    run_grid as _pipeline_run_grid,
)
from seet.stats import block_bootstrap_ci  # noqa: E402  (used by print_status)


# =====================================================================
# Track A configuration (SPX-specific constants)
# =====================================================================

PANEL_NAME = "spx_extended_2011"
FEATURE_SET_ID = "spx_full"
PRICE_COL = "SPX"
VIX_COL = "VIX"

STRESS_DEFS = [
    {"name": "h5_d03",  "h": 5,  "d": 0.03},
    {"name": "h10_d05", "h": 10, "d": 0.05},
    {"name": "h20_d07", "h": 20, "d": 0.07},
]

INITIAL_TRAIN_END = pd.Timestamp("2014-12-31")

SEEDS = (42, 43, 44, 45, 46)
PRIMARY_SEED = 42


# =====================================================================
# Backward-compatible run_grid wrapper
# =====================================================================

def run_grid(
    panel_df: pd.DataFrame,
    features_df: pd.DataFrame,
    out_exp_dir: Path,
    out_table_dir: Path,
    stress_defs: list[dict] | None = None,
    seeds: tuple[int, ...] | None = None,
    initial_train_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Track A-flavored wrapper around pipeline.run_grid.

    Preserves the original positional signature so scripts and tests
    that have been calling this function unchanged continue to work.
    Track A's price column ("SPX") and the project-default deterministic
    / stochastic model registries are passed through; the
    model_kwargs_factory defaults to a no-op (every baseline uses its
    own constructor defaults, which point to "VIX" and "SPX" — the
    Track A behavior)."""
    out_exp_dir = Path(out_exp_dir)
    out_table_dir = Path(out_table_dir)
    out_table_dir.mkdir(parents=True, exist_ok=True)
    return _pipeline_run_grid(
        panel_df,
        features_df,
        out_exp_dir,
        price_col=PRICE_COL,
        stress_defs=stress_defs if stress_defs is not None else STRESS_DEFS,
        seeds=seeds if seeds is not None else SEEDS,
        primary_seed=PRIMARY_SEED,
        initial_train_end=(
            initial_train_end if initial_train_end is not None else INITIAL_TRAIN_END
        ),
        deterministic_models=DETERMINISTIC_MODELS,
        stochastic_models=STOCHASTIC_MODELS,
        save_predictions=True,
    )


# =====================================================================
# STATUS printing (Track A specific narrative)
# =====================================================================

def print_status(per_fold_df: pd.DataFrame, fold_def_df: pd.DataFrame) -> None:
    print()
    print("===== TRACK A STATUS =====")
    for sd in STRESS_DEFS:
        sd_name = sd["name"]
        sd_folds = fold_def_df[fold_def_df["stress_def"] == sd_name]
        print(f"\n[{sd_name}] folds = {len(sd_folds)}")
        for model in ALL_MODELS:
            sub = per_fold_df[
                (per_fold_df["stress_def"] == sd_name)
                & (per_fold_df["model"] == model)
            ]
            auc_per_fold = (
                sub.groupby("fold_id")["auc"].mean().to_numpy()
            )
            lift_per_fold = (
                sub.groupby("fold_id")["drawdown_lift"].mean().to_numpy()
            )
            auc_ci = block_bootstrap_ci(auc_per_fold, n_boot=N_BOOT, seed=BOOTSTRAP_SEED)
            lift_ci = block_bootstrap_ci(lift_per_fold, n_boot=N_BOOT, seed=BOOTSTRAP_SEED)
            print(
                f"  {model:<26}  "
                f"AUC mean={auc_ci['mean']:.3f} "
                f"CI=[{auc_ci['ci_low']:.3f},{auc_ci['ci_high']:.3f}]   "
                f"lift mean={lift_ci['mean']:.3f} "
                f"CI=[{lift_ci['ci_low']:.3f},{lift_ci['ci_high']:.3f}]"
            )

    print("\nFailure-mode probes:")
    print(f"  F1 (AUC-CIs overlap, lift-CIs disjoint): {failure_probe_f1(per_fold_df)}")
    print(f"  F2 (sign mismatch AUC vs lift):          {failure_probe_f2(per_fold_df)}")
    f3 = failure_probe_f3(per_fold_df)
    f3_str = ", ".join(f"{k}: {v:.3f}" for k, v in f3.items())
    print(f"  F3 (Spearman Brier-rank vs lift-rank):   {f3_str}")
    print("==========================")


# =====================================================================
# Main entry points
# =====================================================================

def _load_full_panels(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_path = repo_root / "data" / "processed" / f"{PANEL_NAME}.csv"
    feat_path = repo_root / "data" / "processed" / "features" / f"{PANEL_NAME}_features.csv"
    panel_df = pd.read_csv(panel_path, parse_dates=["Date"])
    features_df = pd.read_csv(feat_path, parse_dates=["Date"])
    if not panel_df["Date"].equals(features_df["Date"]):
        raise ValueError("Panel and features Date columns are not aligned.")
    return panel_df, features_df


def main_full() -> int:
    panel_df, features_df = _load_full_panels(REPO_ROOT)
    out_exp = REPO_ROOT / "experiments" / "track_a_headline"
    out_tab = REPO_ROOT / "outputs" / "track_a"

    per_fold_df, fold_def_df, predictions_by_def = run_grid(
        panel_df, features_df, out_exp, out_tab,
        stress_defs=STRESS_DEFS, seeds=SEEDS,
        initial_train_end=INITIAL_TRAIN_END,
    )

    build_table_with_ci(per_fold_df, L1_METRICS).to_csv(
        out_tab / "table_layer1.csv", index=False
    )
    build_table_with_ci(per_fold_df, L2_METRICS).to_csv(
        out_tab / "table_layer2.csv", index=False
    )
    build_layer3_table(per_fold_df, predictions_by_def).to_csv(
        out_tab / "table_layer3.csv", index=False
    )
    build_layer4_table(per_fold_df).to_csv(
        out_tab / "table_layer4.csv", index=False
    )
    build_pairwise_files(per_fold_df, out_tab)

    plot_lift_with_ci(per_fold_df, out_tab / "fig_lift_with_ci.pdf",
                      stress_defs=STRESS_DEFS)
    target_sd = "h10_d05"
    if target_sd in predictions_by_def and len(predictions_by_def[target_sd]) > 0:
        plot_reliability(predictions_by_def[target_sd], out_tab / "fig_reliability.pdf")
        plot_pr_curves(predictions_by_def[target_sd], out_tab / "fig_pr_curves.pdf")

    print_status(per_fold_df, fold_def_df)
    return 0


def main_smoke() -> int:
    """Synthetic 1-fold, 1-seed sanity check."""
    rng = np.random.default_rng(0)
    n = 400
    dates = pd.date_range("2014-01-01", periods=n, freq="B")
    spx = 1000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n)))
    vix = 15.0 + 5.0 * rng.standard_normal(n).cumsum() / np.sqrt(n)
    vix = np.clip(vix, 8.0, 80.0)
    panel_df = pd.DataFrame({"Date": dates, "SPX": spx, "VIX": vix})

    feat_cols = [f"feat_{i}" for i in range(35)]
    feat_data = rng.standard_normal((n, 35))
    features_df = pd.DataFrame(feat_data, columns=feat_cols)
    features_df.insert(0, "Date", dates)

    out_exp = REPO_ROOT / "experiments" / "track_a_headline_smoke"
    out_tab = REPO_ROOT / "outputs" / "track_a_smoke"

    per_fold_df, fold_def_df, predictions_by_def = run_grid(
        panel_df, features_df, out_exp, out_tab,
        stress_defs=[{"name": "h5_d03", "h": 5, "d": 0.03}],
        seeds=(42,),
        initial_train_end=pd.Timestamp("2014-06-30"),
    )
    if per_fold_df.empty:
        print("SMOKE: no per-fold rows produced (likely no folds fit).")
        return 1
    build_table_with_ci(per_fold_df, L1_METRICS).to_csv(
        out_tab / "table_layer1.csv", index=False
    )
    print(f"SMOKE: per_fold_metrics rows = {len(per_fold_df)}, fold_defs = {len(fold_def_df)}")
    return 0


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track A runner")
    parser.add_argument("--smoke", action="store_true", help="run synthetic smoke")
    parser.add_argument("--full", action="store_true", help="run the full headline grid")
    args = parser.parse_args(argv)
    if args.smoke and args.full:
        parser.error("Pick one of --smoke / --full.")
    if args.smoke:
        return main_smoke()
    if args.full:
        return main_full()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(cli())
