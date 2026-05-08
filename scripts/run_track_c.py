"""Track C: multi-asset replication of the headline grid on NDX (with VXN)
and RUT (with RVX).

Uses seet.pipeline.run_grid with the asset-agnostic API:
  - price_col is set per asset (always "INDEX" for the Track C panels)
  - model_kwargs_factory points the column-aware baselines
    (VIXPercentileRaw, VIXPercentileCalibrated, HarRvThreshold) at
    the generic INDEX/VOL column names
  - LightGbmTuned uses DEFAULT hyperparameters (no Track B HP transfer
    — per the Track C spec, the cross-asset generalization claim is
    tested without retuning)

Outputs:
  experiments/track_c_multiasset/per_fold_metrics_{ndx,rut}.csv
  experiments/track_c_multiasset/fold_definitions_{ndx,rut}.csv
  outputs/track_c/table_replication.csv
  outputs/track_c/fig_replication.pdf

Run from the repo root:
    python scripts/run_track_c.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from seet.baselines import ALL_MODELS, DETERMINISTIC_MODELS, STOCHASTIC_MODELS  # noqa: E402
from seet.pipeline import (  # noqa: E402
    ALPHA,
    BOOTSTRAP_SEED,
    N_BOOT,
    _per_fold_means,
    _palette,
    failure_probe_f1,
    failure_probe_f2,
    failure_probe_f3,
    run_grid,
)
from seet.run_track_a import (  # noqa: E402
    INITIAL_TRAIN_END,
    PRIMARY_SEED,
    SEEDS,
    STRESS_DEFS,
)
from seet.stats import block_bootstrap_ci  # noqa: E402


ASSETS = [
    {"name": "NDX", "panel": "ndx_panel", "feature_set": "asset_minimal"},
    {"name": "RUT", "panel": "rut_panel", "feature_set": "asset_minimal"},
]
TRACK_C_EXP_DIR = REPO_ROOT / "experiments" / "track_c_multiasset"
TRACK_C_OUT_DIR = REPO_ROOT / "outputs" / "track_c"


def track_c_kwargs_factory(model_name: str, **_) -> dict:
    """Point the column-aware baselines at the generic Track C names."""
    if model_name in ("VIXPercentileRaw", "VIXPercentileCalibrated"):
        return {"vix_col": "VOL"}
    if model_name == "HarRvThreshold":
        return {"price_col": "INDEX"}
    return {}


def _load_asset_panel_and_features(panel_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_path = REPO_ROOT / "data" / "processed" / f"{panel_name}.csv"
    feat_path = REPO_ROOT / "data" / "processed" / "features" / f"{panel_name}_features.csv"
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)
    if not feat_path.exists():
        raise FileNotFoundError(feat_path)
    panel = pd.read_csv(panel_path, parse_dates=["Date"])
    feat = pd.read_csv(feat_path, parse_dates=["Date"])
    if not panel["Date"].equals(feat["Date"]):
        raise ValueError(f"{panel_name}: Date columns not aligned")
    return panel, feat


def run_for_asset(asset: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the headline grid for one asset; persist per_fold_metrics
    and fold_definitions into TRACK_C_EXP_DIR with asset-suffixed names."""
    panel_name = asset["panel"]
    asset_label = asset["name"].lower()
    print(f"\n[{asset['name']}] loading panel and features ({panel_name})...")
    panel, feat = _load_asset_panel_and_features(panel_name)
    print(
        f"[{asset['name']}] panel rows={len(panel)}, "
        f"features cols={len([c for c in feat.columns if c != 'Date'])}, "
        f"first_date={panel['Date'].iloc[0].date()}, "
        f"last_date={panel['Date'].iloc[-1].date()}"
    )

    TRACK_C_EXP_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    per_fold_df, fold_def_df, _ = run_grid(
        panel, feat, TRACK_C_EXP_DIR,
        price_col="INDEX",
        stress_defs=STRESS_DEFS,
        seeds=SEEDS,
        primary_seed=PRIMARY_SEED,
        initial_train_end=INITIAL_TRAIN_END,
        deterministic_models=DETERMINISTIC_MODELS,
        stochastic_models=STOCHASTIC_MODELS,
        model_kwargs_factory=track_c_kwargs_factory,
        save_predictions=False,
    )
    elapsed = time.time() - t0
    print(
        f"[{asset['name']}] grid done in {elapsed:.1f}s; "
        f"per_fold rows = {len(per_fold_df)}, folds = {len(fold_def_df)}"
    )

    # Pipeline wrote per_fold_metrics.csv and fold_definitions.csv into
    # TRACK_C_EXP_DIR — rename to asset-suffixed names so two asset runs
    # don't overwrite each other.
    shutil.move(
        TRACK_C_EXP_DIR / "per_fold_metrics.csv",
        TRACK_C_EXP_DIR / f"per_fold_metrics_{asset_label}.csv",
    )
    shutil.move(
        TRACK_C_EXP_DIR / "fold_definitions.csv",
        TRACK_C_EXP_DIR / f"fold_definitions_{asset_label}.csv",
    )
    return per_fold_df, fold_def_df


def build_replication_table(
    per_fold_by_asset: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """One tidy row per (asset, stress_def, model) with mean AUC, mean
    drawdown_lift, both 95% bootstrap CIs, and a lift_above_one flag
    (True iff the CI lower bound on lift exceeds 1.0)."""
    rows: list[dict] = []
    for asset_name, per_fold_df in per_fold_by_asset.items():
        auc_means = _per_fold_means(per_fold_df, "auc")
        lift_means = _per_fold_means(per_fold_df, "drawdown_lift")
        for sd_name in per_fold_df["stress_def"].unique():
            for model in per_fold_df["model"].unique():
                auc_vals = auc_means[
                    (auc_means["stress_def"] == sd_name)
                    & (auc_means["model"] == model)
                ]["auc"].to_numpy()
                lift_vals = lift_means[
                    (lift_means["stress_def"] == sd_name)
                    & (lift_means["model"] == model)
                ]["drawdown_lift"].to_numpy()
                auc_ci = block_bootstrap_ci(
                    auc_vals, n_boot=N_BOOT, alpha=ALPHA, seed=BOOTSTRAP_SEED
                )
                lift_ci = block_bootstrap_ci(
                    lift_vals, n_boot=N_BOOT, alpha=ALPHA, seed=BOOTSTRAP_SEED
                )
                lift_above_one = (
                    not np.isnan(lift_ci["ci_low"]) and lift_ci["ci_low"] > 1.0
                )
                rows.append(
                    {
                        "asset": asset_name,
                        "stress_def": sd_name,
                        "model": model,
                        "mean_AUC": auc_ci["mean"],
                        "ci_AUC_low": auc_ci["ci_low"],
                        "ci_AUC_high": auc_ci["ci_high"],
                        "mean_lift": lift_ci["mean"],
                        "ci_lift_low": lift_ci["ci_low"],
                        "ci_lift_high": lift_ci["ci_high"],
                        "lift_above_one": bool(lift_above_one),
                        "n_folds_valid": auc_ci["n_valid"],
                    }
                )
    return pd.DataFrame(rows)


def plot_replication_smallmultiples(
    per_fold_by_asset: dict[str, pd.DataFrame], out_path: Path
) -> None:
    """One panel per asset, grouped bars by (stress_def, model) showing
    mean lift ± 95% bootstrap CI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    asset_names = list(per_fold_by_asset.keys())
    sd_order = [sd["name"] for sd in STRESS_DEFS]
    model_order = list(ALL_MODELS)
    n_models = len(model_order)
    width = 0.8 / n_models
    colors = _palette(n_models)
    x = np.arange(len(sd_order))

    fig, axes = plt.subplots(
        1, len(asset_names), figsize=(7.5 * len(asset_names), 5),
        sharey=True,
    )
    if len(asset_names) == 1:
        axes = [axes]

    for ax_idx, asset_name in enumerate(asset_names):
        ax = axes[ax_idx]
        per_fold_df = per_fold_by_asset[asset_name]
        means = _per_fold_means(per_fold_df, "drawdown_lift")
        for i, model in enumerate(model_order):
            bar_means: list[float] = []
            bar_los: list[float] = []
            bar_his: list[float] = []
            for sd in sd_order:
                vals = means[
                    (means["stress_def"] == sd)
                    & (means["model"] == model)
                ]["drawdown_lift"].to_numpy()
                if vals.size == 0:
                    bar_means.append(np.nan)
                    bar_los.append(np.nan)
                    bar_his.append(np.nan)
                    continue
                ci = block_bootstrap_ci(
                    vals, n_boot=N_BOOT, alpha=ALPHA, seed=BOOTSTRAP_SEED
                )
                bar_means.append(ci["mean"])
                bar_los.append(ci["mean"] - ci["ci_low"])
                bar_his.append(ci["ci_high"] - ci["mean"])

            bar_means_a = np.array(bar_means, dtype=float)
            nan_mask = np.isnan(bar_means_a)
            plot_means = np.where(nan_mask, 0.0, bar_means_a)
            yerr_lower = np.where(nan_mask, 0.0, np.array(bar_los, dtype=float))
            yerr_upper = np.where(nan_mask, 0.0, np.array(bar_his, dtype=float))

            ax.bar(
                x + (i - n_models / 2 + 0.5) * width,
                plot_means, width,
                yerr=[yerr_lower, yerr_upper], capsize=2,
                label=model if ax_idx == 0 else None,
                color=colors[i],
                edgecolor="black", linewidth=0.4, alpha=0.95,
            )
        ax.axhline(
            1.0, color="gray", linestyle="--", linewidth=0.8,
            zorder=0.5, label="lift = 1" if ax_idx == 0 else None,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(sd_order)
        ax.set_xlabel("Stress definition (h, d)")
        if ax_idx == 0:
            ax.set_ylabel("Drawdown lift (mean ± 95% bootstrap CI)")
        ax.set_title(f"{asset_name}")
        ax.grid(axis="y", linestyle=":", linewidth=0.4)

    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Track C: cross-asset replication of drawdown lift")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def _qualitative_generalization(
    track_a_f2: int, track_c_f2: int,
    track_c_f1: str, track_c_f3: dict[str, float],
) -> str:
    """One-word summary: 'fully', 'partially', or 'fails to' generalize."""
    f1_fires = track_c_f1 != "none"
    f2_meaningful = track_c_f2 >= max(5, int(0.3 * track_a_f2))
    f3_near_zero = any(
        not np.isnan(v) and abs(v) <= 0.6 for v in track_c_f3.values()
    )
    if f1_fires and f2_meaningful and f3_near_zero:
        return "fully generalizes"
    if f1_fires or f2_meaningful:
        return "partially generalizes"
    return "fails to generalize"


def print_status(
    per_fold_by_asset: dict[str, pd.DataFrame],
    track_a_f2_baseline: int,
    elapsed_min: float,
) -> None:
    print()
    print("===== TRACK C STATUS =====")
    print(f"total wall time: {elapsed_min:.1f} min")
    summaries: dict[str, str] = {}
    for asset_name, per_fold_df in per_fold_by_asset.items():
        f1 = failure_probe_f1(per_fold_df)
        f2 = failure_probe_f2(per_fold_df)
        f3 = failure_probe_f3(per_fold_df)
        f3_str = ", ".join(f"{k}: {v:.3f}" for k, v in f3.items())
        print()
        print(f"[{asset_name}]")
        print(f"  F1 (AUC-CIs overlap, lift-CIs disjoint): {f1}")
        print(f"  F2 (sign mismatch AUC vs lift):          {f2}")
        print(f"  F3 (Spearman Brier-rank vs lift-rank):   {f3_str}")
        verdict = _qualitative_generalization(
            track_a_f2=track_a_f2_baseline,
            track_c_f2=f2,
            track_c_f1=f1,
            track_c_f3=f3,
        )
        summaries[asset_name] = verdict
        print(f"  qualitative verdict: framework {verdict}")

    print()
    print("Generalization summary (one line per asset):")
    for asset_name, verdict in summaries.items():
        print(f"  {asset_name}: framework {verdict}")
    print("==========================")


def main() -> int:
    t_phase = time.time()

    per_fold_by_asset: dict[str, pd.DataFrame] = {}
    for asset in ASSETS:
        per_fold_df, _ = run_for_asset(asset)
        per_fold_by_asset[asset["name"]] = per_fold_df

    TRACK_C_OUT_DIR.mkdir(parents=True, exist_ok=True)
    rep_table = build_replication_table(per_fold_by_asset)
    rep_table.to_csv(TRACK_C_OUT_DIR / "table_replication.csv", index=False)
    plot_replication_smallmultiples(
        per_fold_by_asset, TRACK_C_OUT_DIR / "fig_replication.pdf"
    )
    print(f"\nWrote {TRACK_C_OUT_DIR / 'table_replication.csv'}")
    print(f"Wrote {TRACK_C_OUT_DIR / 'fig_replication.pdf'}")

    elapsed_min = (time.time() - t_phase) / 60.0

    # Track A's committed F2 was 37 (post-Track-B). Use that as the
    # benchmark for "did F2 reproduce" judgment.
    print_status(per_fold_by_asset, track_a_f2_baseline=37, elapsed_min=elapsed_min)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track C multi-asset replication runner")
    parser.parse_args()
    sys.exit(main())
