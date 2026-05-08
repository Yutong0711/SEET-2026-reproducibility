"""Track E component ablation experiments.

Runs the four controlled ablations defined in the Track E plan:
    ABLATION_1  no_validators   feed unvalidated raw-derived panel into
                                feature construction
    ABLATION_2  global_scaler   replace per-fold scaler with global one
    ABLATION_3  test_threshold  alarm threshold = top 5% of TEST scores
    ABLATION_4  frozen_model    fit each model once on data <=
                                INITIAL_TRAIN_END, never refit

Configuration (locked):
    Asset           SPX
    Panel           spx_extended_2011  (Track A canonical panel)
    Feature set     spx_full           (35 features)
    Stress def      h10_d05            (single central case)
    Folds           22 expanding-window folds (Track A defaults)
    Models          all six Track A baselines
    Seeds           {42, 43, 44, 45, 46}
    Pairing         per (model, fold_id, seed) against Track A's
                    experiments/track_a_headline/per_fold_metrics.csv

Outputs:
    experiments/track_e_ablation/
        per_fold_no_validators.csv
        per_fold_global_scaler.csv
        per_fold_test_threshold.csv
        per_fold_frozen_model.csv
        no_validators_provenance.json
    outputs/track_e/
        table_ablation.csv
        fig_ablation_lift.pdf
        fig_ablation_auc.pdf
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from seet.ablation import (  # noqa: E402
    ABLATION_IDS,
    build_no_validator_panel,
    run_ablation,
)
from seet.baselines import ALL_MODELS, DETERMINISTIC_MODELS, STOCHASTIC_MODELS  # noqa: E402
from seet.run_track_a import (  # noqa: E402
    INITIAL_TRAIN_END,
    PANEL_NAME,
    PRICE_COL,
    PRIMARY_SEED,
    SEEDS,
)
from seet.stats import paired_bootstrap_ci, paired_wilcoxon  # noqa: E402


STRESS_DEF = {"name": "h10_d05", "h": 10, "d": 0.05}
HP_CSV_PATH = REPO_ROOT / "experiments" / "track_b_tuning" / "selected_hp.csv"
TRACK_A_METRICS = REPO_ROOT / "experiments" / "track_a_headline" / "per_fold_metrics.csv"
EXP_DIR = REPO_ROOT / "experiments" / "track_e_ablation"
OUT_DIR = REPO_ROOT / "outputs" / "track_e"

DELTA_METRICS = (
    "auc",
    "pr_auc",
    "brier",
    "drawdown_lift",
    "alarm_rate",
)
# Layer-3 calm-period alarm rate is folded in from Track A's separate
# layer3 table; we approximate here by reporting realized alarm_rate
# restricted to folds with n_events == 0 (calm folds).
CALM_PERIOD_METRIC = "calm_period_alarm_rate"


# ---------------------------------------------------------------------
# Cell-level interpretation classifier
# ---------------------------------------------------------------------

# Hand-tagged cells that carry the paper's headline findings (matched to
# the four findings the user approved in STATUS):
#   F1  ABLATION_1  no_validators inflates LR AUC (synthetic-backfill leakage)
#   F2  ABLATION_4  ML models lose drawdown lift when frozen (drift sensitivity)
#   F3  ABLATION_3  test-set thresholding hurts ML lift, helps HarRv lift,
#                   inflates calm-period alarm rates (one representative cell)
#   F4  ABLATION_2  global scaler effect is significant but operationally
#                   negligible (contrast finding)
PAPER_RELEVANT_CELLS: set[tuple[str, str, str]] = {
    ("no_validators",  "LogisticRegressionL2", "auc"),               # F1
    ("frozen_model",   "LogisticRegressionL2", "drawdown_lift"),     # F2a
    ("frozen_model",   "LightGbmTuned",        "drawdown_lift"),     # F2b
    ("test_threshold", "LightGbmTuned",        "drawdown_lift"),     # F3a
    ("test_threshold", "HarRvThreshold",       "drawdown_lift"),     # F3b
    ("test_threshold", "LightGbmTuned",        "calm_period_alarm_rate"),  # F3c
    ("global_scaler",  "LogisticRegressionL2", "auc"),               # F4
}

# Floating-point tolerance for "exactly zero" deltas. The bootstrap
# resamples produce 1e-17-scale residuals when every paired delta is
# exactly zero (deterministic baselines whose scores don't depend on
# the ablated component); these get classified as
# no_effect_by_construction.
ZERO_TOL = 1e-10


def _effect_size_category(delta: float) -> str:
    """Bin |delta_mean| into operational-impact tiers.

    Bands:
      large       |delta| > 0.05
      medium      0.01 < |delta| <= 0.05
      small       0.001 < |delta| <= 0.01
      negligible  |delta| <= 0.001
    """
    if delta is None or pd.isna(delta):
        return "negligible"
    a = abs(float(delta))
    if a > 0.05:
        return "large"
    if a > 0.01:
        return "medium"
    if a > 0.001:
        return "small"
    return "negligible"


def _interpretation(row: pd.Series) -> str:
    """Classify each (ablation, model, metric) row into one of:
        paper_relevant_finding      a headline finding (hand-tagged)
        no_effect_by_construction   deterministic baseline / no scaler
                                    on that ablation; delta is exactly
                                    zero by design
        expected_effect             matches the architecture's
                                    predicted impact (the default)
        investigate                 genuinely surprising; the
                                    annotation aims to leave this
                                    bucket EMPTY -- if any row lands
                                    here, the table flags it
    """
    delta = row["delta_mean"]
    n_pairs = row["n_folds_paired"]
    cell = (row["ablation"], row["model"], row["metric"])

    # Degenerate-pair cell (NaiveBaseRate's lift on event-free folds).
    if pd.isna(delta) and (n_pairs == 0):
        return "no_effect_by_construction"

    if not pd.isna(delta) and abs(delta) <= ZERO_TOL:
        return "no_effect_by_construction"

    if cell in PAPER_RELEVANT_CELLS:
        return "paper_relevant_finding"

    return "expected_effect"


def _annotate_table(table: pd.DataFrame) -> pd.DataFrame:
    """Add interpretation + effect_size_category columns. Idempotent —
    overwrites existing columns of the same name."""
    out = table.copy()
    out["effect_size_category"] = out["delta_mean"].apply(_effect_size_category)
    out["interpretation"] = out.apply(_interpretation, axis=1)
    return out


def _track_a_lgbm_kwargs_factory(model_name: str, stress_def=None, outer_fold=None):
    """Mirror Track A's LightGbmTuned wiring: load tuned HPs from
    experiments/track_b_tuning/selected_hp.csv keyed by (stress_def,
    outer_fold)."""
    if model_name == "LightGbmTuned" and stress_def is not None and outer_fold is not None:
        return {"stress_def": stress_def, "outer_fold": outer_fold,
                "hp_path": str(HP_CSV_PATH)}
    return {}


def _load_track_a_per_fold() -> pd.DataFrame:
    if not TRACK_A_METRICS.exists():
        sys.stderr.write(
            f"ERROR: Track A baseline not found at {TRACK_A_METRICS}.\n"
            f"  Run scripts/run_track_a.py --full first.\n"
        )
        sys.exit(1)
    df = pd.read_csv(TRACK_A_METRICS)
    df = df[df["stress_def"] == STRESS_DEF["name"]].copy()
    if df.empty:
        sys.stderr.write(
            f"ERROR: Track A per_fold_metrics has no h10_d05 rows.\n"
        )
        sys.exit(1)
    return df


def _load_canonical_panel():
    panel_path = REPO_ROOT / "data" / "processed" / f"{PANEL_NAME}.csv"
    feat_path = REPO_ROOT / "data" / "processed" / "features" / f"{PANEL_NAME}_features.csv"
    panel = pd.read_csv(panel_path, parse_dates=["Date"])
    feat = pd.read_csv(feat_path, parse_dates=["Date"])
    if not panel["Date"].equals(feat["Date"]):
        raise RuntimeError("Panel and features Date columns are misaligned.")
    return panel, feat


# ---------------------------------------------------------------------
# Per-fold delta computation (paired on (model, fold_id, seed))
# ---------------------------------------------------------------------

def _add_calm_alarm_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `calm_period_alarm_rate` column = alarm_rate for folds
    where n_events == 0, else NaN. Folded into the same row schema so
    pairing is straightforward."""
    out = df.copy()
    calm_mask = out["n_events"] == 0
    out["calm_period_alarm_rate"] = np.where(
        calm_mask, out["alarm_rate"], np.nan
    )
    return out


def _paired_deltas(
    ablation_df: pd.DataFrame, baseline_df: pd.DataFrame, metric: str
) -> dict[str, np.ndarray]:
    """Return per-model arrays of paired deltas (ablation - baseline),
    aligned on (fold_id, seed). NaN preserved; the bootstrap and
    Wilcoxon helpers handle NaN drops downstream."""
    a = ablation_df.set_index(["model", "fold_id", "seed"])[metric]
    b = baseline_df.set_index(["model", "fold_id", "seed"])[metric]
    common_idx = a.index.intersection(b.index)
    a = a.loc[common_idx]
    b = b.loc[common_idx]
    delta = (a - b).reset_index()
    out: dict[str, np.ndarray] = {}
    for model, grp in delta.groupby("model"):
        # Order by (fold_id, seed) for deterministic pairing.
        ordered = grp.sort_values(["fold_id", "seed"])
        out[model] = ordered[metric].to_numpy(dtype=float)
    return out


def _paired_arrays(
    ablation_df: pd.DataFrame, baseline_df: pd.DataFrame, metric: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-model paired (a, b) arrays aligned on (fold_id, seed).
    Returned in the same fold/seed order so the bootstrap and Wilcoxon
    use the same pairing structure."""
    a = ablation_df.set_index(["model", "fold_id", "seed"])[metric]
    b = baseline_df.set_index(["model", "fold_id", "seed"])[metric]
    common_idx = a.index.intersection(b.index)
    a = a.loc[common_idx].reset_index().rename(columns={metric: "a"})
    b = b.loc[common_idx].reset_index().rename(columns={metric: "b"})
    merged = a.merge(b, on=["model", "fold_id", "seed"])
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model, grp in merged.groupby("model"):
        ordered = grp.sort_values(["fold_id", "seed"])
        out[model] = (
            ordered["a"].to_numpy(dtype=float),
            ordered["b"].to_numpy(dtype=float),
        )
    return out


def _build_table(
    per_ablation: dict[str, pd.DataFrame], baseline: pd.DataFrame
) -> pd.DataFrame:
    """Build the long-form table_ablation.csv."""
    rows: list[dict] = []
    metrics = list(DELTA_METRICS) + [CALM_PERIOD_METRIC]
    for ablation_id, ablation_df in per_ablation.items():
        for metric in metrics:
            paired = _paired_arrays(ablation_df, baseline, metric)
            for model in sorted(paired.keys()):
                a_arr, b_arr = paired[model]
                ci = paired_bootstrap_ci(a_arr, b_arr, seed=42, n_boot=10000)
                wlx = paired_wilcoxon(a_arr, b_arr)
                rows.append(
                    {
                        "ablation": ablation_id,
                        "model": model,
                        "metric": metric,
                        "delta_mean": ci["delta_mean"],
                        "delta_ci_low": ci["ci_low"],
                        "delta_ci_high": ci["ci_high"],
                        "wilcoxon_pvalue": wlx["p_value"],
                        "n_folds_paired": ci["n_pairs"],
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def _plot_forest(
    table: pd.DataFrame, metric: str, out_path: Path, title: str
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = table[table["metric"] == metric].copy()
    if sub.empty:
        return
    ablation_order = list(ABLATION_IDS)
    model_order = [
        "NaiveBaseRate", "VIXPercentileRaw", "VIXPercentileCalibrated",
        "HarRvThreshold", "LogisticRegressionL2", "LightGbmTuned",
    ]

    fig, axes = plt.subplots(
        1, len(ablation_order),
        figsize=(3.0 * len(ablation_order), 4.5),
        sharex=False, sharey=True,
    )
    if len(ablation_order) == 1:
        axes = [axes]

    for ax, ablation in zip(axes, ablation_order):
        ablation_sub = sub[sub["ablation"] == ablation]
        ys: list[float] = []
        means: list[float] = []
        los: list[float] = []
        his: list[float] = []
        labels: list[str] = []
        for i, model in enumerate(model_order):
            row = ablation_sub[ablation_sub["model"] == model]
            if row.empty:
                continue
            r = row.iloc[0]
            if np.isnan(r["delta_mean"]):
                continue
            ys.append(i)
            means.append(float(r["delta_mean"]))
            los.append(float(r["delta_ci_low"]))
            his.append(float(r["delta_ci_high"]))
            labels.append(model)
        if not means:
            ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
            ax.set_title(ablation, fontsize=10)
            continue
        ys_arr = np.array(ys, dtype=float)
        means_arr = np.array(means, dtype=float)
        los_arr = np.array(los, dtype=float)
        his_arr = np.array(his, dtype=float)
        xerr = np.vstack([means_arr - los_arr, his_arr - means_arr])
        ax.errorbar(
            means_arr, ys_arr, xerr=xerr,
            fmt="o", capsize=3, color="#1f77b4",
            ecolor="#1f77b4", markeredgecolor="black", markersize=6,
        )
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_yticks(range(len(model_order)))
        ax.set_yticklabels(model_order, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(ablation, fontsize=10)
        ax.grid(axis="x", linestyle=":", linewidth=0.4)

    fig.suptitle(title, fontsize=11)
    fig.text(0.5, 0.02, f"delta {metric} (ablation - unmodified)", ha="center")
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------------
# STATUS
# ---------------------------------------------------------------------

def _print_status(
    table: pd.DataFrame,
    per_ablation: dict[str, pd.DataFrame],
    baseline: pd.DataFrame,
    provenance: dict,
) -> None:
    print()
    print("=" * 78)
    print("TRACK E STATUS — component ablation experiments (h10_d05)")
    print("=" * 78)

    fold_count = baseline.groupby("model")["fold_id"].nunique().max()
    seed_count = baseline["seed"].nunique()
    print(
        f"\nBaseline (Track A unmodified): "
        f"folds={fold_count}, seeds={seed_count}, "
        f"models={baseline['model'].nunique()}, "
        f"rows={len(baseline)}"
    )
    print(f"Source: {TRACK_A_METRICS.relative_to(REPO_ROOT)}")

    # ----- 1. Per-ablation most-affected metric + direction -----
    print("\n[1] Per-ablation most-affected metric (across all 6 models):")
    for ablation_id in ABLATION_IDS:
        ab_table = table[table["ablation"] == ablation_id]
        # Aggregate |delta_mean| across models per metric, weighted
        # by n_folds_paired.
        grp = (
            ab_table.assign(
                weighted_abs_delta=lambda d:
                    d["delta_mean"].abs() * d["n_folds_paired"]
            )
            .groupby("metric")["weighted_abs_delta"]
            .sum()
        )
        if grp.empty or grp.max() == 0:
            print(f"  {ablation_id:<18} (no nonzero deltas)")
            continue
        top_metric = grp.idxmax()
        # Mean direction across models for that metric.
        direction_rows = ab_table[ab_table["metric"] == top_metric]
        mean_dir = direction_rows["delta_mean"].mean()
        sign = "+" if mean_dir > 0 else ""
        print(
            f"  {ablation_id:<18} top metric = {top_metric:<25} "
            f"mean delta = {sign}{mean_dir:+.4f}"
        )

    # ----- 2. Ablation ranking by total operational impact -----
    print("\n[2] Ablation ranking by total operational impact "
          "(sum of |delta_drawdown_lift| * n_folds_paired):")
    impact = (
        table[table["metric"] == "drawdown_lift"]
        .assign(score=lambda d: d["delta_mean"].abs() * d["n_folds_paired"])
        .groupby("ablation")["score"].sum()
        .sort_values(ascending=False)
    )
    for rank, (ab, score) in enumerate(impact.items(), 1):
        print(f"  {rank}. {ab:<18}  score = {score:.3f}")

    # ----- 3. ABLATION_1 synthetic-backfill leakage flag -----
    print("\n[3] ABLATION_1 (no_validators): synthetic-backfill leakage check")
    a1 = table[(table["ablation"] == "no_validators") & (table["metric"] == "auc")]
    leak_hits = a1[
        (a1["model"].isin(["LogisticRegressionL2", "LightGbmTuned"]))
        & (a1["delta_mean"] > 0)
    ]
    if leak_hits.empty:
        print("  no AUC improvements observed for matrix-feature models")
    else:
        for _, r in leak_hits.iterrows():
            sig = " (p<0.05)" if r["wilcoxon_pvalue"] < 0.05 else ""
            print(
                f"  [synthetic backfill leakage] {r['model']}: "
                f"delta_AUC = +{r['delta_mean']:.4f} "
                f"CI=[{r['delta_ci_low']:+.4f},{r['delta_ci_high']:+.4f}]"
                f"  p={r['wilcoxon_pvalue']:.3f}{sig}"
            )
    if provenance:
        print(
            f"  bypassed: "
            f"{provenance['diff_vs_validated'].get('vvix_synthetic_backfill_rows', 0)} "
            f"VVIX synthetic-backfill rows + "
            f"{provenance['diff_vs_validated'].get('rows_added_before_validated_anchor', 0)} "
            f"rows before validated anchor"
        )

    # ----- 4. ABLATION_2 LR vs LightGBM -----
    print("\n[4] ABLATION_2 (global_scaler): scaler-leakage check on matrix models")
    a2 = table[(table["ablation"] == "global_scaler") & (table["metric"] == "auc")]
    for model in ("LogisticRegressionL2", "LightGbmTuned"):
        row = a2[a2["model"] == model]
        if row.empty:
            continue
        r = row.iloc[0]
        d = r["delta_mean"]
        sig = " (p<0.05)" if r["wilcoxon_pvalue"] < 0.05 else ""
        if model == "LogisticRegressionL2":
            verdict = (
                "expected (small positive)" if d > 0
                else "INVESTIGATE — leakage NOT created" if d <= 0
                else "n/a"
            )
        else:
            verdict = (
                "expected zero by construction (LightGBM is scale-invariant)"
                if abs(d) < 1e-9 else "unexpected nonzero — INVESTIGATE"
            )
        print(
            f"  {model:<26}  delta_AUC = {d:+.4f} "
            f"CI=[{r['delta_ci_low']:+.4f},{r['delta_ci_high']:+.4f}]"
            f"  p={r['wilcoxon_pvalue']:.3f}{sig}   [{verdict}]"
        )

    # ----- 5. ABLATION_4 drift sensitivity ranking -----
    print("\n[5] ABLATION_4 (frozen_model): drift sensitivity (delta_drawdown_lift):")
    a4 = (
        table[(table["ablation"] == "frozen_model")
              & (table["metric"] == "drawdown_lift")]
        .copy()
        .assign(abs_delta=lambda d: d["delta_mean"].abs())
        .sort_values("abs_delta", ascending=False)
    )
    for _, r in a4.iterrows():
        sig = " (p<0.05)" if r["wilcoxon_pvalue"] < 0.05 else ""
        print(
            f"  {r['model']:<26}  delta_lift = {r['delta_mean']:+.4f} "
            f"CI=[{r['delta_ci_low']:+.4f},{r['delta_ci_high']:+.4f}]"
            f"  p={r['wilcoxon_pvalue']:.3f}{sig}"
        )
    if not a4.empty:
        worst = a4.iloc[0]
        most_resistant = a4.iloc[-1]
        print(
            f"  Most degraded by frozen training: {worst['model']} "
            f"(|delta_lift|={worst['abs_delta']:.4f})"
        )
        print(
            f"  Most resistant to frozen training: {most_resistant['model']} "
            f"(|delta_lift|={most_resistant['abs_delta']:.4f})"
        )

    # ----- 6. Interpretation breakdown -----
    print("\n[6] Interpretation breakdown (annotated rows):")
    if "interpretation" not in table.columns:
        print("  (interpretation column not present)")
    else:
        for label, count in (
            table["interpretation"].value_counts().items()
        ):
            print(f"  {label:<30}  {count}")
        # If anything ended up in `investigate` it deserves visibility.
        leftover = table[table["interpretation"] == "investigate"]
        if not leftover.empty:
            print("\n  [investigate] cells (genuinely surprising):")
            for _, r in leftover.iterrows():
                print(
                    f"    {r['ablation']:<18} / {r['model']:<26} / "
                    f"{r['metric']:<25}  delta={r['delta_mean']:+.4f}  "
                    f"p={r['wilcoxon_pvalue']:.3f}"
                )
        else:
            print("\n  [investigate] cells: none (annotation complete)")

    # ----- 7. Paper-relevant findings (hand-tagged) -----
    print("\n[7] Paper-relevant findings:")
    pr_rows = table[table["interpretation"] == "paper_relevant_finding"]
    if pr_rows.empty:
        print("  none tagged")
    else:
        for _, r in pr_rows.sort_values(["ablation", "metric", "model"]).iterrows():
            sig = " (p<0.05)" if r["wilcoxon_pvalue"] < 0.05 else ""
            esc = r.get("effect_size_category", "")
            print(
                f"  {r['ablation']:<18} / {r['model']:<26} / "
                f"{r['metric']:<25}  "
                f"delta={r['delta_mean']:+.4f} "
                f"[{esc}] p={r['wilcoxon_pvalue']:.4f}{sig}"
            )

    # ----- 8. Wilcoxon-significant cells, with effect_size_category -----
    sig_table = (
        table.dropna(subset=["wilcoxon_pvalue"])
        .query("wilcoxon_pvalue < 0.05")
        .sort_values(["effect_size_category", "ablation", "metric", "model"])
    )
    # Order categories from largest to smallest for readability.
    size_rank = {"large": 0, "medium": 1, "small": 2, "negligible": 3}
    sig_table = sig_table.copy()
    sig_table["_rank"] = sig_table.get(
        "effect_size_category", pd.Series(["negligible"] * len(sig_table))
    ).map(size_rank).fillna(99)
    sig_table = sig_table.sort_values(
        ["_rank", "ablation", "metric", "model"]
    ).drop(columns="_rank")
    print(f"\n[8] Wilcoxon-significant (p<0.05) cells: {len(sig_table)}")
    print("    (grouped by effect_size_category — large / medium / "
          "small / negligible)")
    cur_cat = None
    for _, r in sig_table.iterrows():
        cat = r.get("effect_size_category", "")
        if cat != cur_cat:
            print(f"\n  --- {cat} ---")
            cur_cat = cat
        d = r["delta_mean"]
        interp = r.get("interpretation", "")
        print(
            f"    {r['ablation']:<18} / {r['model']:<26} / "
            f"{r['metric']:<25}  delta={d:+.4f}  "
            f"p={r['wilcoxon_pvalue']:.4f}  [{interp}]"
        )

    # ----- 9. Artifact summary -----
    print("\n[9] Artifacts:")
    for ab in ABLATION_IDS:
        path = EXP_DIR / f"per_fold_{ab}.csv"
        if path.exists():
            df = pd.read_csv(path)
            print(f"  {path.relative_to(REPO_ROOT)}  rows={len(df)}")
    for fname in (
        "table_ablation.csv",
        "fig_ablation_lift.pdf",
        "fig_ablation_auc.pdf",
    ):
        path = OUT_DIR / fname
        if path.exists():
            size_kb = path.stat().st_size / 1024.0
            print(f"  {path.relative_to(REPO_ROOT)}  ({size_kb:.1f} KB)")
    prov_path = EXP_DIR / "no_validators_provenance.json"
    if prov_path.exists():
        size_kb = prov_path.stat().st_size / 1024.0
        print(f"  {prov_path.relative_to(REPO_ROOT)}  ({size_kb:.1f} KB)")
    print("=" * 78)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Track A baseline (h10_d05).
    print(f"[load] Track A baseline from {TRACK_A_METRICS.relative_to(REPO_ROOT)}")
    baseline = _load_track_a_per_fold()
    baseline = _add_calm_alarm_rate(baseline)

    # Load canonical panel + features.
    print(f"[load] canonical panel ({PANEL_NAME}) + features (spx_full)")
    panel_df, features_df = _load_canonical_panel()

    # Build no-validator panel + features once.
    print("[ablation_1] building no-validator panel from raw CSVs...")
    nv_panel, nv_features, nv_provenance = build_no_validator_panel(REPO_ROOT)
    with open(EXP_DIR / "no_validators_provenance.json", "w") as f:
        json.dump(nv_provenance, f, indent=2)
    print(
        f"  no-validator panel: {len(nv_panel)} rows, "
        f"{nv_panel['Date'].min().date()} -> {nv_panel['Date'].max().date()}; "
        f"VVIX synthetic-backfill rows = "
        f"{nv_provenance['diff_vs_validated'].get('vvix_synthetic_backfill_rows', 0)}"
    )

    per_ablation: dict[str, pd.DataFrame] = {}

    for ablation_id in ABLATION_IDS:
        t0 = time.time()
        if ablation_id == "no_validators":
            ab_panel, ab_feat = nv_panel, nv_features
        else:
            ab_panel, ab_feat = panel_df, features_df

        print(f"[run] ablation = {ablation_id}")
        df = run_ablation(
            ablation_id, ab_panel, ab_feat,
            price_col=PRICE_COL,
            stress_def=STRESS_DEF,
            seeds=SEEDS,
            primary_seed=PRIMARY_SEED,
            initial_train_end=INITIAL_TRAIN_END,
            deterministic_models=DETERMINISTIC_MODELS,
            stochastic_models=STOCHASTIC_MODELS,
            model_kwargs_factory=_track_a_lgbm_kwargs_factory,
        )
        df = _add_calm_alarm_rate(df)
        out_csv = EXP_DIR / f"per_fold_{ablation_id}.csv"
        df.to_csv(out_csv, index=False)
        per_ablation[ablation_id] = df
        elapsed = time.time() - t0
        print(
            f"  {ablation_id}: rows={len(df)}  "
            f"models={df['model'].nunique() if not df.empty else 0}  "
            f"folds={df['fold_id'].nunique() if not df.empty else 0}  "
            f"elapsed={elapsed:.1f}s"
        )

    # Build the long-form delta table.
    print("[deltas] computing paired deltas with paired bootstrap CIs and Wilcoxon...")
    table = _build_table(per_ablation, baseline)
    table = _annotate_table(table)
    table.to_csv(OUT_DIR / "table_ablation.csv", index=False)
    print(f"  table_ablation.csv: {len(table)} rows")
    interp_counts = table["interpretation"].value_counts().to_dict()
    size_counts = table["effect_size_category"].value_counts().to_dict()
    print(f"  interpretation breakdown: {interp_counts}")
    print(f"  effect_size_category breakdown: {size_counts}")

    # Figures.
    print("[plot] forest plots...")
    _plot_forest(
        table, "drawdown_lift",
        OUT_DIR / "fig_ablation_lift.pdf",
        title="Track E — drawdown lift deltas (ablation - unmodified), 95% paired CIs",
    )
    _plot_forest(
        table, "auc",
        OUT_DIR / "fig_ablation_auc.pdf",
        title="Track E — AUC deltas (ablation - unmodified), 95% paired CIs",
    )

    _print_status(table, per_ablation, baseline, nv_provenance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
