"""Track F — failure injection / mutation testing.

Runs five corruption modes (C1 duplicate_dates, C2 missing_values,
C3 stale_quotes, C4 extreme_jumps, C5 calendar_gaps), each at three
rates (1%, 5%, 10%) and five seeds (42–46). For every
(corruption, rate, seed) cell the script:

  1. Applies the corruption to a copy of the canonical panel.
  2. Runs the V1+V2 validator on the corrupted panel and records
     precision/recall/F1 vs ground-truth flags.
  3. Runs the full h10_d05 pipeline TWICE — once with validators
     ENABLED (flagged rows dropped before features) and once
     SILENCED (validators bypassed; corrupted data flows through).
  4. Computes per-fold deltas (ENABLED - SILENCED) on AUC, PR-AUC,
     Brier, drawdown_lift, alarm_rate.

Pairing is per (model, fold_id, seed) within the same corrupted
panel. CIs use a paired bootstrap (10k resamples, fold-level
resampling preserves pairing). Significance is paired Wilcoxon.

Configuration (locked):
    Asset       SPX
    Panel       spx_extended_2011  (canonical)
    Features    spx_full           (35 features)
    Stress def  h=10, d=5%
    Folds       22 expanding-window folds (Track A defaults)
    Models      LogisticRegressionL2, LightGbmTuned
    Seeds       42–46

Outputs:
    experiments/track_f_injection/
        validator_coverage.csv         (75 rows: 5 cor × 3 rates × 5 seeds)
        silenced_impact.csv            (150 rows: × 2 models)
        corruption_provenance.json     (per-cell metadata)
        per_fold_*.csv                 (regenerable, gitignored if any)
    outputs/track_f/
        table_validator_coverage.csv
        table_silenced_impact.csv
        fig_coverage.pdf
        fig_silenced_impact.pdf
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

from seet.baselines import (  # noqa: E402
    DETERMINISTIC_MODELS,
    STOCHASTIC_MODELS,
)
from seet.features import build_features  # noqa: E402
from seet.injection import (  # noqa: E402
    CORRUPTION_IDS,
    GAP_MECHANISM,
    corrupt_panel,
)
from seet.pipeline import run_grid  # noqa: E402
from seet.run_track_a import (  # noqa: E402
    INITIAL_TRAIN_END,
    PANEL_NAME,
    PRICE_COL,
    PRIMARY_SEED,
    SEEDS,
)
from seet.stats import paired_bootstrap_ci, paired_wilcoxon  # noqa: E402
from seet.validators import (  # noqa: E402
    apply_treatment,
    coverage_metrics,
    validate_panel,
)


STRESS_DEF = {"name": "h10_d05", "h": 10, "d": 0.05}
RATES = (0.01, 0.05, 0.10)
TRACK_F_SEEDS = (42, 43, 44, 45, 46)
TRACK_F_MODELS = ("LogisticRegressionL2", "LightGbmTuned")
TRACK_F_DETERMINISTIC = ("LogisticRegressionL2",)
TRACK_F_STOCHASTIC = ("LightGbmTuned",)
HP_CSV_PATH = REPO_ROOT / "experiments" / "track_b_tuning" / "selected_hp.csv"
EXP_DIR = REPO_ROOT / "experiments" / "track_f_injection"
OUT_DIR = REPO_ROOT / "outputs" / "track_f"


def _track_a_lgbm_kwargs_factory(model_name, stress_def=None, outer_fold=None):
    if model_name == "LightGbmTuned" and stress_def is not None and outer_fold is not None:
        return {"stress_def": stress_def, "outer_fold": outer_fold,
                "hp_path": str(HP_CSV_PATH)}
    return {}


def _load_panel_and_manifest():
    panel = pd.read_csv(
        REPO_ROOT / "data" / "processed" / f"{PANEL_NAME}.csv",
        parse_dates=["Date"],
    )
    with open(REPO_ROOT / "data" / "manifests" / "processed_manifest.json") as f:
        manifest = json.load(f)
    return panel, manifest[PANEL_NAME]


# ---------------------------------------------------------------------
# Single-cell runner: one (corruption_id, rate, seed) cell
# ---------------------------------------------------------------------

def _run_pipeline_branch(
    panel_df: pd.DataFrame,
    features_df: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """Run pipeline.run_grid and return the per-fold DataFrame.
    `out_dir` is created and used for the per-fold CSV; predictions
    are not persisted (set save_predictions=False) — Track F only
    needs the per-fold metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold_df, _, _ = run_grid(
        panel_df, features_df, out_dir,
        price_col=PRICE_COL,
        stress_defs=[STRESS_DEF],
        seeds=TRACK_F_SEEDS,
        primary_seed=PRIMARY_SEED,
        initial_train_end=INITIAL_TRAIN_END,
        deterministic_models=TRACK_F_DETERMINISTIC,
        stochastic_models=TRACK_F_STOCHASTIC,
        model_kwargs_factory=_track_a_lgbm_kwargs_factory,
        save_predictions=False,
    )
    return per_fold_df


def _run_one_cell(
    corruption_id: str,
    rate: float,
    seed: int,
    canonical_panel: pd.DataFrame,
    manifest_entry: dict,
    cell_workdir: Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    """Run one (corruption, rate, seed) cell. Returns:
        coverage_row : dict (one row of validator_coverage.csv)
        enabled_per_fold : DataFrame (h10_d05 per-fold, ENABLED branch)
        silenced_per_fold : DataFrame (h10_d05 per-fold, SILENCED branch)
        metadata : corruption metadata (sans large fields, for the
                   provenance record)
    """
    # 1. Corrupt the panel.
    corrupted, gt_flags, meta = corrupt_panel(
        canonical_panel, corruption_id, rate, seed,
        manifest_entry=manifest_entry,
    )

    # 2. Validate.
    v_result = validate_panel(corrupted, manifest_entry, enabled=True)
    v_flags = v_result["flags"]
    cov = coverage_metrics(v_flags, gt_flags)
    coverage_row = {
        "corruption_type": corruption_id,
        "rate": float(rate),
        "seed": int(seed),
        "precision": cov["precision"],
        "recall": cov["recall"],
        "f1": cov["f1"],
        "n_flagged_truth": cov["n_truth"],
        "n_flagged_validator": cov["n_validator"],
        "n_true_positive": cov["n_true_positive"],
        "gap_mechanism": GAP_MECHANISM[corruption_id],
    }

    # 3. ENABLED branch: apply per-rule treatments (V1: dedup,
    #    V2: drop unexpected-NaN rows), then build features on the
    #    post-treatment panel, then run the pipeline.
    enabled_panel = apply_treatment(corrupted, v_result, manifest_entry)
    enabled_features, _ = build_features(enabled_panel, "spx_full")
    enabled_per_fold = _run_pipeline_branch(
        enabled_panel, enabled_features, cell_workdir / "enabled",
    )
    enabled_per_fold = enabled_per_fold.assign(
        corruption_type=corruption_id, rate=float(rate),
        injection_seed=int(seed), branch="enabled",
    )

    # 4. SILENCED branch: build features on the corrupted panel
    #    directly, then run the pipeline.
    silenced_features, _ = build_features(corrupted, "spx_full")
    silenced_per_fold = _run_pipeline_branch(
        corrupted, silenced_features, cell_workdir / "silenced",
    )
    silenced_per_fold = silenced_per_fold.assign(
        corruption_type=corruption_id, rate=float(rate),
        injection_seed=int(seed), branch="silenced",
    )

    # Trim metadata for the provenance file (keep summary only).
    meta_summary = {
        k: v for k, v in meta.items()
        if k not in ("first_corrupted_cells", "first_runs", "first_jumps", "removed_dates")
    }
    return coverage_row, enabled_per_fold, silenced_per_fold, meta_summary


# ---------------------------------------------------------------------
# Per-fold delta computation
# ---------------------------------------------------------------------

DELTA_METRICS = ("auc", "pr_auc", "brier", "drawdown_lift", "alarm_rate")


def _per_fold_deltas(
    enabled_df: pd.DataFrame, silenced_df: pd.DataFrame
) -> dict[str, dict[str, np.ndarray]]:
    """For each model, compute paired arrays of (enabled, silenced)
    per-fold values aligned on (fold_id, seed). Returns:

        out[model][metric] -> np.ndarray of deltas (enabled - silenced)
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    a = enabled_df.set_index(["model", "fold_id", "seed"])
    b = silenced_df.set_index(["model", "fold_id", "seed"])
    common = a.index.intersection(b.index)
    a = a.loc[common]
    b = b.loc[common]
    for model in TRACK_F_MODELS:
        if model not in a.index.get_level_values("model"):
            continue
        a_m = a.xs(model, level="model").sort_index()
        b_m = b.xs(model, level="model").sort_index()
        out[model] = {
            metric: (a_m[metric] - b_m[metric]).to_numpy(dtype=float)
            for metric in DELTA_METRICS
            if metric in a_m.columns and metric in b_m.columns
        }
    return out


# ---------------------------------------------------------------------
# Aggregation: per-cell summaries for the output tables
# ---------------------------------------------------------------------

def _aggregate_coverage(coverage_rows: list[dict]) -> pd.DataFrame:
    """table_validator_coverage.csv: one row per (corruption_type, rate)
    with mean + 95% bootstrap CI of precision, recall, f1."""
    df = pd.DataFrame(coverage_rows)
    rows: list[dict] = []
    for (cid, rate), grp in df.groupby(["corruption_type", "rate"]):
        gap = GAP_MECHANISM[cid]
        for metric in ("precision", "recall", "f1"):
            vals = grp[metric].to_numpy(dtype=float)
            valid = vals[~np.isnan(vals)]
            mean = float(valid.mean()) if valid.size else float("nan")
            if valid.size >= 2:
                rng = np.random.default_rng(42)
                boots = np.empty(10000, dtype=float)
                for k in range(10000):
                    idx = rng.integers(0, valid.size, size=valid.size)
                    boots[k] = valid[idx].mean()
                ci_lo = float(np.percentile(boots, 2.5))
                ci_hi = float(np.percentile(boots, 97.5))
            else:
                ci_lo = ci_hi = float("nan")
            rows.append({
                "corruption_type": cid,
                "rate": float(rate),
                "metric": metric,
                "mean": mean,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "n_seeds": int(grp.shape[0]),
                "gap_mechanism": gap,
            })
    return pd.DataFrame(rows)


def _aggregate_silenced(silenced_rows: list[dict]) -> pd.DataFrame:
    """table_silenced_impact.csv: one row per (corruption, rate, model,
    metric). Aggregates the per-cell paired deltas across the 5 seeds."""
    df = pd.DataFrame(silenced_rows)
    return df


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def _fig_coverage(coverage_table: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = coverage_table[coverage_table["metric"] == "recall"]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for color, cid in zip(colors, CORRUPTION_IDS):
        grp = sub[sub["corruption_type"] == cid].sort_values("rate")
        if grp.empty:
            continue
        rates = grp["rate"].to_numpy()
        means = grp["mean"].to_numpy()
        los = grp["ci_low"].to_numpy()
        his = grp["ci_high"].to_numpy()
        # NaN-safe error bars: zero-width when CI undefined.
        yerr_lo = np.where(np.isnan(means - los), 0.0, means - los)
        yerr_hi = np.where(np.isnan(his - means), 0.0, his - means)
        plot_means = np.where(np.isnan(means), 0.0, means)
        ax.errorbar(
            rates, plot_means, yerr=[yerr_lo, yerr_hi],
            fmt="o-", color=color, label=f"{cid}  ({GAP_MECHANISM[cid]})",
            capsize=3, markersize=6,
        )
    ax.set_xlabel("Corruption rate")
    ax.set_ylabel("Validator recall (mean ± 95% bootstrap CI)")
    ax.set_title("Track F — V1+V2 detection coverage by corruption type")
    ax.set_ylim(-0.05, 1.10)
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.6)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.6)
    ax.grid(linestyle=":", linewidth=0.4)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def _fig_silenced_impact(silenced_table: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = silenced_table[silenced_table["metric"] == "drawdown_lift"]
    if sub.empty:
        return

    rates = sorted(sub["rate"].unique())
    fig, axes = plt.subplots(1, len(rates), figsize=(4.0 * len(rates), 5.0),
                             sharey=True)
    if len(rates) == 1:
        axes = [axes]
    cell_keys = [
        (cid, model)
        for cid in CORRUPTION_IDS
        for model in TRACK_F_MODELS
    ]
    y_labels = [f"{cid} / {model}" for cid, model in cell_keys]
    y_pos = np.arange(len(cell_keys))

    for ax, rate in zip(axes, rates):
        rate_sub = sub[sub["rate"] == rate]
        means = []
        los = []
        his = []
        for cid, model in cell_keys:
            row = rate_sub[
                (rate_sub["corruption_type"] == cid)
                & (rate_sub["model"] == model)
            ]
            if row.empty or pd.isna(row.iloc[0]["delta_mean"]):
                means.append(np.nan)
                los.append(np.nan)
                his.append(np.nan)
            else:
                r = row.iloc[0]
                means.append(float(r["delta_mean"]))
                los.append(float(r["delta_ci_low"]))
                his.append(float(r["delta_ci_high"]))
        means = np.array(means)
        los = np.array(los)
        his = np.array(his)
        nan_mask = np.isnan(means)
        plot_means = np.where(nan_mask, 0.0, means)
        xerr_lo = np.where(nan_mask, 0.0, plot_means - los)
        xerr_hi = np.where(nan_mask, 0.0, his - plot_means)

        ax.errorbar(plot_means, y_pos, xerr=[xerr_lo, xerr_hi],
                    fmt="o", capsize=3, color="#1f77b4", markersize=5,
                    markeredgecolor="black", linewidth=0.8)
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"rate = {rate:.0%}", fontsize=10)
        ax.grid(axis="x", linestyle=":", linewidth=0.4)

    fig.suptitle(
        "Track F — delta drawdown lift (ENABLED - SILENCED), 95% paired CIs",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------------
# STATUS
# ---------------------------------------------------------------------

def _print_status(
    coverage_table: pd.DataFrame, silenced_table: pd.DataFrame
) -> None:
    print()
    print("=" * 78)
    print("TRACK F STATUS — failure injection / mutation testing")
    print("=" * 78)

    # ----- (a) Detection coverage -----
    print("\n[a] Detection coverage (recall by corruption type at each rate):")
    rec = coverage_table[coverage_table["metric"] == "recall"]
    print(f"  {'corruption':<18} {'rate':>6} {'recall':>8} "
          f"{'CI':>22}  mechanism")
    print("  " + "-" * 70)
    for cid in CORRUPTION_IDS:
        for rate in sorted(rec[rec["corruption_type"] == cid]["rate"].unique()):
            r = rec[
                (rec["corruption_type"] == cid)
                & (rec["rate"] == rate)
            ].iloc[0]
            mean_str = f"{r['mean']:.3f}" if not pd.isna(r["mean"]) else "n/a"
            if pd.isna(r["ci_low"]) or pd.isna(r["ci_high"]):
                ci_str = "[n/a]"
            else:
                ci_str = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
            print(
                f"  {cid:<18} {rate:>6.0%} {mean_str:>8} {ci_str:>22}  "
                f"{r['gap_mechanism']}"
            )

    # Weakest by recall: lowest mean recall across rates.
    weakest = (
        rec.dropna(subset=["mean"])
        .groupby("corruption_type")["mean"].mean()
        .sort_values()
    )
    if not weakest.empty:
        print(f"\n  Weakest detection: {weakest.index[0]} "
              f"(mean recall across rates = {weakest.iloc[0]:.3f})")

    # ----- (b) Operational cost when validator catches the corruption -----
    print("\n[b] Operational cost of silenced detection — corruption types "
          "the validator CATCHES (C1, C2):")
    caught = ("duplicate_dates", "missing_values")
    cost_caught = silenced_table[
        (silenced_table["corruption_type"].isin(caught))
        & (silenced_table["metric"].isin(("auc", "drawdown_lift")))
    ].sort_values(["corruption_type", "rate", "metric", "model"])
    print(f"  {'corruption':<18} {'rate':>6} {'model':<26} {'metric':<14} "
          f"{'delta':>10} {'CI':>22} {'p':>8}")
    print("  " + "-" * 100)
    for _, r in cost_caught.iterrows():
        d = r["delta_mean"]
        sig = "*" if (not pd.isna(r["wilcoxon_pvalue"])
                      and r["wilcoxon_pvalue"] < 0.05) else " "
        d_str = f"{d:+.4f}" if not pd.isna(d) else "n/a"
        if pd.isna(r["delta_ci_low"]) or pd.isna(r["delta_ci_high"]):
            ci_str = "[n/a]"
        else:
            ci_str = f"[{r['delta_ci_low']:+.4f},{r['delta_ci_high']:+.4f}]"
        p_str = (
            f"{r['wilcoxon_pvalue']:.3f}"
            if not pd.isna(r["wilcoxon_pvalue"]) else "n/a"
        )
        print(
            f"  {r['corruption_type']:<18} {r['rate']:>6.0%} {r['model']:<26} "
            f"{r['metric']:<14} {d_str:>10} {ci_str:>22} {p_str:>8}{sig}"
        )

    # ----- (c) Architectural validation gap -----
    print("\n[c] Architectural validation gap — corruption types the v1 "
          "architecture CANNOT detect:")
    gap_types = [
        cid for cid in CORRUPTION_IDS
        if GAP_MECHANISM[cid] in ("silent_ignore", "removed_before_inspection")
    ]
    print(f"  Gap types: {', '.join(gap_types)}")
    cost_gap = silenced_table[
        (silenced_table["corruption_type"].isin(gap_types))
        & (silenced_table["metric"].isin(("auc", "drawdown_lift")))
    ].sort_values(["corruption_type", "rate", "metric", "model"])
    print(f"\n  Operational impact of these uncaught corruptions on the "
          f"matrix-feature models")
    print(f"  (delta is ENABLED - SILENCED, but for these gap types ENABLED")
    print(f"  ≡ SILENCED w.r.t. the validator — the corruption flows through")
    print(f"  in BOTH branches, so any nonzero delta reflects downstream NaN")
    print(f"  hygiene in fit_predict, not validator action):")
    print(f"  {'corruption':<18} {'rate':>6} {'model':<26} {'metric':<14} "
          f"{'delta':>10} {'p':>8}")
    print("  " + "-" * 90)
    for _, r in cost_gap.iterrows():
        d = r["delta_mean"]
        sig = "*" if (not pd.isna(r["wilcoxon_pvalue"])
                      and r["wilcoxon_pvalue"] < 0.05) else " "
        d_str = f"{d:+.4f}" if not pd.isna(d) else "n/a"
        p_str = (
            f"{r['wilcoxon_pvalue']:.3f}"
            if not pd.isna(r["wilcoxon_pvalue"]) else "n/a"
        )
        print(
            f"  {r['corruption_type']:<18} {r['rate']:>6.0%} {r['model']:<26} "
            f"{r['metric']:<14} {d_str:>10} {p_str:>8}{sig}"
        )

    # Paper-relevant headline.
    zero_recall_cells = rec[
        (rec["mean"] < 0.10) | (rec["mean"].isna())
    ]
    paper_findings = sorted(zero_recall_cells["corruption_type"].unique())
    print(f"\n[paper finding] Validator recall < 0.10 (or undefined) at any "
          f"rate, across {len(paper_findings)} corruption type(s):")
    for cid in paper_findings:
        print(f"  - {cid}  ({GAP_MECHANISM[cid]})")
    print(
        "\n  These are the v1 architecture's blind spots: corruptions of "
        "these\n  types can flow through the pipeline undetected. Track F "
        "quantifies\n  the operational impact of each in section [c] above."
    )

    # ----- artifact summary -----
    print("\n[artifacts]")
    for fname in (
        "experiments/track_f_injection/validator_coverage.csv",
        "experiments/track_f_injection/silenced_impact.csv",
        "experiments/track_f_injection/corruption_provenance.json",
        "outputs/track_f/table_validator_coverage.csv",
        "outputs/track_f/table_silenced_impact.csv",
        "outputs/track_f/fig_coverage.pdf",
        "outputs/track_f/fig_silenced_impact.pdf",
    ):
        path = REPO_ROOT / fname
        if path.exists():
            size_kb = path.stat().st_size / 1024.0
            print(f"  {fname}  ({size_kb:.1f} KB)")
    print("=" * 78)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    canonical_panel, manifest_entry = _load_panel_and_manifest()
    print(f"[load] canonical panel ({PANEL_NAME}): {len(canonical_panel)} rows, "
          f"{canonical_panel['Date'].min().date()} -> "
          f"{canonical_panel['Date'].max().date()}")

    coverage_rows: list[dict] = []
    silenced_rows: list[dict] = []
    provenance: dict = {}

    n_cells_total = len(CORRUPTION_IDS) * len(RATES) * len(TRACK_F_SEEDS)
    cell_n = 0
    t0_global = time.time()

    for cid in CORRUPTION_IDS:
        for rate in RATES:
            cell_enabled_dfs: list[pd.DataFrame] = []
            cell_silenced_dfs: list[pd.DataFrame] = []
            for seed in TRACK_F_SEEDS:
                cell_n += 1
                t0 = time.time()
                cell_workdir = EXP_DIR / "_workdir" / f"{cid}_{rate:.2f}_{seed}"
                cov_row, en_df, si_df, meta_summary = _run_one_cell(
                    cid, rate, seed,
                    canonical_panel, manifest_entry, cell_workdir,
                )
                coverage_rows.append(cov_row)
                cell_enabled_dfs.append(en_df)
                cell_silenced_dfs.append(si_df)
                provenance[f"{cid}/{rate:.2f}/{seed}"] = meta_summary
                elapsed = time.time() - t0
                print(
                    f"  [{cell_n:>2}/{n_cells_total}] {cid:<18} rate={rate:.2f} "
                    f"seed={seed}  recall={cov_row['recall']:.3f} "
                    f"  enabled_rows={len(en_df)} silenced_rows={len(si_df)} "
                    f"  elapsed={elapsed:.1f}s"
                )

            # Aggregate the 5 seeds at this (cid, rate) into per-fold deltas.
            enabled_concat = pd.concat(cell_enabled_dfs, ignore_index=True)
            silenced_concat = pd.concat(cell_silenced_dfs, ignore_index=True)
            for seed in TRACK_F_SEEDS:
                en_seed = enabled_concat[enabled_concat["injection_seed"] == seed]
                si_seed = silenced_concat[silenced_concat["injection_seed"] == seed]
                deltas = _per_fold_deltas(en_seed, si_seed)
                for model, metric_dict in deltas.items():
                    for metric, delta_arr in metric_dict.items():
                        # Pair the same (model, fold_id, seed) tuple between
                        # enabled and silenced for the bootstrap inputs.
                        a = en_seed.set_index(
                            ["model", "fold_id", "seed"]
                        )[metric]
                        b = si_seed.set_index(
                            ["model", "fold_id", "seed"]
                        )[metric]
                        common = a.index.intersection(b.index)
                        a_arr = a.loc[common]
                        b_arr = b.loc[common]
                        a_arr = a_arr[
                            a_arr.index.get_level_values("model") == model
                        ].to_numpy(dtype=float)
                        b_arr = b_arr[
                            b_arr.index.get_level_values("model") == model
                        ].to_numpy(dtype=float)
                        if a_arr.size == 0 or b_arr.size == 0:
                            continue
                        # Per-seed deltas are stored once per (cid, rate, seed,
                        # model, metric) — averaged at write time so the cell
                        # in silenced_impact.csv aggregates the 5 seeds.
                        silenced_rows.append({
                            "corruption_type": cid,
                            "rate": float(rate),
                            "seed": int(seed),
                            "model": model,
                            "metric": metric,
                            "_a_arr": a_arr.tolist(),
                            "_b_arr": b_arr.tolist(),
                        })

    # Now collapse silenced_rows into one row per (cid, rate, model, metric).
    # We use the paired bootstrap on the concatenated per-fold-per-seed
    # arrays of (a, b), which preserves pairing.
    df_si = pd.DataFrame(silenced_rows)
    if df_si.empty:
        sys.stderr.write("WARNING: no silenced_rows produced.\n")
    summary_rows: list[dict] = []
    for (cid, rate, model, metric), grp in df_si.groupby(
        ["corruption_type", "rate", "model", "metric"]
    ):
        a_full: list[float] = []
        b_full: list[float] = []
        for _, r in grp.iterrows():
            a_full.extend(r["_a_arr"])
            b_full.extend(r["_b_arr"])
        a_arr = np.array(a_full, dtype=float)
        b_arr = np.array(b_full, dtype=float)
        ci = paired_bootstrap_ci(a_arr, b_arr, seed=42, n_boot=10000)
        wlx = paired_wilcoxon(a_arr, b_arr)
        summary_rows.append({
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
    silenced_summary = pd.DataFrame(summary_rows)

    # ---- Persist per-cell coverage rows.
    cov_df = pd.DataFrame(coverage_rows)
    cov_df.to_csv(EXP_DIR / "validator_coverage.csv", index=False)
    silenced_summary.to_csv(EXP_DIR / "silenced_impact.csv", index=False)

    # ---- Provenance.
    with open(EXP_DIR / "corruption_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2, default=str)

    # ---- Aggregated output tables.
    coverage_table = _aggregate_coverage(coverage_rows)
    coverage_table.to_csv(OUT_DIR / "table_validator_coverage.csv", index=False)
    silenced_summary.to_csv(OUT_DIR / "table_silenced_impact.csv", index=False)

    # ---- Figures.
    _fig_coverage(coverage_table, OUT_DIR / "fig_coverage.pdf")
    _fig_silenced_impact(silenced_summary, OUT_DIR / "fig_silenced_impact.pdf")

    elapsed_total = time.time() - t0_global
    print(f"\n[total wall time] {elapsed_total/60.0:.1f} min")

    _print_status(coverage_table, silenced_summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
