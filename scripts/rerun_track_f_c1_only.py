"""Re-run only the C1 (duplicate_dates) cells of Track F after the
true-duplicate fix in src/seet/injection.py.

The earlier C1 implementation copied values from a RANDOM source row,
which produced synthetic stress events when a future-dated source
row's SPX got inserted at an earlier target date. Diagnosed by
scripts/diagnose_track_f_c1_all_folds.py: 11 of 33 valid cells had
SILENCED lift >= 4.0 (P50 = 2.94, max = 17.57), driving the
aggregate delta_lift = -4.17 in silenced_impact.csv. The fix in
_corrupt_duplicate_dates uses true duplicates (same Date + same
values), preserving V1's perfect-recall detection while removing the
synthetic-event mechanism.

This script:
  1. Re-runs the 15 C1 cells (3 rates × 5 seeds) with the fixed
     corruption, overwriting experiments/track_f_injection/_workdir/
     duplicate_dates_*/.
  2. Re-runs scripts/rerun_track_f_postprocess.py-style branch_failure
     accounting for C1 only.
  3. Updates the C1 rows of experiments/track_f_injection/silenced_impact.csv
     and outputs/track_f/table_silenced_impact.csv (preserving the
     non-C1 rows byte-for-byte).
  4. Re-runs scripts/compute_track_f_silenced_vs_clean.py for C1 (no-op
     since that script only handles C3/C4/C5; C1 vs Track A delta is
     out of scope for that table).

Other corruption types (C2/C3/C4/C5) are NOT re-run.

Usage:
    python scripts/rerun_track_f_c1_only.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from seet.baselines import DETERMINISTIC_MODELS, STOCHASTIC_MODELS  # noqa: E402
from seet.features import build_features  # noqa: E402
from seet.injection import GAP_MECHANISM, corrupt_panel  # noqa: E402
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
TRACK_F_DETERMINISTIC = ("LogisticRegressionL2",)
TRACK_F_STOCHASTIC = ("LightGbmTuned",)
HP_CSV_PATH = REPO_ROOT / "experiments" / "track_b_tuning" / "selected_hp.csv"
EXP_DIR = REPO_ROOT / "experiments" / "track_f_injection"
OUT_DIR = REPO_ROOT / "outputs" / "track_f"
DELTA_METRICS = ("auc", "pr_auc", "brier", "drawdown_lift", "alarm_rate")


def _track_a_lgbm_kwargs_factory(model_name, stress_def=None, outer_fold=None):
    if model_name == "LightGbmTuned" and stress_def is not None and outer_fold is not None:
        return {"stress_def": stress_def, "outer_fold": outer_fold,
                "hp_path": str(HP_CSV_PATH)}
    return {}


def _run_pipeline_branch(panel_df, features_df, out_dir):
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


def main() -> int:
    canonical = pd.read_csv(
        REPO_ROOT / "data" / "processed" / f"{PANEL_NAME}.csv",
        parse_dates=["Date"],
    )
    with open(REPO_ROOT / "data" / "manifests" / "processed_manifest.json") as f:
        manifest_entry = json.load(f)[PANEL_NAME]

    print(f"[load] canonical panel: {len(canonical)} rows")
    print(f"[fix] _corrupt_duplicate_dates now uses true-duplicate semantics")

    # ---- Re-run the 15 C1 cells ---------------------------------
    coverage_rows: list[dict] = []
    silenced_rows: list[dict] = []
    cell_n = 0
    n_total = len(RATES) * len(TRACK_F_SEEDS)
    t0_global = time.time()
    cid = "duplicate_dates"
    for rate in RATES:
        cell_enabled_dfs: list[pd.DataFrame] = []
        cell_silenced_dfs: list[pd.DataFrame] = []
        for seed in TRACK_F_SEEDS:
            cell_n += 1
            t0 = time.time()
            cell_workdir = EXP_DIR / "_workdir" / f"{cid}_{rate:.2f}_{seed}"
            corrupted, gt, meta = corrupt_panel(
                canonical, cid, rate, seed, manifest_entry=manifest_entry
            )
            v = validate_panel(corrupted, manifest_entry, enabled=True)
            cov = coverage_metrics(v["flags"], gt)
            coverage_rows.append({
                "corruption_type": cid,
                "rate": float(rate),
                "seed": int(seed),
                "precision": cov["precision"],
                "recall": cov["recall"],
                "f1": cov["f1"],
                "n_flagged_truth": cov["n_truth"],
                "n_flagged_validator": cov["n_validator"],
                "n_true_positive": cov["n_true_positive"],
                "gap_mechanism": GAP_MECHANISM[cid],
            })

            enabled_panel = apply_treatment(corrupted, v, manifest_entry)
            enabled_features, _ = build_features(enabled_panel, "spx_full")
            en_df = _run_pipeline_branch(
                enabled_panel, enabled_features, cell_workdir / "enabled"
            )
            en_df = en_df.assign(
                corruption_type=cid, rate=float(rate),
                injection_seed=int(seed), branch="enabled",
            )

            silenced_features, _ = build_features(corrupted, "spx_full")
            si_df = _run_pipeline_branch(
                corrupted, silenced_features, cell_workdir / "silenced"
            )
            si_df = si_df.assign(
                corruption_type=cid, rate=float(rate),
                injection_seed=int(seed), branch="silenced",
            )
            cell_enabled_dfs.append(en_df)
            cell_silenced_dfs.append(si_df)
            elapsed = time.time() - t0
            print(
                f"  [{cell_n:>2}/{n_total}] {cid} rate={rate:.2f} seed={seed} "
                f"recall={cov['recall']:.3f}  elapsed={elapsed:.1f}s"
            )

        # Pair across the 5 injection seeds at this (cid, rate).
        en_concat = pd.concat(cell_enabled_dfs, ignore_index=True)
        si_concat = pd.concat(cell_silenced_dfs, ignore_index=True)
        for inj_seed in TRACK_F_SEEDS:
            en_seed = en_concat[en_concat["injection_seed"] == inj_seed]
            si_seed = si_concat[si_concat["injection_seed"] == inj_seed]
            for model in ("LogisticRegressionL2", "LightGbmTuned"):
                for metric in DELTA_METRICS:
                    a = en_seed.set_index(
                        ["model", "fold_id", "seed"]
                    )[metric] if metric in en_seed.columns else None
                    b = si_seed.set_index(
                        ["model", "fold_id", "seed"]
                    )[metric] if metric in si_seed.columns else None
                    if a is None or b is None:
                        continue
                    common = a.index.intersection(b.index)
                    a = a.loc[common]
                    b = b.loc[common]
                    a_arr = a[
                        a.index.get_level_values("model") == model
                    ].to_numpy(dtype=float)
                    b_arr = b[
                        b.index.get_level_values("model") == model
                    ].to_numpy(dtype=float)
                    if a_arr.size == 0 or b_arr.size == 0:
                        continue
                    silenced_rows.append({
                        "corruption_type": cid,
                        "rate": float(rate),
                        "seed": int(inj_seed),
                        "model": model,
                        "metric": metric,
                        "_a_arr": a_arr.tolist(),
                        "_b_arr": b_arr.tolist(),
                    })

    df_si = pd.DataFrame(silenced_rows)
    summary_rows: list[dict] = []
    for (cid_, rate, model, metric), grp in df_si.groupby(
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
            "corruption_type": cid_,
            "rate": float(rate),
            "model": model,
            "metric": metric,
            "delta_mean": ci["delta_mean"],
            "delta_ci_low": ci["ci_low"],
            "delta_ci_high": ci["ci_high"],
            "wilcoxon_pvalue": wlx["p_value"],
            "n_pairs": ci["n_pairs"],
        })
    new_si_df = pd.DataFrame(summary_rows)

    # ---- Splice the new C1 rows into silenced_impact.csv -----------
    si_path = EXP_DIR / "silenced_impact.csv"
    full = pd.read_csv(si_path)
    others = full[full["corruption_type"] != cid]
    spliced = pd.concat([others, new_si_df], ignore_index=True).sort_values(
        ["corruption_type", "rate", "model", "metric"]
    )
    spliced.to_csv(si_path, index=False)
    spliced.to_csv(OUT_DIR / "table_silenced_impact.csv", index=False)
    print(f"\n[splice] {si_path.relative_to(REPO_ROOT)} updated "
          f"({len(others)} non-C1 rows preserved + {len(new_si_df)} new C1 rows)")

    # ---- Update validator_coverage.csv (C1 rows) -------------------
    cov_path = EXP_DIR / "validator_coverage.csv"
    cov_full = pd.read_csv(cov_path)
    cov_others = cov_full[cov_full["corruption_type"] != cid]
    cov_new = pd.DataFrame(coverage_rows)
    cov_spliced = pd.concat(
        [cov_others, cov_new], ignore_index=True
    ).sort_values(["corruption_type", "rate", "seed"])
    cov_spliced.to_csv(cov_path, index=False)
    print(f"[splice] {cov_path.relative_to(REPO_ROOT)} updated "
          f"({len(cov_others)} non-C1 rows preserved + {len(cov_new)} new C1 rows)")

    elapsed_total = time.time() - t0_global
    print(f"\n[total wall time] {elapsed_total/60.0:.1f} min")

    # ---- Print fixed C1 deltas -------------------------------------
    print("\n" + "=" * 78)
    print("C1 silenced_impact (POST-FIX, true-duplicate semantics)")
    print("=" * 78)
    show = new_si_df[new_si_df["metric"].isin(("auc", "drawdown_lift"))].sort_values(
        ["rate", "metric", "model"]
    )
    print(f"  {'rate':>5} {'model':<26} {'metric':<14} {'delta':>10} "
          f"{'CI':>22} {'p':>8} {'n':>5}")
    print("  " + "-" * 90)
    for _, r in show.iterrows():
        d = r["delta_mean"]
        d_str = f"{d:+10.4f}" if not pd.isna(d) else "       n/a"
        if pd.isna(r["delta_ci_low"]) or pd.isna(r["delta_ci_high"]):
            ci_str = "[n/a]"
        else:
            ci_str = f"[{r['delta_ci_low']:+.4f},{r['delta_ci_high']:+.4f}]"
        sig = "*" if (
            not pd.isna(r["wilcoxon_pvalue"])
            and r["wilcoxon_pvalue"] < 0.05
        ) else " "
        p_str = (
            f"{r['wilcoxon_pvalue']:>7.3f}"
            if not pd.isna(r["wilcoxon_pvalue"]) else "    n/a"
        )
        print(
            f"  {r['rate']:>5.0%} {r['model']:<26} {r['metric']:<14} "
            f"{d_str} {ci_str:>22} {p_str}{sig} {int(r['n_pairs']):>5}"
        )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
