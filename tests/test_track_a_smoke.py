"""Smoke test for src/seet/run_track_a.py.

Generates a small synthetic panel, runs one fold for one stress_def,
and verifies that all expected output files land on disk and the
per-fold dataframe is non-empty.

Run from the repo root:

    pytest -q tests/test_track_a_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from seet.run_track_a import (  # noqa: E402
    L1_METRICS,
    L2_METRICS,
    build_layer3_table,
    build_layer4_table,
    build_pairwise_files,
    build_table_with_ci,
    plot_lift_with_ci,
    plot_pr_curves,
    plot_reliability,
    run_grid,
)


def _make_synthetic(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2014-01-01", periods=n, freq="B")
    spx = 1000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n)))
    vix = np.clip(15.0 + 5.0 * rng.standard_normal(n).cumsum() / np.sqrt(n), 8, 80)
    panel = pd.DataFrame({"Date": dates, "SPX": spx, "VIX": vix})
    feat_cols = [f"f{i}" for i in range(35)]
    feat = rng.standard_normal((n, 35))
    features = pd.DataFrame(feat, columns=feat_cols)
    features.insert(0, "Date", dates)
    return panel, features


def test_smoke_runner_writes_all_expected_files(tmp_path):
    panel, features = _make_synthetic()
    exp_dir = tmp_path / "experiments" / "track_a_smoke"
    out_dir = tmp_path / "outputs" / "track_a_smoke"

    per_fold_df, fold_def_df, predictions_by_def = run_grid(
        panel,
        features,
        exp_dir,
        out_dir,
        stress_defs=[{"name": "h5_d03", "h": 5, "d": 0.03}],
        seeds=(42,),
        initial_train_end=pd.Timestamp("2014-06-30"),
    )

    assert not per_fold_df.empty, "smoke run produced no per-fold rows"
    assert not fold_def_df.empty, "smoke run produced no fold definitions"
    assert "h5_d03" in predictions_by_def
    assert not predictions_by_def["h5_d03"].empty

    # Per-fold and fold-def CSVs land in the experiments dir.
    assert (exp_dir / "per_fold_metrics.csv").exists()
    assert (exp_dir / "fold_definitions.csv").exists()
    assert (exp_dir / "predictions" / "h5_d03.parquet").exists()

    # Aggregated tables write to disk
    build_table_with_ci(per_fold_df, L1_METRICS).to_csv(out_dir / "table_layer1.csv", index=False)
    build_table_with_ci(per_fold_df, L2_METRICS).to_csv(out_dir / "table_layer2.csv", index=False)
    build_layer3_table(per_fold_df, predictions_by_def).to_csv(out_dir / "table_layer3.csv", index=False)
    build_layer4_table(per_fold_df).to_csv(out_dir / "table_layer4.csv", index=False)
    build_pairwise_files(per_fold_df, out_dir)
    assert (out_dir / "table_layer1.csv").exists()
    assert (out_dir / "table_layer2.csv").exists()
    assert (out_dir / "table_layer3.csv").exists()
    assert (out_dir / "table_layer4.csv").exists()
    # At least one pairwise file per (metric, stress_def) — we only have one stress_def
    pairwise_files = list(out_dir.glob("pairwise_pvalues_*.csv"))
    assert any("AUC_h5_d03" in f.name for f in pairwise_files)
    assert any("lift_h5_d03" in f.name for f in pairwise_files)

    # Figures
    plot_lift_with_ci(per_fold_df, out_dir / "fig_lift_with_ci.pdf")
    plot_reliability(predictions_by_def["h5_d03"], out_dir / "fig_reliability.pdf")
    plot_pr_curves(predictions_by_def["h5_d03"], out_dir / "fig_pr_curves.pdf")
    assert (out_dir / "fig_lift_with_ci.pdf").exists()
    assert (out_dir / "fig_reliability.pdf").exists()
    assert (out_dir / "fig_pr_curves.pdf").exists()


def test_smoke_per_fold_schema_has_all_metrics(tmp_path):
    panel, features = _make_synthetic()
    exp_dir = tmp_path / "experiments" / "track_a_smoke"
    out_dir = tmp_path / "outputs" / "track_a_smoke"
    per_fold_df, _, _ = run_grid(
        panel,
        features,
        exp_dir,
        out_dir,
        stress_defs=[{"name": "h5_d03", "h": 5, "d": 0.03}],
        seeds=(42,),
        initial_train_end=pd.Timestamp("2014-06-30"),
    )
    expected = {"stress_def", "model", "fold_id", "seed",
                "auc", "pr_auc", "brier", "ece",
                "alarm_rate", "alarm_precision",
                "drawdown_lift", "severity_lift",
                "n_alarms", "n_events"}
    missing = expected - set(per_fold_df.columns)
    assert not missing, f"per_fold_metrics missing columns: {sorted(missing)}"
