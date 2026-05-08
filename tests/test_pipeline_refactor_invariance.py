"""Invariance and parameterization tests for the seet.pipeline refactor.

These tests are the gate that protects Track A's committed results
from drift introduced by moving run_grid + helpers from
seet.run_track_a into seet.pipeline.

Two layers of testing:

1. INVARIANCE — run pipeline.run_grid with the SAME inputs that
   produced the committed Track A per_fold_metrics.csv (one stress_def,
   one seed, deterministic baselines only) and assert every output
   metric matches the committed value within 1e-9 tolerance.

2. PARAMETERIZATION — run pipeline.run_grid on a synthetic panel with
   generic INDEX/VOL columns and a model_kwargs_factory that points
   the column-aware baselines at those names. Confirm the run completes
   and produces sensibly-shaped output for every baseline.

LightGbmTuned is excluded from the invariance test because its committed
rows in per_fold_metrics.csv reflect the Track B tuned hyperparameters
(via apply_track_b_to_track_a.py), not the placeholder HPs that the
default constructor produces. The five deterministic baselines have
not been touched since their original Track A run, so they are the
right comparison target.

Run from the repo root:

    pytest -q tests/test_pipeline_refactor_invariance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from seet.baselines import ALL_MODELS, DETERMINISTIC_MODELS  # noqa: E402
from seet.pipeline import run_grid  # noqa: E402


METRIC_COLUMNS = [
    "auc", "pr_auc", "brier", "ece",
    "alarm_rate", "alarm_precision",
    "drawdown_lift", "severity_lift",
    "threshold", "n_alarms", "n_events",
]
TOLERANCE = 1e-9


def _have_track_a_artifacts() -> bool:
    panel = REPO_ROOT / "data" / "processed" / "spx_extended_2011.csv"
    feat = REPO_ROOT / "data" / "processed" / "features" / "spx_extended_2011_features.csv"
    metrics = REPO_ROOT / "experiments" / "track_a_headline" / "per_fold_metrics.csv"
    return panel.exists() and feat.exists() and metrics.exists()


# =====================================================================
# 1. INVARIANCE
# =====================================================================

def test_invariance_one_stress_def_deterministic_baselines(tmp_path):
    """Run pipeline.run_grid with stress_def=h5_d03 and seed=42 only,
    five deterministic baselines, and compare every cell to the
    committed Track A per_fold_metrics.csv values.

    This is the gate the user explicitly demanded ("Confirm refactor
    does not change SPX results by re-running ONE Track A configuration
    and asserting metrics match within numerical tolerance"). If this
    test fails, the refactor has changed Track A's behavior and we
    must stop and report before proceeding."""
    if not _have_track_a_artifacts():
        pytest.skip("Track A panel/features/metrics not on disk")

    from seet.run_track_a import (
        INITIAL_TRAIN_END, PANEL_NAME, PRICE_COL, PRIMARY_SEED, STRESS_DEFS,
    )

    panel_path = REPO_ROOT / "data" / "processed" / f"{PANEL_NAME}.csv"
    feat_path = REPO_ROOT / "data" / "processed" / "features" / f"{PANEL_NAME}_features.csv"
    metrics_path = REPO_ROOT / "experiments" / "track_a_headline" / "per_fold_metrics.csv"

    panel_df = pd.read_csv(panel_path, parse_dates=["Date"])
    features_df = pd.read_csv(feat_path, parse_dates=["Date"])
    committed = pd.read_csv(metrics_path)

    target_sd = STRESS_DEFS[0]  # h5_d03

    per_fold_df, _, _ = run_grid(
        panel_df, features_df, tmp_path / "invariance",
        price_col=PRICE_COL,
        stress_defs=[target_sd],
        seeds=(42,),
        primary_seed=PRIMARY_SEED,
        initial_train_end=INITIAL_TRAIN_END,
        deterministic_models=DETERMINISTIC_MODELS,
        stochastic_models=(),
        save_predictions=False,
    )
    assert not per_fold_df.empty, "pipeline.run_grid produced zero rows"

    n_compared = 0
    for _, new_row in per_fold_df.iterrows():
        match = committed[
            (committed["stress_def"] == new_row["stress_def"])
            & (committed["model"] == new_row["model"])
            & (committed["fold_id"] == new_row["fold_id"])
            & (committed["seed"] == new_row["seed"])
        ]
        if match.empty:
            continue
        committed_row = match.iloc[0]
        for col in METRIC_COLUMNS:
            new_val = new_row[col]
            old_val = committed_row[col]
            if pd.isna(new_val) and pd.isna(old_val):
                continue
            if pd.isna(new_val) or pd.isna(old_val):
                pytest.fail(
                    f"NaN mismatch at "
                    f"({new_row['stress_def']}, {new_row['model']}, "
                    f"fold {new_row['fold_id']}, seed {new_row['seed']}, {col}): "
                    f"committed={old_val}, new={new_val}"
                )
            diff = abs(float(new_val) - float(old_val))
            assert diff < TOLERANCE, (
                f"Metric drift at "
                f"({new_row['stress_def']}, {new_row['model']}, "
                f"fold {new_row['fold_id']}, seed {new_row['seed']}, {col}): "
                f"committed={old_val}, new={new_val}, |diff|={diff:.2e}"
            )
            n_compared += 1

    assert n_compared > 0, "Comparison loop matched no rows; check CSV alignment"


# =====================================================================
# 2. PARAMETERIZATION
# =====================================================================

def _synthetic_track_c_inputs(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2014-01-01", periods=n, freq="B")
    index = 1000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n)))
    vol = np.clip(15.0 + rng.standard_normal(n).cumsum() / np.sqrt(n) * 5, 8, 80)
    panel_df = pd.DataFrame({"Date": dates, "INDEX": index, "VOL": vol})
    feat_cols = [f"f{i}" for i in range(12)]
    feat_data = rng.standard_normal((n, 12))
    features_df = pd.DataFrame(feat_data, columns=feat_cols)
    features_df.insert(0, "Date", dates)
    return panel_df, features_df


def _track_c_kwargs_factory(model_name: str, **_) -> dict:
    if model_name in ("VIXPercentileRaw", "VIXPercentileCalibrated"):
        return {"vix_col": "VOL"}
    if model_name == "HarRvThreshold":
        return {"price_col": "INDEX"}
    return {}


def test_run_grid_accepts_generic_columns(tmp_path):
    """pipeline.run_grid runs end-to-end on a synthetic panel with
    generic INDEX/VOL columns. All six baselines produce non-empty
    rows; no exception is raised."""
    panel_df, features_df = _synthetic_track_c_inputs()
    per_fold_df, fold_def_df, _ = run_grid(
        panel_df, features_df, tmp_path / "track_c_synthetic",
        price_col="INDEX",
        stress_defs=[{"name": "h5_d03", "h": 5, "d": 0.03}],
        seeds=(42,),
        primary_seed=42,
        initial_train_end=pd.Timestamp("2014-06-30"),
        model_kwargs_factory=_track_c_kwargs_factory,
        save_predictions=False,
    )
    assert not per_fold_df.empty
    assert not fold_def_df.empty
    # Every baseline should appear at least once.
    assert set(per_fold_df["model"].unique()) == set(ALL_MODELS), (
        f"Missing models: {set(ALL_MODELS) - set(per_fold_df['model'].unique())}"
    )


def test_model_kwargs_factory_routes_kwargs_to_constructor(tmp_path):
    """The factory's return values must reach the baseline constructor.
    We verify by passing a deliberately wrong column name and asserting
    the affected baseline produces NaN scores (its requested column is
    absent), while the unaffected baselines work."""
    panel_df, features_df = _synthetic_track_c_inputs()

    def factory_with_bad_vix_col(model_name, **_):
        if model_name in ("VIXPercentileRaw", "VIXPercentileCalibrated"):
            return {"vix_col": "DOES_NOT_EXIST"}
        if model_name == "HarRvThreshold":
            return {"price_col": "INDEX"}
        return {}

    # The VIXPercentileRaw / Calibrated constructors raise ValueError when
    # the requested column is missing — confirm pipeline.run_grid surfaces
    # that error rather than silently swallowing it.
    with pytest.raises(ValueError, match="DOES_NOT_EXIST"):
        run_grid(
            panel_df, features_df, tmp_path / "track_c_bad_kwarg",
            price_col="INDEX",
            stress_defs=[{"name": "h5_d03", "h": 5, "d": 0.03}],
            seeds=(42,),
            primary_seed=42,
            initial_train_end=pd.Timestamp("2014-06-30"),
            model_kwargs_factory=factory_with_bad_vix_col,
            save_predictions=False,
        )


def test_default_model_kwargs_factory_is_no_op(tmp_path):
    """Without an explicit model_kwargs_factory, run_grid uses the
    project default which returns {} for every model — equivalent to
    the original Track A behavior. This test runs on a synthetic SPX-
    style panel (INDEX/VIX/SPX columns) so the default-named baselines
    can find their data."""
    rng = np.random.default_rng(0)
    n = 400
    dates = pd.date_range("2014-01-01", periods=n, freq="B")
    spx = 1000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n)))
    vix = np.clip(15.0 + rng.standard_normal(n).cumsum() / np.sqrt(n) * 5, 8, 80)
    panel_df = pd.DataFrame({"Date": dates, "SPX": spx, "VIX": vix})
    feat_cols = [f"f{i}" for i in range(35)]
    features_df = pd.DataFrame(rng.standard_normal((n, 35)), columns=feat_cols)
    features_df.insert(0, "Date", dates)

    per_fold_df, _, _ = run_grid(
        panel_df, features_df, tmp_path / "default_factory",
        price_col="SPX",
        stress_defs=[{"name": "h5_d03", "h": 5, "d": 0.03}],
        seeds=(42,),
        primary_seed=42,
        initial_train_end=pd.Timestamp("2014-06-30"),
        save_predictions=False,
    )
    assert not per_fold_df.empty
    # All six baselines should run (default kwargs work for SPX-named cols)
    assert set(per_fold_df["model"].unique()) == set(ALL_MODELS)


# =====================================================================
# 3. RE-EXPORTS — run_track_a.py back-compat
# =====================================================================

def test_run_track_a_reexports_resolve():
    """The symbols apply_track_b_to_track_a.py and test_track_a_smoke.py
    import from seet.run_track_a must still resolve after the refactor."""
    from seet import run_track_a as ra
    expected = [
        "ALL_MODELS", "INITIAL_TRAIN_END", "L1_METRICS", "L2_METRICS",
        "PANEL_NAME", "PRICE_COL", "PRIMARY_SEED", "SEEDS", "STRESS_DEFS",
        "build_folds_with_init", "build_layer3_table", "build_layer4_table",
        "build_pairwise_files", "build_table_with_ci", "compute_layer1",
        "compute_layer2", "compute_stress_labels",
        "failure_probe_f1", "failure_probe_f2", "failure_probe_f3",
        "plot_lift_with_ci", "plot_pr_curves", "plot_reliability",
        "run_grid",
    ]
    missing = [name for name in expected if not hasattr(ra, name)]
    assert not missing, f"run_track_a missing expected symbols: {missing}"
