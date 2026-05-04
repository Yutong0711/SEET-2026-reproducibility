"""Unit tests for src/seet/baselines.py.

Run from the repo root:

    pytest -q tests/test_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from seet.baselines import (  # noqa: E402
    ALL_MODELS,
    DETERMINISTIC_MODELS,
    HarRvThreshold,
    LightGbmTuned,
    LogisticRegressionL2,
    NaiveBaseRate,
    STOCHASTIC_MODELS,
    VIXPercentileCalibrated,
    VIXPercentileRaw,
    make_model,
)


N = 200
N_FEATURES = 35
SEED = 42


def _synthetic_panel(seed: int = SEED, n: int = N) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a (panel, y) pair the DataFrame-aware baselines can consume."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2014-01-01", periods=n, freq="B")
    spx = 1000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n)))
    vix = np.clip(15.0 + 5.0 * rng.standard_normal(n).cumsum() / np.sqrt(n), 8, 80)
    panel = pd.DataFrame({"Date": dates, "SPX": spx, "VIX": vix})
    # Plant a positive class rate of ~10%.
    y = (rng.random(n) < 0.10).astype(int)
    return panel, y


def _synthetic_features(seed: int = SEED, n: int = N) -> tuple[pd.DataFrame, np.ndarray]:
    """Build an (X, y) pair where X is a 35-feature ndarray-shaped DataFrame."""
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(N_FEATURES)]
    X = pd.DataFrame(rng.standard_normal((n, N_FEATURES)), columns=cols)
    y = (rng.random(n) < 0.10).astype(int)
    return X, y


# ----------------------------------------------------------------------
# Shape and protocol checks (parametrized over every model)
# ----------------------------------------------------------------------

def _build_inputs(model_name: str):
    if model_name in ("VIXPercentileRaw", "VIXPercentileCalibrated", "HarRvThreshold", "NaiveBaseRate"):
        return _synthetic_panel()
    return _synthetic_features()


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_fit_predict_shape(model_name):
    X, y = _build_inputs(model_name)
    model = make_model(model_name, seed=SEED)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_predict_proba_rows_sum_to_one(model_name):
    X, y = _build_inputs(model_name)
    model = make_model(model_name, seed=SEED)
    model.fit(X, y)
    proba = model.predict_proba(X)
    sums = proba.sum(axis=1)
    valid = ~np.isnan(sums)
    if not valid.any():
        pytest.skip(f"{model_name}: all-NaN proba (degenerate); not testable")
    assert np.allclose(sums[valid], 1.0, atol=1e-9), (
        f"{model_name}: rows do not sum to 1.0; got max abs diff "
        f"{np.max(np.abs(sums[valid] - 1.0))}"
    )


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_predict_proba_in_unit_interval(model_name):
    X, y = _build_inputs(model_name)
    model = make_model(model_name, seed=SEED)
    model.fit(X, y)
    proba = model.predict_proba(X)
    valid = ~np.isnan(proba).any(axis=1)
    if not valid.any():
        pytest.skip(f"{model_name}: all-NaN proba")
    p = proba[valid]
    assert (p >= 0.0 - 1e-12).all()
    assert (p <= 1.0 + 1e-12).all()


# ----------------------------------------------------------------------
# Per-baseline behavioral checks
# ----------------------------------------------------------------------

def test_naive_base_rate_returns_constant():
    X, _ = _synthetic_panel()
    y = np.zeros(len(X), dtype=int)
    y[:20] = 1  # 10% positive rate
    model = NaiveBaseRate().fit(X, y)
    proba = model.predict_proba(X)
    assert np.allclose(proba[:, 1], 0.10, atol=1e-9)
    assert np.allclose(proba[:, 0], 0.90, atol=1e-9)


def test_vix_percentile_raw_increasing_in_vix():
    """Higher VIX should map to higher percentile score on training data."""
    panel, y = _synthetic_panel()
    model = VIXPercentileRaw().fit(panel, y)
    proba = model.predict_proba(panel)[:, 1]
    # Spearman-style monotonicity: rank-ordered VIX matches rank-ordered scores.
    vix_rank = pd.Series(panel["VIX"]).rank().to_numpy()
    score_rank = pd.Series(proba).rank().to_numpy()
    rho = np.corrcoef(vix_rank, score_rank)[0, 1]
    assert rho > 0.99, f"raw percentile not monotone in VIX; rho={rho}"


def test_vix_percentile_calibrated_runs_with_isotonic():
    """Calibrated baseline should produce a finite, [0,1]-bounded score."""
    panel, _ = _synthetic_panel()
    # Plant a clear monotone relationship: y = (VIX > median).
    y = (panel["VIX"] > panel["VIX"].median()).astype(int).to_numpy()
    model = VIXPercentileCalibrated().fit(panel, y)
    proba = model.predict_proba(panel)[:, 1]
    valid = ~np.isnan(proba)
    assert valid.any()
    assert (proba[valid] >= 0).all() and (proba[valid] <= 1).all()


def test_har_rv_threshold_handles_real_panel_shape():
    """HAR-RV should fit and predict on a continuous price series."""
    panel, y = _synthetic_panel()
    model = HarRvThreshold(price_col="SPX").fit(panel, y)
    proba = model.predict_proba(panel)[:, 1]
    # First 22 rows have NaN due to 22-day rolling mean; others should be finite.
    finite = proba[22:]
    finite = finite[~np.isnan(finite)]
    assert finite.size > 0
    assert (finite >= 0.0).all() and (finite <= 1.0).all()


def test_logistic_regression_handles_single_class():
    """All-zero training labels: should not raise; predicts constant 0."""
    X, _ = _synthetic_features()
    y = np.zeros(len(X), dtype=int)
    model = LogisticRegressionL2(seed=SEED).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba[:, 1], 0.0)


def test_lightgbm_handles_single_class():
    X, _ = _synthetic_features()
    y = np.zeros(len(X), dtype=int)
    model = LightGbmTuned(seed=SEED).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba[:, 1], 0.0)


def test_lightgbm_same_seed_is_deterministic():
    """Same seed -> bit-equal predictions. (Different seeds may or may
    not diverge depending on hyperparameters: the Track A placeholder
    HPs disable bagging and feature sub-sampling and use a very strict
    min_child_samples, which makes LightGBM fully deterministic. We
    only guarantee same-seed reproducibility here, which is what the
    runner needs for reproducible re-runs.)"""
    X, y = _synthetic_features()
    a = LightGbmTuned(seed=42).fit(X, y).predict_proba(X)[:, 1]
    b = LightGbmTuned(seed=42).fit(X, y).predict_proba(X)[:, 1]
    assert np.allclose(a, b), "same seed should give bit-equal predictions"


def test_logistic_regression_seed_does_not_change_lbfgs_result():
    """LR with lbfgs is deterministic; seed change shouldn't matter."""
    X, y = _synthetic_features()
    a = LogisticRegressionL2(seed=42).fit(X, y).predict_proba(X)[:, 1]
    b = LogisticRegressionL2(seed=43).fit(X, y).predict_proba(X)[:, 1]
    assert np.allclose(a, b, atol=1e-10), (
        "LR with class_weight='balanced' on lbfgs should be seed-independent"
    )


# ----------------------------------------------------------------------
# Registry sanity
# ----------------------------------------------------------------------

def test_registry_partition():
    assert set(DETERMINISTIC_MODELS).isdisjoint(STOCHASTIC_MODELS)
    assert set(ALL_MODELS) == set(DETERMINISTIC_MODELS) | set(STOCHASTIC_MODELS)
    assert len(ALL_MODELS) == 6


def test_make_model_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        make_model("definitely_not_a_model")
