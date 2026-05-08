"""Tests for src/seet/drift.py."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from seet.drift import (  # noqa: E402
    feature_level_drift,
    fold_level_drift_summary,
    psi,
    sym_kl,
)


# ---------------------------------------------------------------------
# psi
# ---------------------------------------------------------------------

def test_psi_identical_distributions_is_zero():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(5000)
    b = rng.standard_normal(5000)
    # Same distribution, large sample -> PSI very close to 0 (the
    # smoothing constant + bin-position randomness leaves residual
    # noise but at the 0.01 level it's a clear "stable" verdict).
    assert psi(a, b) < 0.02


def test_psi_increases_with_shift_magnitude():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(5000)
    cur_small = rng.standard_normal(5000) + 0.2
    cur_med = rng.standard_normal(5000) + 0.5
    cur_large = rng.standard_normal(5000) + 1.5
    p_small = psi(ref, cur_small)
    p_med = psi(ref, cur_med)
    p_large = psi(ref, cur_large)
    assert p_small > 0
    assert p_med > p_small
    assert p_large > p_med
    # The 1.5-sigma shift should be unambiguously "high drift".
    assert p_large > 0.25


def test_psi_returns_nan_on_all_nan_input():
    a = np.full(100, np.nan)
    b = np.random.default_rng(0).standard_normal(100)
    assert np.isnan(psi(a, b))
    assert np.isnan(psi(b, a))


# ---------------------------------------------------------------------
# sym_kl
# ---------------------------------------------------------------------

def test_sym_kl_is_symmetric():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(5000)
    b = rng.standard_normal(5000) + 0.7
    ab = sym_kl(a, b)
    ba = sym_kl(b, a)
    assert abs(ab - ba) < 1e-9
    assert ab > 0


def test_sym_kl_identical_is_near_zero():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(5000)
    b = rng.standard_normal(5000)
    assert sym_kl(a, b) < 0.05


# ---------------------------------------------------------------------
# feature_level_drift
# ---------------------------------------------------------------------

def test_feature_level_drift_one_row_per_feature():
    rng = np.random.default_rng(0)
    cols = [f"f{i}" for i in range(7)]
    train = pd.DataFrame(rng.standard_normal((1000, 7)), columns=cols)
    test = pd.DataFrame(
        rng.standard_normal((300, 7)) + 0.5, columns=cols
    )
    out = feature_level_drift(train, test, cols)
    assert list(out.columns) == ["feature", "psi", "sym_kl"]
    assert len(out) == 7
    assert set(out["feature"]) == set(cols)
    # All features got the same shift; PSI should all be > 0.
    assert (out["psi"] > 0).all()


def test_feature_level_drift_handles_missing_column():
    rng = np.random.default_rng(0)
    train = pd.DataFrame(rng.standard_normal((100, 2)), columns=["a", "b"])
    test = pd.DataFrame(rng.standard_normal((100, 2)), columns=["a", "b"])
    # Ask for a feature that doesn't exist.
    out = feature_level_drift(train, test, ["a", "missing", "b"])
    assert len(out) == 3
    missing_row = out[out["feature"] == "missing"].iloc[0]
    assert np.isnan(missing_row["psi"])
    assert np.isnan(missing_row["sym_kl"])


def test_feature_level_drift_nan_safety():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({
        "a": rng.standard_normal(200),
        "b": np.full(200, np.nan),
    })
    test = pd.DataFrame({
        "a": rng.standard_normal(200) + 0.3,
        "b": np.full(200, np.nan),
    })
    out = feature_level_drift(train, test, ["a", "b"])
    a_row = out[out["feature"] == "a"].iloc[0]
    b_row = out[out["feature"] == "b"].iloc[0]
    assert a_row["psi"] > 0
    assert np.isnan(b_row["psi"])
    assert np.isnan(b_row["sym_kl"])


# ---------------------------------------------------------------------
# fold_level_drift_summary
# ---------------------------------------------------------------------

def test_fold_level_drift_summary_aggregates_correctly():
    rng = np.random.default_rng(0)
    cols = [f"f{i}" for i in range(5)]
    train = pd.DataFrame(rng.standard_normal((2000, 5)), columns=cols)
    # Heavy shift on a subset of features so we get high_drift
    # entries with deterministic ordering.
    shifts = np.array([0.1, 0.2, 1.5, 2.0, 0.05])
    test = pd.DataFrame(
        rng.standard_normal((1000, 5)) + shifts, columns=cols
    )
    summary = fold_level_drift_summary(train, test, cols)

    # Schema check.
    expected_keys = {
        "mean_psi", "max_psi", "mean_kl", "max_kl",
        "n_high_drift", "high_drift_features",
        "n_features", "n_features_valid",
    }
    assert set(summary.keys()) == expected_keys

    # Sanity: max_psi >= mean_psi; both > 0 for a shifted batch.
    assert summary["max_psi"] >= summary["mean_psi"] > 0
    assert summary["n_features"] == 5
    assert summary["n_features_valid"] == 5

    # f3 (shift=2.0) should be the strongest drifter; f2 next.
    assert summary["high_drift_features"][0] == "f3"
    assert summary["high_drift_features"][1] == "f2"
    assert summary["n_high_drift"] >= 2


def test_fold_level_drift_summary_empty_features():
    """All features all-NaN -> all-NaN summary, zero high_drift."""
    train = pd.DataFrame({"a": np.full(50, np.nan), "b": np.full(50, np.nan)})
    test = pd.DataFrame({"a": np.full(50, np.nan), "b": np.full(50, np.nan)})
    summary = fold_level_drift_summary(train, test, ["a", "b"])
    assert np.isnan(summary["mean_psi"])
    assert np.isnan(summary["max_psi"])
    assert summary["n_high_drift"] == 0
    assert summary["high_drift_features"] == []
    assert summary["n_features"] == 2
    assert summary["n_features_valid"] == 0
