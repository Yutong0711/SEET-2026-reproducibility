"""Unit tests for src/seet/stats.py.

Run from the repo root:

    pytest -q tests/test_stats.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from seet.stats import (  # noqa: E402
    block_bootstrap_ci,
    paired_wilcoxon,
    pairwise_table,
    summarize_metric,
)


# ----------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------

def test_bootstrap_recovers_known_mean():
    """100 trials of N(1, 1) of length 50 — at least 90/100 of the
    nominal 95% bootstrap CIs must contain the true mean of 1.0.

    Threshold of 90 leaves headroom below the ~92-94% empirical
    coverage of the percentile bootstrap on Gaussian data with n=50;
    test is fully deterministic via fixed seeds."""
    n_trials = 100
    n_per_trial = 50
    true_mean = 1.0
    n_contain = 0
    for trial_seed in range(n_trials):
        # Different seeds for data and bootstrap so they don't align.
        data_rng = np.random.default_rng(trial_seed + 100)
        values = data_rng.normal(loc=true_mean, scale=1.0, size=n_per_trial)
        result = block_bootstrap_ci(
            values, block_size=1, n_boot=1000, alpha=0.05, seed=trial_seed
        )
        if result["ci_low"] <= true_mean <= result["ci_high"]:
            n_contain += 1
    assert n_contain >= 90, (
        f"Only {n_contain}/{n_trials} CIs contained the true mean "
        f"(expected at least 90 for nominal 95% coverage)."
    )


def test_bootstrap_handles_nan():
    """Array with 5 NaNs out of 50 returns a valid CI and records
    n_valid=45 / n_input=50."""
    rng = np.random.default_rng(42)
    values = rng.normal(loc=1.0, scale=1.0, size=50)
    nan_indices = rng.choice(50, size=5, replace=False)
    values[nan_indices] = np.nan

    result = block_bootstrap_ci(values, block_size=1, n_boot=500, seed=42)
    assert result["n_input"] == 50
    assert result["n_valid"] == 45
    assert not np.isnan(result["mean"])
    assert not np.isnan(result["ci_low"])
    assert not np.isnan(result["ci_high"])
    assert result["ci_low"] < result["ci_high"]


# ----------------------------------------------------------------------
# Paired Wilcoxon
# ----------------------------------------------------------------------

def test_paired_wilcoxon_identical_returns_high_p():
    """Identical arrays of length 30 -> p_value = 1.0 with a note."""
    a = np.arange(30, dtype=float)
    b = a.copy()
    result = paired_wilcoxon(a, b)
    assert result["p_value"] == 1.0
    assert result["median_diff"] == 0.0
    assert result["n_pairs"] == 30
    assert "note" in result


def test_paired_wilcoxon_detects_shift():
    """b = a + 1 with a ~ N(0, 1), n=30 -> p_value < 0.001."""
    rng = np.random.default_rng(42)
    a = rng.normal(size=30)
    b = a + 1.0
    result = paired_wilcoxon(a, b)
    assert result["n_pairs"] == 30
    assert result["p_value"] < 0.001
    # By construction, a - b = -1 for every pair, so median_diff is -1.
    assert np.isclose(result["median_diff"], -1.0)


def test_paired_wilcoxon_all_nan_returns_one():
    """All-NaN inputs return p_value=1.0 with a note, no exception."""
    a = np.full(20, np.nan)
    b = np.full(20, np.nan)
    result = paired_wilcoxon(a, b)
    assert result["p_value"] == 1.0
    assert result["n_pairs"] == 0
    assert "note" in result


# ----------------------------------------------------------------------
# Pairwise table
# ----------------------------------------------------------------------

def test_pairwise_table_is_symmetric():
    """4 models -> 4x4 DataFrame, symmetric, NaN on diagonal."""
    rng = np.random.default_rng(0)
    per_fold = {f"model_{k}": rng.normal(size=20) for k in range(4)}
    table = pairwise_table(per_fold, metric_name="auc")

    assert table.shape == (4, 4)
    assert list(table.index) == list(per_fold.keys())
    assert list(table.columns) == list(per_fold.keys())
    for i in range(4):
        assert np.isnan(table.iloc[i, i]), "diagonal must be NaN"
    for i in range(4):
        for j in range(i + 1, 4):
            assert table.iloc[i, j] == table.iloc[j, i], (
                f"asymmetric at ({i}, {j})"
            )
    # metric_name stored in attrs
    assert table.attrs.get("metric_name") == "auc"


# ----------------------------------------------------------------------
# Summarize metric
# ----------------------------------------------------------------------

def test_summarize_metric_schema():
    """summarize_metric returns a dict with exactly the documented keys
    and the right counts."""
    rng = np.random.default_rng(42)
    values = rng.normal(loc=1.0, scale=1.0, size=20)

    result = summarize_metric(
        values, model_name="test_model", metric_name="test_metric"
    )

    expected_keys = {
        "model", "metric", "n_folds", "n_valid",
        "mean", "ci_low", "ci_high", "std",
    }
    assert expected_keys.issubset(set(result.keys()))
    assert result["model"] == "test_model"
    assert result["metric"] == "test_metric"
    assert result["n_folds"] == 20
    assert result["n_valid"] == 20
    assert not np.isnan(result["mean"])
    assert not np.isnan(result["std"])
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]
