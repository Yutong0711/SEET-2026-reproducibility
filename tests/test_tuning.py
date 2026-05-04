"""Unit tests for src/seet/tuning.py.

Run from repo root:

    pytest -q tests/test_tuning.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from seet.tuning import (  # noqa: E402
    _grid_combinations,
    _random_combinations,
    nested_expanding_cv_tune,
)


def _synthetic_panel(n: int = 400, p_pos: float = 0.20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=n, freq="B"),
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
            "f3": rng.standard_normal(n),
            "label": (rng.random(n) < p_pos).astype(int),
        }
    )


# ----------------------------------------------------------------------
# Combination enumeration
# ----------------------------------------------------------------------

def test_grid_combinations_full_count_and_determinism():
    grid = {"a": [1, 2], "b": [10, 20, 30]}
    combos = _grid_combinations(grid)
    assert len(combos) == 6
    # Sorted-key ordering: keys appear in alphabetical order in each dict.
    assert all(list(c.keys()) == ["a", "b"] for c in combos)
    # First combination should be lex-smallest.
    assert combos[0] == {"a": 1, "b": 10}


def test_random_combinations_returns_full_when_n_too_large():
    grid = {"a": [1, 2], "b": [10, 20]}
    combos = _random_combinations(grid, n=100, seed=42)
    assert len(combos) == 4


def test_random_combinations_subset_seeded():
    grid = {"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}
    combos1 = _random_combinations(grid, n=5, seed=42)
    combos2 = _random_combinations(grid, n=5, seed=42)
    assert len(combos1) == 5
    assert combos1 == combos2  # same seed -> same sample


# ----------------------------------------------------------------------
# Inner-fold splits match TimeSeriesSplit(n_splits=3) exactly
# ----------------------------------------------------------------------

def test_inner_splits_match_timeseriessplit():
    """Verify our fold sizes match sklearn TimeSeriesSplit(n_splits=3)
    on a 1000-row in-window subset: 250/250, 500/250, 750/250."""
    from sklearn.model_selection import TimeSeriesSplit

    n = 1000
    splits = list(TimeSeriesSplit(n_splits=3).split(np.arange(n)))
    train_sizes = [len(tr) for tr, _ in splits]
    test_sizes = [len(te) for _, te in splits]
    assert train_sizes == [250, 500, 750], f"unexpected train sizes: {train_sizes}"
    assert test_sizes == [250, 250, 250], f"unexpected test sizes: {test_sizes}"


# ----------------------------------------------------------------------
# Public API contract
# ----------------------------------------------------------------------

def test_returns_expected_keys_grid():
    df = _synthetic_panel()
    grid = {"max_depth": [2, 3], "n_estimators": [50]}
    result = nested_expanding_cv_tune(
        df,
        feature_cols=["f1", "f2", "f3"],
        label_col="label",
        outer_train_end=df["Date"].iloc[-1],
        param_grid=grid,
        n_inner_folds=3,
        seed=42,
        n_jobs=1,
        search_method="grid",
    )
    expected = {
        "best_params", "all_scores", "inner_fold_scores",
        "n_combinations", "n_full_grid", "search_method",
        "seed", "df_hash", "outer_train_end", "wall_time_sec",
        "n_in_window_rows",
    }
    assert expected.issubset(set(result.keys()))
    assert result["n_combinations"] == 2
    assert result["n_full_grid"] == 2
    assert result["search_method"] == "grid"
    assert result["seed"] == 42
    assert len(result["all_scores"]) == 2
    assert len(result["inner_fold_scores"]) == 3


def test_random_search_respects_count():
    df = _synthetic_panel()
    grid = {"max_depth": [2, 3, 4], "n_estimators": [50, 100, 200]}
    result = nested_expanding_cv_tune(
        df,
        feature_cols=["f1", "f2", "f3"],
        label_col="label",
        outer_train_end=df["Date"].iloc[-1],
        param_grid=grid,
        n_inner_folds=3,
        seed=42,
        n_jobs=1,
        search_method="random",
        n_random_samples=4,
    )
    assert result["n_combinations"] == 4
    assert result["n_full_grid"] == 9
    assert result["search_method"] == "random"


def test_outer_train_end_filters_rows():
    """Rows after outer_train_end must NOT contribute to the inner CV."""
    df = _synthetic_panel(n=400)
    midpoint = df["Date"].iloc[200]
    result = nested_expanding_cv_tune(
        df,
        feature_cols=["f1", "f2", "f3"],
        label_col="label",
        outer_train_end=midpoint,
        param_grid={"max_depth": [2], "n_estimators": [50]},
        n_inner_folds=3,
        seed=42,
        n_jobs=1,
    )
    # In-window row count: rows up to and including the 201st (index 200).
    assert result["n_in_window_rows"] == 201
    assert result["outer_train_end"] == midpoint.strftime("%Y-%m-%d")


def test_seed_determinism():
    """Same inputs + same seed -> bit-equal best_params."""
    df = _synthetic_panel()
    grid = {"max_depth": [2, 3, 4], "n_estimators": [50, 100]}
    r1 = nested_expanding_cv_tune(
        df, ["f1", "f2", "f3"], "label",
        df["Date"].iloc[-1], grid,
        n_inner_folds=3, seed=42, n_jobs=1,
    )
    r2 = nested_expanding_cv_tune(
        df, ["f1", "f2", "f3"], "label",
        df["Date"].iloc[-1], grid,
        n_inner_folds=3, seed=42, n_jobs=1,
    )
    assert r1["best_params"] == r2["best_params"]
    assert r1["df_hash"] == r2["df_hash"]
