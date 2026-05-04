"""Nested expanding-window hyperparameter tuning for Track B.

Standalone module — no imports from other seet/ modules. Used by Track B
to select LightGBM hyperparameters per (stress_def, outer_fold) before
re-running Track A's LightGbmTuned rows with the chosen values.

Public API
----------
nested_expanding_cv_tune(df, feature_cols, label_col, outer_train_end,
                          param_grid, scoring='pr_auc', n_inner_folds=3,
                          seed=42, n_jobs=4, search_method='auto',
                          n_random_samples=80) -> dict

Inner-fold splitting uses sklearn.model_selection.TimeSeriesSplit
(n_splits=3) directly. With n_splits=3 the realised splits are
(approximately, modulo integer division on N):

    fold 0: train [0, N/4),    test [N/4,  N/2)
    fold 1: train [0, N/2),    test [N/2,  3N/4)
    fold 2: train [0, 3N/4),   test [3N/4, N)

Every observation is eligible to land in a test fold (no "always train"
warmup region).

Conventions
-----------
- Scoring: PR-AUC (sklearn.metrics.average_precision_score).
- NaN PR-AUC on degenerate inner folds (single-class train or test)
  drops out of the inner-fold mean via np.nanmean. If all 3 inner folds
  are NaN, the combination's score is NaN and it cannot win argmax.
- Tie-breaking on equal mean PR-AUC: first-in-iteration order
  (deterministic; for grid that's itertools.product order, for random
  that's the rng-sorted index order).
- class_weight='balanced' is forced into every LightGBM fit. The grid
  does not include class_weight; that's a fixed convention.
- Parallelism: joblib with backend='threading' (LightGBM releases the
  GIL internally; this avoids Windows process-spawn overhead on short
  per-fit tasks).
"""
from __future__ import annotations

import hashlib
import itertools
import time
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

import lightgbm as lgb


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _df_hash(df: pd.DataFrame) -> str:
    """Stable sha256 of df contents (used for provenance)."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()


def _grid_combinations(param_grid: dict) -> list[dict]:
    """Cartesian product of values, with sorted keys for determinism."""
    keys = sorted(param_grid.keys())
    values = [list(param_grid[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _random_combinations(param_grid: dict, n: int, seed: int) -> list[dict]:
    """Sample n random combinations without replacement (or all if n>=full)."""
    all_combos = _grid_combinations(param_grid)
    if n >= len(all_combos):
        return all_combos
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(all_combos), size=n, replace=False).tolist())
    return [all_combos[i] for i in indices]


def _fit_and_score(
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> float:
    """Fit one LightGBM with `params` on (X[train_idx], y[train_idx]) and
    return PR-AUC on (X[test_idx], y[test_idx]). Returns NaN on
    degenerate train or test folds (single class)."""
    y_tr = y[train_idx].astype(int)
    y_te = y[test_idx].astype(int)
    if np.unique(y_tr).size < 2 or np.unique(y_te).size < 2:
        return float("nan")
    final_params = {"class_weight": "balanced", **params}
    model = lgb.LGBMClassifier(
        **final_params,
        random_state=seed,
        n_jobs=1,
        verbose=-1,
        verbosity=-1,
    )
    model.fit(X[train_idx], y_tr)
    proba = model.predict_proba(X[test_idx])
    if proba.shape[1] < 2:
        return float("nan")
    return float(average_precision_score(y_te, proba[:, 1]))


def _evaluate_combo(
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> dict:
    """Run all inner-fold fits for one combination, return summary dict."""
    fold_scores: list[float] = []
    for train_idx, test_idx in splits:
        fold_scores.append(_fit_and_score(params, X, y, train_idx, test_idx, seed))
    valid = [s for s in fold_scores if not np.isnan(s)]
    mean_score = float(np.mean(valid)) if valid else float("nan")
    return {
        "params": params,
        "fold_scores": fold_scores,
        "mean_score": mean_score,
    }


# ----------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------

def nested_expanding_cv_tune(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    outer_train_end: Any,
    param_grid: dict,
    scoring: str = "pr_auc",
    n_inner_folds: int = 3,
    seed: int = 42,
    n_jobs: int = 4,
    search_method: str = "auto",
    n_random_samples: int = 80,
) -> dict:
    """Nested expanding-window CV hyperparameter tuning.

    Filters df to rows with Date <= outer_train_end, drops rows with
    NaN in feature_cols or label_col, then runs an inner CV via
    sklearn TimeSeriesSplit(n_splits=n_inner_folds). For each
    parameter combination, fits LightGBM on each inner train fold,
    scores PR-AUC on the inner test fold, averages across inner folds
    via nanmean. Best combination is the one with the highest mean
    inner PR-AUC (ties broken by iteration order).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Date, feature_cols, label_col.
    feature_cols : list[str]
    label_col : str
    outer_train_end : timestamp-like
    param_grid : dict[str, list]
        e.g. {"max_depth": [2, 3], "n_estimators": [50, 200]}
    scoring : str
        Only "pr_auc" supported.
    n_inner_folds : int
        Default 3. Passed to TimeSeriesSplit.
    seed : int
        Random seed for LightGBM and (if applicable) random search.
    n_jobs : int
        joblib parallelism. Default 4.
    search_method : {"auto", "grid", "random"}
        "auto" falls back to "grid". Caller decides via the wall-time
        gate which to pass in.
    n_random_samples : int
        Used when search_method="random".

    Returns
    -------
    dict with keys: best_params, all_scores, inner_fold_scores,
        n_combinations, n_full_grid, search_method, seed, df_hash,
        outer_train_end, wall_time_sec, n_in_window_rows.
    """
    if scoring != "pr_auc":
        raise NotImplementedError(f"scoring={scoring!r} not supported")

    # Filter to in-window data and drop NaN rows.
    df_in = df.copy()
    df_in["Date"] = pd.to_datetime(df_in["Date"])
    df_in = df_in[df_in["Date"] <= pd.Timestamp(outer_train_end)]
    df_in = df_in.dropna(subset=list(feature_cols) + [label_col])
    df_in = df_in.sort_values("Date").reset_index(drop=True)

    n = len(df_in)
    if n < 4 * n_inner_folds:
        raise ValueError(
            f"In-window data has only {n} rows after dropna; need at least "
            f"{4 * n_inner_folds} for {n_inner_folds}-fold CV."
        )

    X = df_in[list(feature_cols)].to_numpy(dtype=float)
    y = df_in[label_col].to_numpy(dtype=float)

    # Inner-fold splits (TimeSeriesSplit semantics).
    tss = TimeSeriesSplit(n_splits=n_inner_folds)
    splits = [(tr, te) for tr, te in tss.split(X)]

    # Combinations.
    method = search_method
    if method == "auto":
        method = "grid"
    if method == "grid":
        combinations = _grid_combinations(param_grid)
    elif method == "random":
        combinations = _random_combinations(param_grid, n_random_samples, seed)
    else:
        raise ValueError(f"Unknown search_method: {method!r}")

    n_full_grid = 1
    for v in param_grid.values():
        n_full_grid *= len(list(v))

    # Run all combinations in parallel. Threading backend keeps overhead low
    # because LightGBM releases the GIL.
    t0 = time.time()
    all_scores = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_evaluate_combo)(combo, X, y, splits, seed)
        for combo in combinations
    )
    wall = time.time() - t0

    # Pick best by mean_score, tie-broken by first-in-iteration-order.
    best_idx = -1
    best_score = -np.inf
    for i, r in enumerate(all_scores):
        m = r["mean_score"]
        if not np.isnan(m) and m > best_score:
            best_score = m
            best_idx = i
    if best_idx == -1:
        # No combination produced a non-NaN score. Pick the first as a
        # placeholder; caller can detect via inner_fold_scores all NaN.
        best_idx = 0

    return {
        "best_params": all_scores[best_idx]["params"],
        "all_scores": all_scores,
        "inner_fold_scores": all_scores[best_idx]["fold_scores"],
        "n_combinations": len(combinations),
        "n_full_grid": int(n_full_grid),
        "search_method": method,
        "seed": int(seed),
        "df_hash": _df_hash(df_in),
        "outer_train_end": pd.Timestamp(outer_train_end).strftime("%Y-%m-%d"),
        "wall_time_sec": float(wall),
        "n_in_window_rows": int(n),
    }
