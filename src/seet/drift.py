"""Distribution-drift utilities for Track G's Layer-4 instrumentation.

Two scalar drift statistics on histogram-discretized features:

    psi(reference, current, n_bins=10)
        Population Stability Index. Standard convention:
            PSI < 0.10  -> stable
            0.10 - 0.25 -> moderate drift
            PSI > 0.25  -> high drift
        See Karakoulas (2004) and Siddiqi (2006). PSI is the
        symmetrized KL divergence between two histograms with a
        small smoothing constant `eps` (1e-6) to avoid log(0) when
        a bin is empty on one side.

    sym_kl(reference, current, n_bins=10)
        Symmetric Kullback-Leibler divergence between two
        histograms. By construction sym_kl(a, b) == sym_kl(b, a)
        within float tolerance. Same `eps` smoothing as PSI.

Both statistics use **quantile-based binning fit on the reference
distribution** (so each reference bin contains ~equal mass) and
both drop NaN before binning. If a feature is all-NaN in either
side, the function returns NaN.

Public API
----------
psi(reference, current, n_bins=10) -> float
sym_kl(reference, current, n_bins=10) -> float
feature_level_drift(train_df, test_df, feature_cols, n_bins=10)
    -> pd.DataFrame  (one row per feature: feature, psi, sym_kl)
fold_level_drift_summary(train_df, test_df, feature_cols, *,
                         n_bins=10, high_threshold=0.25)
    -> dict (mean_psi, max_psi, mean_kl, max_kl,
             n_high_drift, high_drift_features)

Cross-asset note: `mean_psi`/`max_psi` aggregates over the columns
in `feature_cols`. For cross-asset comparison the caller should
either supply the same `feature_cols` to both sides, or accept
that "drift over THIS asset's pipeline" is the operationally
meaningful number (see CLAUDE.md Track G summary).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-6  # smoothing for empty bins; standard PSI/KL practice


def _build_reference_bins(
    reference: np.ndarray, n_bins: int
) -> np.ndarray:
    """Quantile-based bin edges fit on `reference`. Returns a
    monotone-increasing array of length n_bins+1 with edges that
    span the full reference support, with -inf and +inf appended
    so the binning is total over R."""
    ref = np.asarray(reference, dtype=float)
    ref = ref[~np.isnan(ref)]
    if ref.size == 0:
        return np.array([-np.inf, np.inf])
    # Inner edges at quantiles 1/n_bins .. (n_bins-1)/n_bins.
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    inner = np.quantile(ref, qs)
    # Drop duplicate edges (degenerate features with mass at a
    # single value) -- they would produce zero-width bins.
    inner = np.unique(inner)
    edges = np.concatenate([[-np.inf], inner, [np.inf]])
    return edges


def _hist_props(
    values: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Histogram a 1D array against the supplied edges and return
    proportions (counts / n_total)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        # Return uniform-zero histogram; caller will detect via
        # the reference side being empty and short-circuit to NaN.
        return np.zeros(edges.size - 1, dtype=float)
    counts, _ = np.histogram(arr, bins=edges)
    return counts.astype(float) / n


def psi(
    reference: Iterable[float],
    current: Iterable[float],
    n_bins: int = 10,
) -> float:
    """Population Stability Index of `current` against `reference`.

    PSI = sum_i (p_curr_i - p_ref_i) * log(p_curr_i / p_ref_i)
    with eps smoothing on empty bins.

    Returns NaN if either side has no valid (non-NaN) values.
    Returns 0.0 (within float tolerance) when reference and
    current are samples from the same distribution.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    edges = _build_reference_bins(ref, n_bins)
    p = _hist_props(ref, edges) + EPS
    q = _hist_props(cur, edges) + EPS
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def sym_kl(
    reference: Iterable[float],
    current: Iterable[float],
    n_bins: int = 10,
) -> float:
    """Symmetric KL divergence on histogram-discretized inputs.

    sym_kl(a, b) = KL(a || b) + KL(b || a), where both KLs use
    the SAME bin edges (fit on the union to keep symmetry exact)
    and eps smoothing on empty bins.

    Returns NaN if either side has no valid values.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    # Use the union of values for binning so the metric is
    # symmetric: swapping reference and current gives the same edges.
    union = np.concatenate([ref, cur])
    edges = _build_reference_bins(union, n_bins)
    p = _hist_props(ref, edges) + EPS
    q = _hist_props(cur, edges) + EPS
    p = p / p.sum()
    q = q / q.sum()
    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    return kl_pq + kl_qp


def feature_level_drift(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Iterable[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Per-feature PSI and sym_kl over (train_df, test_df).

    Returns a DataFrame with columns:
        feature, psi, sym_kl
    one row per feature in `feature_cols`. Either statistic is
    NaN when the feature has no valid values in one or both sides.
    """
    cols = list(feature_cols)
    rows: list[dict] = []
    for col in cols:
        if col not in train_df.columns or col not in test_df.columns:
            rows.append({"feature": col, "psi": float("nan"),
                         "sym_kl": float("nan")})
            continue
        ref_vals = train_df[col].to_numpy(dtype=float)
        cur_vals = test_df[col].to_numpy(dtype=float)
        rows.append({
            "feature": col,
            "psi": psi(ref_vals, cur_vals, n_bins=n_bins),
            "sym_kl": sym_kl(ref_vals, cur_vals, n_bins=n_bins),
        })
    return pd.DataFrame(rows)


def fold_level_drift_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Iterable[str],
    *,
    n_bins: int = 10,
    high_threshold: float = 0.25,
) -> dict:
    """Aggregate PSI and sym_kl across the supplied features.

    Returns a dict:
        mean_psi             : mean PSI over features (NaN-aware)
        max_psi              : max  PSI over features
        mean_kl              : mean sym_kl over features
        max_kl               : max  sym_kl over features
        n_high_drift         : count of features with PSI > high_threshold
        high_drift_features  : list of feature names sorted by descending
                               PSI, restricted to PSI > high_threshold
        n_features           : total features evaluated (NaN-included)
        n_features_valid     : features with non-NaN PSI
    """
    df = feature_level_drift(
        train_df, test_df, feature_cols, n_bins=n_bins
    )
    valid = df.dropna(subset=["psi"])
    if valid.empty:
        return {
            "mean_psi": float("nan"),
            "max_psi": float("nan"),
            "mean_kl": float("nan"),
            "max_kl": float("nan"),
            "n_high_drift": 0,
            "high_drift_features": [],
            "n_features": int(len(df)),
            "n_features_valid": 0,
        }
    high = (
        valid[valid["psi"] > high_threshold]
        .sort_values("psi", ascending=False)
    )
    return {
        "mean_psi": float(valid["psi"].mean()),
        "max_psi": float(valid["psi"].max()),
        "mean_kl": float(valid["sym_kl"].mean()),
        "max_kl": float(valid["sym_kl"].max()),
        "n_high_drift": int(len(high)),
        "high_drift_features": high["feature"].tolist(),
        "n_features": int(len(df)),
        "n_features_valid": int(len(valid)),
    }
