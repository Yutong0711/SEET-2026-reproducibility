"""Statistical inference utilities for the SEET 2026 evaluation pipeline.

Standalone module — no imports from other seet/ modules. Used by every
track for headline metrics, confidence intervals, and pairwise model
comparisons.

Public API
----------
block_bootstrap_ci(values, block_size=1, n_boot=1000, alpha=0.05, seed=42)
    Moving-block bootstrap CI for the mean of `values`. block_size=1
    reduces to the standard non-parametric bootstrap.

paired_bootstrap_ci(a_values, b_values, block_size=1, n_boot=1000,
                    alpha=0.05, seed=42)
    Paired bootstrap CI for the mean of paired deltas (a - b). Resamples
    fold INDICES (preserving pairing structure); each resampled fold
    contributes both the unmodified value (b) and the ablation value (a),
    and the per-resample mean is computed as mean(a[idx] - b[idx]). Use
    this for paired comparisons (Track E ablation deltas).

paired_wilcoxon(a_values, b_values, alternative='two-sided')
    Paired Wilcoxon signed-rank test on per-fold metric vectors.

summarize_metric(per_fold_values, model_name, metric_name,
                 block_size=1, n_boot=1000, seed=42)
    One-row summary suitable for a results table — mean, 95% CI, std,
    valid-fold count.

pairwise_table(per_fold_dict, metric_name, alpha=0.05)
    Square DataFrame of pairwise Wilcoxon p-values across models.

Conventions (project-wide; see CLAUDE.md):
- Random seed defaults to 42.
- Confidence intervals are 95% by default (alpha=0.05).
- Paired Wilcoxon is two-sided unless the caller overrides.
- NaN handling is explicit: dropped before analysis, with pre-drop and
  post-drop counts recorded in the result dict so the caller can see
  exactly how many folds contributed.
- Edge cases (all-NaN, all-zero-difference) return a soft `p_value=1.0`
  with a `note` key rather than raising — calm folds with identical
  baseline behavior are an expected operating point, not a programmer
  error.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import scipy.stats


# ----------------------------------------------------------------------
# Bootstrap CI
# ----------------------------------------------------------------------

def block_bootstrap_ci(
    values,
    block_size: int = 1,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Moving-block bootstrap confidence interval for the mean.

    Parameters
    ----------
    values : array-like
        1D numpy array or pandas Series of per-fold metric values.
        NaNs are dropped before resampling.
    block_size : int, default 1
        Block length for the moving-block bootstrap. With block_size=1
        this is the standard i.i.d. bootstrap. block_size > 1 preserves
        local serial structure (useful when folds are temporally
        adjacent and exhibit autocorrelation).
    n_boot : int, default 1000
        Number of bootstrap resamples.
    alpha : float, default 0.05
        Two-sided coverage level: the returned CI is the
        (alpha/2, 1-alpha/2) percentile pair.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        mean        : float, mean of the (NaN-dropped) input
        ci_low      : float, lower percentile of bootstrap means
        ci_high     : float, upper percentile of bootstrap means
        n_boot      : int, number of resamples actually performed
        block_size  : int, block size actually used (clamped to n_valid
                      when the input is shorter than the requested block)
        seed        : int, the seed used
        n_input     : int, length of `values` before NaN drop
        n_valid     : int, length after NaN drop (= number of values
                      actually resampled)
        alpha       : float, the alpha used
        note        : str (optional) — present when an edge case
                      degraded the CI (e.g. all-NaN input)
    """
    arr = np.asarray(values, dtype=float).ravel()
    n_input = int(arr.size)
    arr = arr[~np.isnan(arr)]
    n_valid = int(arr.size)

    base = {
        "n_boot": int(n_boot),
        "block_size": int(max(1, block_size)),
        "seed": int(seed),
        "n_input": n_input,
        "n_valid": n_valid,
        "alpha": float(alpha),
    }

    if n_valid == 0:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            **base,
            "note": "all values NaN; CI undefined",
        }

    b = max(1, int(block_size))
    if b > n_valid:
        b = n_valid  # block cannot exceed valid-data length
    n_blocks = n_valid - b + 1
    n_per_resample = (n_valid + b - 1) // b  # ceil(n_valid / b)

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        starts = rng.integers(0, n_blocks, size=n_per_resample)
        idx = (starts[:, None] + np.arange(b)).ravel()[:n_valid]
        boot_means[i] = arr[idx].mean()

    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.percentile(boot_means, 100.0 * alpha / 2.0)),
        "ci_high": float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0))),
        **base,
        "block_size": int(b),
    }


# ----------------------------------------------------------------------
# Paired bootstrap CI on deltas
# ----------------------------------------------------------------------

def paired_bootstrap_ci(
    a_values,
    b_values,
    block_size: int = 1,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Paired bootstrap CI for the mean of paired deltas (a - b).

    Pairing semantics
    -----------------
    Resamples fold INDICES (with optional block structure), not the
    pre-computed deltas. Each resampled fold index contributes BOTH
    a_values[idx] and b_values[idx] to the resample, and the
    per-resample mean is computed as

        mean( a_values[idx_i] - b_values[idx_i] )

    where {idx_i} are the resampled fold indices. This preserves the
    pairing structure between unmodified and ablation arms by
    construction: a fold that is sampled twice contributes its delta
    twice; a fold that is not sampled contributes nothing.

    NaN handling: any pair where either side is NaN is dropped before
    resampling, identically to paired_wilcoxon.

    Parameters
    ----------
    a_values, b_values : array-like
        Equal-length per-fold metric arrays. The CI is on the mean of
        (a - b). For Track E ablations, a_values is the ablation arm
        and b_values is the unmodified Track A baseline.
    block_size, n_boot, alpha, seed : as in block_bootstrap_ci.

    Returns
    -------
    dict with keys:
        delta_mean   : float, mean of (a - b) over valid pairs
        ci_low       : float, lower percentile of bootstrap delta means
        ci_high      : float, upper percentile of bootstrap delta means
        n_pairs      : int, number of valid (non-NaN) paired observations
        n_input      : int, length of inputs before NaN drop
        n_boot       : int, number of resamples performed
        block_size   : int, block size used (clamped to n_pairs)
        seed         : int, the seed used
        alpha        : float, the alpha used
        note         : str (optional) — present in degenerate cases
                       (no valid pairs; in that case all numeric fields
                       are NaN).
    """
    a = np.asarray(a_values, dtype=float).ravel()
    b = np.asarray(b_values, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"a_values and b_values must have the same length; "
            f"got {a.shape} vs {b.shape}"
        )
    n_input = int(a.size)

    mask = ~(np.isnan(a) | np.isnan(b))
    a_clean = a[mask]
    b_clean = b[mask]
    n_pairs = int(a_clean.size)

    base = {
        "n_pairs": n_pairs,
        "n_input": n_input,
        "n_boot": int(n_boot),
        "block_size": int(max(1, block_size)),
        "seed": int(seed),
        "alpha": float(alpha),
    }

    if n_pairs == 0:
        return {
            "delta_mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            **base,
            "note": "no valid pairs (all NaN); CI undefined",
        }

    bsz = max(1, int(block_size))
    if bsz > n_pairs:
        bsz = n_pairs
    n_blocks = n_pairs - bsz + 1
    n_per_resample = (n_pairs + bsz - 1) // bsz  # ceil

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        starts = rng.integers(0, n_blocks, size=n_per_resample)
        idx = (starts[:, None] + np.arange(bsz)).ravel()[:n_pairs]
        # Pairing preserved: same idx applied to both arrays before diff.
        boot_means[i] = (a_clean[idx] - b_clean[idx]).mean()

    return {
        "delta_mean": float((a_clean - b_clean).mean()),
        "ci_low": float(np.percentile(boot_means, 100.0 * alpha / 2.0)),
        "ci_high": float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0))),
        **base,
        "block_size": int(bsz),
    }


# ----------------------------------------------------------------------
# Paired Wilcoxon
# ----------------------------------------------------------------------

def paired_wilcoxon(
    a_values,
    b_values,
    alternative: str = "two-sided",
) -> dict:
    """Paired Wilcoxon signed-rank test on per-fold metric vectors.

    Parameters
    ----------
    a_values, b_values : array-like
        Equal-length per-fold metrics for two models. NaN pairs (where
        either side is NaN) are dropped.
    alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
        Passed through to scipy.stats.wilcoxon.

    Returns
    -------
    dict with keys:
        statistic   : float, the Wilcoxon W statistic (NaN if undefined)
        p_value     : float in [0, 1]
        n_pairs     : int, number of valid (non-NaN) paired observations
        median_diff : float, median of (a - b) over valid pairs
        alternative : str, the alternative used
        note        : str (optional) — present in degenerate cases:
                      no valid pairs, or all paired differences zero.
                      In both cases p_value is set to 1.0 (soft fail,
                      not an exception, because calm-fold ties are
                      expected in this work).

    Notes
    -----
    No exception is raised for the all-NaN or all-zero-diff cases;
    callers can rely on a stable dict shape across folds.
    """
    a = np.asarray(a_values, dtype=float).ravel()
    b = np.asarray(b_values, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"a_values and b_values must have the same length; "
            f"got {a.shape} vs {b.shape}"
        )

    mask = ~(np.isnan(a) | np.isnan(b))
    a_clean = a[mask]
    b_clean = b[mask]
    n_pairs = int(a_clean.size)

    if n_pairs == 0:
        return {
            "statistic": float("nan"),
            "p_value": 1.0,
            "n_pairs": 0,
            "median_diff": float("nan"),
            "alternative": alternative,
            "note": "no valid pairs (all NaN)",
        }

    diffs = a_clean - b_clean
    median_diff = float(np.median(diffs))

    if np.all(diffs == 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n_pairs": n_pairs,
            "median_diff": 0.0,
            "alternative": alternative,
            "note": "all paired differences are zero",
        }

    result = scipy.stats.wilcoxon(
        a_clean, b_clean, alternative=alternative, zero_method="wilcox"
    )
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_pairs": n_pairs,
        "median_diff": median_diff,
        "alternative": alternative,
    }


# ----------------------------------------------------------------------
# One-row summary
# ----------------------------------------------------------------------

def summarize_metric(
    per_fold_values,
    model_name: str,
    metric_name: str,
    block_size: int = 1,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """One-row results-table summary for a single (model, metric) pair.

    Returns
    -------
    dict with keys:
        model    : str
        metric   : str
        n_folds  : int, len(per_fold_values)
        n_valid  : int, count after dropping NaNs
        mean     : float, mean of valid values
        ci_low   : float, lower 95% bootstrap CI
        ci_high  : float, upper 95% bootstrap CI
        std      : float, sample std (ddof=1) of valid values, or NaN
                   if fewer than 2 valid folds
    """
    arr = np.asarray(per_fold_values, dtype=float).ravel()
    n_folds = int(arr.size)
    valid = arr[~np.isnan(arr)]
    n_valid = int(valid.size)

    if n_valid == 0:
        return {
            "model": str(model_name),
            "metric": str(metric_name),
            "n_folds": n_folds,
            "n_valid": 0,
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "std": float("nan"),
        }

    boot = block_bootstrap_ci(
        arr, block_size=block_size, n_boot=n_boot, seed=seed
    )
    std = float(np.std(valid, ddof=1)) if n_valid >= 2 else float("nan")

    return {
        "model": str(model_name),
        "metric": str(metric_name),
        "n_folds": n_folds,
        "n_valid": n_valid,
        "mean": boot["mean"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "std": std,
    }


# ----------------------------------------------------------------------
# Pairwise Wilcoxon table
# ----------------------------------------------------------------------

def pairwise_table(
    per_fold_dict: Mapping[str, "np.ndarray | pd.Series | list"],
    metric_name: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Pairwise Wilcoxon p-values across models.

    Parameters
    ----------
    per_fold_dict : Mapping[str, array-like]
        Maps model name -> 1D array of per-fold metric values. All
        models must use the same fold structure (so positions align).
    metric_name : str
        Stored in `df.attrs['metric_name']` for downstream display.
    alpha : float, default 0.05
        Stored in `df.attrs['alpha']` for downstream significance
        flagging by callers.

    Returns
    -------
    pandas.DataFrame
        Square (n_models × n_models) table indexed and columned by
        model name. Off-diagonal entries are two-sided paired-Wilcoxon
        p-values comparing the row model to the column model. Diagonal
        is NaN. By construction, table[i,j] == table[j,i] (symmetric).
    """
    model_names = list(per_fold_dict.keys())
    n = len(model_names)

    table = pd.DataFrame(
        np.full((n, n), np.nan, dtype=float),
        index=model_names,
        columns=model_names,
    )

    for i in range(n):
        for j in range(i + 1, n):
            res = paired_wilcoxon(
                per_fold_dict[model_names[i]],
                per_fold_dict[model_names[j]],
                alternative="two-sided",
            )
            p = res["p_value"]
            table.iloc[i, j] = p
            table.iloc[j, i] = p

    table.attrs["metric_name"] = str(metric_name)
    table.attrs["alpha"] = float(alpha)
    return table
