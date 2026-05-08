"""Failure injection / mutation testing for SEET 2026 Track F.

Five corruption modes that mutate a copy of a processed panel and
return (corrupted_panel, ground_truth_flags, mechanism_metadata).
The canonical on-disk panel is never modified — every corruption
operates on a fresh `pd.DataFrame.copy()` of the input.

Each (corruption_id, rate, seed) triple has a deterministic RNG so
the experiment is fully reproducible.

Public API
----------
CORRUPTION_IDS : tuple[str, ...]
    The five canonical mutation classes.

corrupt_panel(panel_df, corruption_id, rate, seed, *,
              manifest_entry=None, value_columns=None)
    -> (corrupted_panel, ground_truth_flags, metadata)

`ground_truth_flags` is a Boolean array aligned to the
*corrupted* panel's row index; for C5 (calendar_gaps) the
corruption removes rows, so the flag array length is the
corrupted panel's length and is all-False — the metadata's
`removed_dates` list documents what was dropped.

`metadata` carries mechanism-specific details: which rows were
duplicated, which (date, column) cells were NaN'd, etc.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


CORRUPTION_IDS: tuple[str, ...] = (
    "duplicate_dates",
    "missing_values",
    "stale_quotes",
    "extreme_jumps",
    "calendar_gaps",
)

# Mechanistic classification used in validator_coverage.csv. Maps
# corruption_id -> mechanism. C1 and C2 are 'detected' if the
# validator catches them (this can be overridden per-row at coverage
# time); C3/C4 always silently flow through; C5 removes the
# offending rows from the panel before the validator can see them.
GAP_MECHANISM: dict[str, str] = {
    "duplicate_dates": "detected",        # V1 catches duplicates
    "missing_values":  "detected",        # V2 catches unexpected NaN
    "stale_quotes":    "silent_ignore",   # no rule
    "extreme_jumps":   "silent_ignore",   # no rule
    "calendar_gaps":   "removed_before_inspection",  # row gone
}

# spx_extended_2011 has 6 value columns. We keep them parameterizable
# so the same module can be reused on other panels in the future.
SPX_EXTENDED_VALUE_COLUMNS: tuple[str, ...] = (
    "SPX", "VIX", "VIX9D", "VIX3M", "VIX6M", "VVIX",
)


def _rng(corruption_id: str, rate: float, seed: int) -> np.random.Generator:
    """Deterministic RNG keyed by (corruption_id, rate, seed)."""
    h = abs(hash((corruption_id, round(rate, 6), int(seed)))) % (2**32)
    return np.random.default_rng(h)


def _excluded_dates_for_v2(manifest_entry: dict | None) -> set[pd.Timestamp]:
    """Set of (date, *) pairs that V2 considers 'expected' NaN —
    we exclude these from the C2 sampling so the recall measurement
    isn't contaminated by the data_quality_notes registry. We
    aggregate across columns for a conservative exclusion (any date
    that's documented for ANY column is excluded)."""
    out: set[pd.Timestamp] = set()
    if not manifest_entry:
        return out
    for note in manifest_entry.get("data_quality_notes", []) or []:
        for d in note.get("missing_dates") or note.get("dates") or []:
            out.add(pd.Timestamp(d))
    return out


# =====================================================================
# C1 — duplicate_dates
# =====================================================================

def _corrupt_duplicate_dates(
    panel_df: pd.DataFrame, rate: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Insert TRUE duplicate-date rows. Each duplicate has the same
    Date AND the same values as the target row -- a literal row
    repetition, the kind of corruption that arises from accidental
    re-ingestion of a CSV chunk or a retry-with-no-deduplication bug
    in a data-feed client.

    Design note (Track F revision). The earlier version of this
    function copied values from a RANDOM source row, which produced
    synthetic stress events whenever a future-dated source row's SPX
    was inserted at an earlier target date: future_drawdown computed
    against the panel's real later prices yielded artificial -50%+
    drawdowns and label=1, giving the SILENCED-branch model fake
    alarms to validate. The leakage was diagnosed in
    scripts/diagnose_track_f_c1_all_folds.py (11 of 33 valid cells
    had SILENCED lift >= 4.0, P50 = 2.94, max = 17.57). The fixed
    semantics (same Date + same values) preserve V1's perfect-recall
    detection while removing the synthetic-event mechanism.
    """
    n = len(panel_df)
    n_dup = max(1, int(np.floor(rate * n)))
    target_idx = rng.choice(n, size=n_dup, replace=False)
    # True duplicate: copy both the Date and the values from the
    # target row (same row, repeated).
    dup_rows = panel_df.iloc[target_idx].copy().reset_index(drop=True)
    corrupted = pd.concat(
        [panel_df, dup_rows], ignore_index=True
    ).sort_values("Date", kind="mergesort").reset_index(drop=True)

    # Ground truth: True for every row that's a duplicate (i.e., its
    # Date appears > 1 time in the corrupted panel). This matches V1's
    # "flag both copies" semantics so precision/recall are aligned.
    ground_truth = corrupted["Date"].duplicated(keep=False).to_numpy(dtype=bool)
    metadata = {
        "n_inserted": int(n_dup),
        "target_dates": [
            d.strftime("%Y-%m-%d")
            for d in panel_df.iloc[target_idx]["Date"]
        ],
        "duplicate_semantics": "true_duplicate (same Date + same values)",
    }
    return corrupted, ground_truth, metadata


# =====================================================================
# C2 — missing_values
# =====================================================================

def _corrupt_missing_values(
    panel_df: pd.DataFrame,
    rate: float,
    rng: np.random.Generator,
    value_columns: Iterable[str],
    manifest_entry: dict | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Set rate * n_rows * n_value_columns randomly chosen
    (row, column) cells to NaN. Cells already NaN are skipped.
    Documented `data_quality_notes` dates are excluded from the
    sampling pool so V2's recall measurement isn't contaminated.
    """
    corrupted = panel_df.copy()
    n = len(panel_df)
    cols = [c for c in value_columns if c in panel_df.columns]
    n_cells_target = max(1, int(np.floor(rate * n * len(cols))))

    excluded = _excluded_dates_for_v2(manifest_entry)
    dates = pd.to_datetime(panel_df["Date"]).to_numpy()
    excluded_arr = np.array(
        sorted(np.datetime64(d) for d in excluded), dtype="datetime64[ns]"
    ) if excluded else np.array([], dtype="datetime64[ns]")
    eligible_rows = (
        ~np.isin(dates, excluded_arr) if excluded_arr.size else np.ones(n, bool)
    )

    eligible_pairs: list[tuple[int, str]] = []
    for col in cols:
        col_isna = panel_df[col].isna().to_numpy(dtype=bool)
        for i in np.where(eligible_rows & ~col_isna)[0]:
            eligible_pairs.append((int(i), col))

    n_cells = min(n_cells_target, len(eligible_pairs))
    chosen = rng.choice(len(eligible_pairs), size=n_cells, replace=False)
    nan_rows: set[int] = set()
    nan_cells: list[tuple[str, str]] = []
    for k in chosen:
        row_idx, col = eligible_pairs[k]
        corrupted.iat[row_idx, corrupted.columns.get_loc(col)] = np.nan
        nan_rows.add(row_idx)
        nan_cells.append(
            (panel_df.iloc[row_idx]["Date"].strftime("%Y-%m-%d"), col)
        )

    ground_truth = np.zeros(n, dtype=bool)
    for r in nan_rows:
        ground_truth[r] = True

    metadata = {
        "n_cells_corrupted": int(n_cells),
        "n_rows_with_corruption": int(len(nan_rows)),
        "excluded_dates_from_sampling": [
            d.strftime("%Y-%m-%d") for d in sorted(excluded)
        ],
        "first_corrupted_cells": nan_cells[:10],
    }
    return corrupted, ground_truth, metadata


# =====================================================================
# C3 — stale_quotes
# =====================================================================

def _corrupt_stale_quotes(
    panel_df: pd.DataFrame,
    rate: float,
    rng: np.random.Generator,
    value_columns: Iterable[str],
    run_length: int = 5,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Pick run heads at random; replace the next (run_length-1)
    rows' value in a randomly chosen value column with the head's
    value (simulating a feed freezing for `run_length` days).

    Total flagged rows ≈ rate * n_rows; run heads count as flagged
    too (they're part of the stale segment by definition).
    """
    corrupted = panel_df.copy()
    n = len(panel_df)
    cols = [c for c in value_columns if c in panel_df.columns]
    target_total = max(run_length, int(np.floor(rate * n)))
    n_runs = max(1, target_total // run_length)
    # Pick run heads from the upper part of the index where there's
    # room for run_length-1 follow-ups.
    max_head = n - run_length
    if max_head <= 0:
        return corrupted, np.zeros(n, dtype=bool), {"n_runs": 0}

    head_choices = rng.choice(max_head, size=min(n_runs, max_head), replace=False)
    ground_truth = np.zeros(n, dtype=bool)
    runs_meta: list[dict] = []
    for head in head_choices:
        col = cols[int(rng.integers(0, len(cols)))]
        head_value = panel_df.iloc[head][col]
        if pd.isna(head_value):
            continue
        for k in range(run_length):
            row = head + k
            if row >= n:
                break
            corrupted.iat[row, corrupted.columns.get_loc(col)] = head_value
            ground_truth[row] = True
        runs_meta.append({
            "head_date": panel_df.iloc[head]["Date"].strftime("%Y-%m-%d"),
            "column": col,
            "length": run_length,
        })

    metadata = {
        "run_length": int(run_length),
        "n_runs": len(runs_meta),
        "n_rows_flagged": int(ground_truth.sum()),
        "first_runs": runs_meta[:5],
    }
    return corrupted, ground_truth, metadata


# =====================================================================
# C4 — extreme_jumps
# =====================================================================

def _corrupt_extreme_jumps(
    panel_df: pd.DataFrame,
    rate: float,
    rng: np.random.Generator,
    value_columns: Iterable[str],
    magnitude_in_mads: float = 6.0,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Sample rate * n_rows rows; on each, multiply a randomly
    chosen value column's value by a jump factor:

        factor = 1.0 + sign * magnitude_in_mads * mad_of_diff(col)

    where mad_of_diff is the median absolute deviation of the
    column's first-difference distribution. magnitude_in_mads = 6
    is well beyond ordinary daily moves for any of the SPX panel
    columns, making the corruption unambiguously extreme.
    """
    corrupted = panel_df.copy()
    n = len(panel_df)
    cols = [c for c in value_columns if c in panel_df.columns]
    n_jumps = max(1, int(np.floor(rate * n)))

    # Pre-compute MAD of each column's first-difference series.
    mads: dict[str, float] = {}
    for col in cols:
        diffs = panel_df[col].diff().to_numpy(dtype=float)
        diffs = diffs[~np.isnan(diffs)]
        if diffs.size == 0:
            mads[col] = 0.0
            continue
        med = float(np.median(diffs))
        mads[col] = float(np.median(np.abs(diffs - med)))

    # Sample rows from the *interior* of the panel so the jump has
    # context on both sides.
    eligible = np.arange(1, n - 1)
    chosen_rows = rng.choice(eligible, size=min(n_jumps, eligible.size), replace=False)
    ground_truth = np.zeros(n, dtype=bool)
    jumps_meta: list[dict] = []
    for r in chosen_rows:
        col = cols[int(rng.integers(0, len(cols)))]
        sign = 1.0 if rng.integers(0, 2) == 1 else -1.0
        mad = mads[col]
        if mad == 0.0:
            continue
        original = float(panel_df.iat[r, panel_df.columns.get_loc(col)])
        jump = sign * magnitude_in_mads * mad
        new_value = original + jump
        corrupted.iat[r, corrupted.columns.get_loc(col)] = new_value
        ground_truth[r] = True
        jumps_meta.append({
            "date": panel_df.iloc[r]["Date"].strftime("%Y-%m-%d"),
            "column": col,
            "original": original,
            "jump": jump,
            "magnitude_in_mads": magnitude_in_mads,
        })

    metadata = {
        "magnitude_in_mads": float(magnitude_in_mads),
        "n_jumps": len(jumps_meta),
        "first_jumps": jumps_meta[:5],
    }
    return corrupted, ground_truth, metadata


# =====================================================================
# C5 — calendar_gaps
# =====================================================================

def _corrupt_calendar_gaps(
    panel_df: pd.DataFrame, rate: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Drop rate * n_rows non-consecutive rows from the panel.
    The remaining rows still have a strictly-monotone Date axis,
    which means V1 cannot detect the gap. The ground-truth flag
    vector is recorded against the *corrupted* (post-drop) panel —
    by construction it is all-False because the corrupted rows
    aren't there to flag. The dropped dates are recorded in
    metadata as the canonical 'evidence' for a downstream auditor
    that compared the panel against an external calendar.
    """
    n = len(panel_df)
    n_drop_target = max(1, int(np.floor(rate * n)))
    # Sample non-adjacent indices: pick every-other index from a
    # randomly chosen offset, then trim to n_drop_target.
    candidate_pool = list(range(n))
    rng.shuffle(candidate_pool)

    dropped: list[int] = []
    used = set()
    for idx in candidate_pool:
        if idx in used or (idx - 1) in used or (idx + 1) in used:
            continue
        dropped.append(idx)
        used.add(idx)
        if len(dropped) >= n_drop_target:
            break

    keep = np.ones(n, dtype=bool)
    keep[dropped] = False
    corrupted = panel_df.iloc[keep].reset_index(drop=True)
    ground_truth = np.zeros(len(corrupted), dtype=bool)

    metadata = {
        "n_dropped": int(len(dropped)),
        "removed_dates": [
            panel_df.iloc[i]["Date"].strftime("%Y-%m-%d") for i in sorted(dropped)
        ][:50],  # first 50 for the provenance file
        "n_removed_dates_total": int(len(dropped)),
    }
    return corrupted, ground_truth, metadata


# =====================================================================
# Public dispatcher
# =====================================================================

def corrupt_panel(
    panel_df: pd.DataFrame,
    corruption_id: str,
    rate: float,
    seed: int,
    *,
    manifest_entry: dict | None = None,
    value_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Apply the named corruption to a copy of `panel_df`.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Must include a 'Date' column. NOT mutated.
    corruption_id : str
        One of CORRUPTION_IDS.
    rate : float in (0, 1]
        Approximate fraction of rows affected.
    seed : int
        Combined with corruption_id and rate to seed the RNG.
    manifest_entry : dict or None
        Used by C2 to exclude documented gap dates from sampling.
    value_columns : Iterable[str] or None
        Columns the corruption operates on. Defaults to
        SPX_EXTENDED_VALUE_COLUMNS for spx_extended_2011.

    Returns
    -------
    corrupted_panel : pd.DataFrame
    ground_truth_flags : np.ndarray of bool, aligned to
        corrupted_panel rows
    metadata : dict — mechanism-specific provenance, including the
        corruption_id, rate, seed, and gap_mechanism for the
        validator_coverage.csv annotation.
    """
    if corruption_id not in CORRUPTION_IDS:
        raise ValueError(
            f"Unknown corruption_id={corruption_id!r}; "
            f"expected one of {CORRUPTION_IDS}"
        )
    if not (0.0 < rate <= 1.0):
        raise ValueError(f"rate must be in (0, 1]; got {rate}")
    if "Date" not in panel_df.columns:
        raise ValueError("panel_df must have a 'Date' column")

    rng = _rng(corruption_id, rate, seed)
    cols = (
        list(value_columns) if value_columns is not None
        else list(SPX_EXTENDED_VALUE_COLUMNS)
    )

    if corruption_id == "duplicate_dates":
        corrupted, gt, meta = _corrupt_duplicate_dates(panel_df, rate, rng)
    elif corruption_id == "missing_values":
        corrupted, gt, meta = _corrupt_missing_values(
            panel_df, rate, rng, cols, manifest_entry
        )
    elif corruption_id == "stale_quotes":
        corrupted, gt, meta = _corrupt_stale_quotes(panel_df, rate, rng, cols)
    elif corruption_id == "extreme_jumps":
        corrupted, gt, meta = _corrupt_extreme_jumps(panel_df, rate, rng, cols)
    elif corruption_id == "calendar_gaps":
        corrupted, gt, meta = _corrupt_calendar_gaps(panel_df, rate, rng)
    else:
        raise AssertionError("unreachable")  # guarded above

    meta = {
        "corruption_id": corruption_id,
        "rate": float(rate),
        "seed": int(seed),
        "gap_mechanism": GAP_MECHANISM[corruption_id],
        "n_rows_input": int(len(panel_df)),
        "n_rows_corrupted_panel": int(len(corrupted)),
        "n_ground_truth_flags": int(gt.sum()),
        **meta,
    }
    return corrupted, gt, meta
