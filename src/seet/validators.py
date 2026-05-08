"""Runtime data validators for the SEET 2026 processed panels.

This module codifies the validation logic that was previously
implicit in build-time and audit-time scripts:

    V1 (strict-monotone Date / duplicate detection)
        Source of the rule: scripts/audit_panels.py lines 146-152.
        Catches: duplicate dates, date backsteps.

    V2 (NaN-region vs data_quality_notes registry)
        Source of the rule: scripts/audit_panels.py classify_region.
        Catches: any NaN in a column where the row's date >=
        the column's truncated_before AND the row's date is not
        in the column's data_quality_notes.

A "validator" here is a callable that takes a panel DataFrame +
the panel's processed_manifest entry and returns:

    flags : np.ndarray of bool, shape (n_rows,)
        True iff the row is flagged by *any* enabled rule.
    per_rule_flags : dict[str, np.ndarray of bool]
        One entry per rule name -> per-row flag vector.
    treatment_dispositions : dict
        Documents what the ENABLED branch should do with flagged
        rows. The standard treatment for both V1 and V2 is "drop"
        (V1 deduplicates keeping first; V2 drops the row).

The module is INTENTIONALLY a thin codification of existing
behavior. Its purpose for Track F is to give the failure-injection
harness a callable validator interface and to make it possible to
silence the validators precisely (validate_panel(..., enabled=False)
returns no flags). The CLAUDE.md Track F summary notes that creating
this module -- making the implicit explicit -- is itself an
architectural improvement surfaced by the mutation-testing
methodology.

Public API
----------
validate_panel(panel_df, manifest_entry, *, enabled=True) -> dict
    Returns a result dict with keys: flags, per_rule_flags,
    treatment_dispositions, n_rows, n_flagged, enabled.

apply_treatment(panel_df, validation_result, manifest_entry) -> pd.DataFrame
    Apply per-rule treatments: V1 deduplicates by Date keeping
    first; V2 drops rows whose unexpected NaN persists after
    deduplication. Returns a new DataFrame.

coverage_metrics(validator_flags, ground_truth_flags) -> dict
    Precision / recall / F1 with explicit edge-case handling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


VALIDATOR_RULES: tuple[str, ...] = ("V1_duplicate_date", "V2_unexpected_nan")


def _v1_duplicate_date(panel_df: pd.DataFrame) -> np.ndarray:
    """V1 detection: any row whose Date appears more than once.

    Returns a Boolean array of length n_rows, True iff that row's
    Date value occurs in another row. We flag *both* copies of a
    duplicate (rather than only the second), matching the audit
    script's diagnostic intent. The treatment in apply_treatment
    deduplicates by keeping the first occurrence.
    """
    return panel_df["Date"].duplicated(keep=False).to_numpy(dtype=bool)


def _data_quality_notes_dates(
    manifest_entry: dict,
) -> dict[str, set[pd.Timestamp]]:
    """Parse manifest_entry['data_quality_notes'] into
    {series: set(missing_dates)}. Returns {} if absent or empty."""
    out: dict[str, set[pd.Timestamp]] = {}
    notes = manifest_entry.get("data_quality_notes") if manifest_entry else None
    if not notes:
        return out
    for note in notes:
        series = note.get("series")
        if not series:
            continue
        dates = note.get("missing_dates") or note.get("dates") or []
        out.setdefault(series, set()).update(pd.Timestamp(d) for d in dates)
    return out


def _truncations(manifest_entry: dict) -> dict[str, pd.Timestamp]:
    """Extract per-column truncated_before timestamps from the
    manifest entry's columns block. Columns without a truncation
    rule do not appear."""
    out: dict[str, pd.Timestamp] = {}
    if not manifest_entry:
        return out
    for col in manifest_entry.get("columns", []) or []:
        threshold = col.get("truncated_before")
        if threshold:
            out[col["name"]] = pd.Timestamp(threshold)
    return out


def _v2_unexpected_nan(
    panel_df: pd.DataFrame, manifest_entry: dict | None
) -> np.ndarray:
    """V2 detection: any row that has a NaN in a column where the
    NaN is *not* explained by either the column's truncation rule
    or the column's data_quality_notes registry.

    The classification is per-cell; a row is flagged iff at least
    one of its cells is unexpected-NaN.
    """
    n = len(panel_df)
    flags = np.zeros(n, dtype=bool)
    if manifest_entry is None:
        # No manifest -> no truncation/notes context. Conservatively
        # flag any NaN as unexpected.
        for col in panel_df.columns:
            if col == "Date":
                continue
            flags |= panel_df[col].isna().to_numpy(dtype=bool)
        return flags

    truncations = _truncations(manifest_entry)
    dq_notes = _data_quality_notes_dates(manifest_entry)
    dates = pd.to_datetime(panel_df["Date"]).to_numpy()

    for col in panel_df.columns:
        if col == "Date":
            continue
        col_isna = panel_df[col].isna().to_numpy(dtype=bool)
        if not col_isna.any():
            continue
        expected = np.zeros(n, dtype=bool)
        thr = truncations.get(col)
        if thr is not None:
            expected |= dates < np.datetime64(thr)
        notes_dates = dq_notes.get(col, set())
        if notes_dates:
            notes_arr = np.array(
                sorted(np.datetime64(d) for d in notes_dates),
                dtype="datetime64[ns]",
            )
            expected |= np.isin(dates, notes_arr)
        unexpected = col_isna & ~expected
        flags |= unexpected
    return flags


def validate_panel(
    panel_df: pd.DataFrame,
    manifest_entry: dict | None = None,
    *,
    enabled: bool = True,
) -> dict:
    """Run V1 + V2 on `panel_df` and return per-row flags.

    When enabled=False, returns an all-False flag vector regardless
    of the panel contents. This is the SILENCED branch for Track F.
    """
    n = len(panel_df)
    if not enabled:
        zero = np.zeros(n, dtype=bool)
        return {
            "flags": zero,
            "per_rule_flags": {r: zero.copy() for r in VALIDATOR_RULES},
            "treatment_dispositions": {r: "drop" for r in VALIDATOR_RULES},
            "n_rows": n,
            "n_flagged": 0,
            "enabled": False,
        }

    v1 = _v1_duplicate_date(panel_df)
    v2 = _v2_unexpected_nan(panel_df, manifest_entry)
    flags = v1 | v2
    return {
        "flags": flags,
        "per_rule_flags": {
            "V1_duplicate_date": v1,
            "V2_unexpected_nan": v2,
        },
        "treatment_dispositions": {
            "V1_duplicate_date": "drop",
            "V2_unexpected_nan": "drop",
        },
        "n_rows": int(n),
        "n_flagged": int(flags.sum()),
        "enabled": True,
    }


def apply_treatment(
    panel_df: pd.DataFrame,
    validation_result: dict | None = None,
    manifest_entry: dict | None = None,
    *,
    flags: np.ndarray | None = None,
) -> pd.DataFrame:
    """Apply the per-rule documented treatments to `panel_df`.

    The two-rule architecture currently in scope:

      V1 (duplicate_date) treatment: deduplicate by Date keeping
          the first occurrence. Flagging both copies is a
          diagnostic stance (the audit script's behavior); the
          operational treatment is to drop the redundancy, not
          both copies.

      V2 (unexpected_nan) treatment: drop rows whose unexpected
          NaN persists after deduplication.

    Two calling forms are supported:

      apply_treatment(panel_df, validation_result, manifest_entry)
        -- preferred. Applies V1 dedup + V2 row-drop in sequence.

      apply_treatment(panel_df, flags=<bool array>)
        -- legacy. Drops every flagged row outright.

    Returns a new DataFrame; `panel_df` is not mutated.
    """
    if flags is not None:
        if len(panel_df) != np.asarray(flags).size:
            raise ValueError(
                f"flags length {np.asarray(flags).size} != panel rows "
                f"{len(panel_df)}"
            )
        keep = ~np.asarray(flags, dtype=bool)
        return panel_df.iloc[keep].reset_index(drop=True)

    out = panel_df.copy()
    out = out.drop_duplicates(subset="Date", keep="first").reset_index(drop=True)
    if manifest_entry is not None:
        v2_after = _v2_unexpected_nan(out, manifest_entry)
        if v2_after.any():
            out = out.iloc[~v2_after].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------
# Coverage accounting (used by Track F)
# ---------------------------------------------------------------------

def coverage_metrics(
    validator_flags: np.ndarray, ground_truth_flags: np.ndarray
) -> dict:
    """Precision / recall / F1 of validator_flags vs ground_truth_flags.

    Edge cases:
      - all-False ground truth: precision = NaN if validator also
        all-False, else 0.0; recall = NaN (no positives); F1 = NaN.
      - all-False validator with positive ground truth: precision = NaN,
        recall = 0.0, F1 = 0.0.

    Returns a dict with keys: precision, recall, f1, n_truth,
    n_validator, n_true_positive, n_false_positive, n_false_negative.
    """
    g = np.asarray(ground_truth_flags, dtype=bool)
    v = np.asarray(validator_flags, dtype=bool)
    if g.shape != v.shape:
        raise ValueError(
            f"ground_truth and validator flag shapes differ: "
            f"{g.shape} vs {v.shape}"
        )
    tp = int((g & v).sum())
    fp = int((~g & v).sum())
    fn = int((g & ~v).sum())
    n_truth = int(g.sum())
    n_validator = int(v.sum())

    precision = float("nan")
    if n_validator > 0:
        precision = tp / n_validator
    recall = float("nan")
    if n_truth > 0:
        recall = tp / n_truth
    f1 = float("nan")
    if not (np.isnan(precision) or np.isnan(recall)) and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    elif n_truth == 0 and n_validator == 0:
        f1 = float("nan")
    elif n_truth > 0 and n_validator == 0:
        f1 = 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_truth": n_truth,
        "n_validator": n_validator,
        "n_true_positive": tp,
        "n_false_positive": fp,
        "n_false_negative": fn,
    }
