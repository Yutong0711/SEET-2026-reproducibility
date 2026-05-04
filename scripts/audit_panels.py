#!/usr/bin/env python3
"""Diagnostic auditor for the processed panels and feature CSVs.

Two-section report:

1. PANEL AUDIT — every panel listed in
   data/manifests/processed_manifest.json. Reports row-count drift,
   monotonic/uniqueness violations, and NaN regions classified as either
   "expected" (matching a documented truncation or data_quality_notes
   entry) or "unexpected".

2. FEATURE AUDIT — every panel listed in
   data/manifests/features_manifest.json (skipped if the manifest is
   absent). For every feature, the auditor computes an expected NaN
   mask as

       warmup_mask  ∪  rolling-OR(input_nan_mask, lookback_window)

   over each input column listed in `depends_on`, and flags any actual
   NaN that is not in the expected set. This catches both look-ahead
   leaks (a feature that's mysteriously *valid* where it shouldn't be
   would not be caught here, but that's what the no-look-ahead test in
   tests/test_features.py guards against) and silent over-NaN regions
   (a feature that's NaN somewhere unexpected — a bug in the formula or
   in the panel data).

Exit code:
    0 if every panel and every feature panel is clean,
    1 if any unexpected NaN region or structural problem is found,
    2 if the processed manifest is missing entirely.

Run:
    python scripts/audit_panels.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    sys.stderr.write(f"Missing dependency: {e}. Activate the .venv first.\n")
    sys.exit(2)


# ======================================================================
# Section 1 helpers — panel audit (unchanged from prior turn)
# ======================================================================

def nan_regions(values: pd.Series, dates: pd.Series) -> list[dict]:
    """Return a list of contiguous-NaN regions for `values`, each as a
    dict with start_idx, end_idx, start_date, end_date, length."""
    is_nan = values.isna().values
    out: list[dict] = []
    i = 0
    n = len(is_nan)
    while i < n:
        if is_nan[i]:
            j = i
            while j < n and is_nan[j]:
                j += 1
            out.append(
                {
                    "start_idx": i,
                    "end_idx": j - 1,
                    "start_date": str(dates.iloc[i]),
                    "end_date": str(dates.iloc[j - 1]),
                    "length": j - i,
                }
            )
            i = j
        else:
            i += 1
    return out


def _expand_dates(region: dict) -> set[str]:
    """Return the date set of the region for length-1 regions; empty
    set otherwise (multi-day regions are conservatively unmatched
    against the documented_missing single-date set)."""
    if region["length"] == 1:
        return {region["start_date"]}
    return set()


def classify_region(
    region: dict,
    truncated_before: str | None,
    documented_missing: set[str],
) -> tuple[str, str]:
    if truncated_before is not None:
        if region["start_idx"] == 0 and region["end_date"] < truncated_before:
            return (
                "expected",
                f"leading truncation block, all dates < {truncated_before}",
            )
    region_dates = _expand_dates(region)
    if region_dates and region_dates.issubset(documented_missing):
        return (
            "expected",
            f"matches data_quality_notes ({len(region_dates)} dated NaN row(s))",
        )
    if truncated_before is None and not documented_missing:
        reason = "column has no documented truncation or data_quality_notes"
    elif truncated_before is not None and region["end_date"] >= truncated_before:
        reason = (
            f"region end {region['end_date']} >= truncation_before "
            f"{truncated_before}"
        )
    else:
        undocumented = region_dates - documented_missing
        if undocumented:
            reason = (
                f"NaN at undocumented date(s): "
                f"{sorted(undocumented)[:5]}"
                + ("..." if len(undocumented) > 5 else "")
            )
        else:
            reason = "region structure does not match any expected pattern"
    return ("unexpected", reason)


def audit_panel(name: str, entry: dict, repo_root: Path) -> dict:
    file_path = repo_root / entry["file_path"]
    manifested_rows = entry["row_count"]
    columns_meta = {c["name"]: c for c in entry["columns"]}
    issues: list[str] = []

    if not file_path.exists():
        return {
            "name": name,
            "ok": False,
            "issues": [f"file missing: {file_path}"],
            "unexpected_regions": [],
        }

    df = pd.read_csv(file_path, parse_dates=["Date"])
    actual_rows = len(df)
    if actual_rows != manifested_rows:
        issues.append(
            f"row_count mismatch: manifest={manifested_rows} actual={actual_rows}"
        )
    if not df["Date"].is_monotonic_increasing:
        issues.append("Date is not monotonically increasing")
    elif (df["Date"].diff().dropna() <= pd.Timedelta(0)).any():
        issues.append("Date is not strictly increasing (duplicates or backsteps)")
    if df["Date"].duplicated().any():
        n_dup = int(df["Date"].duplicated().sum())
        issues.append(f"Date has {n_dup} duplicate values")

    documented_missing_by_series: dict[str, set[str]] = {}
    for note in entry.get("data_quality_notes", []) or []:
        series = note.get("series")
        if not series:
            continue
        documented_missing_by_series.setdefault(series, set()).update(
            note.get("missing_dates", [])
        )

    iso_dates = df["Date"].dt.strftime("%Y-%m-%d")
    unexpected: list[dict] = []
    for col_name, col_meta in columns_meta.items():
        if col_name == "Date":
            continue
        expected_dtype = col_meta.get("dtype", "float64")
        if expected_dtype == "float64" and df[col_name].dtype != "float64":
            issues.append(
                f"{col_name} dtype is {df[col_name].dtype}, expected float64"
            )
        regions = nan_regions(df[col_name], iso_dates)
        truncated_before = col_meta.get("truncated_before")
        documented = documented_missing_by_series.get(col_name, set())
        for r in regions:
            status, reason = classify_region(r, truncated_before, documented)
            if status == "unexpected":
                unexpected.append(
                    {
                        "panel": name,
                        "column": col_name,
                        "start_date": r["start_date"],
                        "end_date": r["end_date"],
                        "length": r["length"],
                        "reason": reason,
                    }
                )

    ok = len(issues) == 0 and len(unexpected) == 0
    return {
        "name": name,
        "ok": ok,
        "manifested_rows": manifested_rows,
        "actual_rows": actual_rows,
        "issues": issues,
        "unexpected_regions": unexpected,
    }


# ======================================================================
# Section 2 helpers — feature audit
# ======================================================================

def _expected_feature_nan_mask(
    panel_df: pd.DataFrame, depends_on: list[str], lookback: int
) -> np.ndarray:
    """Conservative expected-NaN mask for a feature.

    Expected NaN at row t iff:
      - t is in the warmup region (t < lookback - 1), OR
      - any input column in depends_on has a NaN anywhere in the
        trailing-`lookback` window ending at t.

    The second clause is the rolling-OR of the input NaN masks over a
    window of size `lookback` with min_periods=1 (so partial windows at
    the start are also evaluated).

    This is conservative: pandas may produce *fewer* NaNs than this if
    a particular operation chooses to skip NaN within the window. Any
    actual NaN that falls outside this mask is genuinely unexpected.
    """
    n = len(panel_df)
    expected = np.zeros(n, dtype=bool)
    warmup = max(0, int(lookback) - 1)
    if warmup > 0:
        expected[:warmup] = True
    for dep in depends_on:
        if dep not in panel_df.columns:
            continue
        nan_int = panel_df[dep].isna().astype(int)
        propagated = (
            nan_int.rolling(int(lookback), min_periods=1).sum() > 0
        ).to_numpy()
        expected = expected | propagated
    return expected


def audit_feature_panel(
    name: str,
    feat_entry: dict,
    panel_df: pd.DataFrame,
    repo_root: Path,
) -> dict:
    """Audit one feature panel against its source panel."""
    file_path = repo_root / feat_entry["file_path"]
    issues: list[str] = []
    unexpected_features: list[dict] = []

    if not file_path.exists():
        return {
            "name": name,
            "ok": False,
            "issues": [f"feature file missing: {file_path}"],
            "unexpected_features": [],
            "manifested_rows": feat_entry.get("row_count"),
            "actual_rows": None,
            "feature_count": feat_entry.get("feature_count", 0),
            "max_warmup": feat_entry.get("max_warmup_rows", 0),
        }

    feat_df = pd.read_csv(file_path, parse_dates=["Date"])
    actual_rows = len(feat_df)
    manifested_rows = feat_entry["row_count"]
    if actual_rows != manifested_rows:
        issues.append(
            f"row_count mismatch: manifest={manifested_rows} actual={actual_rows}"
        )

    # Date alignment with source panel
    if not panel_df["Date"].equals(feat_df["Date"]):
        issues.append("Date column does not align with source panel")

    iso_dates = feat_df["Date"].dt.strftime("%Y-%m-%d")
    specs = feat_entry.get("features", [])

    for spec in specs:
        col = spec["name"]
        lookback = int(spec["lookback_window"])
        deps = list(spec.get("depends_on", []))

        if col not in feat_df.columns:
            issues.append(f"feature column missing on disk: {col}")
            continue

        actual_nan = feat_df[col].isna().to_numpy()
        expected_nan = _expected_feature_nan_mask(panel_df, deps, lookback)

        # Unexpected: actual NaN at positions where we did not predict it
        unexpected_mask = actual_nan & ~expected_nan
        if unexpected_mask.any():
            idxs = np.where(unexpected_mask)[0]
            sample_dates = [iso_dates.iloc[int(i)] for i in idxs[:5]]
            unexpected_features.append(
                {
                    "feature": col,
                    "n_unexpected": int(idxs.size),
                    "first_unexpected": iso_dates.iloc[int(idxs[0])],
                    "last_unexpected": iso_dates.iloc[int(idxs[-1])],
                    "sample_dates": sample_dates,
                }
            )

    ok = len(issues) == 0 and len(unexpected_features) == 0
    return {
        "name": name,
        "ok": ok,
        "manifested_rows": manifested_rows,
        "actual_rows": actual_rows,
        "issues": issues,
        "unexpected_features": unexpected_features,
        "feature_count": feat_entry.get("feature_count", len(specs)),
        "max_warmup": feat_entry.get("max_warmup_rows", 0),
    }


# ======================================================================
# Main
# ======================================================================

def _print_panel_section(manifest: dict, repo_root: Path) -> bool:
    """Print the panel-audit section. Return True iff any panel had a
    problem."""
    print()
    print("===== PANEL AUDIT =====")
    any_problem = False
    for name in sorted(manifest.keys()):
        result = audit_panel(name, manifest[name], repo_root)
        any_problem = any_problem or not result["ok"]
        flag = "OK" if result["ok"] else "REVIEW"
        print(
            f"\n[{flag}] {name}  "
            f"rows={result.get('actual_rows', '?')}/"
            f"{result.get('manifested_rows', '?')}"
        )
        for iss in result["issues"]:
            print(f"   structural: {iss}")
        if result["unexpected_regions"]:
            print(
                f"   unexpected NaN regions ({len(result['unexpected_regions'])}):"
            )
            for r in result["unexpected_regions"]:
                if r["length"] == 1:
                    print(
                        f"     - {r['column']}: {r['start_date']} "
                        f"(1 row); {r['reason']}"
                    )
                else:
                    print(
                        f"     - {r['column']}: {r['start_date']} → "
                        f"{r['end_date']} ({r['length']} rows); {r['reason']}"
                    )
        elif not result["issues"]:
            print("   (no unexpected NaN regions)")
    print()
    print("=======================")
    return any_problem


def _print_feature_section(
    feat_manifest: dict, processed_manifest: dict, repo_root: Path
) -> bool:
    """Print the feature-audit section. Return True iff any panel had a
    problem."""
    print()
    print("===== FEATURE AUDIT =====")
    any_problem = False
    panels_cache: dict[str, pd.DataFrame] = {}
    for name in sorted(feat_manifest.keys()):
        if name not in processed_manifest:
            print(f"\n[REVIEW] {name}: source panel not in processed_manifest")
            any_problem = True
            continue
        if name not in panels_cache:
            panel_path = repo_root / processed_manifest[name]["file_path"]
            panels_cache[name] = pd.read_csv(panel_path, parse_dates=["Date"])
        result = audit_feature_panel(
            name, feat_manifest[name], panels_cache[name], repo_root
        )
        any_problem = any_problem or not result["ok"]
        flag = "OK" if result["ok"] else "REVIEW"
        print(
            f"\n[{flag}] {name}  "
            f"rows={result.get('actual_rows', '?')}/"
            f"{result.get('manifested_rows', '?')}  "
            f"features={result.get('feature_count', '?')}  "
            f"max_warmup={result.get('max_warmup', '?')}"
        )
        for iss in result["issues"]:
            print(f"   structural: {iss}")
        if result["unexpected_features"]:
            print(
                f"   unexpected feature NaNs "
                f"({len(result['unexpected_features'])} feature(s)):"
            )
            for u in result["unexpected_features"]:
                print(
                    f"     - {u['feature']}: {u['n_unexpected']} unexpected "
                    f"NaN row(s); first {u['first_unexpected']}, "
                    f"last {u['last_unexpected']}; "
                    f"sample={u['sample_dates']}"
                )
        elif not result["issues"]:
            print("   (no unexpected feature NaNs)")
    print()
    print("=========================")
    return any_problem


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent

    pm_path = repo_root / "data" / "manifests" / "processed_manifest.json"
    if not pm_path.exists():
        sys.stderr.write(
            f"Manifest not found: {pm_path}. Run build_processed.py first.\n"
        )
        return 2
    with pm_path.open("r", encoding="utf-8") as f:
        processed_manifest = json.load(f)

    panel_problem = _print_panel_section(processed_manifest, repo_root)

    fm_path = repo_root / "data" / "manifests" / "features_manifest.json"
    if fm_path.exists():
        with fm_path.open("r", encoding="utf-8") as f:
            feat_manifest = json.load(f)
        feature_problem = _print_feature_section(
            feat_manifest, processed_manifest, repo_root
        )
    else:
        print()
        print("(features_manifest.json not present — feature audit skipped)")
        feature_problem = False

    return 1 if (panel_problem or feature_problem) else 0


if __name__ == "__main__":
    sys.exit(main())
