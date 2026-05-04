#!/usr/bin/env python3
"""Diagnostic auditor for the processed panels.

Loads each panel listed in data/manifests/processed_manifest.json and
reports:

  * whether the on-disk row count matches the manifest,
  * whether the Date column is strictly monotonically increasing,
  * whether Date values are unique,
  * for each numeric column, the contiguous NaN regions and whether
    each region is "expected" (matches a documented truncation) or
    "unexpected" (calls for review).

A NaN region is "expected" iff EITHER:

  (a) the column has a truncation_before threshold, the region begins at
      index 0, and the entire region is strictly before that threshold
      (i.e., it is the leading truncation block); OR
  (b) every date in the region is listed in a panel-level
      data_quality_notes entry whose `series` matches the column.

Anything else — a NaN block after the threshold, an isolated NaN that
is not documented in data_quality_notes, or any NaN at all in a column
with no truncation and no data_quality_notes — is "unexpected" and is
printed with its date range and length so it can be inspected.

Exit code:
    0 if all panels are clean (only expected NaN regions),
    1 if any panel has unexpected NaN regions or structural problems.

Run:
    python scripts/audit_panels.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as e:
    sys.stderr.write(f"Missing dependency: {e}. Activate the .venv first.\n")
    sys.exit(2)


def nan_regions(values: pd.Series, dates: pd.Series) -> list[dict]:
    """Return a list of contiguous-NaN regions for `values`, each as a
    dict with start_idx, end_idx, start_date, end_date, length.
    Indices are inclusive."""
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


def classify_region(
    region: dict,
    truncated_before: str | None,
    documented_missing: set[str],
) -> tuple[str, str]:
    """Return (status, reason) for a NaN region.

    status: 'expected' | 'unexpected'
    """
    # First check the truncation rule.
    if truncated_before is not None:
        if region["start_idx"] == 0 and region["end_date"] < truncated_before:
            return (
                "expected",
                f"leading truncation block, all dates < {truncated_before}",
            )
    # Then check the data_quality_notes rule.
    region_dates = _expand_dates(region)
    if region_dates and region_dates.issubset(documented_missing):
        return (
            "expected",
            f"matches data_quality_notes ({len(region_dates)} dated NaN row(s))",
        )
    # Build the most informative "unexpected" reason.
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


def _expand_dates(region: dict) -> set[str]:
    """Region carries start_date and end_date but not the full date list;
    we approximate the membership set as the start/end pair when the
    region length is 1 (which is the only case we need precise membership
    for, since multi-day clustered NaN regions in our panels would still
    be caught as 'unexpected' by the documented-set check)."""
    if region["length"] == 1:
        return {region["start_date"]}
    # For multi-day regions we don't expand to every date; instead, return
    # an empty set so the issubset check in classify_region fails, marking
    # the region as unexpected. The audit's printed summary will still show
    # the start/end so the user can inspect.
    return set()


def audit_panel(name: str, entry: dict, repo_root: Path) -> dict:
    """Run all checks on one panel. Returns a result dict."""
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

    # Strictly monotonic, unique dates
    if not df["Date"].is_monotonic_increasing:
        issues.append("Date is not monotonically increasing")
    elif (df["Date"].diff().dropna() <= pd.Timedelta(0)).any():
        issues.append("Date is not strictly increasing (has duplicates or backsteps)")
    if df["Date"].duplicated().any():
        n_dup = int(df["Date"].duplicated().sum())
        issues.append(f"Date has {n_dup} duplicate values")

    # Build per-series documented-missing sets from data_quality_notes
    documented_missing_by_series: dict[str, set[str]] = {}
    for note in entry.get("data_quality_notes", []) or []:
        series = note.get("series")
        if not series:
            continue
        documented_missing_by_series.setdefault(series, set()).update(
            note.get("missing_dates", [])
        )

    # NaN regions per numeric column
    iso_dates = df["Date"].dt.strftime("%Y-%m-%d")
    unexpected: list[dict] = []
    for col_name, col_meta in columns_meta.items():
        if col_name == "Date":
            continue
        # dtype check
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


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    manifest_path = repo_root / "data" / "manifests" / "processed_manifest.json"
    if not manifest_path.exists():
        sys.stderr.write(
            f"Manifest not found: {manifest_path}. Run build_processed.py first.\n"
        )
        return 2

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    print()
    print("===== PANEL AUDIT =====")
    any_problem = False
    for name in sorted(manifest.keys()):
        result = audit_panel(name, manifest[name], repo_root)
        any_problem = any_problem or not result["ok"]
        flag = "OK" if result["ok"] else "REVIEW"
        print(
            f"\n[{flag}] {name}  "
            f"rows={result.get('actual_rows', '?')}/{result.get('manifested_rows', '?')}"
        )
        for iss in result["issues"]:
            print(f"   structural: {iss}")
        if result["unexpected_regions"]:
            print(f"   unexpected NaN regions ({len(result['unexpected_regions'])}):")
            for r in result["unexpected_regions"]:
                if r["length"] == 1:
                    print(
                        f"     - {r['column']}: {r['start_date']} (1 row); {r['reason']}"
                    )
                else:
                    print(
                        f"     - {r['column']}: {r['start_date']} → {r['end_date']} "
                        f"({r['length']} rows); {r['reason']}"
                    )
        elif not result["issues"]:
            print("   (no unexpected NaN regions)")
    print()
    print("=======================")
    return 1 if any_problem else 0


if __name__ == "__main__":
    sys.exit(main())
