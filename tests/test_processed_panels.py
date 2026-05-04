"""Smoke test for the processed panels.

Runs structural and integrity checks against every panel listed in
data/manifests/processed_manifest.json. Tests are parameterized over the
panel names so a new panel automatically gets the full suite without
test edits.

Run from the repo root:

    pytest tests/test_processed_panels.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "processed_manifest.json"


def _load_manifest() -> dict:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _panel_names() -> list[str]:
    """Read panel names at collection time so pytest parameterizes correctly."""
    return sorted(_load_manifest().keys())


PANEL_NAMES = _panel_names()


@pytest.fixture(scope="module")
def manifest() -> dict:
    return _load_manifest()


@pytest.fixture(scope="module")
def loaded() -> dict[str, pd.DataFrame]:
    """Load every panel CSV once. Returns {panel_name: DataFrame}."""
    m = _load_manifest()
    out: dict[str, pd.DataFrame] = {}
    for name, entry in m.items():
        path = REPO_ROOT / entry["file_path"]
        out[name] = pd.read_csv(path, parse_dates=["Date"])
    return out


# ------------------------------------------------------------------
# Structural checks
# ------------------------------------------------------------------

@pytest.mark.parametrize("name", PANEL_NAMES)
def test_row_count_matches_manifest(name, manifest, loaded):
    actual = len(loaded[name])
    expected = manifest[name]["row_count"]
    assert actual == expected, (
        f"{name}: actual rows {actual} != manifest {expected}"
    )


@pytest.mark.parametrize("name", PANEL_NAMES)
def test_dates_strictly_monotonic_increasing(name, loaded):
    df = loaded[name]
    diffs = df["Date"].diff().dropna()
    assert (diffs > pd.Timedelta(0)).all(), (
        f"{name}: Date column is not strictly increasing"
    )


@pytest.mark.parametrize("name", PANEL_NAMES)
def test_no_duplicate_dates(name, loaded):
    df = loaded[name]
    n_dup = int(df["Date"].duplicated().sum())
    assert n_dup == 0, f"{name}: {n_dup} duplicate Date values"


@pytest.mark.parametrize("name", PANEL_NAMES)
def test_first_last_date_match_manifest(name, manifest, loaded):
    df = loaded[name]
    actual_first = df["Date"].iloc[0].strftime("%Y-%m-%d")
    actual_last = df["Date"].iloc[-1].strftime("%Y-%m-%d")
    assert actual_first == manifest[name]["first_date"], (
        f"{name}: first_date {actual_first} != manifest {manifest[name]['first_date']}"
    )
    assert actual_last == manifest[name]["last_date"], (
        f"{name}: last_date {actual_last} != manifest {manifest[name]['last_date']}"
    )


@pytest.mark.parametrize("name", PANEL_NAMES)
def test_columns_match_manifest(name, manifest, loaded):
    df = loaded[name]
    expected = [c["name"] for c in manifest[name]["columns"]]
    actual = list(df.columns)
    assert actual == expected, (
        f"{name}: columns {actual} != manifest {expected}"
    )


@pytest.mark.parametrize("name", PANEL_NAMES)
def test_dtypes_match_manifest(name, manifest, loaded):
    df = loaded[name]
    for col in manifest[name]["columns"]:
        if col["name"] == "Date":
            # parse_dates promotes Date to datetime64; manifest dtype is "date"
            assert "datetime" in str(df["Date"].dtype), (
                f"{name}.Date dtype is {df['Date'].dtype}, expected datetime"
            )
            continue
        expected = col["dtype"]
        actual = str(df[col["name"]].dtype)
        assert actual == expected, (
            f"{name}.{col['name']} dtype is {actual}, expected {expected}"
        )


@pytest.mark.parametrize("name", PANEL_NAMES)
def test_non_null_counts_match_manifest(name, manifest, loaded):
    df = loaded[name]
    for col in manifest[name]["columns"]:
        if col["name"] == "Date":
            continue
        actual = int(df[col["name"]].notna().sum())
        expected = col["non_null_count"]
        assert actual == expected, (
            f"{name}.{col['name']}: actual non_null={actual} != manifest {expected}"
        )


# ------------------------------------------------------------------
# Truncation regions
# ------------------------------------------------------------------

@pytest.mark.parametrize("name", PANEL_NAMES)
def test_truncation_regions_are_null(name, manifest, loaded):
    """For columns with truncated_before, every row with Date < threshold
    must be NaN in that column."""
    df = loaded[name]
    for col in manifest[name]["columns"]:
        if col["name"] == "Date":
            continue
        tb = col.get("truncated_before")
        if tb is None:
            continue
        threshold = pd.Timestamp(tb)
        pre = df[df["Date"] < threshold]
        if len(pre) == 0:
            continue
        n_non_null = int(pre[col["name"]].notna().sum())
        assert n_non_null == 0, (
            f"{name}.{col['name']}: {n_non_null} non-null values before "
            f"truncation threshold {tb}"
        )


@pytest.mark.parametrize("name", PANEL_NAMES)
def test_first_valid_date_matches_manifest(name, manifest, loaded):
    """For each column the manifest's first_valid_date matches the on-disk
    first non-null date. This is the positive complement to
    test_truncation_regions_are_null."""
    df = loaded[name]
    for col in manifest[name]["columns"]:
        if col["name"] == "Date":
            continue
        expected = col.get("first_valid_date")
        mask = df[col["name"]].notna()
        if expected is None:
            assert int(mask.sum()) == 0, (
                f"{name}.{col['name']}: manifest says no valid data, but found {int(mask.sum())}"
            )
            continue
        actual = df.loc[mask, "Date"].min().strftime("%Y-%m-%d")
        assert actual == expected, (
            f"{name}.{col['name']}: first_valid_date {actual} != manifest {expected}"
        )


# ------------------------------------------------------------------
# Data-quality notes (e.g., the 5 RVX gaps in rut_2009)
# ------------------------------------------------------------------

@pytest.mark.parametrize("name", PANEL_NAMES)
def test_data_quality_missing_dates_match(name, manifest, loaded):
    """If a panel records data_quality_notes with missing_dates for a
    series, the on-disk NaN set for that series must be exactly:
        (pre-truncation rows, if any) union (documented missing_dates).
    """
    df = loaded[name]
    notes = manifest[name].get("data_quality_notes", [])
    if not notes:
        pytest.skip(f"{name}: no data_quality_notes")
    cols_meta = {c["name"]: c for c in manifest[name]["columns"]}
    for note in notes:
        if "missing_dates" not in note or "series" not in note:
            continue
        col_name = note["series"]
        if col_name not in df.columns:
            pytest.fail(f"{name}: data_quality_notes references unknown column {col_name}")
        actual_missing = set(
            df.loc[df[col_name].isna(), "Date"].dt.strftime("%Y-%m-%d")
        )
        documented_missing = set(note["missing_dates"])
        tb = cols_meta[col_name].get("truncated_before")
        if tb is None:
            expected = documented_missing
        else:
            pre_mask = df["Date"] < pd.Timestamp(tb)
            pre_dates = set(df.loc[pre_mask, "Date"].dt.strftime("%Y-%m-%d"))
            expected = pre_dates | documented_missing
        assert actual_missing == expected, (
            f"{name}.{col_name}: NaN set differs from expected.\n"
            f"  unexpected (in actual not expected): {sorted(actual_missing - expected)}\n"
            f"  missing  (in expected not actual): {sorted(expected - actual_missing)}"
        )


# ------------------------------------------------------------------
# SHA-256 sanity (panel file unchanged since manifest write)
# ------------------------------------------------------------------

@pytest.mark.parametrize("name", PANEL_NAMES)
def test_panel_sha256_matches_manifest(name, manifest):
    """If the panel CSV has been edited since the manifest was written,
    the sha256 will not match. This catches accidental in-place edits."""
    import hashlib
    entry = manifest[name]
    path = REPO_ROOT / entry["file_path"]
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    expected = entry["sha256"]
    assert actual == expected, (
        f"{name}: sha256 {actual} != manifest {expected} "
        f"(panel CSV was modified after manifest write)"
    )
