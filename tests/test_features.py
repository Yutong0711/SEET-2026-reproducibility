"""Tests for feature engineering: structural sanity, warmup, no-look-ahead,
data-quality NaN propagation.

Run from the repo root:

    pytest tests/test_features.py -v
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from seet.features import build_features  # noqa: E402


PROCESSED_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "processed_manifest.json"
FEATURES_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "features_manifest.json"


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


PANEL_NAMES = sorted(_load_manifest(FEATURES_MANIFEST_PATH).keys())


@pytest.fixture(scope="module")
def processed_manifest() -> dict:
    return _load_manifest(PROCESSED_MANIFEST_PATH)


@pytest.fixture(scope="module")
def features_manifest() -> dict:
    return _load_manifest(FEATURES_MANIFEST_PATH)


@pytest.fixture(scope="module")
def panels(processed_manifest) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name in PANEL_NAMES:
        path = REPO_ROOT / processed_manifest[name]["file_path"]
        out[name] = pd.read_csv(path, parse_dates=["Date"])
    return out


@pytest.fixture(scope="module")
def features(features_manifest) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name in PANEL_NAMES:
        path = REPO_ROOT / features_manifest[name]["file_path"]
        out[name] = pd.read_csv(path, parse_dates=["Date"])
    return out


# ----------------------------------------------------------------------
# Structural sanity
# ----------------------------------------------------------------------

@pytest.mark.parametrize("name", PANEL_NAMES)
def test_features_row_count_matches_panel(name, processed_manifest, features):
    expected = processed_manifest[name]["row_count"]
    actual = len(features[name])
    assert actual == expected, (
        f"{name}: features rows {actual} != panel rows {expected}"
    )


@pytest.mark.parametrize("name", PANEL_NAMES)
def test_features_dates_match_panel(name, panels, features):
    p = panels[name]["Date"].dt.strftime("%Y-%m-%d").tolist()
    f = features[name]["Date"].dt.strftime("%Y-%m-%d").tolist()
    assert p == f, f"{name}: feature Date column does not match panel Date column"


@pytest.mark.parametrize("name", PANEL_NAMES)
def test_features_columns_match_manifest(name, features_manifest, features):
    expected = ["Date"] + [s["name"] for s in features_manifest[name]["features"]]
    actual = list(features[name].columns)
    assert actual == expected, (
        f"{name}: columns {actual} != manifest {expected}"
    )


@pytest.mark.parametrize("name", PANEL_NAMES)
def test_features_dtypes_are_float64(name, features):
    df = features[name]
    for col in df.columns:
        if col == "Date":
            continue
        assert df[col].dtype == "float64", (
            f"{name}.{col} dtype is {df[col].dtype}, expected float64"
        )


@pytest.mark.parametrize("name", PANEL_NAMES)
def test_features_sha256_matches_manifest(name, features_manifest):
    entry = features_manifest[name]
    path = REPO_ROOT / entry["file_path"]
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == entry["sha256"], (
        f"{name}: feature CSV sha256 != manifest"
    )


@pytest.mark.parametrize("name", PANEL_NAMES)
def test_source_panel_sha256_matches(name, processed_manifest, features_manifest):
    """The features manifest pins the source-panel sha; if that drifts, we'd
    compute features over a stale panel."""
    expected = processed_manifest[name]["sha256"]
    actual = features_manifest[name]["source_panel_sha256"]
    assert actual == expected, (
        f"{name}: source_panel_sha256 in features_manifest is stale"
    )


# ----------------------------------------------------------------------
# Warmup
# ----------------------------------------------------------------------

@pytest.mark.parametrize("name", PANEL_NAMES)
def test_warmup_region_is_nan(name, features_manifest, features):
    """For each feature, every row in [0, lookback_window-1) must be NaN."""
    fdf = features[name]
    for spec in features_manifest[name]["features"]:
        col = spec["name"]
        warmup = spec["lookback_window"] - 1
        if warmup <= 0:
            continue
        head = fdf[col].iloc[:warmup]
        assert head.isna().all(), (
            f"{name}.{col}: lookback_window={spec['lookback_window']} implies "
            f"first {warmup} rows must be NaN; "
            f"found {int(head.notna().sum())} non-NaN"
        )


# ----------------------------------------------------------------------
# Data-quality NaN propagation
# ----------------------------------------------------------------------

@pytest.mark.parametrize("name", PANEL_NAMES)
def test_data_quality_nan_propagates_to_features(
    name, processed_manifest, features_manifest, features
):
    """At every documented data_quality_notes missing date for a series,
    every feature whose depends_on includes that series must be NaN.

    This is the contract: panel-level NaN at the gap date propagates to
    every feature that touches that column at that exact row, regardless
    of lookback window. (Features may also be NaN at *trailing* rows due
    to rolling-window propagation — that's not tested here, only the
    point-of-gap propagation is asserted.)
    """
    notes = processed_manifest[name].get("data_quality_notes", [])
    if not notes:
        pytest.skip(f"{name}: no data_quality_notes")

    fdf = features[name]
    specs = features_manifest[name]["features"]

    for note in notes:
        if "missing_dates" not in note or "series" not in note:
            continue
        col = note["series"]
        missing = pd.to_datetime(note["missing_dates"])
        affected = [s for s in specs if col in s["depends_on"]]
        if not affected:
            continue
        for d in missing:
            row = fdf[fdf["Date"] == d]
            if row.empty:
                continue
            for s in affected:
                v = row[s["name"]].iloc[0]
                assert pd.isna(v), (
                    f"{name}.{s['name']} at {d.strftime('%Y-%m-%d')}: "
                    f"expected NaN (depends on {col} which is documented "
                    f"NaN here), got {v!r}"
                )


# ----------------------------------------------------------------------
# No-look-ahead regression
# ----------------------------------------------------------------------

def _values_equivalent(a: float, b: float, atol: float = 1e-12) -> bool:
    """Treat NaN==NaN as equal; otherwise require abs diff < atol."""
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(a - b) <= atol


@pytest.mark.parametrize("name", PANEL_NAMES)
def test_no_lookahead(name, processed_manifest, features_manifest, panels, features):
    """For 50 random dates, recompute every feature using only data with
    Date <= t (slice the panel to rows [0..t]) and assert the last row of
    the recomputed table equals the materialized features at row t.

    A failure here means a feature was using data from the future (a leak),
    or there is a non-determinism in the build pipeline.
    """
    pdf = panels[name]
    fdf = features[name]
    fset = features_manifest[name]["feature_set_id"]
    specs = features_manifest[name]["features"]

    n = len(pdf)
    # Skip rows in the warmup zone — they're trivially NaN. We sample
    # from rows past the largest lookback.
    max_lookback = max(s["lookback_window"] for s in specs)
    if n <= max_lookback:
        pytest.skip(f"{name}: panel too short for sampling beyond warmup")

    rng = np.random.default_rng(seed=42)
    sample_size = min(50, n - max_lookback)
    sample_indices = sorted(
        rng.choice(np.arange(max_lookback, n), size=sample_size, replace=False).tolist()
    )

    for t in sample_indices:
        truncated_panel = pdf.iloc[: t + 1].reset_index(drop=True)
        truncated_features, _ = build_features(truncated_panel, fset)
        recomputed_last = truncated_features.iloc[-1]
        materialized = fdf.iloc[t]
        for s in specs:
            col = s["name"]
            v_re = float(recomputed_last[col]) if pd.notna(recomputed_last[col]) else np.nan
            v_ma = float(materialized[col]) if pd.notna(materialized[col]) else np.nan
            assert _values_equivalent(v_re, v_ma), (
                f"{name}.{col} at index t={t} "
                f"(date={fdf['Date'].iloc[t].strftime('%Y-%m-%d')}): "
                f"truncated-recompute={v_re}, materialized={v_ma} "
                f"— possible look-ahead leak."
            )


# ----------------------------------------------------------------------
# Manifest schema sanity
# ----------------------------------------------------------------------

@pytest.mark.parametrize("name", PANEL_NAMES)
def test_manifest_feature_specs_are_well_formed(name, features_manifest):
    """Every feature spec must carry the contract fields with the right
    types so downstream consumers (audit, model code) can rely on them."""
    REQUIRED_KEYS = {
        "name", "feature_group", "formula", "depends_on",
        "lookback_window", "schema_version",
    }
    for s in features_manifest[name]["features"]:
        missing = REQUIRED_KEYS - set(s.keys())
        assert not missing, f"{name}.{s.get('name', '?')}: missing keys {missing}"
        assert isinstance(s["name"], str)
        assert s["feature_group"] in (1, 2, 3, 4)
        assert isinstance(s["formula"], str) and s["formula"]
        assert isinstance(s["depends_on"], list) and all(
            isinstance(c, str) for c in s["depends_on"]
        )
        assert isinstance(s["lookback_window"], int) and s["lookback_window"] >= 1
        assert s["schema_version"] == 1
