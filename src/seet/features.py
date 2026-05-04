"""Feature engineering on top of the SEET 2026 processed panels.

Public API
----------
build_features(panel_df, feature_set_id) -> (features_df, specs)
    Compute all features for a panel and return both the materialized
    DataFrame and the per-feature spec list (suitable for the manifest).

    panel_df : pd.DataFrame
        Must contain a 'Date' column plus every column referenced by the
        chosen feature set's depends_on lists.

    feature_set_id : str
        One of {"spx_full", "spx_core", "ndx_minimal", "rut_minimal"}.

    Returns
    -------
    features_df : pd.DataFrame with columns ['Date', <feature names...>],
        same row count as panel_df, same Date order.
    specs : list of dicts. Each dict has keys
        'name', 'feature_group', 'formula', 'depends_on',
        'lookback_window', 'schema_version'.

Design notes
------------
1. NO LOOK-AHEAD. Every rolling/shift uses pandas defaults, which are
   trailing-only. .rolling(N) at row t covers rows [t-N+1 .. t].
2. WARMUP IS NaN. For window N, the first N-1 rows are NaN. We never
   fill warmup. A feature's `lookback_window` is the number of trailing
   rows of input the feature needs (including the current row). So
   `lookback_window - 1` = warmup-row count.
3. NaN PROPAGATES. We use pandas defaults: any NaN in a rolling window
   propagates to the result. Direct `.shift(k)` and `.diff(k)` produce
   NaN at row t if either input is NaN. Group 3 indicators
   (curve_inversion, short_end_spike) are explicitly NaN if either
   operand is NaN, rather than coercing False.
4. ANNUALIZATION. Realized vol uses log returns squared via
   rolling().std(), then multiplied by sqrt(252). Pandas's rolling.std()
   uses sample std (ddof=1) which matches the canonical definition.

Feature sets
------------
spx_full       — every group (used for spx_extended_2011)
spx_core       — Groups 1, 2 (VIX, VIX3M only), partial 3 (no VIX9D /
                 VIX6M), 4 (used for spx_core_2007)
ndx_minimal    — Group 1 (NDX) + Group 2 (VXN)
rut_minimal    — Group 1 (RUT) + Group 2 (RVX)

Group 4 (vol-of-vol) is conceptually distinct from Group 2 (term/level)
but uses identical mechanics. We tag VVIX features as feature_group=4
in the manifest while still computing them with the Group 2 helpers.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


SQRT_252 = float(np.sqrt(252.0))
SCHEMA_VERSION = 1


def _spec(
    name: str,
    group: int,
    formula: str,
    depends_on: list[str],
    lookback_window: int,
    compute: Callable[[pd.DataFrame], pd.Series],
) -> dict:
    """Build one feature spec. The compute callable is private and is
    stripped before serialization."""
    return {
        "name": name,
        "feature_group": int(group),
        "formula": formula,
        "depends_on": list(depends_on),
        "lookback_window": int(lookback_window),
        "schema_version": SCHEMA_VERSION,
        "_compute": compute,
    }


# ----------------------------------------------------------------------
# Group 1 — returns and realized vol on a price column
# ----------------------------------------------------------------------

def _group1(price: str) -> list[dict]:
    p = price
    return [
        _spec("ret_1d",  1, f"{p}.pct_change(1)",  [p],  2,
              lambda df, p=p: df[p].pct_change(1)),
        _spec("ret_5d",  1, f"{p}.pct_change(5)",  [p],  6,
              lambda df, p=p: df[p].pct_change(5)),
        _spec("ret_10d", 1, f"{p}.pct_change(10)", [p], 11,
              lambda df, p=p: df[p].pct_change(10)),
        _spec("ret_20d", 1, f"{p}.pct_change(20)", [p], 21,
              lambda df, p=p: df[p].pct_change(20)),
        _spec("logret_1d", 1, f"log({p} / {p}.shift(1))", [p], 2,
              lambda df, p=p: np.log(df[p] / df[p].shift(1))),
        _spec("rv_10d", 1,
              f"log({p}/{p}.shift(1)).rolling(10).std() * sqrt(252)",
              [p], 11,
              lambda df, p=p:
                  np.log(df[p] / df[p].shift(1)).rolling(10).std() * SQRT_252),
        _spec("rv_20d", 1,
              f"log({p}/{p}.shift(1)).rolling(20).std() * sqrt(252)",
              [p], 21,
              lambda df, p=p:
                  np.log(df[p] / df[p].shift(1)).rolling(20).std() * SQRT_252),
        _spec("drawdown_20d", 1,
              f"{p} / {p}.rolling(20).max() - 1", [p], 20,
              lambda df, p=p: df[p] / df[p].rolling(20).max() - 1.0),
    ]


# ----------------------------------------------------------------------
# Group 2 — vol level transformations on a vol column
# Group 4 reuses the same mechanics but tags features feature_group=4.
# ----------------------------------------------------------------------

def _group2_or_4(vol: str, group_id: int) -> list[dict]:
    v = vol
    pre = v.lower()
    return [
        _spec(f"{pre}_chg_1d", group_id, f"{v}.diff(1)", [v], 2,
              lambda df, v=v: df[v].diff(1)),
        _spec(f"{pre}_chg_5d", group_id, f"{v}.diff(5)", [v], 6,
              lambda df, v=v: df[v].diff(5)),
        _spec(f"{pre}_pctchg_5d", group_id, f"{v}.pct_change(5)", [v], 6,
              lambda df, v=v: df[v].pct_change(5)),
        _spec(f"{pre}_pctile_252d", group_id,
              f"{v}.rolling(252).rank(pct=True)", [v], 252,
              lambda df, v=v: df[v].rolling(252).rank(pct=True)),
    ]


# ----------------------------------------------------------------------
# Group 3 — term structure (SPX panels only)
# ----------------------------------------------------------------------

def _binary_with_nan(a: pd.Series, b: pd.Series) -> pd.Series:
    """(a > b) as float (0.0/1.0); NaN when either operand is NaN.

    pandas's default `>` between a NaN and a number is False, which
    would silently coerce NaN → 0 here. We need to preserve NaN
    propagation per the project's data-quality rule.
    """
    out = (a > b).astype(float)
    out[a.isna() | b.isna()] = np.nan
    return out


def _group3(use_short_end: bool, use_six_month: bool) -> list[dict]:
    specs: list[dict] = [
        _spec("vix3m_minus_vix", 3, "VIX3M - VIX",
              ["VIX3M", "VIX"], 1,
              lambda df: df["VIX3M"] - df["VIX"]),
        _spec("ratio_vix_vix3m", 3, "VIX / VIX3M",
              ["VIX", "VIX3M"], 1,
              lambda df: df["VIX"] / df["VIX3M"]),
        _spec("curve_inversion", 3,
              "int(VIX > VIX3M); NaN if either NaN",
              ["VIX", "VIX3M"], 1,
              lambda df: _binary_with_nan(df["VIX"], df["VIX3M"])),
    ]
    if use_six_month:
        specs.append(_spec("vix6m_minus_vix", 3, "VIX6M - VIX",
                           ["VIX6M", "VIX"], 1,
                           lambda df: df["VIX6M"] - df["VIX"]))
    if use_short_end:
        specs += [
            _spec("vix9d_minus_vix", 3, "VIX9D - VIX",
                  ["VIX9D", "VIX"], 1,
                  lambda df: df["VIX9D"] - df["VIX"]),
            _spec("ratio_vix9d_vix", 3, "VIX9D / VIX",
                  ["VIX9D", "VIX"], 1,
                  lambda df: df["VIX9D"] / df["VIX"]),
            _spec("short_end_spike", 3,
                  "int(VIX9D > VIX); NaN if either NaN",
                  ["VIX9D", "VIX"], 1,
                  lambda df: _binary_with_nan(df["VIX9D"], df["VIX"])),
        ]
    return specs


# ----------------------------------------------------------------------
# Feature-set assembly
# ----------------------------------------------------------------------

def _assemble_spx_full() -> list[dict]:
    specs: list[dict] = []
    specs += _group1("SPX")
    for v in ("VIX", "VIX9D", "VIX3M", "VIX6M"):
        specs += _group2_or_4(v, group_id=2)
    specs += _group3(use_short_end=True, use_six_month=True)
    specs += _group2_or_4("VVIX", group_id=4)
    return specs


def _assemble_spx_core() -> list[dict]:
    specs: list[dict] = []
    specs += _group1("SPX")
    for v in ("VIX", "VIX3M"):
        specs += _group2_or_4(v, group_id=2)
    specs += _group3(use_short_end=False, use_six_month=False)
    specs += _group2_or_4("VVIX", group_id=4)
    return specs


def _assemble_ndx_minimal() -> list[dict]:
    return _group1("NDX") + _group2_or_4("VXN", group_id=2)


def _assemble_rut_minimal() -> list[dict]:
    return _group1("RUT") + _group2_or_4("RVX", group_id=2)


FEATURE_SETS: dict[str, Callable[[], list[dict]]] = {
    "spx_full":     _assemble_spx_full,
    "spx_core":     _assemble_spx_core,
    "ndx_minimal":  _assemble_ndx_minimal,
    "rut_minimal":  _assemble_rut_minimal,
}


def feature_specs(feature_set_id: str) -> list[dict]:
    """Return the spec list for a feature set without the private compute
    callables. Useful for tests and audit code that need to know the
    schema without computing."""
    if feature_set_id not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set_id: {feature_set_id!r}")
    return [
        {k: v for k, v in s.items() if not k.startswith("_")}
        for s in FEATURE_SETS[feature_set_id]()
    ]


def build_features(
    panel_df: pd.DataFrame, feature_set_id: str
) -> tuple[pd.DataFrame, list[dict]]:
    """Compute all features for `panel_df` under `feature_set_id`.

    Returns (features_df, specs) where features_df has columns
    ['Date', <each feature>] and specs is the manifest-ready list (no
    compute callables).
    """
    if feature_set_id not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set_id: {feature_set_id!r}")
    if "Date" not in panel_df.columns:
        raise ValueError("panel_df must have a 'Date' column")

    specs = FEATURE_SETS[feature_set_id]()

    needed = set()
    for s in specs:
        needed.update(s["depends_on"])
    missing = needed - set(panel_df.columns)
    if missing:
        raise ValueError(
            f"Panel is missing columns required by {feature_set_id!r}: "
            f"{sorted(missing)}"
        )

    out = pd.DataFrame({"Date": panel_df["Date"].values})
    for s in specs:
        series = s["_compute"](panel_df)
        # Defensive: ensure float64 dtype for all features (Group 3
        # indicators are float-with-NaN, not int).
        out[s["name"]] = pd.to_numeric(series, errors="coerce").astype("float64")

    clean_specs = [
        {k: v for k, v in s.items() if not k.startswith("_")}
        for s in specs
    ]
    return out, clean_specs
