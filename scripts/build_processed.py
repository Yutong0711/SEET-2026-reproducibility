#!/usr/bin/env python3
"""Build the processed panels for SEET 2026 from the raw layer.

Builds four panels in data/processed/, registers them in
data/manifests/processed_manifest.json, and prints a STATUS block.

The raw layer is left as-downloaded (yfinance/CBOE may include synthetic
backfill before an index's actual launch). Truncations are applied here,
in the processed layer, so each series carries NaN for dates before its
real launch:

    VIX9D : NaN before 2011-02-23  (CBOE launch)
    VIX3M : NaN before 2007-12-04  (CBOE launch as VXV)
    VVIX  : NaN before 2012-04-01  (launched March 2012; 1-month buffer)

Other series (SPX, NDX, RUT, VIX, VIX6M, VXN, RVX) are passed through
without truncation: they have no synthetic-backfill issue in our window.

Panels:

    spx_core_2007.csv      anchor SPX,    from 2007-01-03,
                           cols: Date, SPX, VIX, VIX3M*, VVIX*
    spx_extended_2011.csv  anchor VIX9D,  from 2011-02-23,
                           cols: Date, SPX, VIX, VIX9D, VIX3M, VIX6M, VVIX*
    ndx_2007.csv           anchor NDX,    from 2007-01-03,
                           cols: Date, NDX, VXN
    rut_2009.csv           anchor RUT,    from 2009-09-16,
                           cols: Date, RUT, RVX

(*) Truncated; column carries NaN before its truncation date even when
    the panel itself extends back further.

Run:
    python scripts/build_processed.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    sys.stderr.write(f"Missing dependency: {e}. Activate the .venv first.\n")
    sys.exit(1)


# Drop-rows-before semantics: applied in the processed layer by setting
# the corresponding Close to NaN for dates strictly before the threshold.
TRUNCATIONS: dict[str, dict] = {
    "VIX9D": {
        "before": "2011-02-23",
        "reason": (
            "VIX9D was launched by CBOE on 2011-02-23; pre-launch values "
            "in the raw feed are CBOE/Yahoo synthetic backfill, not "
            "real-time available signal."
        ),
    },
    "VIX3M": {
        "before": "2007-12-04",
        "reason": (
            "VIX3M (originally published as VXV) was launched on "
            "2007-12-04; pre-launch values in the raw feed are "
            "CBOE/Yahoo synthetic backfill."
        ),
    },
    "VVIX": {
        "before": "2012-04-01",
        "reason": (
            "VVIX was launched by CBOE in March 2012; the first ~1 month "
            "is dropped to allow clean data, so values before 2012-04-01 "
            "are removed."
        ),
    },
}


PANELS: list[dict] = [
    {
        "name": "spx_core_2007",
        "anchor": "SPX",
        "start_date": "2007-01-03",
        "series": ["SPX", "VIX", "VIX3M", "VVIX"],
        "purpose": (
            "Reduced SPX vol-family panel for crisis-period analyses. "
            "SPX and VIX are real throughout; VIX3M is real from "
            "2007-12-04, VVIX from 2012-04-01."
        ),
    },
    {
        "name": "spx_extended_2011",
        "anchor": "VIX9D",
        "start_date": "2011-02-23",
        "series": ["SPX", "VIX", "VIX9D", "VIX3M", "VIX6M", "VVIX"],
        "purpose": (
            "Full SPX vol-family panel anchored at the VIX9D launch date. "
            "VVIX still carries NaN until 2012-04-01."
        ),
    },
    {
        "name": "ndx_2007",
        "anchor": "NDX",
        "start_date": "2007-01-03",
        "series": ["NDX", "VXN"],
        "purpose": (
            "NDX/VXN panel; neither series has a backfill issue in this "
            "window."
        ),
    },
    {
        "name": "rut_2009",
        "anchor": "RUT",
        "start_date": "2009-09-16",
        "series": ["RUT", "RVX"],
        "purpose": (
            "RUT/RVX panel; window is limited by RVX availability (CBOE "
            "manual import begins 2009-09-16)."
        ),
    },
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_series(label: str, raw_dir: Path) -> pd.DataFrame:
    """Load data/raw/<LABEL>.csv as a DataFrame with columns Date and
    <LABEL>. Truncations are applied here: pre-launch Close values are
    set to NaN. The date row itself is preserved (no rows are dropped
    at this stage)."""
    df = pd.read_csv(raw_dir / f"{label}.csv", parse_dates=["Date"])
    df = df[["Date", "Close"]].rename(columns={"Close": label})
    if label in TRUNCATIONS:
        threshold = pd.Timestamp(TRUNCATIONS[label]["before"])
        df.loc[df["Date"] < threshold, label] = np.nan
    return df


def build_panel(spec: dict, raw_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Return (panel_df, column_schemas). The anchor series defines the
    date grid; other series are left-merged onto it."""
    anchor = spec["anchor"]
    start_date = pd.Timestamp(spec["start_date"])
    series_list = spec["series"]

    anchor_df = load_series(anchor, raw_dir)
    panel = anchor_df[anchor_df["Date"] >= start_date].reset_index(drop=True)

    for s in series_list:
        if s == anchor:
            continue
        sdf = load_series(s, raw_dir)
        panel = panel.merge(sdf, on="Date", how="left")

    panel = panel[["Date"] + series_list]

    column_schemas: list[dict] = [
        {"name": "Date", "dtype": "date", "format": "YYYY-MM-DD"}
    ]
    for s in series_list:
        non_null_mask = panel[s].notna()
        non_null = int(non_null_mask.sum())
        if non_null > 0:
            first_valid = panel.loc[non_null_mask, "Date"].min().strftime("%Y-%m-%d")
            last_valid = panel.loc[non_null_mask, "Date"].max().strftime("%Y-%m-%d")
        else:
            first_valid = None
            last_valid = None
        col = {
            "name": s,
            "dtype": "float64",
            "raw_source": f"data/raw/{s}.csv",
            "non_null_count": non_null,
            "first_valid_date": first_valid,
            "last_valid_date": last_valid,
            "truncated_before": TRUNCATIONS[s]["before"] if s in TRUNCATIONS else None,
            "truncation_reason": TRUNCATIONS[s]["reason"] if s in TRUNCATIONS else None,
        }
        column_schemas.append(col)

    panel["Date"] = panel["Date"].dt.strftime("%Y-%m-%d")
    return panel, column_schemas


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / "data" / "raw"
    processed_dir = repo_root / "data" / "processed"
    manifests_dir = repo_root / "data" / "manifests"
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    required_raw = sorted({s for spec in PANELS for s in spec["series"]})
    missing = [r for r in required_raw if not (raw_dir / f"{r}.csv").exists()]
    if missing:
        sys.stderr.write(
            f"Missing raw files: {missing}. Run scripts/fetch_raw.py first.\n"
        )
        return 1

    manifest: dict[str, dict] = {}
    statuses: list[tuple] = []
    ts = utc_now_iso()

    for spec in PANELS:
        name = spec["name"]
        out_path = processed_dir / f"{name}.csv"

        panel, schema = build_panel(spec, raw_dir)
        panel.to_csv(out_path, index=False)
        digest = sha256_of(out_path)
        n = int(len(panel))
        first_date = panel["Date"].iloc[0]
        last_date = panel["Date"].iloc[-1]

        manifest[name] = {
            "file_path": str(out_path.relative_to(repo_root)).replace("\\", "/"),
            "sha256": digest,
            "row_count": n,
            "first_date": first_date,
            "last_date": last_date,
            "anchor_series": spec["anchor"],
            "purpose": spec["purpose"],
            "columns": schema,
            "build_timestamp_utc": ts,
        }
        statuses.append(
            (name, n, first_date, last_date, digest[:12], len(spec["series"]))
        )

    manifest_path = manifests_dir / "processed_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print()
    print("===== PROCESSED STATUS =====")
    header = (
        f"{'PANEL':<22}{'ROWS':>8}  {'FIRST':<12}{'LAST':<12}"
        f"{'SHA256[:12]':<14}{'#FEAT':>6}"
    )
    print(header)
    print("-" * len(header))
    for name, n, first, last, sha12, nfeat in statuses:
        print(
            f"{name:<22}{n:>8}  {first:<12}{last:<12}{sha12:<14}{nfeat:>6}"
        )
    print("============================")
    print(f"manifest: {manifest_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
