#!/usr/bin/env python3
"""Normalize CBOE RVX_History.csv into the project's standard Date,Close format.

Reads:   data/raw/RVX_History.csv      (CBOE export: DATE,OPEN,HIGH,LOW,CLOSE)
Writes:  data/raw/RVX.csv              (Date,Close, ISO dates, NaN closes dropped)
Updates: data/manifests/raw_manifest.json  (replaces the prior RVX entry)

Use this after manually downloading RVX_History.csv from CBOE because yfinance
returned an empty frame for ^RVX. Run from inside the project venv:

    .\\.venv\\Scripts\\Activate.ps1
    python scripts\\import_rvx_from_cboe.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as e:
    sys.stderr.write(f"Missing dependency: {e}. Activate .venv first.\n")
    sys.exit(1)


CBOE_NOTE = (
    "CBOE manual download (yfinance returned empty frame for ^RVX); "
    "source: https://www.cboe.com/us/indices/dashboard/RVX/"
)


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


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "data" / "raw" / "RVX_History.csv"
    dst = repo_root / "data" / "raw" / "RVX.csv"
    manifest_path = repo_root / "data" / "manifests" / "raw_manifest.json"

    if not src.exists():
        sys.stderr.write(f"Source not found: {src}\n")
        return 1
    if not manifest_path.exists():
        sys.stderr.write(f"Manifest not found: {manifest_path}\n")
        return 1

    df = pd.read_csv(src)
    cols_upper = {c.upper(): c for c in df.columns}
    if "DATE" not in cols_upper or "CLOSE" not in cols_upper:
        sys.stderr.write(
            f"Expected DATE and CLOSE columns in {src.name}; got {list(df.columns)}\n"
        )
        return 1

    out = df[[cols_upper["DATE"], cols_upper["CLOSE"]]].copy()
    out.columns = ["Date", "Close"]
    out["Date"] = pd.to_datetime(out["Date"], format="%m/%d/%Y", errors="coerce")
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")

    bad_dates = int(out["Date"].isna().sum())
    bad_closes = int(out["Close"].isna().sum())
    out = out.dropna(subset=["Date", "Close"])
    out = out.drop_duplicates(subset=["Date"], keep="last")
    out = out.sort_values("Date").reset_index(drop=True)
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")

    if out.empty:
        sys.stderr.write("No usable rows after parsing; aborting.\n")
        return 1

    out.to_csv(dst, index=False)

    src_sha = sha256_of(src)
    dst_sha = sha256_of(dst)
    n = int(len(out))
    first_date = out["Date"].iloc[0]
    last_date = out["Date"].iloc[-1]
    ts = utc_now_iso()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest["RVX"] = {
        "source_ticker": "^RVX",
        "source_path": str(src.relative_to(repo_root)).replace("\\", "/"),
        "source_sha256": src_sha,
        "source_note": CBOE_NOTE,
        "file_path": str(dst.relative_to(repo_root)).replace("\\", "/"),
        "sha256": dst_sha,
        "row_count": n,
        "first_date": first_date,
        "last_date": last_date,
        "download_timestamp_utc": ts,
        "rows_dropped_bad_date": bad_dates,
        "rows_dropped_bad_close": bad_closes,
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print()
    print("===== STATUS (RVX update) =====")
    header = (
        f"{'TICKER':<8}{'STATUS':<8}{'ROWS':>8}  "
        f"{'FIRST':<12}{'LAST':<12}{'SHA256[:12]':<14}NOTE"
    )
    print(header)
    print("-" * len(header))
    note = f"from {src.name} (src sha {src_sha[:12]})"
    print(
        f"{'RVX':<8}{'OK':<8}{n:>8}  "
        f"{first_date:<12}{last_date:<12}{dst_sha[:12]:<14}{note}"
    )
    print("===============================")
    print(f"manifest updated: {manifest_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
