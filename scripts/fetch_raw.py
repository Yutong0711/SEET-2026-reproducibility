#!/usr/bin/env python3
"""Fetch raw daily close-price series via yfinance.

Saves one CSV per ticker to data/raw/, writes data/manifests/raw_manifest.json,
and prints a STATUS block. Does NOT interpolate or fabricate data on failure;
empty/failed tickers are reported and skipped.

Run:
    python scripts/fetch_raw.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

try:
    import pandas as pd
    import yfinance as yf
except ImportError as e:
    sys.stderr.write(
        f"Missing dependency: {e}. Activate the venv (.venv) and ensure setup ran.\n"
    )
    sys.exit(1)


# (yfinance source ticker, output label used as filename and manifest key)
TICKERS: list[tuple[str, str]] = [
    ("^GSPC",  "SPX"),
    ("^NDX",   "NDX"),
    ("^RUT",   "RUT"),
    ("^VIX",   "VIX"),
    ("^VIX9D", "VIX9D"),
    ("^VIX3M", "VIX3M"),
    ("^VIX6M", "VIX6M"),
    ("^VVIX",  "VVIX"),
    ("^VXN",   "VXN"),
    ("^RVX",   "RVX"),
]

START_DATE = "2007-01-01"


def sha256_of(path: Path) -> str:
    """Compute SHA-256 of the file at `path` by streaming 1 MiB chunks."""
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


def fetch_one(symbol: str) -> "pd.DataFrame | None":
    """Return a DataFrame indexed by date with at least a 'Close' column,
    or None on failure / empty result.
    """
    end_exclusive = (dt.date.today() + dt.timedelta(days=1)).isoformat()
    df = yf.download(
        symbol,
        start=START_DATE,
        end=end_exclusive,
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if df is None or df.empty:
        return None
    # yfinance may return a MultiIndex on columns for a single symbol.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if "Close" not in df.columns:
        return None
    return df


def write_csv(df: "pd.DataFrame", out_path: Path) -> "pd.DataFrame":
    out = df[["Close"]].copy()
    out.index.name = "Date"
    out = out.reset_index()[["Date", "Close"]]
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["Close"]).reset_index(drop=True)
    out.to_csv(out_path, index=False)
    return out


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / "data" / "raw"
    manifests_dir = repo_root / "data" / "manifests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    statuses: list[tuple] = []  # (label, status, rows, first, last, sha12, note)

    for symbol, label in TICKERS:
        out_path = raw_dir / f"{label}.csv"
        rel_path = str(out_path.relative_to(repo_root)).replace("\\", "/")
        ts = utc_now_iso()
        note = ""

        try:
            df = fetch_one(symbol)
        except Exception as e:  # network / parser / etc.
            df = None
            note = f"download error: {e!s}"

        if df is None:
            if not note:
                note = "empty frame"
            manifest[label] = {
                "source_ticker": symbol,
                "file_path": rel_path,
                "sha256": None,
                "row_count": 0,
                "first_date": None,
                "last_date": None,
                "download_timestamp_utc": ts,
                "error": note,
            }
            statuses.append((label, "FAIL", 0, "-", "-", "-", note))
            continue

        out_df = write_csv(df, out_path)
        if out_df.empty:
            note = "all rows had null Close"
            manifest[label] = {
                "source_ticker": symbol,
                "file_path": rel_path,
                "sha256": None,
                "row_count": 0,
                "first_date": None,
                "last_date": None,
                "download_timestamp_utc": ts,
                "error": note,
            }
            statuses.append((label, "FAIL", 0, "-", "-", "-", note))
            continue

        digest = sha256_of(out_path)
        first_date = out_df["Date"].iloc[0]
        last_date = out_df["Date"].iloc[-1]
        n = int(len(out_df))

        manifest[label] = {
            "source_ticker": symbol,
            "file_path": rel_path,
            "sha256": digest,
            "row_count": n,
            "first_date": first_date,
            "last_date": last_date,
            "download_timestamp_utc": ts,
        }
        statuses.append((label, "OK", n, first_date, last_date, digest[:12], ""))

    manifest_path = manifests_dir / "raw_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    # STATUS block
    print()
    print("===== STATUS =====")
    header = (
        f"{'TICKER':<8}{'STATUS':<8}{'ROWS':>8}  "
        f"{'FIRST':<12}{'LAST':<12}{'SHA256[:12]':<14}NOTE"
    )
    print(header)
    print("-" * len(header))
    for label, status, n, first, last, sha12, note in statuses:
        print(
            f"{label:<8}{status:<8}{n:>8}  "
            f"{str(first):<12}{str(last):<12}{sha12:<14}{note}"
        )
    print("==================")
    print(f"manifest: {manifest_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
