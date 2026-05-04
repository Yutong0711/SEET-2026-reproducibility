#!/usr/bin/env python3
"""Fetch raw daily close-price series via yfinance, with a CBOE RVX fallback.

Saves one CSV per ticker to data/raw/, writes data/manifests/raw_manifest.json,
and prints a STATUS block. Does NOT interpolate or fabricate data on failure;
empty/failed tickers are reported and skipped.

For ^RVX specifically: if yfinance returns empty AND
data/raw/RVX_History.csv (CBOE export, schema DATE,OPEN,HIGH,LOW,CLOSE) is
present, the CBOE file is normalized into data/raw/RVX.csv as a fallback.
If neither yfinance nor the CBOE file is available, a clear pointer is
emitted and RVX is recorded as FAIL.

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
    import numpy as np  # noqa: F401  (imported for downstream consistency)
    import pandas as pd
    import yfinance as yf
except ImportError as e:
    sys.stderr.write(
        f"Missing dependency: {e}. Activate the venv (.venv) and ensure setup ran.\n"
    )
    sys.exit(1)


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

CBOE_RVX_URL = (
    "https://www.cboe.com/tradable_products/vix/rvx_historical_data/"
)
CBOE_RVX_NOTE = (
    f"CBOE fallback (yfinance returned empty for ^RVX); source: {CBOE_RVX_URL}"
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


def fetch_one(symbol: str) -> "pd.DataFrame | None":
    """Return a date-indexed DataFrame with at least a 'Close' column,
    or None on empty result.
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
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if "Close" not in df.columns:
        return None
    return df


def write_csv_from_yf(df: "pd.DataFrame", out_path: Path) -> "pd.DataFrame":
    out = df[["Close"]].copy()
    out.index.name = "Date"
    out = out.reset_index()[["Date", "Close"]]
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["Close"]).reset_index(drop=True)
    out.to_csv(out_path, index=False)
    return out


def try_cboe_rvx_fallback(raw_dir: Path, repo_root: Path) -> "dict | None":
    """If data/raw/RVX_History.csv exists and parses, return a dict with:
        - df: normalized DataFrame (Date as ISO str, Close as float)
        - source_path: relative POSIX path to the CBOE CSV
        - source_sha256: sha256 of the CBOE CSV
        - source_note: provenance note
        - rows_dropped_bad_date: int
        - rows_dropped_bad_close: int
    Otherwise return None.
    """
    src = raw_dir / "RVX_History.csv"
    if not src.exists():
        return None
    try:
        df = pd.read_csv(src)
    except Exception:
        return None
    cols_upper = {c.upper(): c for c in df.columns}
    if "DATE" not in cols_upper or "CLOSE" not in cols_upper:
        return None

    out = df[[cols_upper["DATE"], cols_upper["CLOSE"]]].copy()
    out.columns = ["Date", "Close"]
    out["Date"] = pd.to_datetime(out["Date"], format="%m/%d/%Y", errors="coerce")
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")

    bad_dates = int(out["Date"].isna().sum())
    bad_closes = int(out["Close"].isna().sum())
    out = out.dropna(subset=["Date", "Close"])
    out = out.drop_duplicates(subset=["Date"], keep="last")
    out = out.sort_values("Date").reset_index(drop=True)
    if out.empty:
        return None
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")

    return {
        "df": out,
        "source_path": str(src.relative_to(repo_root)).replace("\\", "/"),
        "source_sha256": sha256_of(src),
        "source_note": CBOE_RVX_NOTE,
        "rows_dropped_bad_date": bad_dates,
        "rows_dropped_bad_close": bad_closes,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / "data" / "raw"
    manifests_dir = repo_root / "data" / "manifests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    statuses: list[tuple] = []

    for symbol, label in TICKERS:
        out_path = raw_dir / f"{label}.csv"
        rel_path = str(out_path.relative_to(repo_root)).replace("\\", "/")
        ts = utc_now_iso()
        note = ""

        try:
            df = fetch_one(symbol)
        except Exception as e:
            df = None
            note = f"download error: {e!s}"

        # ---- Primary path: yfinance succeeded with non-empty data ----
        if df is not None:
            out_df = write_csv_from_yf(df, out_path)
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
            continue

        # ---- yfinance failed/empty. Special-case RVX with CBOE fallback ----
        if not note:
            note = "empty frame"

        if symbol == "^RVX":
            fb = try_cboe_rvx_fallback(raw_dir, repo_root)
            if fb is not None:
                out_df = fb["df"]
                out_df.to_csv(out_path, index=False)
                digest = sha256_of(out_path)
                first_date = out_df["Date"].iloc[0]
                last_date = out_df["Date"].iloc[-1]
                n = int(len(out_df))
                manifest[label] = {
                    "source_ticker": symbol,
                    "source_path": fb["source_path"],
                    "source_sha256": fb["source_sha256"],
                    "source_note": fb["source_note"],
                    "file_path": rel_path,
                    "sha256": digest,
                    "row_count": n,
                    "first_date": first_date,
                    "last_date": last_date,
                    "download_timestamp_utc": ts,
                    "rows_dropped_bad_date": fb["rows_dropped_bad_date"],
                    "rows_dropped_bad_close": fb["rows_dropped_bad_close"],
                }
                statuses.append(
                    (label, "OK", n, first_date, last_date, digest[:12], "CBOE fallback")
                )
                continue
            # No CBOE file available: tell the user, then fall through to FAIL
            sys.stderr.write(
                "[RVX] yfinance returned empty for ^RVX and "
                "data/raw/RVX_History.csv is not present.\n"
                f"  Download the CBOE CSV from\n    {CBOE_RVX_URL}\n"
                "  place it at data/raw/RVX_History.csv, then re-run this "
                "script.\n"
            )
            note = "empty frame; no CBOE fallback available"

        # ---- Standard FAIL path ----
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

    manifest_path = manifests_dir / "raw_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

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
