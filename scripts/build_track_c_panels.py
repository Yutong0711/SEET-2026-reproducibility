"""Build Track C multi-asset panels: ndx_panel.csv, rut_panel.csv.

Each output panel has generic columns Date, INDEX, VOL where INDEX is
the equity-index level and VOL is the corresponding implied-volatility
level. Rows with any missing value are dropped (Track C policy; tighter
than the Track A processed-panel build which carries documented NaNs).

The existing data/processed/{ndx_2007,rut_2009}.csv panels are
unchanged. These new panels exist alongside them and are used only by
Track C's asset-agnostic pipeline.

Run:
    python scripts/build_track_c_panels.py
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
    sys.stderr.write(f"Missing dependency: {e}. Activate the .venv first.\n")
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parent.parent

ASSETS = [
    {
        "name": "ndx_panel",
        "index_raw": "NDX",
        "vol_raw": "VXN",
        "purpose": "NDX index + VXN implied vol; Track C generalization (no backfill issues in this window).",
    },
    {
        "name": "rut_panel",
        "index_raw": "RUT",
        "vol_raw": "RVX",
        "purpose": "RUT index + RVX implied vol; Track C generalization. RVX has 5 documented gap dates; rows with missing values are dropped per Track C policy.",
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


def build_panel(asset: dict, raw_dir: Path, out_dir: Path) -> dict:
    """Build one (INDEX, VOL) panel and return its manifest entry."""
    idx_path = raw_dir / f"{asset['index_raw']}.csv"
    vol_path = raw_dir / f"{asset['vol_raw']}.csv"
    if not idx_path.exists():
        raise FileNotFoundError(idx_path)
    if not vol_path.exists():
        raise FileNotFoundError(vol_path)

    idx = pd.read_csv(idx_path, parse_dates=["Date"])
    vol = pd.read_csv(vol_path, parse_dates=["Date"])
    idx = idx.rename(columns={"Close": "INDEX"})[["Date", "INDEX"]]
    vol = vol.rename(columns={"Close": "VOL"})[["Date", "VOL"]]

    n_idx_full = len(idx)
    n_vol_full = len(vol)

    panel = idx.merge(vol, on="Date", how="inner")
    n_after_merge = len(panel)
    n_dropped_na_idx = int(panel["INDEX"].isna().sum())
    n_dropped_na_vol = int(panel["VOL"].isna().sum())
    panel = panel.dropna(subset=["INDEX", "VOL"]).reset_index(drop=True)
    panel = panel.sort_values("Date").reset_index(drop=True)
    n_final = len(panel)

    # Write output (ISO date strings)
    out_path = out_dir / f"{asset['name']}.csv"
    out = panel.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out.to_csv(out_path, index=False)

    return {
        "file_path": str(out_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "sha256": sha256_of(out_path),
        "row_count": int(n_final),
        "first_date": out["Date"].iloc[0],
        "last_date": out["Date"].iloc[-1],
        "anchor_series": "INDEX",
        "purpose": asset["purpose"],
        "columns": [
            {"name": "Date", "dtype": "date", "format": "YYYY-MM-DD"},
            {
                "name": "INDEX",
                "dtype": "float64",
                "raw_source": str(idx_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "non_null_count": int(panel["INDEX"].notna().sum()),
                "first_valid_date": out["Date"].iloc[0],
                "last_valid_date": out["Date"].iloc[-1],
                "truncated_before": None,
                "truncation_reason": None,
            },
            {
                "name": "VOL",
                "dtype": "float64",
                "raw_source": str(vol_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "non_null_count": int(panel["VOL"].notna().sum()),
                "first_valid_date": out["Date"].iloc[0],
                "last_valid_date": out["Date"].iloc[-1],
                "truncated_before": None,
                "truncation_reason": None,
            },
        ],
        "build_timestamp_utc": utc_now_iso(),
        "track_c_build_notes": {
            "raw_index_rows": n_idx_full,
            "raw_vol_rows": n_vol_full,
            "rows_after_inner_join": int(n_after_merge),
            "rows_dropped_na_index": n_dropped_na_idx,
            "rows_dropped_na_vol": n_dropped_na_vol,
            "rows_final": int(n_final),
        },
    }


def main() -> int:
    raw_dir = REPO_ROOT / "data" / "raw"
    out_dir = REPO_ROOT / "data" / "processed"
    manifests_dir = REPO_ROOT / "data" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    pm_path = manifests_dir / "processed_manifest.json"
    if pm_path.exists():
        with pm_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {}

    statuses: list[tuple] = []
    for asset in ASSETS:
        entry = build_panel(asset, raw_dir, out_dir)
        manifest[asset["name"]] = entry
        statuses.append(
            (
                asset["name"],
                entry["row_count"],
                entry["first_date"],
                entry["last_date"],
                entry["sha256"][:12],
            )
        )

    with pm_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print()
    print("===== TRACK C PANEL BUILD STATUS =====")
    header = f"{'PANEL':<14}{'ROWS':>8}  {'FIRST':<12}{'LAST':<12}{'SHA[:12]':<14}"
    print(header)
    print("-" * len(header))
    for name, n, first, last, sha12 in statuses:
        print(f"{name:<14}{n:>8}  {first:<12}{last:<12}{sha12:<14}")
    print("======================================")
    print(f"manifest updated: {pm_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
