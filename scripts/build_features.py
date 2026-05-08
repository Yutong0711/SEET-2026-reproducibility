#!/usr/bin/env python3
"""Build features for all four processed panels.

Reads:   data/processed/<panel>.csv
         data/manifests/processed_manifest.json
Writes:  data/processed/features/<panel>_features.csv  (one per panel)
         data/manifests/features_manifest.json

Run:
    python scripts/build_features.py
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
sys.path.insert(0, str(REPO_ROOT / "src"))

from seet.features import build_features, SCHEMA_VERSION  # noqa: E402


PANEL_TO_FEATURE_SET: dict[str, str] = {
    "spx_extended_2011": "spx_full",
    "spx_core_2007":     "spx_core",
    "ndx_2007":          "ndx_minimal",
    "rut_2009":          "rut_minimal",
    # Track C: generic INDEX/VOL panels for the multi-asset
    # replication. Use the asset_minimal feature set which is
    # mechanically the same as ndx_minimal/rut_minimal but expects
    # generic column names INDEX (price) and VOL (volatility).
    "ndx_panel":         "asset_minimal",
    "rut_panel":         "asset_minimal",
}


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
    processed_dir = REPO_ROOT / "data" / "processed"
    features_dir = processed_dir / "features"
    manifests_dir = REPO_ROOT / "data" / "manifests"
    features_dir.mkdir(parents=True, exist_ok=True)

    pm_path = manifests_dir / "processed_manifest.json"
    if not pm_path.exists():
        sys.stderr.write(
            f"Processed manifest not found: {pm_path}. "
            "Run scripts/build_processed.py first.\n"
        )
        return 1
    with pm_path.open("r", encoding="utf-8") as f:
        processed_manifest = json.load(f)

    features_manifest: dict = {}
    statuses: list[tuple] = []
    ts = utc_now_iso()

    for panel_name in sorted(PANEL_TO_FEATURE_SET.keys()):
        fset_id = PANEL_TO_FEATURE_SET[panel_name]
        if panel_name not in processed_manifest:
            sys.stderr.write(
                f"Panel {panel_name!r} not in processed_manifest; skipping.\n"
            )
            continue
        panel_entry = processed_manifest[panel_name]
        panel_path = REPO_ROOT / panel_entry["file_path"]
        panel_df = pd.read_csv(panel_path, parse_dates=["Date"])

        features_df, specs = build_features(panel_df, fset_id)

        out_df = features_df.copy()
        out_df["Date"] = pd.to_datetime(out_df["Date"]).dt.strftime("%Y-%m-%d")
        out_path = features_dir / f"{panel_name}_features.csv"
        out_df.to_csv(out_path, index=False)
        digest = sha256_of(out_path)

        n_rows = int(len(out_df))
        n_features = len(specs)
        nan_count = int(out_df.drop(columns=["Date"]).isna().sum().sum())
        warmup_max = max((s["lookback_window"] for s in specs), default=1) - 1

        features_manifest[panel_name] = {
            "feature_set_id": fset_id,
            "source_panel": panel_name,
            "source_panel_sha256": panel_entry["sha256"],
            "file_path": str(out_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": digest,
            "row_count": n_rows,
            "first_date": out_df["Date"].iloc[0],
            "last_date": out_df["Date"].iloc[-1],
            "feature_count": n_features,
            "max_warmup_rows": int(warmup_max),
            "schema_version": SCHEMA_VERSION,
            "build_timestamp_utc": ts,
            "features": specs,
        }
        statuses.append(
            (panel_name, fset_id, n_rows, n_features, warmup_max, nan_count, digest[:12])
        )

    fm_path = manifests_dir / "features_manifest.json"
    with fm_path.open("w", encoding="utf-8") as f:
        json.dump(features_manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print()
    print("===== FEATURES STATUS =====")
    header = (
        f"{'PANEL':<22}{'SET':<14}{'ROWS':>6}  {'FEATS':>6}  "
        f"{'WARMUP':>7}  {'NANS':>10}  SHA[:12]"
    )
    print(header)
    print("-" * len(header))
    for name, fset, n, nf, wm, nans, sha12 in statuses:
        print(
            f"{name:<22}{fset:<14}{n:>6}  {nf:>6}  "
            f"{wm:>7}  {nans:>10}  {sha12}"
        )
    print("===========================")
    print(f"manifest: {fm_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
