"""Track G — drift instrumentation for Layer 4.

Computes per-fold drift summaries (PSI / sym_kl) for SPX, NDX, RUT
using existing Track A / Track C fold definitions and feature CSVs;
joins with the corresponding per_fold_metrics.csv files; computes
Spearman correlations between drift metrics and operational metrics;
renders three figures; prints a STATUS block.

Inputs (read-only):
    data/processed/spx_extended_2011.csv + features/spx_extended_2011_features.csv
    data/processed/ndx_panel.csv         + features/ndx_panel_features.csv
    data/processed/rut_panel.csv         + features/rut_panel_features.csv
    experiments/track_a_headline/{fold_definitions, per_fold_metrics}.csv
    experiments/track_c_multiasset/{fold_definitions_ndx, per_fold_metrics_ndx,
                                    fold_definitions_rut, per_fold_metrics_rut}.csv

Outputs:
    experiments/track_g_drift/fold_drift_spx.csv
    experiments/track_g_drift/fold_drift_ndx.csv
    experiments/track_g_drift/fold_drift_rut.csv
    experiments/track_g_drift/feature_drift_per_fold.csv      (SPX only)
    outputs/track_g/table_drift_correlations.csv
    outputs/track_g/fig_drift_vs_metric.pdf      (2x2 grid)
    outputs/track_g/fig_drift_timeline.pdf       (SPX, h10_d05)
    outputs/track_g/fig_feature_drift_heatmap.pdf (SPX, h10_d05)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from seet.drift import (  # noqa: E402
    feature_level_drift,
    fold_level_drift_summary,
)


# Asset registry for the drift loop.
ASSETS = {
    "spx": {
        "panel_path": REPO_ROOT / "data" / "processed" / "spx_extended_2011.csv",
        "features_path": REPO_ROOT / "data" / "processed" / "features"
        / "spx_extended_2011_features.csv",
        "fold_def_path": REPO_ROOT / "experiments" / "track_a_headline"
        / "fold_definitions.csv",
        "per_fold_path": REPO_ROOT / "experiments" / "track_a_headline"
        / "per_fold_metrics.csv",
    },
    "ndx": {
        "panel_path": REPO_ROOT / "data" / "processed" / "ndx_panel.csv",
        "features_path": REPO_ROOT / "data" / "processed" / "features"
        / "ndx_panel_features.csv",
        "fold_def_path": REPO_ROOT / "experiments" / "track_c_multiasset"
        / "fold_definitions_ndx.csv",
        "per_fold_path": REPO_ROOT / "experiments" / "track_c_multiasset"
        / "per_fold_metrics_ndx.csv",
    },
    "rut": {
        "panel_path": REPO_ROOT / "data" / "processed" / "rut_panel.csv",
        "features_path": REPO_ROOT / "data" / "processed" / "features"
        / "rut_panel_features.csv",
        "fold_def_path": REPO_ROOT / "experiments" / "track_c_multiasset"
        / "fold_definitions_rut.csv",
        "per_fold_path": REPO_ROOT / "experiments" / "track_c_multiasset"
        / "per_fold_metrics_rut.csv",
    },
}

EXP_DIR = REPO_ROOT / "experiments" / "track_g_drift"
OUT_DIR = REPO_ROOT / "outputs" / "track_g"
TARGET_ALARM_RATE = 0.05

REGIME_EVENTS = [
    ("2015-08-24", "Aug 2015 China"),
    ("2018-02-05", "Feb 2018 volmageddon"),
    ("2018-12-24", "Dec 2018 selloff"),
    ("2020-03-16", "Mar 2020 COVID"),
    ("2022-01-31", "Jan 2022 rate-hike repricing"),
    ("2023-03-13", "Mar 2023 regional bank stress"),
]


# ---------------------------------------------------------------------
# Step 2 — per-fold drift summaries
# ---------------------------------------------------------------------

def compute_fold_drift_for_asset(
    asset: str,
    panel_df: pd.DataFrame,
    features_df: pd.DataFrame,
    fold_def_df: pd.DataFrame,
    per_fold_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (asset, stress_def, fold_id) with the drift summary
    plus the realized alarm rate averaged across the 6 baselines."""
    feat_cols = [c for c in features_df.columns if c != "Date"]
    feat_dates = pd.to_datetime(features_df["Date"]).to_numpy()

    rows: list[dict] = []
    for _, fd_row in fold_def_df.iterrows():
        sd = fd_row["stress_def"]
        fid = int(fd_row["fold_id"])
        train_end = pd.Timestamp(fd_row["train_end"])
        test_start = pd.Timestamp(fd_row["test_start"])
        test_end = pd.Timestamp(fd_row["test_end"])

        train_mask = feat_dates <= np.datetime64(train_end)
        test_mask = (
            (feat_dates >= np.datetime64(test_start))
            & (feat_dates <= np.datetime64(test_end))
        )
        train_df = features_df.loc[train_mask, feat_cols]
        test_df = features_df.loc[test_mask, feat_cols]
        if len(train_df) < 30 or len(test_df) < 5:
            continue

        summary = fold_level_drift_summary(train_df, test_df, feat_cols)

        cell_metrics = per_fold_df[
            (per_fold_df["stress_def"] == sd)
            & (per_fold_df["fold_id"] == fid)
        ]
        if cell_metrics.empty:
            realized_alarm = float("nan")
        else:
            # Average across (model, seed) — drift is model-agnostic.
            realized_alarm = float(
                pd.to_numeric(cell_metrics["alarm_rate"], errors="coerce")
                .dropna().mean()
            )
        gap_signed = (
            realized_alarm - TARGET_ALARM_RATE
            if not np.isnan(realized_alarm) else float("nan")
        )
        gap_abs = (
            abs(gap_signed) if not np.isnan(gap_signed) else float("nan")
        )

        rows.append({
            "asset": asset,
            "stress_def": sd,
            "fold_id": fid,
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
            "mean_psi": summary["mean_psi"],
            "max_psi": summary["max_psi"],
            "mean_kl": summary["mean_kl"],
            "max_kl": summary["max_kl"],
            "n_high_drift": summary["n_high_drift"],
            "high_drift_features": json.dumps(summary["high_drift_features"]),
            "n_features": summary["n_features"],
            "n_features_valid": summary["n_features_valid"],
            "realized_alarm_rate_avg": realized_alarm,
            "alarm_rate_gap_signed": gap_signed,
            "alarm_rate_gap_abs": gap_abs,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Step 3 — SPX per-feature heatmap data
# ---------------------------------------------------------------------

def compute_spx_feature_drift(
    panel_df: pd.DataFrame,
    features_df: pd.DataFrame,
    fold_def_df: pd.DataFrame,
) -> pd.DataFrame:
    feat_cols = [c for c in features_df.columns if c != "Date"]
    feat_dates = pd.to_datetime(features_df["Date"]).to_numpy()

    rows: list[dict] = []
    for _, fd_row in fold_def_df.iterrows():
        sd = fd_row["stress_def"]
        fid = int(fd_row["fold_id"])
        train_end = pd.Timestamp(fd_row["train_end"])
        test_start = pd.Timestamp(fd_row["test_start"])
        test_end = pd.Timestamp(fd_row["test_end"])
        train_mask = feat_dates <= np.datetime64(train_end)
        test_mask = (
            (feat_dates >= np.datetime64(test_start))
            & (feat_dates <= np.datetime64(test_end))
        )
        train_df = features_df.loc[train_mask, feat_cols]
        test_df = features_df.loc[test_mask, feat_cols]
        if len(train_df) < 30 or len(test_df) < 5:
            continue
        per_feat = feature_level_drift(train_df, test_df, feat_cols)
        for _, r in per_feat.iterrows():
            rows.append({
                "stress_def": sd,
                "fold_id": fid,
                "train_end": train_end.strftime("%Y-%m-%d"),
                "feature": r["feature"],
                "psi": r["psi"],
                "sym_kl": r["sym_kl"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Step 4 — drift × operational correlations
# ---------------------------------------------------------------------

DRIFT_METRICS = ("mean_psi", "max_psi", "mean_kl", "max_kl", "n_high_drift")
OP_METRICS = (
    "auc",
    "pr_auc",
    "drawdown_lift",
    "alarm_rate",
    "alarm_rate_gap_abs",
    "alarm_rate_gap_signed",
)


def compute_correlations(
    fold_drift_dfs: dict[str, pd.DataFrame],
    per_fold_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    from scipy.stats import spearmanr

    rows: list[dict] = []
    for asset, drift_df in fold_drift_dfs.items():
        per_fold = per_fold_dfs[asset].copy()
        per_fold["alarm_rate_gap_signed"] = (
            per_fold["alarm_rate"] - TARGET_ALARM_RATE
        )
        per_fold["alarm_rate_gap_abs"] = per_fold["alarm_rate_gap_signed"].abs()

        for sd in sorted(drift_df["stress_def"].unique()):
            ddf = drift_df[drift_df["stress_def"] == sd]
            pf = per_fold[per_fold["stress_def"] == sd]
            joined = pf.merge(
                ddf[
                    ["fold_id", "mean_psi", "max_psi",
                     "mean_kl", "max_kl", "n_high_drift"]
                ],
                on="fold_id",
                how="inner",
            )
            for d_metric in DRIFT_METRICS:
                for op_metric in OP_METRICS:
                    sub = joined[[d_metric, op_metric]].dropna()
                    if len(sub) < 5:
                        rows.append({
                            "asset": asset,
                            "stress_def": sd,
                            "drift_metric": d_metric,
                            "operational_metric": op_metric,
                            "spearman_rho": float("nan"),
                            "p_value": float("nan"),
                            "n_pairs": int(len(sub)),
                        })
                        continue
                    rho, p = spearmanr(sub[d_metric], sub[op_metric])
                    rows.append({
                        "asset": asset,
                        "stress_def": sd,
                        "drift_metric": d_metric,
                        "operational_metric": op_metric,
                        "spearman_rho": float(rho),
                        "p_value": float(p),
                        "n_pairs": int(len(sub)),
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Step 5 — figures
# ---------------------------------------------------------------------

def _palette():
    return {
        "h5_d03":  "#1f77b4",  # blue
        "h10_d05": "#d62728",  # red
        "h20_d07": "#2ca02c",  # green
    }


def _markers():
    return {"spx": "o", "ndx": "s", "rut": "^"}


def fig_drift_vs_metric(
    fold_drift_dfs: dict[str, pd.DataFrame],
    per_fold_dfs: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """2x2 grid as specified by the user:
        TL  mean_psi vs drawdown_lift
        TR  mean_psi vs alarm_rate_gap_abs
        BL  max_psi  vs auc
        BR  n_high_drift vs drawdown_lift
    Color = stress_def, marker = asset; Spearman rho + p in upper-right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    panels = [
        ("mean_psi",     "drawdown_lift",       "Top-left:   mean_psi vs drawdown_lift"),
        ("mean_psi",     "alarm_rate_gap_abs",  "Top-right:  mean_psi vs |alarm_rate - 0.05|"),
        ("max_psi",      "auc",                 "Bottom-left: max_psi vs AUC"),
        ("n_high_drift", "drawdown_lift",       "Bottom-right: n_high_drift vs drawdown_lift"),
    ]

    palette = _palette()
    markers = _markers()

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes_flat = axes.ravel()
    for ax, (d_metric, op_metric, title) in zip(axes_flat, panels):
        all_xs: list[float] = []
        all_ys: list[float] = []
        for asset, drift_df in fold_drift_dfs.items():
            per_fold = per_fold_dfs[asset].copy()
            per_fold["alarm_rate_gap_abs"] = (
                (per_fold["alarm_rate"] - TARGET_ALARM_RATE).abs()
            )
            for sd in ("h5_d03", "h10_d05", "h20_d07"):
                ddf = drift_df[drift_df["stress_def"] == sd]
                pf = per_fold[per_fold["stress_def"] == sd]
                joined = pf.merge(
                    ddf[["fold_id", d_metric]], on="fold_id", how="inner"
                )
                xs = pd.to_numeric(joined[d_metric], errors="coerce")
                ys = pd.to_numeric(joined[op_metric], errors="coerce")
                mask = xs.notna() & ys.notna()
                xs = xs[mask].to_numpy()
                ys = ys[mask].to_numpy()
                if xs.size == 0:
                    continue
                ax.scatter(
                    xs, ys, color=palette[sd], marker=markers[asset],
                    alpha=0.55, s=24, edgecolor="black", linewidth=0.3,
                    label=f"{asset.upper()} {sd}",
                )
                all_xs.extend(xs.tolist())
                all_ys.extend(ys.tolist())
        # Spearman across the pooled data shown in the panel.
        if len(all_xs) >= 5:
            rho, p = spearmanr(all_xs, all_ys)
            ax.text(
                0.97, 0.96, f"ρ = {rho:+.3f}\n(p = {p:.3f})",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white",
                          edgecolor="gray", alpha=0.85),
            )
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(d_metric)
        ax.set_ylabel(op_metric)
        ax.grid(linestyle=":", linewidth=0.4)

    # Single legend below all panels.
    handles_color = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=palette[sd], markeredgecolor="black",
                   markersize=8, label=sd)
        for sd in ("h5_d03", "h10_d05", "h20_d07")
    ]
    handles_marker = [
        plt.Line2D([0], [0], marker=markers[asset], color="w",
                   markerfacecolor="lightgray", markeredgecolor="black",
                   markersize=8, label=asset.upper())
        for asset in ("spx", "ndx", "rut")
    ]
    fig.legend(
        handles=handles_color + handles_marker,
        loc="lower center", ncol=6, fontsize=8,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Track G — drift vs operational metrics (2×2 headline)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def fig_drift_timeline(
    spx_drift_df: pd.DataFrame, out_path: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = spx_drift_df[spx_drift_df["stress_def"] == "h10_d05"].copy()
    sub["test_start_dt"] = pd.to_datetime(sub["test_start"])
    sub = sub.sort_values("test_start_dt")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        sub["test_start_dt"], sub["mean_psi"],
        marker="o", color="#d62728", linewidth=1.4, markersize=4,
        label="SPX h10_d05 mean_psi",
    )
    # n_high_drift as light bars on a secondary axis.
    ax2 = ax.twinx()
    ax2.bar(
        sub["test_start_dt"], sub["n_high_drift"],
        width=120, color="lightgray", alpha=0.6,
        label="n_high_drift (right axis)", zorder=0,
    )
    ax2.set_ylabel("n_high_drift", color="gray")
    ax2.tick_params(axis="y", colors="gray")
    ax2.set_ylim(0, max(int(sub["n_high_drift"].max()) + 2, 5))

    # PSI threshold lines.
    ax.axhline(0.10, color="orange", linestyle=":", linewidth=0.8)
    ax.axhline(0.25, color="red", linestyle=":", linewidth=0.8)
    ax.text(
        sub["test_start_dt"].iloc[-1], 0.10, " moderate (PSI 0.10)",
        color="orange", fontsize=8, va="bottom", ha="right",
    )
    ax.text(
        sub["test_start_dt"].iloc[-1], 0.25, " high (PSI 0.25)",
        color="red", fontsize=8, va="bottom", ha="right",
    )

    # Regime markers.
    ymax = max(float(sub["mean_psi"].max()) * 1.15, 0.30)
    ax.set_ylim(0, ymax)
    panel_min = sub["test_start_dt"].min()
    panel_max = sub["test_start_dt"].max()
    for dt_str, label in REGIME_EVENTS:
        dt = pd.Timestamp(dt_str)
        if dt < panel_min or dt > panel_max:
            continue
        ax.axvline(dt, color="black", linestyle="--",
                   linewidth=0.7, alpha=0.5)
        ax.text(
            dt, ymax * 0.92, " " + label,
            rotation=90, fontsize=7, va="top", ha="left", alpha=0.75,
        )

    ax.set_xlabel("test window start date")
    ax.set_ylabel("mean PSI")
    ax.set_title(
        "Track G — SPX mean_psi over time with regime markers (h10_d05)",
        fontsize=10,
    )
    ax.grid(linestyle=":", linewidth=0.4)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def fig_feature_drift_heatmap(
    feat_drift_df: pd.DataFrame, fold_def_df: pd.DataFrame, out_path: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = feat_drift_df[feat_drift_df["stress_def"] == "h10_d05"].copy()
    if sub.empty:
        return
    pivot = sub.pivot(index="feature", columns="fold_id", values="psi")
    # Keep features in deterministic insertion order from the spx_full
    # spec by reading the source features CSV's column order.
    src_features = pd.read_csv(
        REPO_ROOT / "data" / "processed" / "features"
        / "spx_extended_2011_features.csv",
        nrows=1,
    )
    feat_order = [c for c in src_features.columns if c != "Date"]
    pivot = pivot.reindex(feat_order)
    pivot = pivot[sorted(pivot.columns)]

    fold_def_sub = fold_def_df[fold_def_df["stress_def"] == "h10_d05"].copy()
    fold_def_sub = fold_def_sub.set_index("fold_id")
    col_labels = [
        f"{int(c)}\n{fold_def_sub.loc[int(c), 'train_end'][:7]}"
        if int(c) in fold_def_sub.index else str(c)
        for c in pivot.columns
    ]

    fig, ax = plt.subplots(figsize=(11, 9))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(
        data, aspect="auto", cmap="Reds",
        vmin=0.0, vmax=max(float(np.nanmax(data)) * 0.85, 0.30),
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(col_labels, rotation=0, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)

    # Annotate cells where PSI > 0.25 with the value.
    for i, feature in enumerate(pivot.index):
        for j, fold_id in enumerate(pivot.columns):
            v = data[i, j]
            if not np.isnan(v) and v > 0.25:
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=6, color="white",
                )
    ax.set_xlabel("fold_id (train_end yyyy-mm)")
    ax.set_ylabel("feature")
    ax.set_title(
        "Track G — SPX per-feature PSI by fold (h10_d05); cells > 0.25 annotated",
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("PSI", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Step 6 — STATUS
# ---------------------------------------------------------------------

def print_status(
    fold_drift_dfs: dict[str, pd.DataFrame],
    correlations: pd.DataFrame,
) -> None:
    print()
    print("=" * 78)
    print("TRACK G STATUS — drift instrumentation for Layer 4")
    print("=" * 78)

    # ----- Headline correlations -----
    print("\n[a] Headline drift × operational-metric Spearman correlations:")
    headline = correlations[
        correlations["drift_metric"].isin(("mean_psi", "max_psi"))
        & correlations["operational_metric"].isin(
            ("auc", "drawdown_lift", "alarm_rate_gap_abs")
        )
        & correlations["stress_def"].eq("h10_d05")
    ].sort_values(["asset", "drift_metric", "operational_metric"])
    print(f"  {'asset':<5} {'drift':<14} {'op_metric':<22} "
          f"{'rho':>7} {'p':>8} {'n':>4}")
    print("  " + "-" * 64)
    for _, r in headline.iterrows():
        rho_s = f"{r['spearman_rho']:+7.3f}" if not pd.isna(r["spearman_rho"]) else "    n/a"
        p_s = f"{r['p_value']:8.3f}" if not pd.isna(r["p_value"]) else "     n/a"
        sig = "*" if (not pd.isna(r["p_value"]) and r["p_value"] < 0.05) else " "
        print(
            f"  {r['asset']:<5} {r['drift_metric']:<14} "
            f"{r['operational_metric']:<22} {rho_s} {p_s}{sig} "
            f"{int(r['n_pairs']):>4}"
        )

    # ----- Layer-4 alarm-gap claim -----
    print("\n[b] Layer-4 methodology claim — does mean_psi correlate with "
          "|realized − target| alarm-rate gap?")
    claim = correlations[
        correlations["drift_metric"].eq("mean_psi")
        & correlations["operational_metric"].eq("alarm_rate_gap_abs")
        & correlations["stress_def"].eq("h10_d05")
    ]
    for _, r in claim.iterrows():
        if pd.isna(r["spearman_rho"]):
            print(f"  {r['asset']:<5}  rho = n/a (insufficient pairs)")
            continue
        sign = "POSITIVE" if r["spearman_rho"] > 0 else "NEGATIVE"
        sig = "p<0.05" if r["p_value"] < 0.05 else "n.s."
        print(
            f"  {r['asset']:<5}  rho = {r['spearman_rho']:+.3f}  "
            f"p = {r['p_value']:.3f}  [{sign}, {sig}]  n={int(r['n_pairs'])}"
        )

    # ----- mean_psi as leading indicator -----
    print("\n[c] mean_psi → drawdown_lift (does drift predict operational "
          "degradation?):")
    leading = correlations[
        correlations["drift_metric"].eq("mean_psi")
        & correlations["operational_metric"].eq("drawdown_lift")
        & correlations["stress_def"].eq("h10_d05")
    ]
    for _, r in leading.iterrows():
        if pd.isna(r["spearman_rho"]):
            print(f"  {r['asset']:<5}  rho = n/a (insufficient pairs)")
            continue
        sign = "POSITIVE (more drift → higher lift?)" if r["spearman_rho"] > 0 else \
               "NEGATIVE (more drift → lower lift, the methodology prediction)"
        sig = "p<0.05" if r["p_value"] < 0.05 else "n.s."
        print(
            f"  {r['asset']:<5}  rho = {r['spearman_rho']:+.3f}  "
            f"p = {r['p_value']:.3f}  [{sign}; {sig}]"
        )

    # ----- Top 3 drift folds across assets -----
    print("\n[d] Three highest-drift folds across all assets (by max_psi):")
    pooled = pd.concat(fold_drift_dfs.values(), ignore_index=True)
    pooled = pooled[pooled["stress_def"] == "h10_d05"]
    top3 = pooled.sort_values("max_psi", ascending=False).head(3)
    for _, r in top3.iterrows():
        feats = json.loads(r["high_drift_features"])
        feats_str = ", ".join(feats[:5])
        if len(feats) > 5:
            feats_str += f", ... ({len(feats)-5} more)"
        print(
            f"  {r['asset'].upper()} fold {int(r['fold_id'])} "
            f"(test {r['test_start']} → {r['test_end']}):"
        )
        print(
            f"    max_psi = {r['max_psi']:.3f}  mean_psi = {r['mean_psi']:.3f}  "
            f"n_high_drift = {int(r['n_high_drift'])}"
        )
        print(f"    high_drift_features: {feats_str if feats else '(none)'}")

    # ----- Regime coincidence (visual sanity check) -----
    print("\n[e] Regime markers vs SPX mean_psi spikes (visual check; see "
          "fig_drift_timeline.pdf):")
    spx_h10 = fold_drift_dfs["spx"]
    spx_h10 = spx_h10[spx_h10["stress_def"] == "h10_d05"].copy()
    spx_h10["test_start_dt"] = pd.to_datetime(spx_h10["test_start"])
    spx_h10["test_end_dt"] = pd.to_datetime(spx_h10["test_end"])
    median_psi = float(spx_h10["mean_psi"].median())
    p75_psi = float(spx_h10["mean_psi"].quantile(0.75))
    for dt_str, label in REGIME_EVENTS:
        dt = pd.Timestamp(dt_str)
        # The fold whose test window contains this date.
        match = spx_h10[
            (spx_h10["test_start_dt"] <= dt) & (spx_h10["test_end_dt"] >= dt)
        ]
        if match.empty:
            print(f"  {label:<32}  (date outside SPX panel range)")
            continue
        m = match.iloc[0]
        spike_class = (
            "elevated" if m["mean_psi"] > p75_psi
            else "median-ish" if m["mean_psi"] > median_psi
            else "below median"
        )
        print(
            f"  {label:<32}  fold {int(m['fold_id']):>2} "
            f"(test {m['test_start']}): mean_psi = {m['mean_psi']:.3f}  "
            f"[{spike_class}]"
        )

    # ----- INVESTIGATE candidates -----
    print("\n[f] [INVESTIGATE] folds where the drift-vs-metric story breaks "
          "the methodology claim:")
    candidates: list[str] = []
    for asset, drift_df in fold_drift_dfs.items():
        sub = drift_df[drift_df["stress_def"] == "h10_d05"]
        per_fold = pd.read_csv(ASSETS[asset]["per_fold_path"])
        per_fold = per_fold[per_fold["stress_def"] == "h10_d05"]
        per_fold = per_fold.groupby("fold_id").agg(
            auc=("auc", "mean"),
            lift=("drawdown_lift", "mean"),
        ).reset_index()
        joined = sub.merge(per_fold, on="fold_id", how="inner")
        # Methodology claim: high drift -> degraded metrics.
        # Anomaly type 1: mean_psi very high (top quartile), but lift > median lift
        # Anomaly type 2: mean_psi very low (bottom quartile), but lift very low
        if len(joined) < 4:
            continue
        psi_q75 = float(joined["mean_psi"].quantile(0.75))
        psi_q25 = float(joined["mean_psi"].quantile(0.25))
        lift_med = float(joined["lift"].median())
        for _, r in joined.iterrows():
            if pd.isna(r["mean_psi"]) or pd.isna(r["lift"]):
                continue
            if r["mean_psi"] >= psi_q75 and r["lift"] > lift_med * 1.3:
                candidates.append(
                    f"  {asset.upper()} fold {int(r['fold_id'])}: "
                    f"high drift (mean_psi={r['mean_psi']:.3f}) "
                    f"but high lift ({r['lift']:.3f}) — drift didn't hurt"
                )
            elif r["mean_psi"] <= psi_q25 and r["lift"] < lift_med * 0.7:
                candidates.append(
                    f"  {asset.upper()} fold {int(r['fold_id'])}: "
                    f"low drift (mean_psi={r['mean_psi']:.3f}) "
                    f"but low lift ({r['lift']:.3f}) — drift didn't matter"
                )
    if candidates:
        for c in candidates[:10]:
            print(c)
        if len(candidates) > 10:
            print(f"  ... ({len(candidates) - 10} more)")
    else:
        print("  none")

    # ----- Artifacts -----
    print("\n[artifacts]")
    for fname in (
        "experiments/track_g_drift/fold_drift_spx.csv",
        "experiments/track_g_drift/fold_drift_ndx.csv",
        "experiments/track_g_drift/fold_drift_rut.csv",
        "experiments/track_g_drift/feature_drift_per_fold.csv",
        "outputs/track_g/table_drift_correlations.csv",
        "outputs/track_g/fig_drift_vs_metric.pdf",
        "outputs/track_g/fig_drift_timeline.pdf",
        "outputs/track_g/fig_feature_drift_heatmap.pdf",
    ):
        path = REPO_ROOT / fname
        if path.exists():
            kb = path.stat().st_size / 1024.0
            print(f"  {fname}  ({kb:.1f} KB)")
    print("=" * 78)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fold_drift_dfs: dict[str, pd.DataFrame] = {}
    per_fold_dfs: dict[str, pd.DataFrame] = {}

    t0 = time.time()
    for asset, paths in ASSETS.items():
        print(f"[load] {asset}")
        panel_df = pd.read_csv(paths["panel_path"], parse_dates=["Date"])
        features_df = pd.read_csv(paths["features_path"], parse_dates=["Date"])
        fold_def_df = pd.read_csv(paths["fold_def_path"])
        per_fold_df = pd.read_csv(paths["per_fold_path"])
        per_fold_dfs[asset] = per_fold_df
        print(f"  panel rows={len(panel_df)}  features={features_df.shape[1]-1}  "
              f"folds={fold_def_df['fold_id'].nunique()}  "
              f"per_fold rows={len(per_fold_df)}")

        df = compute_fold_drift_for_asset(
            asset, panel_df, features_df, fold_def_df, per_fold_df
        )
        fold_drift_dfs[asset] = df
        out_path = EXP_DIR / f"fold_drift_{asset}.csv"
        df.to_csv(out_path, index=False)
        print(f"  -> {out_path.relative_to(REPO_ROOT)}: {len(df)} rows")

    # SPX per-feature heatmap data.
    print("[step 3] SPX per-feature drift...")
    spx = ASSETS["spx"]
    panel_df = pd.read_csv(spx["panel_path"], parse_dates=["Date"])
    features_df = pd.read_csv(spx["features_path"], parse_dates=["Date"])
    fold_def_df = pd.read_csv(spx["fold_def_path"])
    feat_drift = compute_spx_feature_drift(panel_df, features_df, fold_def_df)
    feat_drift.to_csv(EXP_DIR / "feature_drift_per_fold.csv", index=False)
    print(f"  feature_drift_per_fold.csv: {len(feat_drift)} rows")

    # Step 4 — correlations.
    print("[step 4] correlations...")
    correlations = compute_correlations(fold_drift_dfs, per_fold_dfs)
    correlations.to_csv(OUT_DIR / "table_drift_correlations.csv", index=False)
    print(f"  table_drift_correlations.csv: {len(correlations)} rows")

    # Step 5 — figures.
    print("[step 5] figures...")
    fig_drift_vs_metric(
        fold_drift_dfs, per_fold_dfs,
        OUT_DIR / "fig_drift_vs_metric.pdf",
    )
    fig_drift_timeline(
        fold_drift_dfs["spx"],
        OUT_DIR / "fig_drift_timeline.pdf",
    )
    fig_feature_drift_heatmap(
        feat_drift, fold_def_df,
        OUT_DIR / "fig_feature_drift_heatmap.pdf",
    )

    elapsed = time.time() - t0
    print(f"\n[total wall time] {elapsed:.1f}s")

    print_status(fold_drift_dfs, correlations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
