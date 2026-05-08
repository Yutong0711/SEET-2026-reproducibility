"""Track D: crisis-period held-out evaluations.

Two single-fold crisis evaluations on SPX, designed to surface failure
mode F2 (silent classification degradation under regime shift):

  Crisis-2008  train_end 2007-12-31, test 2008-01-01 -> 2009-12-31
  Crisis-2020  train_end 2019-12-31, test 2020-01-01 -> 2020-12-31

The Crisis-2008 fold uses the spx_crisis_2008 feature set (11 features:
8 Group-1 on SPX + 3 Group-2 on VIX without pctile_252d) because the
spx_full / spx_core feature sets are unavailable at train_end=2007-12-31
(VIX9D / VVIX / VIX3M / pctile_252d all lack sufficient history).

Per the Track D D3 override, Crisis-2020 is run TWICE:
  - HEADLINE: spx_full (35 features) — same as Track A
  - ROBUSTNESS: spx_crisis_2008 (11 features) — symmetric with Crisis-2008

Both Crisis-2020 runs land as separate rows in
per_fold_metrics_crisis_2020.csv with a configuration column
distinguishing them.

Outputs:
  experiments/track_d_crisis/feature_availability.json
  experiments/track_d_crisis/per_fold_metrics_crisis_2008.csv
  experiments/track_d_crisis/per_fold_metrics_crisis_2020.csv
  experiments/track_d_crisis/fold_definitions_crisis_{2008,2020}.csv
  outputs/track_d/table_crisis.csv
  outputs/track_d/fig_crisis_scores.pdf
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from seet.baselines import (  # noqa: E402
    ALL_MODELS,
    DETERMINISTIC_MODELS,
    STOCHASTIC_MODELS,
    LightGbmTuned,
    make_model,
)
from seet.features import build_features  # noqa: E402
from seet.pipeline import (  # noqa: E402
    compute_stress_labels,
    run_grid,
)
from seet.run_track_a import (  # noqa: E402
    PRIMARY_SEED,
    SEEDS,
    STRESS_DEFS,
)


# =====================================================================
# Configuration
# =====================================================================

CRISES: list[dict] = [
    {
        "label": "crisis_2008",
        "panel_name": "spx_core_2007",
        "feature_set_id": "spx_crisis_2008",
        "is_headline": True,
        "is_recovery": False,
        "fold": {
            "fold_id": 1,
            "train_end": pd.Timestamp("2007-12-31"),
            "test_start": pd.Timestamp("2008-01-01"),
            "test_end": pd.Timestamp("2009-12-31"),
        },
    },
    {
        "label": "crisis_2020",
        "panel_name": "spx_extended_2011",
        "feature_set_id": "spx_full",
        "is_headline": True,
        "is_recovery": False,
        "fold": {
            "fold_id": 1,
            "train_end": pd.Timestamp("2019-12-31"),
            "test_start": pd.Timestamp("2020-01-01"),
            "test_end": pd.Timestamp("2020-12-31"),
        },
    },
    {
        "label": "crisis_2020",
        "panel_name": "spx_extended_2011",
        "feature_set_id": "spx_crisis_2008",
        "is_headline": False,
        "is_recovery": False,  # symmetric robustness check
        "fold": {
            "fold_id": 1,
            "train_end": pd.Timestamp("2019-12-31"),
            "test_start": pd.Timestamp("2020-01-01"),
            "test_end": pd.Timestamp("2020-12-31"),
        },
    },
    {
        # F4 RECOVERY: only the matrix-feature baselines (LR, LightGbm)
        # are re-fit. The four panel-DataFrame baselines already have
        # valid spx_full results (they don't touch the 35-feature
        # matrix), so we don't re-run them.
        "label": "crisis_2020",
        "panel_name": "spx_extended_2011",
        "feature_set_id": "spx_no_vvix",
        "is_headline": False,
        "is_recovery": True,
        "deterministic_models": ("LogisticRegressionL2",),
        "stochastic_models": ("LightGbmTuned",),
        "fold": {
            "fold_id": 1,
            "train_end": pd.Timestamp("2019-12-31"),
            "test_start": pd.Timestamp("2020-01-01"),
            "test_end": pd.Timestamp("2020-12-31"),
        },
    },
]

PRICE_COL = "SPX"
N_CALM_DAYS = 60       # last N training days = "calm pre-crisis baseline"
N_FIRST_REGIME = 30    # first N test days for the regime-shift split

EXP_DIR = REPO_ROOT / "experiments" / "track_d_crisis"
OUT_DIR = REPO_ROOT / "outputs" / "track_d"


# =====================================================================
# Loading helpers
# =====================================================================

def _load_panel(panel_name: str) -> pd.DataFrame:
    return pd.read_csv(
        REPO_ROOT / "data" / "processed" / f"{panel_name}.csv",
        parse_dates=["Date"],
    )


def _load_or_build_features(
    panel_df: pd.DataFrame, panel_name: str, feature_set_id: str
) -> pd.DataFrame:
    """For spx_full on spx_extended_2011 we use the existing committed
    feature CSV. For other (panel, feature_set_id) combinations we
    build features in-memory."""
    cached_path = (
        REPO_ROOT / "data" / "processed" / "features"
        / f"{panel_name}_features.csv"
    )
    standard_panel_to_set = {
        "spx_extended_2011": "spx_full",
        "spx_core_2007":     "spx_core",
        "ndx_2007":          "ndx_minimal",
        "rut_2009":          "rut_minimal",
    }
    if (
        cached_path.exists()
        and standard_panel_to_set.get(panel_name) == feature_set_id
    ):
        return pd.read_csv(cached_path, parse_dates=["Date"])
    features_df, _ = build_features(panel_df, feature_set_id)
    return features_df


# =====================================================================
# Feature-availability provenance
# =====================================================================

def _write_feature_availability_json() -> Path:
    payload = {
        "crisis_2008": {
            "panel": "spx_core_2007",
            "feature_set_id": "spx_crisis_2008",
            "n_features": 11,
            "train_end": "2007-12-31",
            "test_start": "2008-01-01",
            "test_end": "2009-12-31",
            "available_features": [
                "ret_1d", "ret_5d", "ret_10d", "ret_20d", "logret_1d",
                "rv_10d", "rv_20d", "drawdown_20d",
                "vix_chg_1d", "vix_chg_5d", "vix_pctchg_5d",
            ],
            "dropped_features": {
                "*_pctile_252d": (
                    "spx_core_2007 panel starts 2007-01-03; only ~251 "
                    "trading days at train_end=2007-12-31, leaving zero "
                    "valid rows after 252-day rolling-rank warmup."
                ),
                "vix3m_*": (
                    "VIX3M (originally VXV) launched 2007-12-04; only "
                    "~20 days of usable history at train_end."
                ),
                "vix9d_*": (
                    "VIX9D launched 2011-02-23; not available at "
                    "train_end=2007-12-31."
                ),
                "vix6m_*": (
                    "VIX6M launched 2008-01-02; not available at "
                    "train_end=2007-12-31."
                ),
                "vvix_*": (
                    "VVIX truncated to 2012-04-01 in the processed "
                    "layer; entirely NaN through Crisis-2008 window."
                ),
                "term_structure_*": (
                    "Group 3 features (vix3m_minus_vix, etc.) require "
                    "VIX3M/VIX9D/VIX6M which are unavailable."
                ),
            },
        },
        "crisis_2020": {
            "headline": {
                "panel": "spx_extended_2011",
                "feature_set_id": "spx_full",
                "n_features": 35,
                "train_end": "2019-12-31",
                "test_start": "2020-01-01",
                "test_end": "2020-12-31",
                "rationale": (
                    "spx_full is the full SPX vol family from Track A; "
                    "all 35 features have ample history at "
                    "train_end=2019-12-31."
                ),
            },
            "robustness_check": {
                "panel": "spx_extended_2011",
                "feature_set_id": "spx_crisis_2008",
                "n_features": 11,
                "train_end": "2019-12-31",
                "test_start": "2020-01-01",
                "test_end": "2020-12-31",
                "rationale": (
                    "Symmetric feature set with Crisis-2008 (11 features) "
                    "for cross-crisis comparability under D3 override."
                ),
            },
            "recovery_for_f4": {
                "panel": "spx_extended_2011",
                "feature_set_id": "spx_no_vvix",
                "n_features": 31,
                "train_end": "2019-12-31",
                "test_start": "2020-01-01",
                "test_end": "2020-12-31",
                "applies_to_models": [
                    "LogisticRegressionL2", "LightGbmTuned",
                ],
                "rationale": (
                    "F4 recovery: spx_full's LR and LightGbmTuned cells "
                    "are NaN under Crisis-2020 because vvix_pctile_252d "
                    "propagates documented VVIX gaps (2019-07-05, "
                    "2020-06-11) through the entire test window. "
                    "Removing the 4 VVIX-derived features eliminates "
                    "the cascade. The four panel-DataFrame baselines "
                    "(NaiveBaseRate, VIXPercentileRaw, "
                    "VIXPercentileCalibrated, HarRvThreshold) keep "
                    "their valid spx_full results — they do not read "
                    "the feature matrix and were unaffected by F4."
                ),
            },
        },
    }
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXP_DIR / "feature_availability.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return out_path


# =====================================================================
# Regime-shift split AUC (per (model, stress_def, seed))
# =====================================================================

def _add_regime_shift_aucs(
    per_fold_df: pd.DataFrame, predictions_by_def: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """For each (stress_def, model, seed) row, compute three AUC values:
    auc_full_window (entire test set), auc_first30 (first 30 OOS trading
    days), auc_rest (days 31+). Add the three columns and return."""
    if per_fold_df.empty:
        return per_fold_df

    aucs_first: list[float] = []
    aucs_rest: list[float] = []
    aucs_full: list[float] = []
    for _, row in per_fold_df.iterrows():
        sd_name = row["stress_def"]
        model = row["model"]
        seed = int(row["seed"])
        preds = predictions_by_def.get(sd_name)
        if preds is None or preds.empty:
            aucs_first.append(np.nan)
            aucs_rest.append(np.nan)
            aucs_full.append(np.nan)
            continue
        sub = preds[(preds["model"] == model) & (preds["seed"] == seed)]
        sub = sub.sort_values("Date").reset_index(drop=True)
        s = sub["score"].to_numpy(dtype=float)
        y = sub["label"].to_numpy(dtype=float)
        valid = ~(np.isnan(s) | np.isnan(y))
        s = s[valid]
        y = y[valid].astype(int)

        def _safe_auc(y_arr: np.ndarray, s_arr: np.ndarray) -> float:
            if y_arr.size == 0 or np.unique(y_arr).size < 2:
                return float("nan")
            return float(roc_auc_score(y_arr, s_arr))

        auc_full = _safe_auc(y, s)
        if y.size >= N_FIRST_REGIME + 5:
            auc_first = _safe_auc(y[:N_FIRST_REGIME], s[:N_FIRST_REGIME])
            auc_rest = _safe_auc(y[N_FIRST_REGIME:], s[N_FIRST_REGIME:])
        else:
            auc_first = float("nan")
            auc_rest = float("nan")
        aucs_first.append(auc_first)
        aucs_rest.append(auc_rest)
        aucs_full.append(auc_full)

    out = per_fold_df.copy()
    out["auc_full_window"] = aucs_full
    out["auc_first30"] = aucs_first
    out["auc_rest"] = aucs_rest
    return out


# =====================================================================
# Calm-period score + LightGBM diagnostics
# =====================================================================

def _model_kwargs(model_name: str) -> dict:
    return {}  # Track D uses Track A's defaults


def _check_f4(
    panel_df: pd.DataFrame,
    features_df: pd.DataFrame,
    fold: dict,
) -> dict:
    """Detect F4 (data-quality cascade rendering test set empty after
    row-drop) for a single (panel, features, fold) configuration.

    F4 fires when, in the test window of `fold`, every row has at least
    one NaN feature in `features_df`. Under the runner's drop-NaN-feature
    policy this means LR and LightGbmTuned cannot evaluate any test
    row, so all their metrics come back NaN. The four panel-DataFrame
    baselines (NaiveBaseRate, VIXPercentileRaw, VIXPercentileCalibrated,
    HarRvThreshold) are unaffected because they read panel columns
    directly, not the feature matrix.
    """
    feat_cols = [c for c in features_df.columns if c != "Date"]
    panel_dates = pd.to_datetime(panel_df["Date"])
    test_mask = (
        (panel_dates >= fold["test_start"])
        & (panel_dates <= fold["test_end"])
    )
    feat_test = features_df.loc[test_mask, feat_cols]
    feat_nan_any = feat_test.isna().any(axis=1)
    n_test_total = int(test_mask.sum())
    n_test_after = int((~feat_nan_any).sum())
    f4_fired = (n_test_total > 0) and (n_test_after == 0)
    always_nan_cols: list[str] = []
    if f4_fired:
        always_nan_cols = (
            feat_test.columns[feat_test.isna().all(axis=0)].tolist()
        )
    return {
        "f4_fired": f4_fired,
        "n_test_total": n_test_total,
        "n_test_after_dropna_features": n_test_after,
        "always_nan_features": always_nan_cols,
    }


def _collect_diagnostics(
    panel_df: pd.DataFrame,
    features_df: pd.DataFrame,
    fold: dict,
    models_to_run: list[str] | None = None,
    n_calm_days: int = N_CALM_DAYS,
) -> pd.DataFrame:
    """For each (stress_def, model, seed) cell:
      - fit the baseline on its training set,
      - compute calm_mean_score / calm_max_score on the last n_calm_days
        of training,
      - record n_train_rows_after_dropna (size of training set after
        dropping rows with any NaN feature),
      - record n_trees_built (LightGbm only; NaN otherwise).

    Deterministic baselines are fit once with seed=PRIMARY_SEED and
    their scalar diagnostics are replicated across all 5 seeds. The
    stochastic baseline (LightGbmTuned) is fit per seed (the
    placeholder HPs make it deterministic across seeds, so the values
    coincide, but we still produce one row per seed for downstream
    join compatibility).
    """
    panel_dates = pd.to_datetime(panel_df["Date"])
    feat_cols = [c for c in features_df.columns if c != "Date"]
    train_mask = (panel_dates <= fold["train_end"]).to_numpy()
    train_idx = np.where(train_mask)[0]
    feat_nan = features_df.iloc[train_idx][feat_cols].isna().any(axis=1).to_numpy()
    train_keep_local = ~feat_nan
    train_idx_kept = train_idx[train_keep_local]
    n_train_after = int(train_keep_local.sum())

    rows: list[dict] = []
    prices = panel_df[PRICE_COL].to_numpy(dtype=float)
    for sd in STRESS_DEFS:
        _, labels = compute_stress_labels(prices, sd["h"], sd["d"])
        label_nan = np.isnan(labels)
        train_label_mask = train_mask & ~label_nan
        train_idx_lab = np.where(train_label_mask)[0]
        feat_nan_lab = features_df.iloc[train_idx_lab][feat_cols].isna().any(axis=1).to_numpy()
        keep_lab = ~feat_nan_lab
        train_idx_lab_kept = train_idx_lab[keep_lab]
        n_train_after_lab = int(keep_lab.sum())

        models_iter = models_to_run if models_to_run is not None else ALL_MODELS
        if n_train_after_lab < 10:
            for model_name in models_iter:
                for seed in SEEDS:
                    rows.append({
                        "stress_def": sd["name"],
                        "model": model_name,
                        "seed": int(seed),
                        "n_train_rows_after_dropna": n_train_after_lab,
                        "n_trees_built": (
                            0 if model_name == "LightGbmTuned" else float("nan")
                        ),
                        "calm_mean_score": float("nan"),
                        "calm_max_score": float("nan"),
                    })
            continue

        X_tr = features_df.iloc[train_idx_lab_kept][feat_cols]
        y_tr = labels[train_idx_lab_kept]

        for model_name in models_iter:
            seeds_to_run = (
                (PRIMARY_SEED,) if model_name in DETERMINISTIC_MODELS else SEEDS
            )
            for seed in seeds_to_run:
                model = make_model(model_name, seed=seed)
                # NaiveBaseRate / VIX percentiles / HAR-RV are panel-DataFrame
                # baselines and need the panel slice to .fit. The matrix
                # baselines (LR, LightGbm) take the feature matrix.
                if model_name in ("LogisticRegressionL2", "LightGbmTuned"):
                    model.fit(X_tr, y_tr)
                    train_proba = model.predict_proba(X_tr)[:, 1]
                else:
                    panel_train = panel_df.iloc[train_idx_lab_kept]
                    model.fit(panel_train, y_tr)
                    train_proba = model.predict_proba(panel_train)[:, 1]

                n_trees = float("nan")
                if model_name == "LightGbmTuned":
                    if isinstance(model, LightGbmTuned) and model.model_ is not None:
                        try:
                            n_trees = int(model.model_.booster_.num_trees())
                        except Exception:
                            n_trees = float("nan")
                    else:
                        n_trees = 0  # single-class fallback => no trees

                # Calm window: last n_calm_days of valid training rows.
                if train_proba.size >= n_calm_days:
                    calm = train_proba[-n_calm_days:]
                else:
                    calm = train_proba
                calm_valid = calm[~np.isnan(calm)]
                if calm_valid.size:
                    calm_mean = float(calm_valid.mean())
                    calm_max = float(calm_valid.max())
                else:
                    calm_mean = float("nan")
                    calm_max = float("nan")

                row = {
                    "stress_def": sd["name"],
                    "model": model_name,
                    "seed": int(seed),
                    "n_train_rows_after_dropna": n_train_after_lab,
                    "n_trees_built": n_trees,
                    "calm_mean_score": calm_mean,
                    "calm_max_score": calm_max,
                }
                rows.append(row)
                # Replicate deterministic results across seed slots.
                if model_name in DETERMINISTIC_MODELS:
                    for replicate_seed in [s for s in SEEDS if s != PRIMARY_SEED]:
                        row_copy = {**row, "seed": int(replicate_seed)}
                        rows.append(row_copy)
    return pd.DataFrame(rows)


# =====================================================================
# F2 detection
# =====================================================================

def _f2_cells(per_fold_df: pd.DataFrame) -> pd.DataFrame:
    """F2 cells deduped to unique (crisis, configuration, stress_def,
    model) triples.

    Track D uses placeholder LightGbm hyperparameters and replicates
    deterministic-baseline rows across the 5 seed slots. That makes
    each (model, crisis, stress_def) cell appear up to 5 times in the
    raw per_fold_df with bit-identical AUC and lift values; counting
    those as 5 independent F2 firings would 5x-inflate the F2 count.
    Here we collapse to one row per unique triple and report
    `seed_replicates` (the number of seeds the cell fired on, between
    1 and 5) so the underlying replication structure is visible."""
    df = per_fold_df.dropna(subset=["auc", "drawdown_lift"])
    cond = (
        ((df["auc"] < 0.5) & (df["drawdown_lift"] > 1.0))
        | ((df["auc"] > 0.5) & (df["drawdown_lift"] < 1.0))
    )
    f2 = df[cond]
    if f2.empty:
        return pd.DataFrame(
            columns=[
                "crisis", "configuration", "stress_def", "model",
                "seed_replicates", "auc", "drawdown_lift",
            ]
        )
    grouped = (
        f2.groupby(
            ["crisis", "configuration", "stress_def", "model"],
            as_index=False,
        )
        .agg(
            seed_replicates=("seed", "nunique"),
            auc=("auc", "mean"),
            drawdown_lift=("drawdown_lift", "mean"),
        )
        .sort_values(
            ["crisis", "configuration", "stress_def", "model"]
        )
        .reset_index(drop=True)
    )
    return grouped


# =====================================================================
# table_crisis.csv (one tidy row per cell)
# =====================================================================

def _build_crisis_table(per_fold_2008: pd.DataFrame, per_fold_2020: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "crisis", "configuration", "stress_def", "model", "seed",
        "n_train", "n_test",
        "auc", "auc_full_window", "auc_first30", "auc_rest",
        "pr_auc", "brier", "ece",
        "alarm_rate", "alarm_precision", "drawdown_lift", "severity_lift",
        "n_alarms", "n_events",
        "n_train_rows_after_dropna", "n_trees_built",
        "calm_mean_score", "calm_max_score",
    ]
    combined = pd.concat([per_fold_2008, per_fold_2020], ignore_index=True)
    keep = [c for c in cols if c in combined.columns]
    return combined[keep].copy()


# =====================================================================
# Figure: stress score timelines, h10_d05, headline runs only
# =====================================================================

def _plot_crisis_scores(
    runs_for_figure: list[dict],  # subset of CRISES (headline only)
    per_fold_by_runkey: dict[tuple[str, str], pd.DataFrame],
    predictions_by_runkey: dict[tuple[str, str], dict[str, pd.DataFrame]],
    panels_by_runkey: dict[tuple[str, str], pd.DataFrame],
    out_path: Path,
    target_sd: str = "h10_d05",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_models = len(ALL_MODELS)
    fig, axes = plt.subplots(
        n_models, len(runs_for_figure),
        figsize=(7.0 * len(runs_for_figure), 2.0 * n_models),
        sharex="col", sharey=True,
    )
    if n_models == 1:
        axes = np.array([axes])
    if len(runs_for_figure) == 1:
        axes = axes.reshape(-1, 1)

    for col, run in enumerate(runs_for_figure):
        run_key = (run["label"], run["feature_set_id"])
        panel = panels_by_runkey[run_key]
        preds = predictions_by_runkey[run_key].get(target_sd)
        if preds is None or preds.empty:
            for ax in axes[:, col]:
                ax.text(0.5, 0.5, "no predictions",
                        ha="center", transform=ax.transAxes)
            continue
        per_fold = per_fold_by_runkey[run_key]
        per_fold_sd = per_fold[per_fold["stress_def"] == target_sd]

        # SPX drawdown over the test window
        train_end = pd.Timestamp(run["fold"]["train_end"])
        test_start = pd.Timestamp(run["fold"]["test_start"])
        test_end = pd.Timestamp(run["fold"]["test_end"])
        panel_dates = pd.to_datetime(panel["Date"])
        test_mask = (panel_dates >= test_start) & (panel_dates <= test_end)
        test_dates = panel_dates[test_mask].reset_index(drop=True)
        test_prices = panel.loc[test_mask, PRICE_COL].to_numpy(dtype=float)
        if test_prices.size == 0:
            continue
        rolling_max = np.maximum.accumulate(test_prices)
        drawdown = test_prices / rolling_max - 1.0  # <= 0

        # Stress event days (label == 1)
        # Pull labels from any model's predictions
        any_model = ALL_MODELS[0]
        sub = preds[(preds["model"] == any_model) & (preds["seed"] == PRIMARY_SEED)]
        sub = sub.sort_values("Date").reset_index(drop=True)
        sub["Date"] = pd.to_datetime(sub["Date"])
        event_dates = sub.loc[sub["label"] == 1, "Date"]

        for row_idx, model in enumerate(ALL_MODELS):
            ax = axes[row_idx, col]
            mp = preds[(preds["model"] == model) & (preds["seed"] == PRIMARY_SEED)]
            mp = mp.sort_values("Date").reset_index(drop=True)
            mp["Date"] = pd.to_datetime(mp["Date"])
            score_dates = mp["Date"]
            scores = mp["score"].to_numpy(dtype=float)

            # Background shading for stress event days
            for d in event_dates:
                ax.axvline(d, color="red", alpha=0.10, linewidth=0.6, zorder=1)

            # Stress score line
            ax.plot(
                score_dates, scores,
                color="#1f77b4", linewidth=1.0, zorder=3,
                label="stress score",
            )

            # Threshold (median across rows for this stress_def + model)
            thr_vals = per_fold_sd.loc[per_fold_sd["model"] == model, "threshold"]
            if not thr_vals.empty:
                thr = float(thr_vals.median())
                if not np.isnan(thr):
                    ax.axhline(
                        thr, color="black", linestyle="--",
                        linewidth=0.7, zorder=4,
                        label="train 95p threshold",
                    )

            # Drawdown overlay on twin axis
            ax2 = ax.twinx()
            ax2.plot(
                test_dates, drawdown,
                color="gray", linewidth=0.8, alpha=0.6, zorder=2,
            )
            ax2.set_ylim(min(drawdown.min(), -0.05) * 1.05, 0.02)
            ax2.set_yticks([0, -0.1, -0.2, -0.3, -0.4, -0.5])
            if col == len(runs_for_figure) - 1:
                ax2.set_ylabel("SPX drawdown", color="gray", fontsize=8)
            ax2.tick_params(axis="y", labelsize=7, colors="gray")

            ax.set_ylim(0.0, 1.0)
            if col == 0:
                ax.set_ylabel(model, fontsize=8)
            if row_idx == 0:
                ax.set_title(
                    f"{run['label']} ({run['feature_set_id']})", fontsize=10
                )
            ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(linestyle=":", linewidth=0.4, axis="y")

    fig.suptitle(
        f"Track D crisis scores ({target_sd}; headline configurations) — "
        f"red bands = stress event days; gray = SPX drawdown",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


# =====================================================================
# STATUS
# =====================================================================

def _print_status(
    per_fold_2008: pd.DataFrame,
    per_fold_2020: pd.DataFrame,
    runs: list[dict],
    f4_by_runkey: dict[tuple[str, str], dict],
) -> None:
    print()
    print("===== TRACK D STATUS =====")

    headline = pd.concat([
        per_fold_2008,
        per_fold_2020[per_fold_2020["configuration"] == "spx_full"],
    ], ignore_index=True)
    robustness = per_fold_2020[
        per_fold_2020["configuration"] == "spx_crisis_2008"
    ]
    recovery = per_fold_2020[
        per_fold_2020["configuration"] == "spx_no_vvix"
    ]

    # Per (model, crisis, stress_def) headline cells, averaging seeds.
    print("\nHeadline per (model, crisis, stress_def) — averaging seeds:")
    grouped = (
        headline.groupby(
            ["crisis", "configuration", "stress_def", "model"], as_index=False
        ).agg(
            auc_full_window=("auc_full_window", "mean"),
            auc_first30=("auc_first30", "mean"),
            auc_rest=("auc_rest", "mean"),
            drawdown_lift=("drawdown_lift", "mean"),
            alarm_rate=("alarm_rate", "mean"),
            calm_mean_score=("calm_mean_score", "mean"),
            n_train_rows_after_dropna=("n_train_rows_after_dropna", "first"),
            n_trees_built=("n_trees_built", "mean"),
        )
    )
    grouped = grouped.sort_values(["crisis", "stress_def", "model"]).reset_index(drop=True)

    cur_key: tuple = ()
    for _, r in grouped.iterrows():
        key = (r["crisis"], r["stress_def"])
        if key != cur_key:
            print(f"\n[{r['crisis']}  {r['stress_def']}]")
            cur_key = key
        n_trees = (
            "—" if pd.isna(r["n_trees_built"]) else f"{int(r['n_trees_built'])}"
        )
        print(
            f"  {r['model']:<26}  "
            f"AUC_full={r['auc_full_window']:.3f}  "
            f"AUC_first30={r['auc_first30']:.3f}  "
            f"AUC_rest={r['auc_rest']:.3f}  "
            f"lift={r['drawdown_lift']:.3f}  "
            f"alarm_rate={r['alarm_rate']:.3f}  "
            f"calm={r['calm_mean_score']:.3f}  "
            f"n_train={int(r['n_train_rows_after_dropna']):>4}  "
            f"n_trees={n_trees}"
        )

    # F2 detection — deduped to unique (crisis, configuration, stress_def, model)
    f2 = _f2_cells(headline)
    n_unique = len(f2)
    print(
        f"\nF2 cells deduped to unique (crisis, configuration, stress_def, "
        f"model) triples: {n_unique}"
    )
    print(
        f"  rule: AUC<0.5 with lift>1, OR AUC>0.5 with lift<1  "
        f"(applied to point estimates; deterministic-baseline replication "
        f"across 5 seeds is collapsed to 1 row per cell with seed_replicates "
        f"recording the count)."
    )
    if f2.empty:
        print("  none")
    else:
        for _, r in f2.iterrows():
            print(
                f"  {r['crisis']:<13}  {r['configuration']:<18}  "
                f"{r['stress_def']:<8}  {r['model']:<26}  "
                f"seed_replicates={int(r['seed_replicates'])}  "
                f"auc={r['auc']:.3f}  lift={r['drawdown_lift']:.3f}"
            )

    # F4 detection (data-quality cascade rendering test set empty
    # after row-drop)
    print(
        "\nF4 cells (data-quality cascade rendering test set empty after "
        "feature-NaN drop):"
    )
    f4_runs_fired = [
        (run, f4_by_runkey[(run["label"], run["feature_set_id"])])
        for run in runs
        if f4_by_runkey[(run["label"], run["feature_set_id"])]["f4_fired"]
    ]
    if not f4_runs_fired:
        print("  none across all 4 Track D run configurations.")
    else:
        for run, f4 in f4_runs_fired:
            print(
                f"  {run['label']} ({run['feature_set_id']}): "
                f"X_te empty after dropna  "
                f"({f4['n_test_after_dropna_features']}/"
                f"{f4['n_test_total']} test rows valid)"
            )
            print(
                f"    Always-NaN test features (root cause): "
                f"{f4['always_nan_features']}"
            )
            print(
                f"    Affected baselines: LogisticRegressionL2, "
                f"LightGbmTuned (matrix-feature models). "
                f"NaiveBaseRate / VIXPercentile* / HarRvThreshold are "
                f"unaffected (panel-DataFrame baselines, do not read "
                f"the feature matrix)."
            )

    # F4 recovery: Crisis-2020 spx_no_vvix
    print(
        "\nspx_no_vvix recovery configuration (Crisis-2020 only; "
        "LR + LightGbm only; the 4 panel-DataFrame baselines retain "
        "their valid spx_full results):"
    )
    if recovery.empty:
        print("  No recovery configuration data.")
    else:
        rec_grouped = (
            recovery.groupby(["stress_def", "model"], as_index=False)
            .agg(
                auc_full_window=("auc_full_window", "mean"),
                auc_first30=("auc_first30", "mean"),
                auc_rest=("auc_rest", "mean"),
                drawdown_lift=("drawdown_lift", "mean"),
                alarm_rate=("alarm_rate", "mean"),
                n_trees_built=("n_trees_built", "mean"),
                n_train_rows_after_dropna=("n_train_rows_after_dropna", "first"),
            )
            .sort_values(["stress_def", "model"])
        )
        cur_sd = ""
        for _, r in rec_grouped.iterrows():
            if r["stress_def"] != cur_sd:
                print(f"\n  [{r['stress_def']}]")
                cur_sd = r["stress_def"]
            n_trees = (
                "—" if pd.isna(r["n_trees_built"])
                else f"{int(r['n_trees_built'])}"
            )
            print(
                f"    {r['model']:<26}  "
                f"AUC_full={r['auc_full_window']:.3f}  "
                f"AUC_first30={r['auc_first30']:.3f}  "
                f"AUC_rest={r['auc_rest']:.3f}  "
                f"lift={r['drawdown_lift']:.3f}  "
                f"alarm_rate={r['alarm_rate']:.3f}  "
                f"n_train={int(r['n_train_rows_after_dropna']):>4}  "
                f"n_trees={n_trees}"
            )

    # Robustness check (Crisis-2020 spx_crisis_2008)
    print("\nRobustness check (Crisis-2020 with spx_crisis_2008 11-feature set):")
    rob_grouped = (
        robustness.groupby(
            ["stress_def", "model"], as_index=False
        ).agg(
            auc_full_window=("auc_full_window", "mean"),
            drawdown_lift=("drawdown_lift", "mean"),
        ).sort_values(["stress_def", "model"])
    )
    cur_sd = ""
    for _, r in rob_grouped.iterrows():
        if r["stress_def"] != cur_sd:
            print(f"\n  [{r['stress_def']}]")
            cur_sd = r["stress_def"]
        print(
            f"    {r['model']:<26}  "
            f"AUC={r['auc_full_window']:.3f}  lift={r['drawdown_lift']:.3f}"
        )

    # Qualitative narrative
    print("\nQualitative narrative (raw material for paper text — bullets only):")
    for crisis_label, sub in headline.groupby("crisis"):
        print(f"\n  ## {crisis_label}")
        for model in ALL_MODELS:
            mb = sub[sub["model"] == model]
            if mb.empty:
                continue
            calm_mean = float(mb["calm_mean_score"].mean())
            calm_max = float(mb["calm_max_score"].mean())
            auc_full = float(mb["auc_full_window"].mean())
            auc_first = float(mb["auc_first30"].mean())
            auc_rest = float(mb["auc_rest"].mean())
            lift = float(mb["drawdown_lift"].mean())
            alarm_rate = float(mb["alarm_rate"].mean())
            alarm_prec = float(mb["alarm_precision"].mean())
            n_trees = mb["n_trees_built"].dropna()
            n_trees_str = (
                f"{int(n_trees.mean())}" if not n_trees.empty else "—"
            )
            f2_fired = (
                ((auc_full < 0.5) and (lift > 1.0))
                or ((auc_full > 0.5) and (lift < 1.0))
            )
            print(f"    - {model}")
            print(
                f"        calm pre-crisis mean score: {calm_mean:.3f}  "
                f"(max in last {N_CALM_DAYS} train days: {calm_max:.3f})"
            )
            print(
                f"        AUC: full {auc_full:.3f}  first-30 {auc_first:.3f}  "
                f"rest {auc_rest:.3f}"
                + (
                    "  [regime shift: first-30 weaker than rest by "
                    f"{auc_rest - auc_first:+.3f}]"
                    if not np.isnan(auc_first - auc_rest) else ""
                )
            )
            print(
                f"        operational: lift {lift:.3f}  alarm_rate {alarm_rate:.3f}  "
                f"alarm_precision {alarm_prec:.3f}"
            )
            if model == "LightGbmTuned":
                print(f"        LightGBM trees built: {n_trees_str}")
            print(
                f"        F2: {'fires' if f2_fired else 'does not fire'}"
            )
    print("\n==========================")


# =====================================================================
# Main
# =====================================================================

def _run_one(run: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """Returns (per_fold_df, fold_def_df, predictions_by_def, panel_df, f4_info)."""
    label = run["label"]
    fset = run["feature_set_id"]
    panel_name = run["panel_name"]
    deterministic = tuple(run.get("deterministic_models", DETERMINISTIC_MODELS))
    stochastic = tuple(run.get("stochastic_models", STOCHASTIC_MODELS))
    models_active = list(deterministic) + list(stochastic)
    is_recovery = bool(run.get("is_recovery", False))
    tag = " [RECOVERY]" if is_recovery else ""
    print(
        f"\n[Run]{tag} {label}  panel={panel_name}  feature_set={fset}  "
        f"models={models_active}"
    )

    panel = _load_panel(panel_name)
    features = _load_or_build_features(panel, panel_name, fset)
    if not panel["Date"].equals(features["Date"]):
        raise ValueError(
            f"{panel_name} / {fset}: panel and features Date columns "
            f"are not aligned"
        )

    # F4 detection — runs always, before run_grid, so we know whether
    # to expect NaN matrix-feature metrics.
    f4 = _check_f4(panel, features, run["fold"])
    if f4["f4_fired"]:
        print(
            f"[Run] *** F4 detected: test set empty after dropna "
            f"({f4['n_test_after_dropna_features']}/{f4['n_test_total']} "
            f"rows valid). Always-NaN features: {f4['always_nan_features']}"
        )
    else:
        print(
            f"[Run] F4 check passed: "
            f"{f4['n_test_after_dropna_features']}/{f4['n_test_total']} "
            f"test rows valid after dropna"
        )

    workdir = EXP_DIR / f"_workdir_{label}_{fset}"
    workdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    per_fold_df, fold_def_df, preds = run_grid(
        panel, features, workdir,
        price_col=PRICE_COL,
        stress_defs=STRESS_DEFS,
        seeds=SEEDS,
        primary_seed=PRIMARY_SEED,
        initial_train_end=run["fold"]["train_end"],  # ignored under folds_override
        deterministic_models=deterministic,
        stochastic_models=stochastic,
        save_predictions=True,
        folds_override=[run["fold"]],
    )
    elapsed = time.time() - t0
    print(
        f"[Run] grid done in {elapsed:.1f}s; "
        f"per_fold rows = {len(per_fold_df)}"
    )

    per_fold_df["crisis"] = label
    per_fold_df["configuration"] = fset
    per_fold_df["is_recovery"] = is_recovery

    # Augment with regime-shift AUCs and diagnostics
    per_fold_df = _add_regime_shift_aucs(per_fold_df, preds)
    diag = _collect_diagnostics(
        panel, features, run["fold"], models_to_run=models_active
    )
    per_fold_df = per_fold_df.merge(
        diag, on=["stress_def", "model", "seed"], how="left"
    )
    return per_fold_df, fold_def_df, preds, panel, f4


def main() -> int:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feat_avail_path = _write_feature_availability_json()
    print(f"[Setup] feature_availability.json -> {feat_avail_path}")

    per_fold_by_runkey: dict[tuple[str, str], pd.DataFrame] = {}
    fold_def_by_runkey: dict[tuple[str, str], pd.DataFrame] = {}
    predictions_by_runkey: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    panels_by_runkey: dict[tuple[str, str], pd.DataFrame] = {}
    f4_by_runkey: dict[tuple[str, str], dict] = {}

    for run in CRISES:
        run_key = (run["label"], run["feature_set_id"])
        per_fold_df, fold_def_df, preds, panel, f4 = _run_one(run)
        per_fold_by_runkey[run_key] = per_fold_df
        fold_def_by_runkey[run_key] = fold_def_df
        predictions_by_runkey[run_key] = preds
        panels_by_runkey[run_key] = panel
        f4_by_runkey[run_key] = f4

    # Combine per-fold metrics per crisis (Crisis-2020 has THREE
    # configurations: spx_full headline, spx_crisis_2008 symmetric
    # robustness, spx_no_vvix recovery for the F4 finding).
    per_fold_2008 = per_fold_by_runkey[("crisis_2008", "spx_crisis_2008")]
    per_fold_2020_full = per_fold_by_runkey[("crisis_2020", "spx_full")]
    per_fold_2020_crisis = per_fold_by_runkey[("crisis_2020", "spx_crisis_2008")]
    per_fold_2020_recovery = per_fold_by_runkey[("crisis_2020", "spx_no_vvix")]
    per_fold_2020 = pd.concat(
        [per_fold_2020_full, per_fold_2020_crisis, per_fold_2020_recovery],
        ignore_index=True,
    )

    per_fold_2008.to_csv(EXP_DIR / "per_fold_metrics_crisis_2008.csv", index=False)
    per_fold_2020.to_csv(EXP_DIR / "per_fold_metrics_crisis_2020.csv", index=False)
    fold_def_by_runkey[("crisis_2008", "spx_crisis_2008")].to_csv(
        EXP_DIR / "fold_definitions_crisis_2008.csv", index=False
    )
    fold_def_by_runkey[("crisis_2020", "spx_full")].to_csv(
        EXP_DIR / "fold_definitions_crisis_2020.csv", index=False
    )
    print(f"[Persist] {EXP_DIR / 'per_fold_metrics_crisis_2008.csv'}")
    print(f"[Persist] {EXP_DIR / 'per_fold_metrics_crisis_2020.csv'}")

    table = _build_crisis_table(per_fold_2008, per_fold_2020)
    table.to_csv(OUT_DIR / "table_crisis.csv", index=False)
    print(f"[Persist] {OUT_DIR / 'table_crisis.csv'}")

    # F2 cells deduped (one row per unique (crisis, configuration,
    # stress_def, model)) with seed_replicates count and mean metrics.
    f2_dedup_headline = _f2_cells(
        pd.concat([
            per_fold_2008,
            per_fold_2020[per_fold_2020["configuration"] == "spx_full"],
        ], ignore_index=True)
    )
    f2_dedup_headline.to_csv(OUT_DIR / "f2_cells.csv", index=False)
    print(f"[Persist] {OUT_DIR / 'f2_cells.csv'}")

    runs_for_figure = [c for c in CRISES if c["is_headline"]]
    _plot_crisis_scores(
        runs_for_figure,
        per_fold_by_runkey,
        predictions_by_runkey,
        panels_by_runkey,
        OUT_DIR / "fig_crisis_scores.pdf",
        target_sd="h10_d05",
    )
    print(f"[Persist] {OUT_DIR / 'fig_crisis_scores.pdf'}")

    # Cleanup workdirs (predictions parquets are intermediate; not committed)
    for run in CRISES:
        wd = EXP_DIR / f"_workdir_{run['label']}_{run['feature_set_id']}"
        if wd.exists():
            shutil.rmtree(wd, ignore_errors=True)

    _print_status(per_fold_2008, per_fold_2020, CRISES, f4_by_runkey)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track D crisis-period evaluation")
    parser.parse_args()
    sys.exit(main())
