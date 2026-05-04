"""Track A runner — headline grid, four-layer evaluation, bootstrap CIs,
paired Wilcoxon tests.

Usage from the repo root:

    python src/seet/run_track_a.py --smoke
    python src/seet/run_track_a.py --full

The --smoke path generates a tiny synthetic panel and runs one fold for
one stress_def to verify the runner end-to-end. The --full path runs the
configured grid against data/processed/spx_extended_2011.csv and the
matching feature CSV, writing per-fold metrics, predictions, aggregated
tables, three figures, and a STATUS block.

Imports only seet.{features,stats,baselines}, never depending on
external state.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from seet.baselines import (  # noqa: E402
    ALL_MODELS,
    DETERMINISTIC_MODELS,
    STOCHASTIC_MODELS,
    make_model,
)
from seet.stats import (  # noqa: E402
    block_bootstrap_ci,
    pairwise_table,
    paired_wilcoxon,
)


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =====================================================================
# Track A configuration
# =====================================================================

PANEL_NAME = "spx_extended_2011"
FEATURE_SET_ID = "spx_full"
PRICE_COL = "SPX"
VIX_COL = "VIX"

STRESS_DEFS = [
    {"name": "h5_d03",  "h": 5,  "d": 0.03},
    {"name": "h10_d05", "h": 10, "d": 0.05},
    {"name": "h20_d07", "h": 20, "d": 0.07},
]

INITIAL_TRAIN_END = pd.Timestamp("2014-12-31")
TEST_WINDOW_MONTHS = 6
STEP_MONTHS = 6

SEEDS = (42, 43, 44, 45, 46)
PRIMARY_SEED = 42

ECE_BINS = 10
THRESHOLD_PERCENTILE = 95.0
N_BOOT = 1000
ALPHA = 0.05
BOOTSTRAP_SEED = 42


# =====================================================================
# Stress-label and fold construction
# =====================================================================

def compute_stress_labels(prices: np.ndarray, h: int, d: float) -> tuple[np.ndarray, np.ndarray]:
    """For each row t, compute future_drawdown(t, h) and binary stress
    label. The last h rows have NaN for both since the forward window
    extends past data."""
    prices = np.asarray(prices, dtype=float)
    n = prices.size
    future_drawdown = np.full(n, np.nan)
    label = np.full(n, np.nan)
    for t in range(n - h):
        future = prices[t + 1: t + h + 1] / prices[t] - 1.0
        future_drawdown[t] = float(future.min())
        label[t] = 1.0 if future_drawdown[t] <= -d else 0.0
    return future_drawdown, label


def build_folds(panel_dates: pd.Series) -> list[dict]:
    """Return the expanding-window folds. Stops when the next test
    window would extend past the panel end (so the last fold's
    test_end <= panel_end)."""
    panel_end = pd.Timestamp(panel_dates.iloc[-1])
    folds: list[dict] = []
    train_end = INITIAL_TRAIN_END
    k = 1
    while True:
        test_start = train_end + pd.Timedelta(days=1)
        test_end = train_end + pd.DateOffset(months=TEST_WINDOW_MONTHS)
        if test_end > panel_end:
            break
        folds.append(
            {
                "fold_id": k,
                "train_end": pd.Timestamp(train_end),
                "test_start": pd.Timestamp(test_start),
                "test_end": pd.Timestamp(test_end),
            }
        )
        k += 1
        train_end = train_end + pd.DateOffset(months=STEP_MONTHS)
    return folds


# =====================================================================
# Metric helpers
# =====================================================================

def _filter_valid(scores: np.ndarray, labels: np.ndarray):
    valid = ~(np.isnan(scores) | np.isnan(labels))
    return scores[valid], labels[valid].astype(int), valid


def compute_ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = ECE_BINS) -> float:
    if scores.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(scores, bins[1:-1]), 0, n_bins - 1)
    n = scores.size
    ece = 0.0
    for b in range(n_bins):
        m = bin_idx == b
        if m.any():
            ece += (m.sum() / n) * abs(scores[m].mean() - labels[m].mean())
    return float(ece)


def compute_layer1(scores: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    s, y, _ = _filter_valid(scores, labels)
    if y.size == 0:
        return {"auc": np.nan, "pr_auc": np.nan, "brier": np.nan, "ece": np.nan}
    if np.unique(y).size < 2:
        auc = np.nan
        prauc = np.nan
    else:
        auc = float(roc_auc_score(y, s))
        prauc = float(average_precision_score(y, s))
    brier = float(brier_score_loss(y, s))
    ece = compute_ece(s, y, ECE_BINS)
    return {"auc": auc, "pr_auc": prauc, "brier": brier, "ece": ece}


def compute_layer2(
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    train_scores: np.ndarray,
    test_future_drawdown: np.ndarray,
) -> dict:
    """Layer 2 with strict-> threshold = 95th percentile of training scores.

    drawdown_lift and severity_lift are MAGNITUDE comparisons — absolute
    values in both numerator and denominator, per the paper's
    methodology section:

        drawdown_lift = |mean(fd over alarm days)| / |mean(fd over OOS)|
        severity_lift = |mean(fd over alarm ∩ event days)|
                        / |mean(fd over event days)|

    Both are always non-negative. lift > 1 means alarms have
    larger-magnitude drawdowns than the OOS / event-day baseline.

    Alarm convention is strict-> (D5): an OOS row is an alarm iff its
    score is strictly greater than the threshold.

    Returns alarm rate, alarm precision, drawdown lift, severity lift,
    plus the raw counts used. NaN for any cell that becomes 0/0.
    """
    train_valid = train_scores[~np.isnan(train_scores)]
    if train_valid.size == 0:
        threshold = np.nan
    else:
        threshold = float(np.percentile(train_valid, THRESHOLD_PERCENTILE))

    s, y, valid_mask = _filter_valid(test_scores, test_labels)
    fd = test_future_drawdown[valid_mask]
    n_test = s.size

    base = {
        "threshold": threshold,
        "alarm_rate": np.nan,
        "alarm_precision": np.nan,
        "drawdown_lift": np.nan,
        "severity_lift": np.nan,
        "n_alarms": 0,
        "n_events": 0,
        "n_test_eff": n_test,
    }
    if np.isnan(threshold) or n_test == 0:
        return base

    alarms = s > threshold
    n_alarms = int(alarms.sum())
    events = y == 1
    n_events = int(events.sum())

    base["alarm_rate"] = float(n_alarms / n_test)
    base["n_alarms"] = n_alarms
    base["n_events"] = n_events

    if n_alarms > 0:
        base["alarm_precision"] = float((alarms & events).sum() / n_alarms)
        denom = float(np.nanmean(fd))
        if denom != 0 and not np.isnan(denom):
            base["drawdown_lift"] = float(
                abs(np.nanmean(fd[alarms])) / abs(denom)
            )

    if n_events > 0:
        alarm_event = alarms & events
        if alarm_event.any():
            event_mean_fd = float(np.nanmean(fd[events]))
            if event_mean_fd != 0 and not np.isnan(event_mean_fd):
                base["severity_lift"] = float(
                    abs(np.nanmean(fd[alarm_event])) / abs(event_mean_fd)
                )
    return base


def bootstrap_std_ci(
    values: Iterable[float],
    n_boot: int = N_BOOT,
    alpha: float = ALPHA,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Bootstrap CI for the sample std (ddof=1). Used for Layer 4."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n < 2:
        return {"std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_valid": int(n)}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = arr[idx].std(ddof=1)
    return {
        "std": float(arr.std(ddof=1)),
        "ci_low": float(np.percentile(boot, 100.0 * alpha / 2.0)),
        "ci_high": float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0))),
        "n_valid": int(n),
    }


# =====================================================================
# Per-fold model fit + predict
# =====================================================================

def _select_dataframe(model_name: str, panel_df: pd.DataFrame, features_df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Pick the correct input table for a model name."""
    if model_name in ("LogisticRegressionL2", "LightGbmTuned"):
        cols = [c for c in features_df.columns if c != "Date"]
        return features_df.loc[mask, cols]
    return panel_df.loc[mask]


def _fit_predict_one(
    model_name: str,
    seed: int,
    panel_df: pd.DataFrame,
    features_df: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one model on the train mask, return (train_scores, test_scores)
    aligned to (train_mask, test_mask) row sets respectively. NaN scores
    are returned where the model cannot evaluate (e.g. NaN features for
    LR/LightGBM)."""
    feat_cols = [c for c in features_df.columns if c != "Date"]
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    n_train_full = train_idx.size
    n_test_full = test_idx.size

    train_scores = np.full(n_train_full, np.nan)
    test_scores = np.full(n_test_full, np.nan)

    if model_name in ("LogisticRegressionL2", "LightGbmTuned"):
        # Drop NaN-feature rows from train; predict-NaN on test rows with NaN features.
        train_feat_nan = features_df.iloc[train_idx][feat_cols].isna().any(axis=1).to_numpy()
        test_feat_nan = features_df.iloc[test_idx][feat_cols].isna().any(axis=1).to_numpy()
        train_keep_local = ~train_feat_nan
        if train_keep_local.sum() < 10:
            return train_scores, test_scores  # all NaN
        X_tr = features_df.iloc[train_idx[train_keep_local]][feat_cols]
        y_tr = labels[train_idx[train_keep_local]]
        if np.unique(y_tr.astype(int)).size < 2:
            # Single-class fallback handled by the model class itself.
            pass
        model = make_model(model_name, seed=seed)
        model.fit(X_tr, y_tr)
        train_scores[train_keep_local] = model.predict_proba(X_tr)[:, 1]
        test_keep_local = ~test_feat_nan
        if test_keep_local.any():
            X_te = features_df.iloc[test_idx[test_keep_local]][feat_cols]
            test_scores[test_keep_local] = model.predict_proba(X_te)[:, 1]
        return train_scores, test_scores

    # Panel-DataFrame baselines: train rows are those in date+label mask.
    X_tr = panel_df.iloc[train_idx]
    X_te = panel_df.iloc[test_idx]
    y_tr = labels[train_idx]
    model = make_model(model_name, seed=seed)
    model.fit(X_tr, y_tr)
    train_scores = model.predict_proba(X_tr)[:, 1]
    test_scores = model.predict_proba(X_te)[:, 1]
    return train_scores, test_scores


# =====================================================================
# Track loop
# =====================================================================

def run_grid(
    panel_df: pd.DataFrame,
    features_df: pd.DataFrame,
    out_exp_dir: Path,
    out_table_dir: Path,
    stress_defs: list[dict] | None = None,
    seeds: tuple[int, ...] | None = None,
    initial_train_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Execute the full Track A grid. Returns (per_fold_df, fold_def_df,
    predictions_by_def)."""
    stress_defs = stress_defs or STRESS_DEFS
    seeds = seeds or SEEDS
    out_exp_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_exp_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_table_dir.mkdir(parents=True, exist_ok=True)

    init_train_end = pd.Timestamp(initial_train_end) if initial_train_end is not None else INITIAL_TRAIN_END
    panel_dates = pd.to_datetime(panel_df["Date"])

    per_fold_rows: list[dict] = []
    fold_def_rows: list[dict] = []
    predictions_by_def: dict[str, list[dict]] = {sd["name"]: [] for sd in stress_defs}

    feat_cols = [c for c in features_df.columns if c != "Date"]
    feat_nan_mask_full = features_df[feat_cols].isna().any(axis=1).to_numpy()

    folds_template = build_folds_with_init(panel_dates, init_train_end)

    for sd in stress_defs:
        sd_name, h, d = sd["name"], sd["h"], sd["d"]
        prices = panel_df[PRICE_COL].to_numpy(dtype=float)
        future_drawdown, labels = compute_stress_labels(prices, h, d)
        label_nan = np.isnan(labels)

        for fold in folds_template:
            train_mask_dates = (panel_dates <= fold["train_end"]).to_numpy()
            test_mask_dates = (
                (panel_dates >= fold["test_start"])
                & (panel_dates <= fold["test_end"])
            ).to_numpy()
            train_mask = train_mask_dates & ~label_nan
            test_mask = test_mask_dates & ~label_nan

            n_dropped_h = int(((train_mask_dates | test_mask_dates) & label_nan).sum())
            n_dropped_nan_features_train = int(
                (train_mask & feat_nan_mask_full).sum()
            )
            n_dropped_nan_features_test = int(
                (test_mask & feat_nan_mask_full).sum()
            )
            fold_def_rows.append(
                {
                    "stress_def": sd_name,
                    "fold_id": fold["fold_id"],
                    "train_end": fold["train_end"].strftime("%Y-%m-%d"),
                    "test_start": fold["test_start"].strftime("%Y-%m-%d"),
                    "test_end": fold["test_end"].strftime("%Y-%m-%d"),
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "n_dropped_h": n_dropped_h,
                    "n_train_dropped_nan_features": n_dropped_nan_features_train,
                    "n_test_dropped_nan_features": n_dropped_nan_features_test,
                }
            )
            if train_mask.sum() < 30 or test_mask.sum() < 5:
                continue

            test_dates = panel_dates[test_mask].reset_index(drop=True)
            test_fd = future_drawdown[test_mask]
            test_labels = labels[test_mask]

            # Run deterministic models once per fold; replicate across seeds.
            for model_name in DETERMINISTIC_MODELS:
                train_scores, test_scores = _fit_predict_one(
                    model_name, PRIMARY_SEED, panel_df, features_df,
                    train_mask, test_mask, labels,
                )
                l1 = compute_layer1(test_scores, test_labels)
                l2 = compute_layer2(test_scores, test_labels, train_scores, test_fd)
                base_row = {
                    "stress_def": sd_name,
                    "model": model_name,
                    "fold_id": fold["fold_id"],
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    **l1,
                    **l2,
                }
                for s in seeds:
                    per_fold_rows.append({**base_row, "seed": s})
                    for date_, score, lab in zip(test_dates, test_scores, test_labels):
                        predictions_by_def[sd_name].append(
                            {
                                "Date": date_,
                                "model": model_name,
                                "seed": s,
                                "fold_id": fold["fold_id"],
                                "score": float(score) if not np.isnan(score) else np.nan,
                                "label": float(lab) if not np.isnan(lab) else np.nan,
                            }
                        )

            # Stochastic models: separate run per seed.
            for model_name in STOCHASTIC_MODELS:
                for s in seeds:
                    train_scores, test_scores = _fit_predict_one(
                        model_name, s, panel_df, features_df,
                        train_mask, test_mask, labels,
                    )
                    l1 = compute_layer1(test_scores, test_labels)
                    l2 = compute_layer2(test_scores, test_labels, train_scores, test_fd)
                    per_fold_rows.append(
                        {
                            "stress_def": sd_name,
                            "model": model_name,
                            "seed": s,
                            "fold_id": fold["fold_id"],
                            "n_train": int(train_mask.sum()),
                            "n_test": int(test_mask.sum()),
                            **l1,
                            **l2,
                        }
                    )
                    for date_, score, lab in zip(test_dates, test_scores, test_labels):
                        predictions_by_def[sd_name].append(
                            {
                                "Date": date_,
                                "model": model_name,
                                "seed": s,
                                "fold_id": fold["fold_id"],
                                "score": float(score) if not np.isnan(score) else np.nan,
                                "label": float(lab) if not np.isnan(lab) else np.nan,
                            }
                        )

    per_fold_df = pd.DataFrame(per_fold_rows)
    fold_def_df = pd.DataFrame(fold_def_rows)
    pred_dfs: dict[str, pd.DataFrame] = {}
    for sd_name, rows in predictions_by_def.items():
        pred_dfs[sd_name] = pd.DataFrame(rows)

    # Persist
    per_fold_df.to_csv(out_exp_dir / "per_fold_metrics.csv", index=False)
    fold_def_df.to_csv(out_exp_dir / "fold_definitions.csv", index=False)
    for sd_name, df in pred_dfs.items():
        df.to_parquet(pred_dir / f"{sd_name}.parquet", index=False)

    return per_fold_df, fold_def_df, pred_dfs


def build_folds_with_init(panel_dates: pd.Series, init_train_end: pd.Timestamp) -> list[dict]:
    """Same as build_folds but with an overridable initial train_end (for the smoke path)."""
    panel_end = pd.Timestamp(panel_dates.iloc[-1])
    folds: list[dict] = []
    train_end = pd.Timestamp(init_train_end)
    k = 1
    while True:
        test_start = train_end + pd.Timedelta(days=1)
        test_end = train_end + pd.DateOffset(months=TEST_WINDOW_MONTHS)
        if test_end > panel_end:
            break
        folds.append(
            {
                "fold_id": k,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        k += 1
        train_end = train_end + pd.DateOffset(months=STEP_MONTHS)
    return folds


# =====================================================================
# Aggregation: per-fold -> per-cell tables with CIs
# =====================================================================

L1_METRICS = ["auc", "pr_auc", "brier", "ece"]
L2_METRICS = ["alarm_rate", "alarm_precision", "drawdown_lift", "severity_lift"]


def _per_fold_means(per_fold_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average across seeds within each (stress_def, model, fold_id) -> one
    value per fold."""
    grp = per_fold_df.groupby(["stress_def", "model", "fold_id"])[metric].mean()
    return grp.reset_index()


def build_table_with_ci(per_fold_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for metric in metrics:
        means = _per_fold_means(per_fold_df, metric)
        for (sd_name, model), grp in means.groupby(["stress_def", "model"]):
            values = grp[metric].to_numpy()
            ci = block_bootstrap_ci(
                values, block_size=1, n_boot=N_BOOT, alpha=ALPHA, seed=BOOTSTRAP_SEED
            )
            rows.append(
                {
                    "stress_def": sd_name,
                    "model": model,
                    "metric": metric,
                    "mean": ci["mean"],
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                    "n_folds_total": ci["n_input"],
                    "n_folds_valid": ci["n_valid"],
                }
            )
    return pd.DataFrame(rows)


def build_layer3_table(per_fold_df: pd.DataFrame, predictions_by_def: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calm-period: rows where realized event count is 0. Uses per-fold
    'n_events' to identify zero-event folds, then aggregates predictions
    on those folds only (averaging across seeds)."""
    rows: list[dict] = []
    n_events_col = "n_events"
    for sd_name, pred_df in predictions_by_def.items():
        sd_per_fold = per_fold_df[per_fold_df["stress_def"] == sd_name]
        # A fold is "calm" if all events==0 across all (model, seed) rows for that fold.
        # Since events depend only on label (not on model), n_events is identical for
        # all (model, seed) within a fold. So just check first row.
        calm_folds = (
            sd_per_fold.groupby("fold_id")[n_events_col].first()
            .pipe(lambda s: s[s == 0]).index.tolist()
        )
        calm_pred = pred_df[pred_df["fold_id"].isin(calm_folds)]
        for model, grp in calm_pred.groupby("model"):
            # Average score across seeds at each (fold_id, Date)
            avg_score = (
                grp.groupby(["fold_id", "Date"])["score"]
                .mean()
                .to_numpy()
            )
            avg_score = avg_score[~np.isnan(avg_score)]
            n = avg_score.size
            if n == 0:
                rows.append(
                    {
                        "stress_def": sd_name,
                        "model": model,
                        "n_calm_folds": len(calm_folds),
                        "n_days": 0,
                        "mean_score": np.nan,
                        "max_score": np.nan,
                        "alarm_count": 0,
                        "alarm_rate": np.nan,
                    }
                )
                continue
            # Threshold for alarm count: same convention as Layer 2,
            # 95th percentile of training scores per fold. We approximate
            # here with the per-fold threshold from Layer 2 (taken from
            # per_fold_metrics.csv). Use the median across seeds as the
            # representative threshold for each fold.
            fold_thresholds = (
                sd_per_fold[sd_per_fold["model"] == model]
                .groupby("fold_id")["threshold"]
                .median()
            )
            # Recover per-day per-fold scores and apply each fold's threshold.
            fold_scores = grp.groupby(["fold_id", "Date"])["score"].mean().reset_index()
            n_alarm = 0
            n_total = 0
            for fid in calm_folds:
                thr = fold_thresholds.get(fid, np.nan)
                if np.isnan(thr):
                    continue
                fold_rows = fold_scores[fold_scores["fold_id"] == fid]
                s = fold_rows["score"].to_numpy()
                s_valid = s[~np.isnan(s)]
                n_alarm += int((s_valid > thr).sum())
                n_total += int(s_valid.size)
            alarm_rate = (n_alarm / n_total) if n_total > 0 else np.nan
            rows.append(
                {
                    "stress_def": sd_name,
                    "model": model,
                    "n_calm_folds": len(calm_folds),
                    "n_days": int(n),
                    "mean_score": float(avg_score.mean()),
                    "max_score": float(avg_score.max()),
                    "alarm_count": int(n_alarm),
                    "alarm_rate": alarm_rate,
                }
            )
    return pd.DataFrame(rows)


def build_layer4_table(per_fold_df: pd.DataFrame) -> pd.DataFrame:
    """Stability: std across folds of L1 and L2 metrics, with bootstrap-of-std CI."""
    rows: list[dict] = []
    metrics = L1_METRICS + L2_METRICS
    for metric in metrics:
        means = _per_fold_means(per_fold_df, metric)
        for (sd_name, model), grp in means.groupby(["stress_def", "model"]):
            values = grp[metric].to_numpy()
            ci = bootstrap_std_ci(values, n_boot=N_BOOT, alpha=ALPHA, seed=BOOTSTRAP_SEED)
            rows.append(
                {
                    "stress_def": sd_name,
                    "model": model,
                    "metric": metric,
                    "std": ci["std"],
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                    "n_folds_valid": ci["n_valid"],
                }
            )
    return pd.DataFrame(rows)


def build_pairwise_files(per_fold_df: pd.DataFrame, out_table_dir: Path) -> dict[str, pd.DataFrame]:
    """One file per (stress_def, metric in {auc, drawdown_lift})."""
    out: dict[str, pd.DataFrame] = {}
    for metric_alias, metric_col in (("AUC", "auc"), ("lift", "drawdown_lift")):
        for sd_name in per_fold_df["stress_def"].unique():
            means = _per_fold_means(
                per_fold_df[per_fold_df["stress_def"] == sd_name], metric_col
            )
            per_model = {
                model: g[metric_col].to_numpy()
                for model, g in means.groupby("model")
            }
            tab = pairwise_table(per_model, metric_name=metric_col, alpha=ALPHA)
            fname = f"pairwise_pvalues_{metric_alias}_{sd_name}.csv"
            tab.to_csv(out_table_dir / fname)
            out[fname] = tab
    return out


# =====================================================================
# Figures
# =====================================================================

def _palette(n: int) -> list[str]:
    base = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    return [base[i % len(base)] for i in range(n)]


def plot_lift_with_ci(per_fold_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = _per_fold_means(per_fold_df, "drawdown_lift")
    cell = (
        means.groupby(["stress_def", "model"])["drawdown_lift"]
        .agg(list)
        .reset_index()
    )
    sd_order = [sd["name"] for sd in STRESS_DEFS]
    model_order = list(ALL_MODELS)

    fig, ax = plt.subplots(figsize=(9, 5))
    n_models = len(model_order)
    width = 0.8 / n_models
    colors = _palette(n_models)
    x = np.arange(len(sd_order))

    for i, model in enumerate(model_order):
        bar_means: list[float] = []
        bar_los: list[float] = []
        bar_his: list[float] = []
        for sd in sd_order:
            row = cell[(cell["stress_def"] == sd) & (cell["model"] == model)]
            if row.empty:
                bar_means.append(np.nan)
                bar_los.append(np.nan)
                bar_his.append(np.nan)
                continue
            vals = np.asarray(row["drawdown_lift"].iloc[0], dtype=float)
            ci = block_bootstrap_ci(
                vals, block_size=1, n_boot=N_BOOT, alpha=ALPHA, seed=BOOTSTRAP_SEED
            )
            bar_means.append(ci["mean"])
            bar_los.append(ci["mean"] - ci["ci_low"])
            bar_his.append(ci["ci_high"] - ci["mean"])

        bar_means_a = np.array(bar_means, dtype=float)
        nan_mask = np.isnan(bar_means_a)
        plot_means = np.where(nan_mask, 0.0, bar_means_a)
        yerr_lower = np.where(nan_mask, 0.0, np.array(bar_los, dtype=float))
        yerr_upper = np.where(nan_mask, 0.0, np.array(bar_his, dtype=float))

        ax.bar(
            x + (i - n_models / 2 + 0.5) * width,
            plot_means,
            width,
            yerr=[yerr_lower, yerr_upper],
            capsize=2,
            label=model,
            color=colors[i],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.95,
        )

    # Reference line at lift = 1: thin gray dashed, drawn behind the bars
    # but ahead of the gridlines. The methodology section interprets
    # lift relative to 1 (alarms equal OOS magnitude on average).
    ax.axhline(
        1.0, color="gray", linestyle="--", linewidth=0.8,
        zorder=0.5, label="lift = 1 (no skill)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sd_order)
    ax.set_xlabel("Stress definition (h, d)")
    ax.set_ylabel("Drawdown lift (mean ± 95% bootstrap CI)")
    ax.set_title("Track A drawdown lift by model and stress definition")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_reliability(predictions: pd.DataFrame, out_path: Path, n_bins: int = 10) -> dict:
    """Reliability diagrams using equal-frequency (quantile) bins.

    Drops NaiveBaseRate (constant scores collapse to a single bin and
    contribute no calibration information). Returns a dict with the
    per-model bin counts so the caller can print STATUS without
    re-deriving them.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models_to_plot = [m for m in ALL_MODELS if m != "NaiveBaseRate"]

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    bin_counts: dict[str, list[int]] = {}

    for i, model in enumerate(models_to_plot):
        ax = axes_flat[i]
        # Diagonal first (zorder=1) so calibration markers render on top.
        ax.plot(
            [0, 1], [0, 1],
            color="gray", linestyle="--", linewidth=0.6, zorder=1,
        )

        sub = predictions[predictions["model"] == model]
        agg = (
            sub.groupby(["fold_id", "Date"])
            .agg(score=("score", "mean"), label=("label", "first"))
            .reset_index()
        )
        s = agg["score"].to_numpy()
        y = agg["label"].to_numpy()
        valid = ~(np.isnan(s) | np.isnan(y))
        s = s[valid]
        y = y[valid]
        if s.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
            ax.set_title(model, fontsize=9)
            bin_counts[model] = []
            continue

        # Equal-frequency bins via pd.qcut; duplicates='drop' collapses
        # tied boundaries into one effective bin, so the actual bin
        # count may be smaller than n_bins on a low-variance score
        # distribution. We record the realised counts for STATUS.
        try:
            bin_idx, bin_edges = pd.qcut(
                s, q=n_bins, retbins=True, duplicates="drop", labels=False,
            )
        except ValueError:
            ax.text(0.5, 0.5, "score distribution degenerate",
                    ha="center", transform=ax.transAxes)
            ax.set_title(model, fontsize=9)
            bin_counts[model] = []
            continue
        bin_idx = np.asarray(bin_idx)
        n_actual_bins = len(bin_edges) - 1

        bin_pred: list[float] = []
        bin_obs: list[float] = []
        bin_n: list[int] = []
        for b in range(n_actual_bins):
            m = bin_idx == b
            if m.any():
                bin_pred.append(float(s[m].mean()))
                bin_obs.append(float(y[m].mean()))
                bin_n.append(int(m.sum()))
        bin_counts[model] = bin_n

        # Marker size scales with sqrt(n) so sparse bins stay visible
        # and dense bins are obvious. Floor of 20 keeps ones visible.
        marker_sizes = [20.0 + 6.0 * float(np.sqrt(n)) for n in bin_n]
        ax.scatter(
            bin_pred, bin_obs, s=marker_sizes,
            alpha=0.85, edgecolor="black", linewidth=0.4, zorder=2,
        )
        ax.set_title(model, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(linestyle=":", linewidth=0.4)

    # Hide the unused 6th panel after dropping NaiveBaseRate.
    for j in range(len(models_to_plot), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.text(0.5, 0.04,
             "Mean predicted probability per bin (h=10, d=5%)",
             ha="center")
    fig.text(0.04, 0.5,
             "Empirical event rate per bin",
             va="center", rotation="vertical")
    fig.suptitle(
        "Reliability diagrams (h=10, d=5%) — equal-frequency bins, "
        "marker size ∝ √n;  NaiveBaseRate omitted (constant predictions)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.06, 0.06, 1.0, 0.94))
    fig.savefig(out_path, format="pdf")
    plt.close(fig)

    return {"bin_counts": bin_counts}


def plot_pr_curves(predictions: pd.DataFrame, out_path: Path) -> dict:
    """PR curves with empirical base-rate reference line.

    Returns a dict with per-model PR-AUC (average precision) values and
    the empirical base rate computed from the pooled OOS labels (the
    baseline that every model must beat to be useful).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve

    # First pass: compute pooled (score, label) per model and PR-AUC.
    per_model: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pr_aucs: dict[str, float] = {}
    for model in ALL_MODELS:
        sub = predictions[predictions["model"] == model]
        agg = (
            sub.groupby(["fold_id", "Date"])
            .agg(score=("score", "mean"), label=("label", "first"))
            .reset_index()
        )
        s = agg["score"].to_numpy()
        y = agg["label"].to_numpy()
        valid = ~(np.isnan(s) | np.isnan(y))
        s = s[valid]
        y = y[valid].astype(int)
        per_model[model] = (s, y)
        if s.size > 0 and np.unique(y).size >= 2:
            pr_aucs[model] = float(average_precision_score(y, s))
        else:
            pr_aucs[model] = float("nan")

    # Empirical base rate: positive class rate on pooled OOS labels.
    # Labels are identical across models for a given (fold, Date), so
    # we can pull from any model. Pick the first model with non-empty
    # data to get the labels.
    base_rate = float("nan")
    for model in ALL_MODELS:
        _, y = per_model[model]
        if y.size > 0:
            base_rate = float(y.mean())
            break

    # Plot
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    colors = _palette(len(ALL_MODELS))
    for i, model in enumerate(ALL_MODELS):
        s, y = per_model[model]
        ap = pr_aucs[model]
        if s.size == 0 or np.unique(y).size < 2:
            ax.plot([], [], label=f"{model} (no data)")
            continue
        precision, recall, _ = precision_recall_curve(y, s)
        ax.plot(
            recall, precision,
            color=colors[i], linewidth=1.4,
            label=f"{model}  (AP = {ap:.3f})",
        )

    # Base-rate reference line; the no-skill PR curve has y = base_rate.
    if not np.isnan(base_rate):
        ax.axhline(
            base_rate, color="gray", linestyle="--", linewidth=0.8,
            label=f"Random baseline (base rate = {100.0 * base_rate:.2f}%)",
        )

    # Y-axis: [0, max(PR-AUC) * 2] so the curves are visually
    # distinguishable instead of all hugging y = 0.
    valid_aucs = [v for v in pr_aucs.values() if not np.isnan(v)]
    ylim_top = max(valid_aucs) * 2.0 if valid_aucs else 1.0
    ylim_top = max(ylim_top, (base_rate * 2.0) if not np.isnan(base_rate) else 0.0)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, ylim_top)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves (h=10, d=5%)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(linestyle=":", linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)

    return {"pr_aucs": pr_aucs, "base_rate": base_rate}


# =====================================================================
# STATUS / failure-mode probes
# =====================================================================

def print_status(per_fold_df: pd.DataFrame, fold_def_df: pd.DataFrame) -> None:
    print()
    print("===== TRACK A STATUS =====")
    for sd in STRESS_DEFS:
        sd_name = sd["name"]
        sd_folds = fold_def_df[fold_def_df["stress_def"] == sd_name]
        print(f"\n[{sd_name}] folds = {len(sd_folds)}")
        # Per-model AUC and lift means + 95% CIs
        for model in ALL_MODELS:
            sub = per_fold_df[
                (per_fold_df["stress_def"] == sd_name)
                & (per_fold_df["model"] == model)
            ]
            auc_per_fold = (
                sub.groupby("fold_id")["auc"].mean().to_numpy()
            )
            lift_per_fold = (
                sub.groupby("fold_id")["drawdown_lift"].mean().to_numpy()
            )
            auc_ci = block_bootstrap_ci(
                auc_per_fold, n_boot=N_BOOT, seed=BOOTSTRAP_SEED
            )
            lift_ci = block_bootstrap_ci(
                lift_per_fold, n_boot=N_BOOT, seed=BOOTSTRAP_SEED
            )
            print(
                f"  {model:<26}  "
                f"AUC mean={auc_ci['mean']:.3f} "
                f"CI=[{auc_ci['ci_low']:.3f},{auc_ci['ci_high']:.3f}]   "
                f"lift mean={lift_ci['mean']:.3f} "
                f"CI=[{lift_ci['ci_low']:.3f},{lift_ci['ci_high']:.3f}]"
            )

    print("\nFailure-mode probes:")
    print(f"  F1 (AUC-CIs overlap, lift-CIs disjoint): {failure_probe_f1(per_fold_df)}")
    print(f"  F2 (sign mismatch AUC vs lift):          {failure_probe_f2(per_fold_df)}")
    f3 = failure_probe_f3(per_fold_df)
    f3_str = ", ".join(f"{k}: {v:.3f}" for k, v in f3.items())
    print(f"  F3 (Spearman Brier-rank vs lift-rank):   {f3_str}")
    print("==========================")


def _ci(values: np.ndarray) -> tuple[float, float, float]:
    ci = block_bootstrap_ci(values, n_boot=N_BOOT, seed=BOOTSTRAP_SEED)
    return ci["mean"], ci["ci_low"], ci["ci_high"]


def failure_probe_f1(per_fold_df: pd.DataFrame) -> str:
    """Pairs of models where AUC CIs overlap but lift CIs do NOT."""
    hits: list[str] = []
    for sd_name, sd_grp in per_fold_df.groupby("stress_def"):
        ci_by_model: dict[str, dict] = {}
        for model in ALL_MODELS:
            sub = sd_grp[sd_grp["model"] == model]
            auc_v = sub.groupby("fold_id")["auc"].mean().to_numpy()
            lift_v = sub.groupby("fold_id")["drawdown_lift"].mean().to_numpy()
            ci_by_model[model] = {
                "auc": _ci(auc_v),
                "lift": _ci(lift_v),
            }
        models = list(ci_by_model.keys())
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a = ci_by_model[models[i]]
                b = ci_by_model[models[j]]
                auc_overlap = not (a["auc"][2] < b["auc"][1] or b["auc"][2] < a["auc"][1])
                lift_disjoint = (a["lift"][2] < b["lift"][1] or b["lift"][2] < a["lift"][1])
                if auc_overlap and lift_disjoint:
                    hits.append(f"{sd_name}: {models[i]} vs {models[j]}")
    return "; ".join(hits) if hits else "none"


def failure_probe_f2(per_fold_df: pd.DataFrame) -> int:
    """Count of (model, stress_def, fold) cells where AUC<0.5 & lift>1
    OR AUC>0.5 & lift<1."""
    df = per_fold_df.groupby(["stress_def", "model", "fold_id"]).agg(
        auc=("auc", "mean"),
        lift=("drawdown_lift", "mean"),
    ).reset_index()
    cond = (
        ((df["auc"] < 0.5) & (df["lift"] > 1.0))
        | ((df["auc"] > 0.5) & (df["lift"] < 1.0))
    )
    return int(cond.sum())


def failure_probe_f3(per_fold_df: pd.DataFrame) -> dict[str, float]:
    """Spearman correlation between mean Brier ranking and mean lift
    ranking across models, per stress_def. Lower Brier is better
    (ascending); higher lift is better (descending). Models with NaN
    in either statistic are excluded — typically NaiveBaseRate,
    whose constant scores produce zero alarms under strict-> and
    therefore an undefined lift."""
    from scipy.stats import spearmanr
    out: dict[str, float] = {}
    for sd_name, grp in per_fold_df.groupby("stress_def"):
        agg = (
            grp.groupby("model")
            .agg(brier=("brier", "mean"), lift=("drawdown_lift", "mean"))
            .reset_index()
        )
        # Drop rows where either statistic is NaN before ranking; this
        # keeps spearmanr from returning NaN under the default
        # nan_policy='propagate'.
        agg = agg.dropna(subset=["brier", "lift"])
        if len(agg) < 2:
            out[sd_name] = float("nan")
            continue
        brier_rank = agg["brier"].rank(ascending=True)
        lift_rank = agg["lift"].rank(ascending=False)
        if brier_rank.nunique() < 2 or lift_rank.nunique() < 2:
            out[sd_name] = float("nan")
            continue
        rho, _ = spearmanr(brier_rank, lift_rank)
        out[sd_name] = float(rho)
    return out


# =====================================================================
# Main entry points
# =====================================================================

def _load_full_panels(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_path = repo_root / "data" / "processed" / f"{PANEL_NAME}.csv"
    feat_path = repo_root / "data" / "processed" / "features" / f"{PANEL_NAME}_features.csv"
    panel_df = pd.read_csv(panel_path, parse_dates=["Date"])
    features_df = pd.read_csv(feat_path, parse_dates=["Date"])
    if not panel_df["Date"].equals(features_df["Date"]):
        raise ValueError("Panel and features Date columns are not aligned.")
    return panel_df, features_df


def main_full() -> int:
    panel_df, features_df = _load_full_panels(REPO_ROOT)
    out_exp = REPO_ROOT / "experiments" / "track_a_headline"
    out_tab = REPO_ROOT / "outputs" / "track_a"

    per_fold_df, fold_def_df, predictions_by_def = run_grid(
        panel_df, features_df, out_exp, out_tab,
        stress_defs=STRESS_DEFS, seeds=SEEDS,
        initial_train_end=INITIAL_TRAIN_END,
    )

    # Aggregated tables
    build_table_with_ci(per_fold_df, L1_METRICS).to_csv(
        out_tab / "table_layer1.csv", index=False
    )
    build_table_with_ci(per_fold_df, L2_METRICS).to_csv(
        out_tab / "table_layer2.csv", index=False
    )
    build_layer3_table(per_fold_df, predictions_by_def).to_csv(
        out_tab / "table_layer3.csv", index=False
    )
    build_layer4_table(per_fold_df).to_csv(
        out_tab / "table_layer4.csv", index=False
    )
    build_pairwise_files(per_fold_df, out_tab)

    # Figures
    plot_lift_with_ci(per_fold_df, out_tab / "fig_lift_with_ci.pdf")
    target_sd = "h10_d05"
    if target_sd in predictions_by_def and len(predictions_by_def[target_sd]) > 0:
        plot_reliability(predictions_by_def[target_sd], out_tab / "fig_reliability.pdf")
        plot_pr_curves(predictions_by_def[target_sd], out_tab / "fig_pr_curves.pdf")

    print_status(per_fold_df, fold_def_df)
    return 0


def main_smoke() -> int:
    """Synthetic 1-fold, 1-seed sanity check."""
    rng = np.random.default_rng(0)
    n = 400
    dates = pd.date_range("2014-01-01", periods=n, freq="B")
    spx = 1000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n)))
    vix = 15.0 + 5.0 * rng.standard_normal(n).cumsum() / np.sqrt(n)
    vix = np.clip(vix, 8.0, 80.0)
    panel_df = pd.DataFrame({"Date": dates, "SPX": spx, "VIX": vix})

    # Synthetic 35 features (random Gaussian, no NaN)
    feat_cols = [f"feat_{i}" for i in range(35)]
    feat_data = rng.standard_normal((n, 35))
    features_df = pd.DataFrame(feat_data, columns=feat_cols)
    features_df.insert(0, "Date", dates)

    out_exp = REPO_ROOT / "experiments" / "track_a_headline_smoke"
    out_tab = REPO_ROOT / "outputs" / "track_a_smoke"

    per_fold_df, fold_def_df, predictions_by_def = run_grid(
        panel_df, features_df, out_exp, out_tab,
        stress_defs=[{"name": "h5_d03", "h": 5, "d": 0.03}],
        seeds=(42,),
        initial_train_end=pd.Timestamp("2014-06-30"),
    )
    if per_fold_df.empty:
        print("SMOKE: no per-fold rows produced (likely no folds fit).")
        return 1
    build_table_with_ci(per_fold_df, L1_METRICS).to_csv(out_tab / "table_layer1.csv", index=False)
    print(f"SMOKE: per_fold_metrics rows = {len(per_fold_df)}, fold_defs = {len(fold_def_df)}")
    return 0


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track A runner")
    parser.add_argument("--smoke", action="store_true", help="run synthetic smoke")
    parser.add_argument("--full", action="store_true", help="run the full headline grid")
    args = parser.parse_args(argv)
    if args.smoke and args.full:
        parser.error("Pick one of --smoke / --full.")
    if args.smoke:
        return main_smoke()
    if args.full:
        return main_full()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(cli())
