"""Track E component ablation modes.

Implements the four controlled ablations described in Track E:

  ablation_1  no_validators   feed unvalidated raw-derived panel into
                              feature construction (synthetic VVIX
                              backfill becomes "real")
  ablation_2  global_scaler   replace per-fold StandardScaler with a
                              single scaler fit on the entire dataset
                              (look-ahead leakage variant; only LR is
                              affected — LightGBM has no scaler in the
                              unmodified pipeline)
  ablation_3  test_threshold  replace training-quantile alarm threshold
                              (95th pct of train scores) with a fixed
                              5% top-of-test threshold (the v1 paper)
  ablation_4  frozen_model    fit each model once on the first training
                              window (data <= INITIAL_TRAIN_END) and
                              apply that frozen model to every later
                              test fold without refitting

All four ablations run on the SAME folds, SAME seed list, SAME
configuration as Track A (h10_d05 only) so per-fold deltas are paired
on (model, fold_id, seed). Ablations 2/3/4 patch the in-memory pipeline
and reuse the canonical Track A panel + features. Ablation 1 requires a
caller-supplied no-validator panel + features (built once by
scripts/run_track_e.py).

Public API
----------
build_no_validator_panel(repo_root) -> (panel_df, features_df, provenance)
    Reconstruct an SPX panel from data/raw/ without applying the
    pre-launch truncations or the data-quality NaN flags. Returns the
    same DataFrame pair shape as data/processed/spx_extended_2011.csv +
    its features CSV, plus a provenance dict that captures exactly
    what the ablation bypassed.

run_ablation(mode, panel_df, features_df, ...) -> per_fold_df
    Run the headline grid (Track A protocol) with ONE component
    swapped out. Mode in {"no_validators","global_scaler",
    "test_threshold","frozen_model"}. Returns a per-fold DataFrame
    with the same schema as Track A's per_fold_metrics.csv plus an
    "ablation_id" column.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from seet.baselines import (
    DETERMINISTIC_MODELS,
    HarRvThreshold,
    LightGbmTuned,
    LogisticRegressionL2,
    NaiveBaseRate,
    STOCHASTIC_MODELS,
    VIXPercentileCalibrated,
    VIXPercentileRaw,
    make_model,
)
from seet.features import build_features
from seet.pipeline import (
    BOOTSTRAP_SEED,
    THRESHOLD_PERCENTILE,
    build_folds_with_init,
    compute_layer1,
    compute_layer2,
    compute_stress_labels,
)


ABLATION_IDS = ("no_validators", "global_scaler", "test_threshold", "frozen_model")


# =====================================================================
# ABLATION_1 setup helper: build a no-validator panel from raw CSVs
# =====================================================================

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_no_validator_panel(
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reconstruct the SPX panel from raw CSVs WITHOUT validator stages.

    The validated panel data/processed/spx_extended_2011.csv applied
    these stages from build_processed.py:
      (1) anchor on VIX9D, left-join SPX, VIX, VIX3M, VIX6M, VVIX
      (2) truncate VVIX before 2012-04-01 (synthetic backfill -> NaN)
      (3) record data_quality_notes for 8 documented VVIX gap dates

    This function applies (1) only. Stage (2) is bypassed: synthetic
    yfinance VVIX values from before 2012-04-01 are kept as if they
    were real exchange data. Stage (3) is bypassed implicitly: raw
    VVIX is missing those rows (the gap dates are absent from
    data/raw/VVIX.csv too), so the left-join still yields NaN at
    those exact dates -- but no `data_quality_notes` block flags them.

    Anchor and date axis: VIX9D's full raw range (2011-01-03 onwards)
    rather than the validated panel's 2011-02-23 cutoff. This keeps
    the ablation honest -- "no validators" really means no validators.
    The fold definition that begins at INITIAL_TRAIN_END=2014-12-31
    is unchanged, so per-fold pairing with Track A is preserved.

    Returns
    -------
    panel_df : pd.DataFrame
        Columns ['Date', 'SPX', 'VIX', 'VIX9D', 'VIX3M', 'VIX6M', 'VVIX'].
    features_df : pd.DataFrame
        spx_full feature set (35 features) computed on panel_df.
    provenance : dict
        Audit trail: raw input shas, output shas, the diff vs the
        validated panel (rows that gained synthetic-backfill values).
    """
    raw_dir = Path(repo_root) / "data" / "raw"
    cols = {
        "VIX9D": "VIX9D",
        "SPX": "SPX",
        "VIX": "VIX",
        "VIX3M": "VIX3M",
        "VIX6M": "VIX6M",
        "VVIX": "VVIX",
    }

    raw_shas = {}
    frames: dict[str, pd.DataFrame] = {}
    for fname, col in cols.items():
        path = raw_dir / f"{fname}.csv"
        df = pd.read_csv(path, parse_dates=["Date"])
        if "Close" in df.columns:
            df = df.rename(columns={"Close": col})
        elif col not in df.columns:
            # raw file may carry an index-named close column; pick the
            # first non-Date column.
            other = [c for c in df.columns if c != "Date"]
            df = df.rename(columns={other[0]: col})
        df = df[["Date", col]]
        frames[col] = df
        raw_shas[fname] = _sha256(path)

    # Anchor on VIX9D (matches the validated panel's anchor).
    panel = frames["VIX9D"].copy()
    for col in ["SPX", "VIX", "VIX3M", "VIX6M", "VVIX"]:
        panel = panel.merge(frames[col], on="Date", how="left")
    panel = panel.sort_values("Date").reset_index(drop=True)

    # Order columns consistently with the validated panel for a clean
    # diff. (The validated panel's column order is
    # Date, SPX, VIX, VIX9D, VIX3M, VIX6M, VVIX.)
    panel = panel[["Date", "SPX", "VIX", "VIX9D", "VIX3M", "VIX6M", "VVIX"]]

    # Build features (spx_full) on the no-validator panel.
    features_df, _ = build_features(panel, "spx_full")

    # Diff vs validated panel for the provenance record.
    validated_path = (
        Path(repo_root) / "data" / "processed" / "spx_extended_2011.csv"
    )
    if validated_path.exists():
        validated = pd.read_csv(validated_path, parse_dates=["Date"])
        validated_dates = set(validated["Date"].tolist())
        ablation_dates = set(panel["Date"].tolist())
        new_dates = sorted(d.strftime("%Y-%m-%d")
                           for d in ablation_dates - validated_dates)

        # Rows present in both panels where the ablation has a non-NaN
        # VVIX but the validated panel has NaN (the synthetic-backfill
        # window 2011-02-23 -> 2012-03-31).
        merged = panel.merge(
            validated[["Date", "VVIX"]].rename(columns={"VVIX": "VVIX_validated"}),
            on="Date", how="inner",
        )
        backfill_mask = (
            merged["VVIX_validated"].isna() & merged["VVIX"].notna()
        )
        backfill_dates = sorted(
            d.strftime("%Y-%m-%d") for d in merged.loc[backfill_mask, "Date"]
        )
        bypassed_summary = {
            "rows_added_before_validated_anchor": len(new_dates),
            "first_added_date": new_dates[0] if new_dates else None,
            "last_added_date": new_dates[-1] if new_dates else None,
            "vvix_synthetic_backfill_rows": int(backfill_mask.sum()),
            "first_synthetic_vvix_date": (
                backfill_dates[0] if backfill_dates else None
            ),
            "last_synthetic_vvix_date": (
                backfill_dates[-1] if backfill_dates else None
            ),
        }
    else:
        bypassed_summary = {
            "note": "validated panel not found; diff not computed"
        }

    provenance = {
        "ablation": "no_validators",
        "anchor": "VIX9D (raw, no truncation)",
        "raw_input_sha256": raw_shas,
        "panel_n_rows": int(len(panel)),
        "panel_first_date": panel["Date"].min().strftime("%Y-%m-%d"),
        "panel_last_date": panel["Date"].max().strftime("%Y-%m-%d"),
        "panel_columns": list(panel.columns),
        "features_n_rows": int(len(features_df)),
        "features_n_cols": int(features_df.shape[1] - 1),
        "bypassed_validators": [
            "pre-launch truncation of VVIX (Date < 2012-04-01 in validated)",
            "data_quality_notes flagging of 8 VVIX gap dates",
            "anchor cutoff to VIX9D real-launch (2011-02-23 in validated)",
        ],
        "diff_vs_validated": bypassed_summary,
    }
    return panel, features_df, provenance


# =====================================================================
# ABLATION_2: Global StandardScaler patch on LR
# =====================================================================

class _LogisticRegressionL2_GlobalScaler(LogisticRegressionL2):
    """LogisticRegressionL2 with the per-fold StandardScaler swapped out
    for a pre-fitted global StandardScaler that was fit on the ENTIRE
    feature matrix (training + every fold's test rows combined).

    The global scaler is supplied via the `global_scaler` constructor
    kwarg. It must already be fit; we just slot it in front of LR
    instead of letting Pipeline refit a new scaler per fold.

    Inheriting from LogisticRegressionL2 keeps the seed handling,
    single-class fallback, and predict_proba semantics aligned with the
    project baseline.
    """

    def __init__(
        self,
        seed: int = 42,
        global_scaler: StandardScaler | None = None,
        **_unused,
    ):
        super().__init__(seed=seed)
        if global_scaler is None:
            raise ValueError(
                "global_scaler is required for ablation_2 LR variant"
            )
        self._global_scaler = global_scaler

    def fit(self, X, y):
        from seet.baselines import _to_array  # local import; private helper
        Xa = _to_array(X)
        y_arr = np.asarray(y).astype(int)
        unique = np.unique(y_arr)
        if unique.size < 2:
            self._single_class = float(unique[0]) if unique.size == 1 else 0.0
            self.pipeline_ = None
            return self
        self._single_class = None
        # Pre-transform with the GLOBAL scaler, then fit LR on the scaled
        # training rows. Skipping the StandardScaler step in the pipeline
        # because we've already applied it.
        Xa_scaled = self._global_scaler.transform(Xa)
        self.pipeline_ = Pipeline([
            ("lr", LogisticRegression(
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
                random_state=self.seed,
            )),
        ])
        self.pipeline_.fit(Xa_scaled, y_arr)
        return self

    def predict_proba(self, X):
        from seet.baselines import _to_array, _binary_proba
        Xa = _to_array(X)
        if self.pipeline_ is None:
            n = Xa.shape[0]
            return _binary_proba(np.full(n, self._single_class))
        Xa_scaled = self._global_scaler.transform(Xa)
        return self.pipeline_.predict_proba(Xa_scaled)


def _fit_global_scaler(features_df: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the entire feature matrix (every row of
    every fold). Drops rows with any NaN feature (these are excluded
    from LR/LightGBM in the unmodified pipeline anyway, so the scaler
    sees the same row population it would in production)."""
    feat_cols = [c for c in features_df.columns if c != "Date"]
    X_full = features_df[feat_cols].dropna(how="any")
    sc = StandardScaler()
    sc.fit(X_full.to_numpy(dtype=float))
    return sc


# =====================================================================
# Per-fold fit + predict (parameterized for ablations 2 & 4)
# =====================================================================

def _fit_predict_one_ablation(
    model_name: str,
    seed: int,
    panel_df: pd.DataFrame,
    features_df: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    labels: np.ndarray,
    *,
    mode: str,
    global_scaler: StandardScaler | None = None,
    frozen_models: dict | None = None,
    model_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit-and-predict tailored to an ablation mode.

    mode == "global_scaler"
        For LogisticRegressionL2, the StandardScaler is replaced by a
        global, pre-fit scaler. Other models are unchanged.
    mode == "frozen_model"
        Skip the .fit() step and call .predict_proba() on the pre-fit
        model from frozen_models[(model_name, seed)]. The fold's
        training data is still used to compute the threshold base
        (training scores), so the L2 alarm rate is comparable to the
        unmodified pipeline; only the model's parameters are frozen.
    Other values of mode delegate to the unmodified _fit_predict_one
    (no patching needed -- these ablations affect aggregation, not
    fitting).
    """
    extra_kwargs = dict(model_kwargs or {})
    feat_cols = [c for c in features_df.columns if c != "Date"]
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    n_train_full = train_idx.size
    n_test_full = test_idx.size

    train_scores = np.full(n_train_full, np.nan)
    test_scores = np.full(n_test_full, np.nan)

    is_matrix = model_name in ("LogisticRegressionL2", "LightGbmTuned")

    if mode == "frozen_model":
        # Use the pre-fit frozen model for this seed.
        frozen = frozen_models.get((model_name, seed))
        if frozen is None:
            return train_scores, test_scores
        if is_matrix:
            train_feat_nan = (
                features_df.iloc[train_idx][feat_cols].isna().any(axis=1).to_numpy()
            )
            test_feat_nan = (
                features_df.iloc[test_idx][feat_cols].isna().any(axis=1).to_numpy()
            )
            train_keep = ~train_feat_nan
            if train_keep.any():
                X_tr = features_df.iloc[train_idx[train_keep]][feat_cols]
                train_scores[train_keep] = frozen.predict_proba(X_tr)[:, 1]
            test_keep = ~test_feat_nan
            if test_keep.any():
                X_te = features_df.iloc[test_idx[test_keep]][feat_cols]
                test_scores[test_keep] = frozen.predict_proba(X_te)[:, 1]
            return train_scores, test_scores
        # Panel-DataFrame baselines: predict_proba on the panel slice.
        X_tr = panel_df.iloc[train_idx]
        X_te = panel_df.iloc[test_idx]
        train_scores = frozen.predict_proba(X_tr)[:, 1]
        test_scores = frozen.predict_proba(X_te)[:, 1]
        return train_scores, test_scores

    if mode == "global_scaler":
        if model_name == "LogisticRegressionL2":
            if global_scaler is None:
                raise RuntimeError("global_scaler not provided for ablation_2")
            train_feat_nan = (
                features_df.iloc[train_idx][feat_cols].isna().any(axis=1).to_numpy()
            )
            test_feat_nan = (
                features_df.iloc[test_idx][feat_cols].isna().any(axis=1).to_numpy()
            )
            train_keep_local = ~train_feat_nan
            if train_keep_local.sum() < 10:
                return train_scores, test_scores
            X_tr = features_df.iloc[train_idx[train_keep_local]][feat_cols]
            y_tr = labels[train_idx[train_keep_local]]
            model = _LogisticRegressionL2_GlobalScaler(
                seed=seed, global_scaler=global_scaler
            )
            model.fit(X_tr, y_tr)
            train_scores[train_keep_local] = model.predict_proba(X_tr)[:, 1]
            test_keep_local = ~test_feat_nan
            if test_keep_local.any():
                X_te = features_df.iloc[test_idx[test_keep_local]][feat_cols]
                test_scores[test_keep_local] = model.predict_proba(X_te)[:, 1]
            return train_scores, test_scores
        # All other models: unchanged.

    # Default behavior: fit a fresh model from the registry.
    if is_matrix:
        train_feat_nan = (
            features_df.iloc[train_idx][feat_cols].isna().any(axis=1).to_numpy()
        )
        test_feat_nan = (
            features_df.iloc[test_idx][feat_cols].isna().any(axis=1).to_numpy()
        )
        train_keep_local = ~train_feat_nan
        if train_keep_local.sum() < 10:
            return train_scores, test_scores
        X_tr = features_df.iloc[train_idx[train_keep_local]][feat_cols]
        y_tr = labels[train_idx[train_keep_local]]
        model = make_model(model_name, seed=seed, **extra_kwargs)
        model.fit(X_tr, y_tr)
        train_scores[train_keep_local] = model.predict_proba(X_tr)[:, 1]
        test_keep_local = ~test_feat_nan
        if test_keep_local.any():
            X_te = features_df.iloc[test_idx[test_keep_local]][feat_cols]
            test_scores[test_keep_local] = model.predict_proba(X_te)[:, 1]
        return train_scores, test_scores

    X_tr = panel_df.iloc[train_idx]
    X_te = panel_df.iloc[test_idx]
    y_tr = labels[train_idx]
    model = make_model(model_name, seed=seed, **extra_kwargs)
    model.fit(X_tr, y_tr)
    train_scores = model.predict_proba(X_tr)[:, 1]
    test_scores = model.predict_proba(X_te)[:, 1]
    return train_scores, test_scores


# =====================================================================
# ABLATION_4: pre-fit each model once on the first training window
# =====================================================================

def _fit_frozen_models(
    panel_df: pd.DataFrame,
    features_df: pd.DataFrame,
    labels: np.ndarray,
    initial_train_end: pd.Timestamp,
    seeds: tuple[int, ...],
    primary_seed: int,
    deterministic_models: tuple[str, ...],
    stochastic_models: tuple[str, ...],
    model_kwargs_factory: Callable[..., dict],
    stress_def_name: str,
    first_outer_fold: int,
) -> dict:
    """Fit each (model, seed) once on data <= initial_train_end and
    return a dict keyed by (model_name, seed) -> fitted model.

    Deterministic models are fit with primary_seed and replicated
    across the seed slots in the returned dict (preserves Track A's
    convention that deterministic-model rows carry every seed).

    For LightGbmTuned, the per-(stress_def, outer_fold=1) tuned HPs are
    used (matching the HPs Track A used for fold 1)."""
    panel_dates = pd.to_datetime(panel_df["Date"])
    feat_cols = [c for c in features_df.columns if c != "Date"]
    train_mask_dates = (panel_dates <= initial_train_end).to_numpy()
    label_nan = np.isnan(labels)
    train_mask = train_mask_dates & ~label_nan

    train_idx = np.where(train_mask)[0]
    train_feat_nan = (
        features_df.iloc[train_idx][feat_cols].isna().any(axis=1).to_numpy()
    )

    frozen: dict = {}

    for model_name in deterministic_models:
        kwargs = model_kwargs_factory(
            model_name, stress_def=stress_def_name, outer_fold=first_outer_fold
        )
        if model_name in ("LogisticRegressionL2",):
            keep = ~train_feat_nan
            if keep.sum() < 10:
                m = None
            else:
                X_tr = features_df.iloc[train_idx[keep]][feat_cols]
                y_tr = labels[train_idx[keep]]
                m = make_model(model_name, seed=primary_seed, **kwargs)
                m.fit(X_tr, y_tr)
        else:
            X_tr = panel_df.iloc[train_idx]
            y_tr = labels[train_idx]
            m = make_model(model_name, seed=primary_seed, **kwargs)
            m.fit(X_tr, y_tr)
        for s in seeds:
            frozen[(model_name, s)] = m

    for model_name in stochastic_models:
        for s in seeds:
            kwargs = model_kwargs_factory(
                model_name, stress_def=stress_def_name, outer_fold=first_outer_fold
            )
            keep = ~train_feat_nan
            if keep.sum() < 10:
                m = None
            else:
                X_tr = features_df.iloc[train_idx[keep]][feat_cols]
                y_tr = labels[train_idx[keep]]
                m = make_model(model_name, seed=s, **kwargs)
                m.fit(X_tr, y_tr)
            frozen[(model_name, s)] = m
    return frozen


# =====================================================================
# Layer-2 with test-set top-5% threshold (ABLATION_3)
# =====================================================================

def _layer2_test_threshold(
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    test_future_drawdown: np.ndarray,
) -> dict:
    """Layer 2 alarms with the v1-paper threshold: top 5% of TEST
    scores (a peeking threshold). Mathematically pins the realized
    alarm rate to ~5% by construction."""
    valid = ~(np.isnan(test_scores) | np.isnan(test_labels))
    s = test_scores[valid]
    y = test_labels[valid].astype(int)
    fd = test_future_drawdown[valid]
    n_test = s.size

    base = {
        "threshold": np.nan,
        "alarm_rate": np.nan,
        "alarm_precision": np.nan,
        "drawdown_lift": np.nan,
        "severity_lift": np.nan,
        "n_alarms": 0,
        "n_events": 0,
        "n_test_eff": n_test,
    }
    if n_test == 0:
        return base
    threshold = float(np.percentile(s, 95.0))
    alarms = s > threshold
    n_alarms = int(alarms.sum())
    events = y == 1
    n_events = int(events.sum())

    base["threshold"] = threshold
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


# =====================================================================
# The public ablation runner
# =====================================================================

def run_ablation(
    mode: str,
    panel_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    price_col: str,
    stress_def: dict,
    seeds: tuple[int, ...],
    primary_seed: int,
    initial_train_end: pd.Timestamp,
    deterministic_models: tuple[str, ...] = DETERMINISTIC_MODELS,
    stochastic_models: tuple[str, ...] = STOCHASTIC_MODELS,
    model_kwargs_factory: Callable[..., dict] | None = None,
    min_train_rows: int = 30,
    min_test_rows: int = 5,
) -> pd.DataFrame:
    """Run the headline grid (h10_d05 only) under one ablation mode.

    Parameters
    ----------
    mode : str
        One of ABLATION_IDS. For "no_validators", caller must pass the
        no-validator (panel_df, features_df) pair (built by
        build_no_validator_panel). For the other three modes, pass the
        Track A canonical pair.
    panel_df, features_df : pd.DataFrame
        Aligned by Date.
    price_col : str
        Panel column for stress label computation (Track E uses "SPX").
    stress_def : dict
        Single stress definition {"name", "h", "d"}. Track E uses the
        h10_d05 central case only.
    seeds, primary_seed, initial_train_end :
        Track A's values are passed verbatim to keep pairing correct.
    deterministic_models, stochastic_models :
        Project default registries from seet.baselines.
    model_kwargs_factory : callable or None
        Same as pipeline.run_grid. Track A passes one that supplies
        the Track B HP CSV path for LightGbmTuned; pass-through here.

    Returns
    -------
    per_fold_df : pd.DataFrame with the same schema as Track A's
        per_fold_metrics.csv, plus an "ablation_id" column. Always for
        the single stress_def supplied; folds are the Track A folds
        derived by build_folds_with_init.
    """
    if mode not in ABLATION_IDS:
        raise ValueError(f"Unknown ablation mode: {mode!r}; expected {ABLATION_IDS}")
    if model_kwargs_factory is None:
        model_kwargs_factory = lambda model_name, **_: {}

    panel_dates = pd.to_datetime(panel_df["Date"])
    folds = build_folds_with_init(panel_dates, initial_train_end)

    sd_name, h, d = stress_def["name"], stress_def["h"], stress_def["d"]
    prices = panel_df[price_col].to_numpy(dtype=float)
    future_drawdown, labels = compute_stress_labels(prices, h, d)
    label_nan = np.isnan(labels)

    # Pre-fit the global scaler if needed.
    global_scaler = None
    if mode == "global_scaler":
        global_scaler = _fit_global_scaler(features_df)

    # Pre-fit the frozen models if needed.
    frozen_models: dict | None = None
    if mode == "frozen_model":
        first_fold_id = folds[0]["fold_id"] if folds else 1
        frozen_models = _fit_frozen_models(
            panel_df, features_df, labels,
            initial_train_end=initial_train_end,
            seeds=seeds, primary_seed=primary_seed,
            deterministic_models=deterministic_models,
            stochastic_models=stochastic_models,
            model_kwargs_factory=model_kwargs_factory,
            stress_def_name=sd_name,
            first_outer_fold=first_fold_id,
        )

    rows: list[dict] = []

    for fold in folds:
        train_mask_dates = (panel_dates <= fold["train_end"]).to_numpy()
        test_mask_dates = (
            (panel_dates >= fold["test_start"])
            & (panel_dates <= fold["test_end"])
        ).to_numpy()
        train_mask = train_mask_dates & ~label_nan
        test_mask = test_mask_dates & ~label_nan
        if train_mask.sum() < min_train_rows or test_mask.sum() < min_test_rows:
            continue

        test_fd = future_drawdown[test_mask]
        test_labels = labels[test_mask]

        for model_name in deterministic_models:
            kwargs = model_kwargs_factory(
                model_name, stress_def=sd_name, outer_fold=fold["fold_id"]
            )
            train_scores, test_scores = _fit_predict_one_ablation(
                model_name, primary_seed, panel_df, features_df,
                train_mask, test_mask, labels,
                mode=mode,
                global_scaler=global_scaler,
                frozen_models=frozen_models,
                model_kwargs=kwargs,
            )
            l1 = compute_layer1(test_scores, test_labels)
            if mode == "test_threshold":
                l2 = _layer2_test_threshold(test_scores, test_labels, test_fd)
            else:
                l2 = compute_layer2(test_scores, test_labels, train_scores, test_fd)
            base_row = {
                "ablation_id": mode,
                "stress_def": sd_name,
                "model": model_name,
                "fold_id": fold["fold_id"],
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                **l1, **l2,
            }
            for s in seeds:
                rows.append({**base_row, "seed": s})

        for model_name in stochastic_models:
            for s in seeds:
                kwargs = model_kwargs_factory(
                    model_name, stress_def=sd_name, outer_fold=fold["fold_id"]
                )
                train_scores, test_scores = _fit_predict_one_ablation(
                    model_name, s, panel_df, features_df,
                    train_mask, test_mask, labels,
                    mode=mode,
                    global_scaler=global_scaler,
                    frozen_models=frozen_models,
                    model_kwargs=kwargs,
                )
                l1 = compute_layer1(test_scores, test_labels)
                if mode == "test_threshold":
                    l2 = _layer2_test_threshold(test_scores, test_labels, test_fd)
                else:
                    l2 = compute_layer2(test_scores, test_labels, train_scores, test_fd)
                rows.append(
                    {
                        "ablation_id": mode,
                        "stress_def": sd_name,
                        "model": model_name,
                        "seed": s,
                        "fold_id": fold["fold_id"],
                        "n_train": int(train_mask.sum()),
                        "n_test": int(test_mask.sum()),
                        **l1, **l2,
                    }
                )

    return pd.DataFrame(rows)
