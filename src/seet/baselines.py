"""Baselines for Track A.

Six classes implementing a uniform interface:

    model.fit(X, y) -> self
    model.predict_proba(X) -> ndarray of shape (n, 2), rows summing to 1

By convention column 0 is P(y=0) and column 1 is P(y=1). All classes
handle the single-class training case gracefully (fall back to the
constant prior).

The DataFrame-aware baselines (VIXPercentileRaw, VIXPercentileCalibrated,
HarRvThreshold) require X to be a pandas.DataFrame with the named
column(s); the matrix-style baselines (NaiveBaseRate, LogisticRegressionL2,
LightGbmTuned) accept either a DataFrame or a 2D ndarray.

Random seed flows through the `seed` argument; deterministic baselines
ignore it. Track A's runner cycles seed across {42..46} for stochastic
models and replicates deterministic results across the seed slots.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _to_array(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float)
    return np.asarray(X, dtype=float)


def _binary_proba(p1) -> np.ndarray:
    """Build (n, 2) probability matrix from a 1D vector of P(y=1)."""
    p1 = np.asarray(p1, dtype=float).ravel()
    p1 = np.clip(p1, 0.0, 1.0)
    return np.column_stack([1.0 - p1, p1])


def _length(X) -> int:
    if hasattr(X, "shape"):
        return int(X.shape[0])
    return int(len(X))


# ----------------------------------------------------------------------
# 1. NaiveBaseRate
# ----------------------------------------------------------------------

class NaiveBaseRate:
    """Predicts the training-period positive class rate, constant for
    every test row. NaN labels are dropped before computing the rate.
    With zero valid labels the rate falls back to 0.0."""

    def __init__(self, **_unused):
        self.base_rate_: float | None = None

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=float)
        valid = y_arr[~np.isnan(y_arr)]
        self.base_rate_ = float(valid.mean()) if valid.size else 0.0
        return self

    def predict_proba(self, X):
        n = _length(X)
        return _binary_proba(np.full(n, self.base_rate_))


# ----------------------------------------------------------------------
# 2. VIXPercentileRaw
# ----------------------------------------------------------------------

class VIXPercentileRaw:
    """Maps today's VIX to its empirical percentile in the training
    distribution. Score in [0, 1]. NaN VIX -> NaN score."""

    def __init__(self, vix_col: str = "VIX", **_unused):
        self.vix_col = vix_col
        self._sorted_train: np.ndarray | None = None

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame) or self.vix_col not in X.columns:
            raise ValueError(
                f"{type(self).__name__} requires a DataFrame with "
                f"column {self.vix_col!r}"
            )
        train_vix = X[self.vix_col].to_numpy(dtype=float)
        train_vix = train_vix[~np.isnan(train_vix)]
        self._sorted_train = np.sort(train_vix)
        return self

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame) or self.vix_col not in X.columns:
            raise ValueError(
                f"{type(self).__name__} requires a DataFrame with "
                f"column {self.vix_col!r}"
            )
        test_vix = X[self.vix_col].to_numpy(dtype=float)
        n = test_vix.size
        n_train = self._sorted_train.size
        p1 = np.full(n, np.nan)
        if n_train == 0:
            return _binary_proba(p1)
        valid = ~np.isnan(test_vix)
        if valid.any():
            ranks = np.searchsorted(self._sorted_train, test_vix[valid], side="right")
            p1[valid] = ranks / n_train
        return _binary_proba(p1)


# ----------------------------------------------------------------------
# 3. VIXPercentileCalibrated
# ----------------------------------------------------------------------

class VIXPercentileCalibrated:
    """VIXPercentileRaw composed with sklearn's IsotonicRegression.
    Falls back to raw scores when there isn't enough data for a
    monotonic fit (constant labels or fewer than 2 valid pairs)."""

    def __init__(self, vix_col: str = "VIX", **_unused):
        self.vix_col = vix_col
        self._raw = VIXPercentileRaw(vix_col=vix_col)
        self._iso: IsotonicRegression | None = None

    def fit(self, X, y):
        self._raw.fit(X, y)
        raw_scores = self._raw.predict_proba(X)[:, 1]
        y_arr = np.asarray(y, dtype=float)
        mask = ~(np.isnan(raw_scores) | np.isnan(y_arr))
        s = raw_scores[mask]
        t = y_arr[mask]
        if s.size < 2 or len(np.unique(t)) < 2:
            self._iso = None
            return self
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(s, t)
        return self

    def predict_proba(self, X):
        raw_scores = self._raw.predict_proba(X)[:, 1]
        if self._iso is None:
            return _binary_proba(raw_scores)
        valid = ~np.isnan(raw_scores)
        p1 = np.full(raw_scores.size, np.nan)
        if valid.any():
            p1[valid] = np.clip(self._iso.transform(raw_scores[valid]), 0.0, 1.0)
        return _binary_proba(p1)


# ----------------------------------------------------------------------
# 4. HarRvThreshold
# ----------------------------------------------------------------------

class HarRvThreshold:
    """HAR-RV (Corsi 2009) on log-returns-squared as a one-step-ahead RV
    forecast, then min-max scaled to [0, 1] using training-window bounds.

    Inputs at row t: RV_d = logret_t**2, RV_w = mean(RV over last 5 days),
    RV_m = mean(RV over last 22 days). OLS fit for RV at t+1.
    Score = clip((predicted_RV - train_min) / (train_max - train_min), 0, 1).
    """

    def __init__(self, price_col: str = "SPX", **_unused):
        self.price_col = price_col
        self.coef_: np.ndarray | None = None
        self.train_pred_min_: float | None = None
        self.train_pred_max_: float | None = None

    @staticmethod
    def _features(prices: np.ndarray) -> np.ndarray:
        """Return (n, 4) design matrix [1, RV_d, RV_w, RV_m]."""
        prev = np.empty_like(prices)
        prev[0] = np.nan
        prev[1:] = prices[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            logret = np.log(prices / prev)
        rv = logret ** 2
        rv_s = pd.Series(rv)
        rv_w = rv_s.rolling(5).mean().to_numpy()
        rv_m = rv_s.rolling(22).mean().to_numpy()
        return np.column_stack([np.ones(prices.size), rv, rv_w, rv_m])

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame) or self.price_col not in X.columns:
            raise ValueError(
                f"{type(self).__name__} requires a DataFrame with "
                f"column {self.price_col!r}"
            )
        prices = X[self.price_col].to_numpy(dtype=float)
        feats = self._features(prices)
        # Target: next-day RV
        rv_d = feats[:, 1]
        target = np.empty_like(rv_d)
        target[:-1] = rv_d[1:]
        target[-1] = np.nan
        mask = ~(np.isnan(feats).any(axis=1) | np.isnan(target))
        if mask.sum() < 4:
            self.coef_ = np.zeros(4)
            self.train_pred_min_ = 0.0
            self.train_pred_max_ = 1.0
            return self
        beta, *_ = np.linalg.lstsq(feats[mask], target[mask], rcond=None)
        self.coef_ = beta
        train_pred = feats[mask] @ beta
        self.train_pred_min_ = float(np.nanmin(train_pred))
        self.train_pred_max_ = float(np.nanmax(train_pred))
        return self

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame) or self.price_col not in X.columns:
            raise ValueError(
                f"{type(self).__name__} requires a DataFrame with "
                f"column {self.price_col!r}"
            )
        prices = X[self.price_col].to_numpy(dtype=float)
        feats = self._features(prices)
        nan_mask = np.isnan(feats).any(axis=1)
        pred = np.full(prices.size, np.nan)
        if (~nan_mask).any():
            pred[~nan_mask] = feats[~nan_mask] @ self.coef_
        rng = self.train_pred_max_ - self.train_pred_min_
        if rng > 0:
            scaled = (pred - self.train_pred_min_) / rng
        else:
            scaled = np.full_like(pred, 0.5)
        scaled = np.clip(scaled, 0.0, 1.0)
        scaled[nan_mask] = np.nan
        return _binary_proba(scaled)


# ----------------------------------------------------------------------
# 5. LogisticRegressionL2
# ----------------------------------------------------------------------

class LogisticRegressionL2:
    """sklearn LogisticRegression(class_weight='balanced') in a
    StandardScaler pipeline. L2 regularization (sklearn default).
    Falls back to constant prediction when training data has only one
    class."""

    def __init__(self, seed: int = 42, **_unused):
        self.seed = int(seed)
        self.pipeline_: Pipeline | None = None
        self._single_class: float | None = None

    def fit(self, X, y):
        Xa = _to_array(X)
        y_arr = np.asarray(y).astype(int)
        unique = np.unique(y_arr)
        if unique.size < 2:
            self._single_class = float(unique[0]) if unique.size == 1 else 0.0
            self.pipeline_ = None
            return self
        self._single_class = None
        # Note: penalty='l2' is the sklearn default; passing it explicitly
        # triggers a deprecation warning in sklearn 1.8+. We rely on the
        # default and configure regularization via C if/when needed.
        self.pipeline_ = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
                random_state=self.seed,
            )),
        ])
        self.pipeline_.fit(Xa, y_arr)
        return self

    def predict_proba(self, X):
        Xa = _to_array(X)
        if self.pipeline_ is None:
            n = Xa.shape[0]
            return _binary_proba(np.full(n, self._single_class))
        return self.pipeline_.predict_proba(Xa)


# ----------------------------------------------------------------------
# 6. LightGbmTuned
# ----------------------------------------------------------------------

class LightGbmTuned:
    """LightGBM. Hyperparameters are either Track A placeholder defaults
    or, if `stress_def`, `outer_fold`, and `hp_path` are all provided,
    loaded from `experiments/track_b_tuning/selected_hp.csv` (or
    equivalent) keyed by (stress_def, outer_fold). Loaded params are
    augmented with `class_weight='balanced'` if absent from the CSV.

    Falls back to a constant-prediction model when training data has
    only one class.

    Attributes
    ----------
    params : dict
        The actual LightGBM kwargs that will be passed to LGBMClassifier.
    _source : {"loaded", "default"}
        Where `params` came from. Useful for tests and provenance.
    """

    DEFAULT_HPS: dict = dict(
        max_depth=2,
        num_leaves=5,
        n_estimators=50,
        learning_rate=0.03,
        min_child_samples=80,
        reg_alpha=2.0,
        reg_lambda=10.0,
        class_weight="balanced",
    )

    def __init__(
        self,
        seed: int = 42,
        stress_def: str | None = None,
        outer_fold: int | None = None,
        hp_path=None,
        **kwargs,
    ):
        self.seed = int(seed)
        self.stress_def = stress_def
        self.outer_fold = (
            int(outer_fold) if outer_fold is not None else None
        )
        self.hp_path = hp_path

        loaded = None
        if (
            self.stress_def is not None
            and self.outer_fold is not None
            and self.hp_path is not None
        ):
            loaded = self._try_load(
                self.hp_path, self.stress_def, self.outer_fold
            )

        if loaded is not None:
            # class_weight='balanced' is the project-wide convention for
            # this baseline; ensure it's present even if the CSV doesn't
            # carry it (the Track B grid does not search over it).
            self.params = {"class_weight": "balanced", **loaded}
            self._source = "loaded"
        else:
            self.params = dict(self.DEFAULT_HPS)
            self._source = "default"

        # Allow explicit kwargs to override anything we just loaded.
        for k, v in kwargs.items():
            if k in self.params or k in self.DEFAULT_HPS:
                self.params[k] = v

        self.model_: lgb.LGBMClassifier | None = None
        self._single_class: float | None = None

    @staticmethod
    def _try_load(
        hp_path, stress_def: str, outer_fold: int
    ) -> "dict | None":
        """Look up the (stress_def, outer_fold) row in the selected_hp
        CSV. Returns the parsed best_params dict, or None on any
        failure (file missing, no matching row, malformed JSON)."""
        import json

        try:
            df = pd.read_csv(hp_path)
        except (FileNotFoundError, pd.errors.EmptyDataError, OSError):
            return None
        required = {"stress_def", "outer_fold", "best_params"}
        if not required.issubset(df.columns):
            return None
        match = df[
            (df["stress_def"] == stress_def)
            & (df["outer_fold"].astype(int) == int(outer_fold))
        ]
        if match.empty:
            return None
        try:
            return json.loads(str(match.iloc[0]["best_params"]))
        except (json.JSONDecodeError, ValueError):
            return None

    def fit(self, X, y):
        Xa = _to_array(X)
        y_arr = np.asarray(y).astype(int)
        unique = np.unique(y_arr)
        if unique.size < 2:
            self._single_class = float(unique[0]) if unique.size == 1 else 0.0
            self.model_ = None
            return self
        self._single_class = None
        self.model_ = lgb.LGBMClassifier(
            **self.params,
            random_state=self.seed,
            n_jobs=1,
            verbose=-1,
            verbosity=-1,
        )
        self.model_.fit(Xa, y_arr)
        return self

    def predict_proba(self, X):
        Xa = _to_array(X)
        if self.model_ is None:
            n = Xa.shape[0]
            return _binary_proba(np.full(n, self._single_class))
        proba = self.model_.predict_proba(Xa)
        if proba.shape[1] == 1:
            single = self.model_.classes_[0]
            p1 = proba[:, 0] if single == 1 else 1.0 - proba[:, 0]
            return _binary_proba(p1)
        return proba


# ----------------------------------------------------------------------
# Registry / factory
# ----------------------------------------------------------------------

DETERMINISTIC_MODELS: tuple[str, ...] = (
    "NaiveBaseRate",
    "VIXPercentileRaw",
    "VIXPercentileCalibrated",
    "HarRvThreshold",
    "LogisticRegressionL2",
)
STOCHASTIC_MODELS: tuple[str, ...] = ("LightGbmTuned",)
ALL_MODELS: tuple[str, ...] = DETERMINISTIC_MODELS + STOCHASTIC_MODELS


_MODEL_CLASSES = {
    "NaiveBaseRate": NaiveBaseRate,
    "VIXPercentileRaw": VIXPercentileRaw,
    "VIXPercentileCalibrated": VIXPercentileCalibrated,
    "HarRvThreshold": HarRvThreshold,
    "LogisticRegressionL2": LogisticRegressionL2,
    "LightGbmTuned": LightGbmTuned,
}


def make_model(name: str, seed: int = 42, **kwargs):
    """Factory for a fresh model instance by name."""
    if name not in _MODEL_CLASSES:
        raise ValueError(
            f"Unknown model name: {name!r}. Known: {sorted(_MODEL_CLASSES)}"
        )
    return _MODEL_CLASSES[name](seed=seed, **kwargs)
