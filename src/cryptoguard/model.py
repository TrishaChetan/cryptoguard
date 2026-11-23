"""Unsupervised behavioural model: learns what *normal* looks like for a server."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import OneClassSVM

from .features import BEHAVIOURAL_FEATURES, DERIVED_FEATURES, WORKLOAD_DRIVERS


@dataclass
class BaselineModel:
    algorithm: str = "isolation_forest"
    n_estimators: int = 300
    max_samples: int | str = 2048
    cusum_k: float = 1.0            # allowance (in score-scale units) before evidence accumulates
    compute_gate: float | None = 0.75  # XAI gate used when calibrating the CUSUM (None = ungated)
    threshold_quantile: float = 0.995
    random_state: int = 42
    cpu_model: str = "ridge"        # ridge | hgb : learns CPU expected from workload
    features: list[str] = field(default_factory=lambda: list(BEHAVIOURAL_FEATURES) + list(DERIVED_FEATURES))

    cpu_regressor: object | None = None
    scaler: RobustScaler | None = None
    estimator: object | None = None
    threshold: float = 0.0
    score_scale: float = 1.0
    score_median: float = 0.0
    cusum_h: float = 0.0
    normal_stats: dict = field(default_factory=dict)
    train_scores: np.ndarray | None = None

    # ------------------------------------------------------------------ train
    def fit(self, normal: pd.DataFrame, calibration: pd.DataFrame | None = None,
            cusum_exclude: np.ndarray | None = None) -> "BaselineModel":
        """Learn the normal profile.

        ``cusum_exclude`` marks training windows that belong to *registered*
        scheduled jobs (maintenance windows). They are still part of the
        learned normal profile, but they do not set the CUSUM decision
        interval, because at run time drift evidence is paused inside those
        windows (see Detector.maintenance)."""
        # 1) learn how much CPU the normal workload *should* need
        drivers = self._drivers(normal)
        if self.cpu_model == "ridge":     # extrapolates sensibly to unseen traffic peaks
            self.cpu_regressor = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        else:
            self.cpu_regressor = HistGradientBoostingRegressor(max_iter=200, max_depth=6,
                                                               random_state=self.random_state)
        self.cpu_regressor.fit(drivers, normal["cpu_percent"].to_numpy(dtype=float))
        normal = self.with_derived(normal)
        # 2) learn the joint distribution of all behavioural features
        X = normal[self.features].to_numpy(dtype=float)
        self.scaler = RobustScaler().fit(X)
        Xs = self.scaler.transform(X)
        if self.algorithm == "isolation_forest":
            self.estimator = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples,
                                             contamination="auto", random_state=self.random_state,
                                             n_jobs=-1).fit(Xs)
        elif self.algorithm == "ocsvm":
            sub = Xs if len(Xs) <= 4000 else Xs[np.random.default_rng(self.random_state)
                                                  .choice(len(Xs), 4000, replace=False)]
            self.estimator = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01).fit(sub)
        else:
            raise ValueError(f"unknown algorithm {self.algorithm}")

        calib = calibration if calibration is not None else normal
        cal_scores = self.score(calib)
        self.threshold = float(np.quantile(cal_scores, self.threshold_quantile))
        self.score_median = float(np.median(cal_scores))
        spread = np.quantile(cal_scores, 0.95) - self.score_median
        self.score_scale = float(max(spread, 1e-6))
        self.train_scores = cal_scores
        self._fit_normal_stats(normal)
        self.calibrate_cusum(normal, calib, cusum_exclude)
        return self

    # ------------------------------------------------------------ CUSUM calib
    def cusum_shares(self, df: pd.DataFrame) -> np.ndarray:
        """XAI compute-share for the windows that would add CUSUM evidence (1.0 elsewhere)."""
        scores = self.score(df)
        share = np.ones(len(df))
        if self.algorithm == "isolation_forest":
            from .explainer import Explainer          # local import: avoids a cycle
            idx = np.flatnonzero([self.cusum_increment(float(s)) > 0 for s in scores])
            if len(idx):
                share[idx] = Explainer(self).compute_share(df.iloc[idx])
        return share

    def calibrate_cusum(self, normal: pd.DataFrame, calib: pd.DataFrame,
                        cusum_exclude: np.ndarray | None = None,
                        shares: tuple[np.ndarray, np.ndarray] | None = None) -> float:
        """Decision interval h = 1.2 x the largest CUSUM value reached on normal data
        (training + calibration), using the same XAI gate as at run time.
        ``shares`` lets callers reuse precomputed compute-shares (they do not
        depend on the gate value)."""
        gated = self.compute_gate is not None
        if shares is None:
            shares = (self.cusum_shares(normal), self.cusum_shares(calib)) if gated else \
                     (np.ones(len(normal)), np.ones(len(calib)))
        keep = np.ones(len(normal), dtype=bool) if cusum_exclude is None else ~np.asarray(cusum_exclude, bool)
        best = 0.0
        for df, sh, mask in ((normal, shares[0], keep), (calib, shares[1], np.ones(len(calib), bool))):
            scores = self.score(df)[mask]
            S = 0.0
            for sc, sv in zip(scores, np.asarray(sh)[mask]):
                inc = self.cusum_increment(float(sc))
                if gated and inc > 0 and sv < self.compute_gate:
                    inc = -self.cusum_k
                S = max(0.0, S + inc)
                best = max(best, S)
        self.cusum_h = float(1.2 * max(best, 1.0))
        return self.cusum_h

    def _fit_normal_stats(self, normal: pd.DataFrame) -> None:
        self.normal_stats = {
            f: {"median": float(normal[f].median()),
                "p05": float(normal[f].quantile(0.05)),
                "p95": float(normal[f].quantile(0.95))}
            for f in self.features
        }
        # per-hour CPU profile lets the explainer say "normal for 03:00 is ~12%"
        if "hour" in normal:
            prof = normal.groupby(normal["hour"].astype(int))["cpu_percent"].median()
            self.normal_stats["_cpu_by_hour"] = {int(k): float(v) for k, v in prof.items()}

    # ------------------------------------------------------------------ score
    @staticmethod
    def _drivers(df: pd.DataFrame) -> np.ndarray:
        return df[WORKLOAD_DRIVERS].to_numpy(dtype=float)

    def with_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the learned feature cpu_excess = CPU used beyond what the workload needs.

        One-sided on purpose: a miner only ever *adds* CPU, while using less
        CPU than expected (caching, idle workers) is harmless."""
        out = df.copy()
        expected = np.clip(self.cpu_regressor.predict(self._drivers(df)), 0, 100)
        out["cpu_expected"] = expected
        out["cpu_excess"] = np.clip(df["cpu_percent"].to_numpy(dtype=float) - expected, 0, None)
        return out

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = self.with_derived(df)
        return self.scaler.transform(df[self.features].to_numpy(dtype=float))

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Anomaly score: higher = more abnormal."""
        Xs = self.transform(df)
        return -self.estimator.score_samples(Xs)

    def behavioural_confidence(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to [0, 1]; 0.5 exactly at the calibrated threshold."""
        z = (np.asarray(scores) - self.threshold) / self.score_scale
        return 0.5 + 0.5 * np.tanh(z)

    # ---------------------------------------------------------------- CUSUM
    def cusum_increment(self, score: float) -> float:
        """Evidence a window adds towards a *low-and-slow* alarm.

        Windows below the 95th percentile of normal add nothing; increments
        are clipped at the point-anomaly threshold, because strong anomalies
        are already caught window-by-window."""
        z = (score - self.score_median) / self.score_scale
        z_max = (self.threshold - self.score_median) / self.score_scale
        return min(z, z_max) - self.cusum_k

    def cusum_path(self, scores: np.ndarray, cap: float | None = None) -> np.ndarray:
        S, out = 0.0, np.empty(len(scores))
        for i, sc in enumerate(scores):
            S = max(0.0, S + self.cusum_increment(float(sc)))
            if cap is not None:
                S = min(S, cap)
            out[i] = S
        return out

    # ------------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "BaselineModel":
        return joblib.load(path)
