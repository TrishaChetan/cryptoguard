"""Explainable-AI layer: turns a model score into reasons a cloud admin can read.

For the Isolation Forest the explanation uses exact TreeSHAP values
(Lundberg et al.). In an Isolation Forest a *short* path means "easy to
isolate" = anomalous, so a feature's contribution to the anomaly is the
negative of its SHAP value on path length. For other models a robust
z-score against the learned normal profile is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import COMPUTE_FEATURES, FEATURE_LABELS
from .model import BaselineModel

try:  # SHAP is optional at runtime
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None


@dataclass
class Reason:
    feature: str
    contribution: float     # share of the anomaly attributed to this feature (0-1)
    text: str


class Explainer:
    def __init__(self, model: BaselineModel, top_k: int = 3):
        self.model = model
        self.top_k = top_k
        self._tree = None
        if shap is not None and model.algorithm == "isolation_forest":
            try:
                self._tree = shap.TreeExplainer(model.estimator)
            except Exception:
                self._tree = None

    @property
    def method(self) -> str:
        return "TreeSHAP" if self._tree is not None else "robust z-score"

    # ------------------------------------------------------------ attribution
    def attribution_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """(n_rows, n_features) contributions towards 'anomalous'."""
        feats = self.model.features
        if self._tree is not None:
            return -np.asarray(self._tree.shap_values(self.model.transform(df)))
        d = self.model.with_derived(df)
        cols = []
        for f in feats:
            st = self.model.normal_stats[f]
            spread = max(st["p95"] - st["p05"], 1e-6)
            cols.append(np.abs(d[f].to_numpy(dtype=float) - st["median"]) / spread)
        return np.column_stack(cols)

    def attributions(self, row: pd.DataFrame) -> dict[str, float]:
        """Per-feature contribution for one window (positive = more anomalous)."""
        out = dict(zip(self.model.features, self.attribution_matrix(row)[0].astype(float)))
        # hour_sin / hour_cos are one concept for the reader
        out["hour_sin"] = out.get("hour_sin", 0.0) + out.pop("hour_cos", 0.0)
        return out

    def compute_share(self, df: pd.DataFrame) -> np.ndarray:
        """Fraction of the (positive) anomaly attribution carried by compute features.

        High -> the abnormality is about CPU behaviour (mining-like);
        low  -> it is about traffic / process churn (workload change)."""
        A = np.clip(self.attribution_matrix(df), 0, None)
        feats = self.model.features
        mask = np.array([f in COMPUTE_FEATURES or f == "hour_cos" for f in feats])
        tot = A.sum(axis=1)
        return np.where(tot > 0, A[:, mask].sum(axis=1) / np.maximum(tot, 1e-12), 0.0)

    # ---------------------------------------------------------------- wording
    def _describe(self, f: str, v: float, hour: float | None) -> str:
        st = self.model.normal_stats.get(f, {})
        med, lo, hi = st.get("median", 0), st.get("p05", 0), st.get("p95", 0)
        if f == "cpu_percent":
            prof = self.model.normal_stats.get("_cpu_by_hour", {})
            if hour is not None and int(hour) in prof:
                return (f"CPU usage is {v:.0f}%, while this server normally runs at about "
                        f"{prof[int(hour)]:.0f}% at {int(hour):02d}:00")
            return f"CPU usage is {v:.0f}% (normal range {lo:.0f}-{hi:.0f}%)"
        if f == "cpu_std":
            if v < lo:
                return (f"CPU load is unusually flat (variability {v:.1f} vs a normal {med:.1f}); "
                        "miners keep cores busy at a constant rate, real workloads are bursty")
            return f"CPU variability is {v:.1f} (normal {lo:.1f}-{hi:.1f})"
        if f == "cpu_sustained":
            return f"CPU has stayed above 80% for {v * 100:.0f}% of the last windows"
        if f == "top_proc_cpu":
            return f"a single process is using {v:.0f}% of total CPU (normal top process: ~{med:.0f}%)"
        if f == "top_proc_share":
            return f"one process accounts for {v * 100:.0f}% of all CPU use (normally ~{med * 100:.0f}%)"
        if f == "cpu_excess":
            return (f"the server is using {v:.0f} percentage points more CPU than its current traffic "
                    f"and process activity can explain (normally under {max(hi, 1):.0f})")
        if f == "hour_sin":
            return "this level of activity is unusual for the time of day"
        label = FEATURE_LABELS.get(f, f)
        direction = "higher" if v > med else "lower"
        return f"{label} is {v:.0f}, {direction} than normal ({lo:.0f}-{hi:.0f})"

    def explain(self, row: pd.DataFrame, context: dict | None = None) -> dict:
        """Return a structured + plain-English explanation for one window."""
        context = context or {}
        row = self.model.with_derived(row)
        contrib = self.attributions(row)
        positive = {k: v for k, v in contrib.items() if v > 0}
        total = sum(positive.values()) or 1.0
        hour = float(row["hour"].iloc[0]) if "hour" in row else None
        reasons = []
        for f, c in sorted(positive.items(), key=lambda kv: -kv[1])[: self.top_k]:
            val = float(row[f].iloc[0])
            reasons.append(Reason(f, c / total, self._describe(f, val, hour)))

        intel = []
        r = row.iloc[0]
        for proc in context.get("suspicious_processes", []):
            for why in proc.get("reasons", []):
                intel.append(f"process '{proc['name']}' (pid {proc['pid']}) {why}")
        if not intel:   # simulation mode has counts but no process objects
            name = context.get("top_process_name", "unknown")
            if r.get("miner_name_hits", 0) > 0:
                intel.append(f"process name '{name}' matches a known miner signature")
            if r.get("pool_port_conns", 0) > 0:
                intel.append(f"{int(r['pool_port_conns'])} connection(s) to a mining-pool (Stratum) port")
            if r.get("blocklist_ip_conns", 0) > 0:
                intel.append("connection to an IP on the mining-pool blocklist")

        parts = [x.text for x in reasons] + intel
        summary = "Looks like cryptojacking because " + "; ".join(parts) + "." if parts else \
                  "Behaviour deviates from the learned baseline."
        share = float(self.compute_share(row)[0])
        return {
            "method": self.method,
            "compute_share": round(share, 3),
            "summary": summary,
            "reasons": [{"feature": x.feature, "weight": round(x.contribution, 3), "text": x.text}
                        for x in reasons],
            "threat_intel": intel,
            "attributions": {k: round(v, 4) for k, v in contrib.items()},
        }
