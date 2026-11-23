"""Decision layer: turns model scores + threat intel into a cryptojacking verdict.

Three ideas are combined:

1. Two time-scales of behavioural evidence
   * point anomaly  - one window is clearly abnormal (Isolation Forest score
                      above the calibrated threshold), confirmed when it repeats
                      in ``k`` of the last ``n`` windows;
   * drift anomaly  - a CUSUM accumulator over the anomaly score catches
                      *throttled* miners whose individual windows only look
                      mildly unusual but stay that way for a long time. Only
                      windows that pass the XAI gate add evidence.

2. Explanation-guided gate (XAI in the loop)
   An anomaly only counts as *cryptojacking-like* when at least
   ``compute_gate`` of its SHAP attribution falls on compute features (CPU
   level, CPU not explained by traffic, flatness, single-process dominance).
   Anomalies explained by traffic or process churn - a marketing surge, a
   CI build - are reported as generic anomalies and never trigger a response.

3. Evidence fusion with threat intel (noisy-OR)
       confidence = 1 - (1 - b) * prod_i (1 - w_i * [evidence_i])
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .features import INTEL_FEATURES
from .model import BaselineModel

DEFAULT_WEIGHTS = {"pool_port_conns": 0.5, "miner_name_hits": 0.5, "blocklist_ip_conns": 0.45}


def make_maintenance(windows: list[dict] | None):
    """Build a maintenance-window test from config entries such as
    ``{"days": [6], "start": "04:00", "end": "05:15"}`` (0 = Monday)."""
    if not windows:
        return None
    parsed = []
    for w in windows:
        sh, sm = (int(x) for x in str(w["start"]).split(":"))
        eh, em = (int(x) for x in str(w["end"]).split(":"))
        parsed.append((set(w.get("days", range(7))), sh + sm / 60, eh + em / 60))

    def inside(row: dict) -> bool:
        wd = row.get("weekday")
        wd = int(wd) if wd is not None and wd == wd else int(row.get("day", 0)) % 7
        hour = float(row.get("hour", -1))
        return any(wd in days and start <= hour < end for days, start, end in parsed)
    return inside


@dataclass
class Verdict:
    score: float
    behavioural_conf: float
    point_anomaly: bool
    cusum: float
    drift_anomaly: bool
    compute_share: float
    mining_like: bool
    persistent: bool
    intel_conf: float
    confidence: float
    alert: bool
    cpu_excess: float = 0.0

    @property
    def behavioural_anomaly(self) -> bool:
        return self.point_anomaly or self.drift_anomaly

    def to_dict(self) -> dict:
        d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}
        d["behavioural_anomaly"] = self.behavioural_anomaly
        return d


class Detector:
    def __init__(self, model: BaselineModel, persistence_k: int = 3, persistence_n: int = 5,
                 intel_weights: dict | None = None, alert_threshold: float = 0.5,
                 use_intel: bool = True, use_persistence: bool = True, use_cusum: bool = True,
                 explainer=None, compute_gate: float | None = None, drift_tau: float = 20.0,
                 maintenance=None):
        self.model = model
        self.k, self.n = persistence_k, persistence_n
        self.weights = dict(DEFAULT_WEIGHTS, **(intel_weights or {}))
        self.alert_threshold = alert_threshold
        self.use_intel = use_intel
        self.use_persistence = use_persistence
        self.use_cusum = use_cusum
        self.explainer = explainer
        gate = compute_gate if compute_gate is not None else getattr(model, "compute_gate", 0.7)
        self.compute_gate = gate if explainer is not None else None
        self.drift_tau = drift_tau          # windows for drift confidence to build up
        self.maintenance = maintenance      # callable(row: dict) -> bool, registered scheduled jobs
        self.reset()

    def reset(self) -> None:
        self._history: deque[bool] = deque(maxlen=self.n)
        self._S = 0.0
        self._drift_run = 0

    # ---------------------------------------------------------------- pieces
    def intel_confidence(self, row: dict) -> float:
        if not self.use_intel:
            return 0.0
        keep = 1.0
        for f in INTEL_FEATURES:
            if float(row.get(f, 0) or 0) > 0:
                keep *= 1.0 - self.weights.get(f, 0.0)
        return 1.0 - keep

    def _needs_share(self, score: float) -> bool:
        """SHAP is only needed for windows that could raise an alarm."""
        return score > self.model.threshold or (self.use_cusum and self.model.cusum_increment(score) > 0)

    def _gate_ok(self, share: float) -> bool:
        return self.compute_gate is None or share >= self.compute_gate

    def _cusum_step(self, score: float, share: float, row: dict | None = None) -> float:
        if not self.use_cusum:
            return 0.0
        h, k = self.model.cusum_h, self.model.cusum_k
        inc = self.model.cusum_increment(score)
        if inc > 0 and not self._gate_ok(share):
            inc = -k                     # abnormal, but not mining-like: no evidence
        if inc > 0 and self.maintenance is not None and row is not None and self.maintenance(row):
            inc = 0.0                    # registered scheduled job: drift evidence paused
        # cap slightly above h so the alarm clears a few windows after the attack stops
        self._S = min(h + 5 * k, max(0.0, self._S + inc))
        return self._S

    def _share(self, frame: pd.DataFrame) -> float:
        if self.compute_gate is None:
            return 1.0
        return float(self.explainer.compute_share(frame)[0])

    def _decide(self, score: float, S: float, row: dict, share: float) -> Verdict:
        b = float(self.model.behavioural_confidence(np.array([score]))[0])
        point = score > self.model.threshold
        drift = bool(self.use_cusum and S > self.model.cusum_h)   # evidence already gated
        gate_ok = self._gate_ok(share)
        mining_like = (point and gate_ok) or drift

        self._history.append(point and gate_ok)
        persistent = (sum(self._history) >= self.k) if self.use_persistence else (point and gate_ok)
        self._drift_run = self._drift_run + 1 if drift else 0

        b_point = b if persistent else 0.0
        b_drift = 0.5 + 0.45 * (1 - math.exp(-self._drift_run / self.drift_tau)) if self._drift_run else 0.0
        b_eff = max(b_point, b_drift)
        if not mining_like:
            b_eff = min(b_eff, 0.49)
        intel = self.intel_confidence(row)
        conf = 1.0 - (1.0 - b_eff) * (1.0 - intel)
        return Verdict(score, b, bool(point), float(S), bool(drift), float(share), bool(mining_like),
                       bool(persistent), intel, conf, conf >= self.alert_threshold,
                       float(row.get("cpu_excess", 0.0)))

    # ------------------------------------------------------------------ APIs
    def process(self, frame: pd.DataFrame) -> Verdict:
        """Streaming use: one window (1-row DataFrame) at a time."""
        score = float(self.model.score(frame)[0])
        share = self._share(frame) if self._needs_share(score) else 0.0
        row = self.model.with_derived(frame).iloc[0].to_dict()
        S = self._cusum_step(score, share, row)
        return self._decide(score, S, row, share)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch use for evaluation; identical decisions to calling process() row by row."""
        self.reset()
        scores = self.model.score(df)
        need = np.array([self._needs_share(float(s)) for s in scores])
        shares = np.zeros(len(df))
        if self.compute_gate is None:
            shares[:] = 1.0
        elif need.any():
            idx = np.flatnonzero(need)
            shares[idx] = self.explainer.compute_share(df.iloc[idx])
        out = []
        for sc, r, c in zip(scores, self.model.with_derived(df).to_dict("records"), shares):
            S = self._cusum_step(float(sc), float(c), r)
            out.append(self._decide(float(sc), S, r, float(c)))
        frame = pd.DataFrame([asdict(v) for v in out], index=df.index)
        frame["behavioural_anomaly"] = frame["point_anomaly"] | frame["drift_anomaly"]
        return frame
