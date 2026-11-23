"""Reproducible evaluation on simulated telemetry.

Compares CryptoGuard against common alternatives:

* static CPU alarm  - CPU > 90 % for 5 consecutive minutes (typical cloud alarm)
* signature only    - any threat-intel hit (miner name, pool port, blocklisted IP)
* OCSVM             - One-Class SVM on behavioural features (+ persistence)
* IForest           - Isolation Forest on behavioural features (+ persistence)
* CryptoGuard       - Isolation Forest + threat-intel fusion + persistence
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .detector import Detector, make_maintenance
from .explainer import Explainer
from .features import INTEL_FEATURES
from .model import BaselineModel
from .simulator import generate_normal, generate_test

ATTACKS = ["noisy", "stealthy", "container"]
HARD_NEG = ["traffic_surge", "ci_build"]
DEFAULT_ALLOW = {"systemd", "sshd", "init", "containerd", "dockerd", "kubelet", "nginx",
                 "gunicorn", "uwsgi", "postgres", "mysqld", "pg_dump", "tar", "gzip", "java"}


def static_cpu_alarm(df: pd.DataFrame, level: float = 90, minutes: int = 5) -> np.ndarray:
    above = (df["cpu_percent"] > level).astype(int)
    run = above.groupby((above != above.shift()).cumsum()).cumsum()
    return (run >= minutes).to_numpy()


def signature_only(df: pd.DataFrame) -> np.ndarray:
    return (df[INTEL_FEATURES].sum(axis=1) > 0).to_numpy()


def _incidents(mask: np.ndarray, gap: int = 6) -> list[tuple[int, int]]:
    """Group alert windows into incidents (alerts closer than `gap` merge)."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    out, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap:
            out.append((s, prev))
            s = i
        prev = i
    out.append((s, prev))
    return out


def score_method(name: str, df: pd.DataFrame, events, alert: np.ndarray,
                 score: np.ndarray | None, days: float) -> dict:
    y = df["label"].to_numpy()
    res = {"method": name,
           "precision": precision_score(y, alert, zero_division=0),
           "recall": recall_score(y, alert, zero_division=0),
           "f1": f1_score(y, alert, zero_division=0),
           "auc": roc_auc_score(y, score) if score is not None else np.nan}
    # event-level
    det, ttd = {k: [] for k in ATTACKS}, {k: [] for k in ATTACKS}
    for e in events:
        if not e.malicious:
            continue
        seg = alert[e.start: e.start + e.duration]
        hit = bool(seg.any())
        det[e.kind].append(hit)
        if hit:
            ttd[e.kind].append(int(np.argmax(seg)) + 1)   # minutes until first alert
    for k in ATTACKS:
        res[f"det_{k}"] = float(np.mean(det[k])) if det[k] else np.nan
        res[f"ttd_{k}"] = float(np.mean(ttd[k])) if ttd[k] else np.nan
    res["event_detection"] = float(np.mean(sum(det.values(), [])))
    all_ttd = sum(ttd.values(), [])
    res["mean_ttd_min"] = float(np.mean(all_ttd)) if all_ttd else np.nan
    # false alarms: incidents that do not overlap any malicious window
    false_inc = [s for s, e in _incidents(alert) if y[s: e + 1].sum() == 0]
    res["false_alerts_per_day"] = len(false_inc) / days
    hn = df["scenario"].isin(HARD_NEG).to_numpy()
    res["hard_negative_fpr"] = float(alert[hn].mean()) if hn.any() else np.nan
    return res


def simulate_response(df: pd.DataFrame, events, verdicts: pd.DataFrame, cfg_resp: dict | None = None) -> dict:
    """Replay the graduated response policy on the simulated stream."""
    r = cfg_resp or {"kill_threshold": 0.75, "isolate_threshold": 0.90}
    allow = DEFAULT_ALLOW
    conf = verdicts["confidence"].to_numpy()
    intel = verdicts["intel_conf"].to_numpy()
    names = df["top_proc_name"].str.lower().to_numpy()
    topcpu = df["top_proc_cpu"].to_numpy()
    share = df["top_proc_share"].to_numpy()
    flagged = (df[INTEL_FEATURES].sum(axis=1) > 0).to_numpy()
    excess = verdicts["cpu_excess"].to_numpy()
    attributable = flagged | ((topcpu >= 20) & (share >= 0.4) & (excess >= 10) & (topcpu >= 0.5 * excess))
    kill = (conf >= r["kill_threshold"]) & attributable & ~np.isin(names, list(allow))
    iso = (conf >= r["isolate_threshold"]) & (intel > 0)
    y = df["label"].to_numpy().astype(bool)

    contained, ttc, killed_ev = [], [], 0
    for e in events:
        if not e.malicious:
            continue
        seg_k = kill[e.start: e.start + e.duration]
        seg_i = iso[e.start: e.start + e.duration]
        act = seg_k | seg_i
        contained.append(bool(act.any()))
        if act.any():
            ttc.append(int(np.argmax(act)) + 1)
        killed_ev += bool(seg_k.any())
    wrong_kill_inc = _incidents(kill & ~y)
    wrong_iso_inc = _incidents(iso & ~y)
    days = len(df) / 1440
    return {"containment_rate": float(np.mean(contained)),
            "mean_ttc_min": float(np.mean(ttc)) if ttc else np.nan,
            "wrong_kills_per_day": len(wrong_kill_inc) / days,
            "wrong_isolations_per_day": len(wrong_iso_inc) / days}


# Registered maintenance window of the simulated server (weekly report job): Sundays 04:00-05:15.
is_weekly_job_window = make_maintenance([{"days": [6], "start": "04:00", "end": "05:15"}])


class _CachedShares:
    """Computes SHAP compute-shares once per row and reuses them across ablations."""

    def __init__(self, explainer: Explainer, df: pd.DataFrame):
        self.explainer = explainer
        self.cache: dict = {}

    def compute_share(self, df: pd.DataFrame) -> np.ndarray:
        missing = [i for i in df.index if i not in self.cache]
        if missing:
            vals = self.explainer.compute_share(df.loc[missing])
            self.cache.update(zip(missing, vals))
        return np.array([self.cache[i] for i in df.index])


def run_seed(seed: int, train_days: int = 7, test_days: int = 3) -> dict:
    train = generate_normal(train_days, seed=seed, start_day=0)
    calib = generate_normal(1, seed=seed + 100, start_day=7)
    test, events = generate_test(test_days, seed=seed + 1000, start_day=8)

    # the weekly report job is registered as a maintenance window (operational setting);
    # it stays in the learned normal profile but does not set the CUSUM interval
    weekly = (train["scenario"] == "weekly_report").to_numpy()
    mw = is_weekly_job_window

    iforest = BaselineModel("isolation_forest", random_state=seed).fit(train, calib, cusum_exclude=weekly)
    shares = (iforest.cusum_shares(train), iforest.cusum_shares(calib))   # gate-independent

    def variant(gate=iforest.compute_gate, exclude=weekly):
        """Same forest, different XAI gate / maintenance setting (only the CUSUM interval changes)."""
        m = copy.copy(iforest)
        m.compute_gate = gate
        m.calibrate_cusum(train, calib, exclude, shares if gate is not None else None)
        return m

    ocsvm = BaselineModel("ocsvm", random_state=seed).fit(train, calib)
    exp = _CachedShares(Explainer(iforest), test)

    results = []

    def add(name, alert, score):
        results.append(score_method(name, test, events, np.asarray(alert), np.asarray(score), test_days))

    add("Static CPU alarm", static_cpu_alarm(test), test["cpu_percent"])
    add("Signature only", signature_only(test), test[INTEL_FEATURES].sum(axis=1))
    v = Detector(ocsvm, use_intel=False, use_cusum=False).run(test)
    add("OCSVM (behavioural)", v["alert"], v["score"])
    v = Detector(iforest, use_intel=False, use_cusum=False).run(test)
    add("IForest (behavioural)", v["alert"], v["score"])
    v = Detector(iforest, use_cusum=False, explainer=exp, maintenance=mw).run(test)
    add("CryptoGuard w/o CUSUM", v["alert"], v["confidence"])
    v = Detector(variant(gate=None), maintenance=mw).run(test)
    add("CryptoGuard w/o XAI gate", v["alert"], v["confidence"])
    v = Detector(iforest, use_intel=False, explainer=exp, maintenance=mw).run(test)
    add("CryptoGuard w/o threat intel", v["alert"], v["confidence"])
    v = Detector(iforest, use_persistence=False, explainer=exp, maintenance=mw).run(test)
    add("CryptoGuard w/o persistence", v["alert"], v["confidence"])
    no_mw = variant(exclude=None)
    v_nomw = Detector(no_mw, explainer=exp).run(test)
    add("CryptoGuard w/o maintenance window", v_nomw["alert"], v_nomw["confidence"])
    v_cg = Detector(iforest, explainer=exp, maintenance=mw).run(test)
    add("CryptoGuard (full)", v_cg["alert"], v_cg["confidence"])
    response = simulate_response(test, events, v_cg)
    response_nomw = simulate_response(test, events, v_nomw)

    # sensitivity of the XAI gate (same forest, CUSUM recalibrated per gate)
    gate_rows = []
    for g in (0.6, 0.7, 0.75, 0.8, 0.9):
        m = variant(gate=g)                 # identical forest -> cached SHAP values stay valid
        vg = Detector(m, explainer=exp, maintenance=mw).run(test)
        row = score_method(f"gate={g}", test, events, vg["alert"].to_numpy(), vg["confidence"].to_numpy(),
                           test_days)
        gate_rows.append(dict(row, gate=g))
    return {"seed": seed, "results": results, "response": response, "response_nomw": response_nomw,
            "gate": gate_rows,
            "artifacts": {"train": train, "test": test, "events": events, "model": iforest,
                          "verdicts": v_cg, "verdicts_static": static_cpu_alarm(test)}}


def explanation_profile(model: BaselineModel, test: pd.DataFrame, verdicts: pd.DataFrame,
                        max_per_kind: int = 60, seed: int = 0) -> pd.DataFrame:
    """Mean share of TreeSHAP attribution per feature, for alerted windows of each scenario."""
    exp = Explainer(model)
    rng = np.random.default_rng(seed)
    rows = []
    for kind in ATTACKS + HARD_NEG:
        idx = test.index[(test["scenario"] == kind) & verdicts["behavioural_anomaly"]]
        if len(idx) == 0:
            continue
        idx = rng.choice(idx, min(max_per_kind, len(idx)), replace=False)
        acc = {}
        for i in idx:
            a = exp.attributions(test.loc[[i]])
            pos = {k: max(v, 0) for k, v in a.items()}
            tot = sum(pos.values()) or 1
            for k, v in pos.items():
                acc[k] = acc.get(k, 0) + v / tot / len(idx)
        acc["scenario"] = kind
        rows.append(acc)
    return pd.DataFrame(rows).set_index("scenario")


def run_experiments(seeds=(0, 1, 2, 3, 4), out_dir: str | Path = "results") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_res, all_resp, all_gate, first = [], [], [], None
    for s in seeds:
        r = run_seed(s)
        for row in r["results"]:
            all_res.append(dict(row, seed=s))
        for row in r["gate"]:
            all_gate.append(dict(row, seed=s))
        all_resp.append(dict(r["response"], seed=s, variant="full"))
        all_resp.append(dict(r["response_nomw"], seed=s, variant="no_maintenance_window"))
        if first is None:
            first = r
        print(f"seed {s} done", flush=True)
    res = pd.DataFrame(all_res)
    resp = pd.DataFrame(all_resp)
    res.to_csv(out / "metrics_per_seed.csv", index=False)
    resp.to_csv(out / "response_per_seed.csv", index=False)
    summary = res.groupby("method", sort=False).agg(["mean", "std"])
    summary.to_csv(out / "metrics_summary.csv")
    resp_summary = resp.drop(columns="seed").groupby("variant").agg(["mean", "std"])
    resp_summary.to_csv(out / "response_summary.csv")
    gate = pd.DataFrame(all_gate)
    gate.to_csv(out / "gate_sensitivity_per_seed.csv", index=False)
    gate.drop(columns=["method", "seed"]).groupby("gate").agg(["mean", "std"]).to_csv(
        out / "gate_sensitivity_summary.csv")

    a = first["artifacts"]
    prof = explanation_profile(a["model"], a["test"], a["verdicts"])
    prof.to_csv(out / "explanation_profile.csv")
    # one worked explanation per attack kind for the paper / README
    exp = Explainer(a["model"])
    examples = {}
    for e in a["events"]:
        if e.malicious and e.kind not in examples:
            seg = a["verdicts"].iloc[e.start: e.start + e.duration]
            if seg["alert"].any():
                i = seg.index[int(np.argmax(seg["alert"].to_numpy()))]
                row = a["test"].loc[[i]]
                examples[e.kind] = {
                    "minute_into_attack": int(i - e.start + 1),
                    "confidence": round(float(a["verdicts"].loc[i, "confidence"]), 3),
                    **exp.explain(row, {"top_process_name": a["test"].loc[i, "top_proc_name"]})}
    (out / "explanation_examples.json").write_text(json.dumps(examples, indent=2))
    meta = {"seeds": list(seeds), "events_first_seed": [asdict(e) for e in a["events"]],
            "train_windows": len(a["train"]), "test_windows": len(a["test"]),
            "threshold": a["model"].threshold}
    (out / "run_info.json").write_text(json.dumps(meta, indent=2))
    a["test"].assign(confidence=a["verdicts"]["confidence"], alert=a["verdicts"]["alert"],
                     static_alert=a["verdicts_static"]).to_csv(out / "test_trace_seed0.csv", index=False)
    return {"summary": summary, "response": resp_summary, "first": first}
