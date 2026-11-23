"""Real-time monitoring loop: collect -> score -> explain -> respond."""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from .collector import TelemetryCollector
from .detector import Detector, make_maintenance
from .explainer import Explainer
from .model import BaselineModel
from .responder import Responder
from .threat_intel import ThreatIntel


def window_to_frame(window) -> pd.DataFrame:
    ts = datetime.fromisoformat(window.timestamp)
    row = dict(window.features)
    row["hour"] = ts.hour + ts.minute / 60
    row["weekday"] = ts.weekday()
    return pd.DataFrame([row])


def build_collector(cfg: dict) -> TelemetryCollector:
    intel = ThreatIntel.load(cfg["intel"]["directory"], cfg["intel"].get("resolve_domains", False))
    c = cfg["collector"]
    return TelemetryCollector(intel, c["window_seconds"], c["sample_seconds"], c["sustained_windows"])


def collect_baseline(cfg: dict, windows: int, out_csv: str, verbose: bool = True) -> pd.DataFrame:
    """Record normal behaviour of *this* server for training."""
    col = build_collector(cfg)
    rows = []
    for i in range(windows):
        w = col.collect_window()
        frame = window_to_frame(w)
        frame.insert(0, "timestamp", w.timestamp)
        rows.append(frame)
        if verbose:
            f = w.features
            print(f"[{i + 1}/{windows}] {w.timestamp} cpu={f['cpu_percent']:.1f}% "
                  f"procs={f['num_procs']} conns={f['net_conns']}", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out_csv, index=False)
    return df


def run_monitor(cfg: dict, model: BaselineModel, max_windows: int | None = None,
                verbose: bool = True, on_event=None) -> None:
    col = build_collector(cfg)
    exp = Explainer(model)
    det = Detector(model, cfg["detector"]["persistence_k"], cfg["detector"]["persistence_n"],
                   cfg["detector"].get("intel_weights"), cfg["response"]["alert_threshold"],
                   explainer=exp, maintenance=make_maintenance(cfg["detector"].get("maintenance_windows")))
    resp = Responder(cfg)
    mode = "DRY-RUN" if resp.dry_run else "ACTIVE RESPONSE"
    if verbose:
        print(f"CryptoGuard monitoring ({mode}, explainer={exp.method}, "
              f"window={cfg['collector']['window_seconds']}s). Ctrl+C to stop.", flush=True)
    n = 0
    try:
        while max_windows is None or n < max_windows:
            w = col.collect_window()
            frame = window_to_frame(w)
            v = det.process(frame).to_dict()
            wd = w.to_dict()
            explanation = None
            if v["confidence"] >= resp.alert_t:
                explanation = exp.explain(frame, wd)
            actions = resp.handle(wd, v, explanation)
            if verbose:
                status = "ALERT" if v["confidence"] >= resp.alert_t else "ok"
                top = wd["top_process"]["name"] if wd["top_process"] else "-"
                print(f"{w.timestamp} cpu={w.features['cpu_percent']:5.1f}% top={top:<16} "
                      f"score={v['score']:.3f} conf={v['confidence']:.2f} [{status}]", flush=True)
                if explanation and any(a["action"] == "notify" for a in actions):
                    print("   WHY: " + explanation["summary"], flush=True)
                for a in actions:
                    if a["action"] in ("kill", "isolate"):
                        print(f"   ACTION: {a['action']} {a.get('name') or a.get('mode')} -> "
                              f"{a['status']}", flush=True)
            if on_event:
                on_event({"window": wd, "verdict": v, "explanation": explanation, "actions": actions})
            n += 1
    except KeyboardInterrupt:
        if verbose:
            print("stopped.")


def dump_json(obj) -> str:
    return json.dumps(obj, indent=2, default=str)
