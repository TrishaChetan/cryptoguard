"""Command-line interface.

    cryptoguard simulate   - generate synthetic training / test telemetry
    cryptoguard train      - learn the normal profile from a CSV of telemetry
    cryptoguard baseline   - record this server's normal behaviour (live)
    cryptoguard monitor    - real-time detection, explanation and response (live)
    cryptoguard replay     - run the full pipeline over a CSV (offline demo)
    cryptoguard evaluate   - reproduce the experiments in the paper
    cryptoguard report     - render the alert log as an HTML report
    cryptoguard release    - remove iptables isolation
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT, load_config


def _model_from_cfg(cfg: dict):
    from .model import BaselineModel
    return BaselineModel(**cfg["model"])


def cmd_simulate(args, cfg):
    from .simulator import generate_normal, generate_test
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    generate_normal(args.days, seed=args.seed).to_csv(out / "normal.csv", index=False)
    generate_normal(1, seed=args.seed + 100, start_day=args.days).to_csv(out / "calib.csv", index=False)
    test, events = generate_test(args.test_days, seed=args.seed + 1000, start_day=args.days + 1)
    test.to_csv(out / "test.csv", index=False)
    (out / "events.json").write_text(json.dumps([asdict(e) for e in events], indent=2))
    print(f"wrote {out}/normal.csv ({args.days} days), calib.csv, test.csv ({args.test_days} days, "
          f"{sum(e.malicious for e in events)} attacks, {sum(not e.malicious for e in events)} hard negatives)")


def cmd_train(args, cfg):
    from .detector import make_maintenance
    normal = pd.read_csv(args.data)
    calib = pd.read_csv(args.calib) if args.calib else None
    mw = make_maintenance(cfg["detector"].get("maintenance_windows"))
    if "weekday" not in normal and "timestamp" in normal:
        normal["weekday"] = pd.to_datetime(normal["timestamp"]).dt.weekday
    exclude = normal.apply(lambda r: mw(r.to_dict()), axis=1).to_numpy() if mw else None
    model = _model_from_cfg(cfg).fit(normal, calib, cusum_exclude=exclude)
    model.save(args.model)
    print(f"trained {model.algorithm} on {len(normal)} windows; threshold={model.threshold:.4f} "
          f"cusum_h={model.cusum_h:.2f} -> {args.model}")


def cmd_baseline(args, cfg):
    from .monitor import collect_baseline
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    collect_baseline(cfg, args.windows, args.out)
    print(f"saved {args.windows} windows to {args.out}. Next: cryptoguard train --data {args.out}")


def _apply_response_flags(args, cfg):
    if getattr(args, "active", False):
        cfg["response"]["dry_run"] = False
    if getattr(args, "isolation", None):
        cfg["response"]["isolation_mode"] = args.isolation
    if getattr(args, "window", None):
        cfg["collector"]["window_seconds"] = args.window


def cmd_monitor(args, cfg):
    from .model import BaselineModel
    from .monitor import run_monitor
    _apply_response_flags(args, cfg)
    if args.log:
        cfg["response"]["notify"]["log_file"] = args.log
    run_monitor(cfg, BaselineModel.load(args.model), max_windows=args.windows)


def cmd_replay(args, cfg):
    """Offline demo: stream a CSV through detector -> explainer -> responder (dry run)."""
    from .detector import Detector, make_maintenance
    from .explainer import Explainer
    from .model import BaselineModel
    from .responder import Responder
    cfg["response"]["dry_run"] = True
    if args.log:
        cfg["response"]["notify"]["log_file"] = args.log
    df = pd.read_csv(args.data)
    model = BaselineModel.load(args.model)
    exp = Explainer(model)
    det = Detector(model, cfg["detector"]["persistence_k"], cfg["detector"]["persistence_n"],
                   cfg["detector"].get("intel_weights"), cfg["response"]["alert_threshold"], explainer=exp,
                   maintenance=make_maintenance(cfg["detector"].get("maintenance_windows")))
    resp = Responder(cfg)
    verdicts = det.run(df)
    n_alerts = 0
    for i, row in df.iterrows():
        v = verdicts.loc[i].to_dict()
        if v["confidence"] < resp.alert_t and resp.incident is None:
            continue
        # simulated process objects (pids are fake: replay never touches real processes)
        name = row.get("top_proc_name", "?")
        top = {"pid": 10000 + int(i), "name": name, "cpu": float(row["top_proc_cpu"]), "cmdline": ""}
        reasons = []
        if row.get("miner_name_hits", 0) > 0:
            reasons.append("name matches a miner signature")
        if row.get("pool_port_conns", 0) > 0:
            reasons.append("connected to a mining-pool port")
        if row.get("blocklist_ip_conns", 0) > 0:
            reasons.append("connected to a blocklisted IP")
        window = {"timestamp": str(row.get("timestamp", i)), "features": {"cpu_percent": float(row["cpu_percent"])},
                  "top_process": top, "suspicious_processes": [dict(top, reasons=reasons)] if reasons else []}
        explanation = exp.explain(df.loc[[i]], {"top_process_name": row.get("top_proc_name")}) \
            if v["confidence"] >= resp.alert_t else None
        actions = resp.handle(window, v, explanation)
        for a in actions:
            if a["action"] == "notify" and a["kind"] == "ALERT":
                n_alerts += 1
                truth = row.get("scenario", "?")
                print(f"\n[{window['timestamp']}] ALERT conf={v['confidence']:.2f} (ground truth: {truth})")
                print("  WHY: " + explanation["summary"])
            elif a["action"] in ("kill", "isolate"):
                print(f"  ACTION (dry-run): {a['action']} {a.get('name') or a.get('mode')}")
            elif a["action"] == "close_incident":
                print(f"  incident {a['id']} closed")
    print(f"\n{n_alerts} incidents raised over {len(df)} windows. Log: {cfg['response']['notify']['log_file']}")


def cmd_evaluate(args, cfg):
    from .evaluate import run_experiments
    res = run_experiments(seeds=tuple(range(args.seeds)), out_dir=args.out)
    cols = ["precision", "recall", "f1", "event_detection", "mean_ttd_min", "false_alerts_per_day"]
    summary = res["summary"]
    table = pd.DataFrame({c: summary[(c, "mean")] for c in cols}).round(3)
    print(table.to_string())
    print("\nautomated response (CryptoGuard):")
    print(res["response"].round(3).to_string())
    print(f"\nresults written to {args.out}/  (figures: python scripts/make_figures.py)")


def cmd_report(args, cfg):
    from .report import render_report
    out = render_report(args.log or cfg["response"]["notify"]["log_file"], args.out)
    print(f"report written to {out}")


def cmd_release(args, cfg):
    import subprocess
    from .responder import iptables_release_commands
    for c in iptables_release_commands():
        if args.dry_run:
            print(" ".join(c))
        else:
            subprocess.run(c, check=False)
    print("isolation rules removed" if not args.dry_run else "(dry run)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="cryptoguard", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML config overriding config/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="generate synthetic telemetry")
    s.add_argument("--days", type=int, default=7)
    s.add_argument("--test-days", type=int, default=3)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out-dir", default=str(PROJECT_ROOT / "data"))
    s.set_defaults(func=cmd_simulate)

    s = sub.add_parser("train", help="learn the normal profile")
    s.add_argument("--data", required=True)
    s.add_argument("--calib", help="separate normal data for threshold calibration")
    s.add_argument("--model", default=str(PROJECT_ROOT / "models" / "cryptoguard.joblib"))
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("baseline", help="record this server's normal behaviour")
    s.add_argument("--windows", type=int, default=360, help="number of windows (360 x 10 s = 1 h)")
    s.add_argument("--window", type=float, help="window length in seconds")
    s.add_argument("--out", default=str(PROJECT_ROOT / "data" / "baseline.csv"))
    s.set_defaults(func=lambda a, c: (_apply_response_flags(a, c), cmd_baseline(a, c)))

    s = sub.add_parser("monitor", help="real-time detection and response")
    s.add_argument("--model", default=str(PROJECT_ROOT / "models" / "cryptoguard.joblib"))
    s.add_argument("--windows", type=int, help="stop after N windows (default: run forever)")
    s.add_argument("--window", type=float, help="window length in seconds")
    s.add_argument("--active", action="store_true", help="really kill / isolate (default: dry run)")
    s.add_argument("--isolation", choices=["none", "iptables", "aws"])
    s.add_argument("--log", help="alert log path")
    s.set_defaults(func=cmd_monitor)

    s = sub.add_parser("replay", help="offline demo over a CSV")
    s.add_argument("--data", required=True)
    s.add_argument("--model", default=str(PROJECT_ROOT / "models" / "cryptoguard.joblib"))
    s.add_argument("--log", help="alert log path")
    s.set_defaults(func=cmd_replay)

    s = sub.add_parser("evaluate", help="reproduce the paper's experiments")
    s.add_argument("--seeds", type=int, default=5)
    s.add_argument("--out", default=str(PROJECT_ROOT / "results"))
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("report", help="HTML report from the alert log")
    s.add_argument("--log")
    s.add_argument("--out", default=str(PROJECT_ROOT / "logs" / "report.html"))
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("release", help="remove iptables isolation")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_release)

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    args.func(args, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
