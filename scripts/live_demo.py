#!/usr/bin/env python3
"""End-to-end live test on the current machine.

1. records a short baseline of this machine's normal behaviour,
2. trains the model on it,
3. starts monitoring with ACTIVE response (kill only; network isolation stays
   off so the demo never cuts your connection),
4. launches the safe fake miner (scripts/fake_miner.py) twice:
      a) noisy   - all cores, 'xmrig' in its command line, fake pool on :3333
      b) stealthy- one core at 50 %, no signature, no pool connection
5. measures time-to-detect and time-to-kill and writes results/live_demo.json.

    python scripts/live_demo.py --baseline-windows 120 --window 5

Keep the machine otherwise quiet while the baseline is recorded: anything you
run then becomes part of "normal".
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cryptoguard.config import load_config  # noqa: E402
from cryptoguard.model import BaselineModel  # noqa: E402
from cryptoguard.monitor import collect_baseline, run_monitor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-windows", type=int, default=60)
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--attack-seconds", type=float, default=240, help="monitoring time per scenario")
    ap.add_argument("--out", default=str(ROOT / "results" / "live_demo.json"))
    a = ap.parse_args()

    cfg = load_config()
    cfg["collector"]["window_seconds"] = a.window
    cfg["response"]["dry_run"] = False
    cfg["response"]["isolation_mode"] = "none"
    cfg["response"]["notify"]["log_file"] = str(ROOT / "logs" / "live_demo_alerts.jsonl")
    Path(cfg["response"]["notify"]["log_file"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["response"]["notify"]["log_file"]).write_text("")

    print(f"== 1. baseline: {a.baseline_windows} windows x {a.window:.0f}s", flush=True)
    base_csv = ROOT / "data" / "live_baseline.csv"
    base_csv.parent.mkdir(exist_ok=True)
    df = collect_baseline(cfg, a.baseline_windows, str(base_csv), verbose=False)
    model = BaselineModel(**cfg["model"]).fit(df)
    print(f"   trained on {len(df)} windows, mean CPU {df['cpu_percent'].mean():.1f}%", flush=True)

    scenarios = [
        ("noisy", ["--threads", "2", "--label", "xmrig", "--pool", "--duration", "600"]),
        ("stealthy", ["--threads", "1", "--throttle", "0.5", "--duration", "600"]),
    ]
    results = {"window_seconds": a.window, "baseline_windows": len(df),
               "baseline_cpu_mean": float(df["cpu_percent"].mean()), "scenarios": []}

    for name, extra in scenarios:
        print(f"== 2. scenario '{name}'", flush=True)
        state = {"start": None, "pid": None, "detect": None, "kill": None, "explanation": None,
                 "confidence": None, "false_alerts_before": 0}

        def launch():
            p = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "fake_miner.py"), *extra],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            state["start"], state["pid"], state["proc"] = time.time(), p.pid, p

        def on_event(ev):
            now = time.time()
            if ev["verdict"]["alert"]:
                if state["start"] is None:
                    state["false_alerts_before"] += 1
                elif state["detect"] is None:
                    state["detect"] = now - state["start"]
                    state["confidence"] = ev["verdict"]["confidence"]
                    state["explanation"] = (ev["explanation"] or {}).get("summary")
            for act in ev["actions"]:
                if act["action"] == "kill" and act.get("pid") == state["pid"] and act["status"] == "terminated":
                    state["kill"] = now - state["start"]

        timer = threading.Timer(3 * a.window, launch)     # 3 clean windows first
        timer.start()
        run_monitor(cfg, model, max_windows=int(3 + a.attack_seconds / a.window), verbose=True, on_event=on_event)
        proc = state.pop("proc", None)
        alive = proc is not None and proc.poll() is None
        if alive:
            proc.kill()
        state.update({"name": name, "miner_still_running_at_end": alive})
        results["scenarios"].append(state)
        print(f"   detect={state['detect']}s kill={state['kill']}s", flush=True)
        time.sleep(2 * a.window)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
