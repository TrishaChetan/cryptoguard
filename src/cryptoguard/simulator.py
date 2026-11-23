"""Synthetic cloud-server telemetry with injected cryptojacking attacks.

Real labelled cryptojacking telemetry from production cloud accounts is not
publicly available, so this module generates realistic 1-minute windows for
a typical web/API server:

Normal behaviour
  * daily traffic cycle (quiet at night, busy in the afternoon), bursty CPU
  * nightly database backup at 02:00 (high CPU + large upload)
  * weekly report job on Sundays at 04:00 (CPU ~90 % for an hour)

Hard negatives (only in the test period - legitimate but never seen in training)
  * marketing traffic surge (high CPU *and* many connections)
  * CI build on the server (CPU ~95 %, many short-lived processes)

Attacks (only in the test period)
  * ``noisy``     - XMRig at full speed, Stratum port 3333/4444/14444
  * ``stealthy``  - throttled miner (+20-35 % CPU), renamed binary,
                    pool reached through TLS on 443
  * ``container`` - miner dropped through an exposed container API,
                    many new processes, generic process name
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import BEHAVIOURAL_FEATURES, INTEL_FEATURES, hour_encoding, top_share

MINUTES_PER_DAY = 24 * 60

LEGIT_TOP = ["gunicorn", "nginx", "python3", "node", "postgres"]
STEALTH_NAMES = ["kworker_u8", "systemd-updater", "dbus-helper", "[kthreadd]"]
CONTAINER_NAMES = ["python3", "sh", "java", "app"]


@dataclass
class Event:
    kind: str           # attack kind or hard-negative kind
    start: int          # minute index
    duration: int
    malicious: bool
    renamed: bool = False       # miner binary renamed (defeats name signatures)
    pool_port: bool = False     # talks Stratum on a well-known port (else TLS/443 proxy)
    blocklisted: bool = False   # pool / proxy IP already on the threat-intel blocklist


def _diurnal(hour: np.ndarray) -> np.ndarray:
    """0 at ~03:00, 1 at ~15:00."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * (hour - 3) / 24)


def _base_frame(n_minutes: int, start_day: int, rng: np.random.Generator) -> pd.DataFrame:
    t = np.arange(n_minutes)
    minute_of_day = t % MINUTES_PER_DAY
    hour = minute_of_day / 60.0
    day = start_day + t // MINUTES_PER_DAY
    d = _diurnal(hour) * rng.uniform(0.85, 1.15, size=n_minutes)

    # short request bursts: load (and therefore CPU) jumps for 1-3 minutes
    burst = rng.random(n_minutes) < 0.02
    burst = burst | np.roll(burst, 1) | (np.roll(burst, 2) & (rng.random(n_minutes) < 0.5))
    load = (0.15 + d) * np.where(burst, rng.uniform(1.5, 2.2, n_minutes), 1.0)

    # CPU is driven by the work the server is doing
    conns = rng.poisson(10 + 85 * load)
    cpu = 4 + 0.30 * conns + rng.normal(0, 3, n_minutes)
    cpu_std = np.abs(rng.normal(6, 2, n_minutes)) + 1.0 + burst * rng.uniform(3, 8, n_minutes)
    top_cpu = cpu * rng.uniform(0.25, 0.55, n_minutes)
    df = pd.DataFrame({
        "minute": t,
        "day": day,
        "hour": hour,
        "cpu_percent": cpu,
        "cpu_std": cpu_std,
        "top_proc_cpu": top_cpu,
        "mem_percent": 38 + 12 * d + rng.normal(0, 2, n_minutes),
        "num_procs": np.round(140 + 25 * d + rng.normal(0, 3, n_minutes)),
        "new_procs": rng.poisson(2 + 5 * load),
        "net_conns": conns,
        "unique_remote_ips": np.minimum(conns, rng.poisson(4 + 0.45 * conns)),
        "bytes_sent_kbps": (25 * conns) * rng.lognormal(0, 0.25, n_minutes),
        "bytes_recv_kbps": (12 * conns) * rng.lognormal(0, 0.25, n_minutes),
        "pool_port_conns": 0,
        "miner_name_hits": 0,
        "blocklist_ip_conns": 0,
        "top_proc_name": rng.choice(LEGIT_TOP, n_minutes),
        "label": 0,
        "scenario": "normal",
        "event_id": -1,
    })
    return df


def _apply_scheduled_jobs(df: pd.DataFrame, rng: np.random.Generator) -> None:
    """Nightly backup + weekly report: legitimate heavy jobs seen in training."""
    for day in df["day"].unique():
        idx0 = df.index[(df["day"] == day)]
        if len(idx0) == 0:
            continue
        first = idx0[0] - df.loc[idx0[0], "minute"] % MINUTES_PER_DAY
        # nightly backup 02:00 for 25-40 min
        s = first + 120 + int(rng.integers(0, 5))
        e = s + int(rng.integers(25, 41))
        sl = df.index[(df.index >= s) & (df.index < e)]
        if len(sl):
            n = len(sl)
            df.loc[sl, "cpu_percent"] = rng.uniform(62, 85, n)
            df.loc[sl, "cpu_std"] = rng.uniform(7, 13, n)
            df.loc[sl, "top_proc_cpu"] = df.loc[sl, "cpu_percent"] * rng.uniform(0.7, 0.9, n)
            df.loc[sl, "top_proc_name"] = "pg_dump"
            df.loc[sl, "bytes_sent_kbps"] = rng.uniform(20000, 45000, n)
            df.loc[sl, "num_procs"] += 4
            df.loc[sl, "scenario"] = "backup"
        # weekly report job, Sundays (day % 7 == 6) at 04:00 for ~60 min
        if day % 7 == 6:
            s = first + 240
            e = s + int(rng.integers(50, 70))
            sl = df.index[(df.index >= s) & (df.index < e)]
            if len(sl):
                n = len(sl)
                df.loc[sl, "cpu_percent"] = rng.uniform(83, 95, n)
                df.loc[sl, "cpu_std"] = rng.uniform(4, 8, n)
                df.loc[sl, "top_proc_cpu"] = df.loc[sl, "cpu_percent"] * rng.uniform(0.8, 0.95, n)
                df.loc[sl, "top_proc_name"] = "python3"
                df.loc[sl, "mem_percent"] = rng.uniform(65, 78, n)
                df.loc[sl, "scenario"] = "weekly_report"


def _inject(df: pd.DataFrame, ev: Event, eid: int, rng: np.random.Generator) -> None:
    sl = df.index[(df.index >= ev.start) & (df.index < ev.start + ev.duration)]
    n = len(sl)
    if n == 0:
        return
    base = df.loc[sl, "cpu_percent"].to_numpy()
    if ev.kind == "noisy":
        df.loc[sl, "cpu_percent"] = np.clip(rng.normal(97, 1.5, n), 90, 100)
        df.loc[sl, "cpu_std"] = rng.uniform(0.3, 2.0, n)
        df.loc[sl, "top_proc_cpu"] = df.loc[sl, "cpu_percent"] * rng.uniform(0.90, 0.97, n)
        df.loc[sl, "top_proc_name"] = "kswapd1" if ev.renamed else "xmrig"
        df.loc[sl, "miner_name_hits"] = 0 if ev.renamed else 1
        if ev.pool_port:
            df.loc[sl, "pool_port_conns"] = rng.integers(1, 3, n)
        df.loc[sl, "unique_remote_ips"] += 1
        df.loc[sl, "net_conns"] += 1
        df.loc[sl, "bytes_sent_kbps"] += rng.uniform(3, 12, n)
        df.loc[sl[:2], "new_procs"] += rng.integers(2, 5, min(2, n))
    elif ev.kind == "stealthy":
        extra = rng.uniform(20, 35)
        df.loc[sl, "cpu_percent"] = np.clip(base + extra + rng.normal(0, 1, n), 0, 100)
        # a constant extra load does not make the system curve flatter
        df.loc[sl, "cpu_std"] = df.loc[sl, "cpu_std"].to_numpy() * rng.uniform(0.75, 1.0, n)
        df.loc[sl, "top_proc_cpu"] = extra + rng.normal(0, 1, n)
        df.loc[sl, "top_proc_name"] = STEALTH_NAMES[eid % len(STEALTH_NAMES)]
        df.loc[sl, "unique_remote_ips"] += 1
        df.loc[sl, "net_conns"] += 1
        df.loc[sl, "bytes_sent_kbps"] += rng.uniform(2, 8, n)
        if ev.blocklisted:
            df.loc[sl, "blocklist_ip_conns"] = 1
    elif ev.kind == "container":
        df.loc[sl, "cpu_percent"] = np.clip(rng.normal(95, 2.5, n), 85, 100)
        df.loc[sl, "cpu_std"] = rng.uniform(0.8, 3.0, n)
        df.loc[sl, "top_proc_cpu"] = df.loc[sl, "cpu_percent"] * rng.uniform(0.45, 0.7, n)
        df.loc[sl, "top_proc_name"] = CONTAINER_NAMES[eid % len(CONTAINER_NAMES)]
        df.loc[sl, "num_procs"] += rng.integers(25, 45)
        df.loc[sl[:3], "new_procs"] += rng.integers(25, 45, min(3, n))
        df.loc[sl, "net_conns"] += rng.integers(4, 9, n)
        df.loc[sl, "unique_remote_ips"] += 2
        if ev.pool_port:
            df.loc[sl, "pool_port_conns"] = (rng.random(n) < 0.6).astype(int) * rng.integers(1, 4, n)
    elif ev.kind == "traffic_surge":         # legitimate, unseen in training
        c = rng.poisson(rng.uniform(250, 330), n)
        df.loc[sl, "net_conns"] = c
        df.loc[sl, "cpu_percent"] = np.clip(4 + 0.27 * c + rng.normal(0, 3, n), 0, 99)
        df.loc[sl, "cpu_std"] = rng.uniform(8, 16, n)
        df.loc[sl, "top_proc_cpu"] = df.loc[sl, "cpu_percent"] * rng.uniform(0.3, 0.5, n)
        df.loc[sl, "top_proc_name"] = "gunicorn"
        df.loc[sl, "unique_remote_ips"] = np.round(c * rng.uniform(0.55, 0.7, n))
        df.loc[sl, "bytes_sent_kbps"] = 25 * c * rng.lognormal(0, 0.25, n)
        df.loc[sl, "bytes_recv_kbps"] = 12 * c * rng.lognormal(0, 0.25, n)
        df.loc[sl, "new_procs"] += rng.poisson(6, n)
    elif ev.kind == "ci_build":              # legitimate, unseen in training
        df.loc[sl, "cpu_percent"] = np.clip(rng.normal(94, 3, n), 80, 100)
        df.loc[sl, "cpu_std"] = rng.uniform(5, 14, n)
        df.loc[sl, "top_proc_cpu"] = df.loc[sl, "cpu_percent"] * rng.uniform(0.15, 0.35, n)
        df.loc[sl, "top_proc_name"] = "cc1plus"
        df.loc[sl, "new_procs"] += rng.poisson(70, n)
        df.loc[sl, "num_procs"] += rng.integers(10, 30, n)
        df.loc[sl, "mem_percent"] = rng.uniform(60, 80, n)
    else:
        raise ValueError(ev.kind)
    df.loc[sl, "scenario"] = ev.kind
    df.loc[sl, "event_id"] = eid
    df.loc[sl, "label"] = int(ev.malicious)


def _finalise(df: pd.DataFrame, sustained_windows: int = 10) -> pd.DataFrame:
    df["cpu_percent"] = df["cpu_percent"].clip(0, 100)
    df["top_proc_cpu"] = np.minimum(df["top_proc_cpu"].clip(0, 100), df["cpu_percent"])
    df["mem_percent"] = df["mem_percent"].clip(0, 100)
    df["top_proc_share"] = [top_share(a, b) for a, b in zip(df["top_proc_cpu"], df["cpu_percent"])]
    df["cpu_sustained"] = (df["cpu_percent"] > 80).astype(float).rolling(
        sustained_windows, min_periods=1).mean()
    enc = [hour_encoding(h) for h in df["hour"]]
    df["hour_sin"] = [e[0] for e in enc]
    df["hour_cos"] = [e[1] for e in enc]
    df["timestamp"] = pd.Timestamp("2026-01-05") + pd.to_timedelta(df["day"] * MINUTES_PER_DAY
                                                                   + (df["minute"] % MINUTES_PER_DAY), unit="min")
    cols = ["timestamp", "day", "hour"] + BEHAVIOURAL_FEATURES + INTEL_FEATURES + \
           ["top_proc_name", "label", "scenario", "event_id"]
    return df[cols].reset_index(drop=True)


def generate_normal(days: int = 7, seed: int = 0, start_day: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _base_frame(days * MINUTES_PER_DAY, start_day, rng)
    _apply_scheduled_jobs(df, rng)
    return _finalise(df)


def generate_test(days: int = 3, seed: int = 1, start_day: int = 7,
                  n_noisy: int = 3, n_stealthy: int = 4, n_container: int = 3,
                  n_surge: int = 2, n_ci: int = 3, p_renamed: float = 0.5, p_pool_port: float = 0.6,
                  p_blocklisted: float = 0.3) -> tuple[pd.DataFrame, list[Event]]:
    """Test period with attacks and hard negatives at random, non-overlapping times.

    ``p_*`` control how often an attacker evades signatures (renamed binary,
    pool reached over TLS/443, IP not yet on the blocklist)."""
    rng = np.random.default_rng(seed)
    n = days * MINUTES_PER_DAY
    df = _base_frame(n, start_day, rng)
    _apply_scheduled_jobs(df, rng)

    specs = (["noisy"] * n_noisy + ["stealthy"] * n_stealthy + ["container"] * n_container
             + ["traffic_surge"] * n_surge + ["ci_build"] * n_ci)
    rng.shuffle(specs)
    events: list[Event] = []
    occupied = np.zeros(n, dtype=bool)
    occupied[df["scenario"] != "normal"] = True       # keep scheduled jobs clean
    for kind in specs:
        dur = {"noisy": (60, 180), "stealthy": (180, 480), "container": (45, 120),
               "traffic_surge": (60, 150), "ci_build": (12, 30)}[kind]
        for _ in range(500):
            d = int(rng.integers(*dur))
            s = int(rng.integers(30, n - d - 30))
            if not occupied[max(0, s - 30): s + d + 30].any():
                occupied[s: s + d] = True
                # how much threat-intel evidence the attacker leaves behind
                ren = kind == "noisy" and rng.random() < p_renamed
                port = (kind == "noisy" and rng.random() < p_pool_port) or \
                       (kind == "container" and rng.random() < p_pool_port)
                blk = kind == "stealthy" and rng.random() < p_blocklisted
                events.append(Event(kind, s, d, malicious=kind in ("noisy", "stealthy", "container"),
                                    renamed=bool(ren), pool_port=bool(port), blocklisted=bool(blk)))
                break
    events.sort(key=lambda e: e.start)
    for eid, ev in enumerate(events):
        _inject(df, ev, eid, rng)
    return _finalise(df), events
