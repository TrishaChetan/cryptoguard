"""Live host telemetry collector (Linux / Windows / macOS via psutil).

Each call to :meth:`TelemetryCollector.collect_window` blocks for one window
(default 10 s), samples CPU once per second and returns one feature row plus
the context the explainer and responder need (top process, suspicious
connections, ...).
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import psutil

from .features import hour_encoding, top_share
from .threat_intel import ThreatIntel


@dataclass
class ProcessInfo:
    pid: int
    name: str
    cmdline: str
    cpu: float          # normalised to 0-100 % of the whole machine
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"pid": self.pid, "name": self.name, "cmdline": self.cmdline[:300],
                "cpu": round(self.cpu, 2), "reasons": self.reasons}


@dataclass
class Window:
    timestamp: str
    features: dict
    top_process: ProcessInfo | None
    suspicious_processes: list[ProcessInfo]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "features": self.features,
            "top_process": self.top_process.to_dict() if self.top_process else None,
            "suspicious_processes": [p.to_dict() for p in self.suspicious_processes],
        }


class TelemetryCollector:
    def __init__(self, intel: ThreatIntel, window_seconds: float = 10,
                 sample_seconds: float = 1, sustained_windows: int = 10):
        self.intel = intel
        self.window_seconds = window_seconds
        self.sample_seconds = sample_seconds
        self.cpu_history: deque[float] = deque(maxlen=sustained_windows)
        self.n_cpus = psutil.cpu_count(logical=True) or 1
        self._procs: dict[int, psutil.Process] = {}
        self._known_pids: set[int] = set(psutil.pids())
        self._own_pid = os.getpid()
        self._last_net = psutil.net_io_counters()
        self._last_net_t = time.time()
        self._prime_processes()

    # ---------------------------------------------------------------- helpers
    def _prime_processes(self) -> None:
        for proc in psutil.process_iter(["pid"]):
            if proc.pid not in self._procs:
                try:
                    proc.cpu_percent(None)        # first call primes the counter
                    self._procs[proc.pid] = proc
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

    def _process_snapshot(self) -> list[ProcessInfo]:
        infos = []
        alive = set()
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            pid = proc.pid
            alive.add(pid)
            cached = self._procs.get(pid)
            if cached is None:                    # new since last window
                try:
                    proc.cpu_percent(None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                self._procs[pid] = proc
                cpu = 0.0
            else:
                try:
                    cpu = cached.cpu_percent(None) / self.n_cpus
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            name = proc.info.get("name") or ""
            cmd = " ".join(proc.info.get("cmdline") or [])
            infos.append(ProcessInfo(pid, name, cmd, cpu))
        for dead in set(self._procs) - alive:
            self._procs.pop(dead, None)
        return infos

    def _connections(self) -> list:
        try:
            return [c for c in psutil.net_connections(kind="inet")
                    if c.status == psutil.CONN_ESTABLISHED and c.raddr]
        except (psutil.AccessDenied, PermissionError):
            # Without root on macOS we can only see our own sockets.
            return []

    # ------------------------------------------------------------------- main
    def collect_window(self) -> Window:
        samples = []
        deadline = time.time() + self.window_seconds
        while time.time() < deadline:
            samples.append(psutil.cpu_percent(interval=self.sample_seconds))
        cpu_mean = float(np.mean(samples)) if samples else 0.0
        cpu_std = float(np.std(samples)) if samples else 0.0
        self.cpu_history.append(cpu_mean)

        procs = self._process_snapshot()
        pids_now = {p.pid for p in procs}
        new_procs = len(pids_now - self._known_pids)
        self._known_pids = pids_now
        by_pid = {p.pid: p for p in procs}

        conns = self._connections()
        remote_ips = {c.raddr.ip for c in conns}
        pool_conns = 0
        block_conns = 0
        for c in conns:
            owner = by_pid.get(c.pid)
            if self.intel.is_pool_port(c.raddr.port):
                pool_conns += 1
                if owner:
                    owner.reasons.append(f"connected to {c.raddr.ip}:{c.raddr.port} (mining-pool port)")
            if self.intel.is_blocklisted(c.raddr.ip):
                block_conns += 1
                if owner:
                    owner.reasons.append(f"connected to blocklisted IP {c.raddr.ip}")

        miner_hits = 0
        for p in procs:
            sig = self.intel.matches_miner(p.name, p.cmdline)
            if sig:
                miner_hits += 1
                p.reasons.append(f"name/command line matches miner signature '{sig}'")

        candidates = [p for p in procs if p.pid != self._own_pid]
        top = max(candidates, key=lambda p: p.cpu) if candidates else None

        now = time.time()
        net = psutil.net_io_counters()
        dt = max(1e-3, now - self._last_net_t)
        sent_kbps = (net.bytes_sent - self._last_net.bytes_sent) * 8 / 1000 / dt
        recv_kbps = (net.bytes_recv - self._last_net.bytes_recv) * 8 / 1000 / dt
        self._last_net, self._last_net_t = net, now

        ts = datetime.now()
        hs, hc = hour_encoding(ts.hour + ts.minute / 60)
        top_cpu = top.cpu if top else 0.0
        features = {
            "cpu_percent": cpu_mean,
            "cpu_std": cpu_std,
            "cpu_sustained": float(np.mean([c > 80 for c in self.cpu_history])),
            "top_proc_cpu": top_cpu,
            "top_proc_share": top_share(top_cpu, cpu_mean),
            "mem_percent": psutil.virtual_memory().percent,
            "num_procs": len(procs),
            "new_procs": new_procs,
            "net_conns": len(conns),
            "unique_remote_ips": len(remote_ips),
            "bytes_sent_kbps": max(0.0, sent_kbps),
            "bytes_recv_kbps": max(0.0, recv_kbps),
            "hour_sin": hs,
            "hour_cos": hc,
            "pool_port_conns": pool_conns,
            "miner_name_hits": miner_hits,
            "blocklist_ip_conns": block_conns,
        }
        suspicious = [p for p in procs if p.reasons]
        return Window(ts.isoformat(timespec="seconds"), features, top, suspicious)
