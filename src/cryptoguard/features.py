"""Feature definitions shared by the live collector, the simulator and the model.

Two groups of signals are produced for every observation window:

* BEHAVIOURAL_FEATURES  - numeric host telemetry that the anomaly model learns
                          a "normal profile" from (no labels needed).
* INTEL_FEATURES        - threat-intelligence evidence (mining-pool ports,
                          known miner binaries, blocklisted IPs). These are
                          almost always zero on a healthy host, so they are
                          fused with the model score instead of being fed
                          into it.
"""

from __future__ import annotations

import math

BEHAVIOURAL_FEATURES = [
    "cpu_percent",        # mean system CPU utilisation in the window (%)
    "cpu_std",            # std-dev of CPU samples in the window (miners are flat)
    "cpu_sustained",      # fraction of recent windows with CPU > 80 %
    "top_proc_cpu",       # CPU % of the single hungriest process
    "top_proc_share",     # top_proc_cpu / cpu_percent  (one process dominating?)
    "mem_percent",        # memory utilisation (%)
    "num_procs",          # number of running processes
    "new_procs",          # processes spawned since the last window
    "net_conns",          # established TCP connections
    "unique_remote_ips",  # distinct remote endpoints
    "bytes_sent_kbps",    # outbound throughput
    "bytes_recv_kbps",    # inbound throughput
    "hour_sin",           # time-of-day encoding, so "normal" can depend on the hour
    "hour_cos",
]

# Learned by the model from the features above (see model.BaselineModel):
# CPU actually used minus CPU the current workload is expected to need.
DERIVED_FEATURES = ["cpu_excess"]

# Workload signals used to predict the CPU a healthy server should need.

WORKLOAD_DRIVERS = ["net_conns", "unique_remote_ips", "bytes_sent_kbps", "bytes_recv_kbps",
                    "new_procs", "num_procs", "hour_sin", "hour_cos"]

# Features that describe *compute* behaviour. Mining is compute-bound, so an
# anomaly explained mostly by these looks like mining; an anomaly explained
# mostly by traffic or process churn looks like a workload change instead.
COMPUTE_FEATURES = ["cpu_percent", "cpu_std", "cpu_sustained", "top_proc_cpu",
                    "top_proc_share", "cpu_excess", "hour_sin"]

INTEL_FEATURES = [
    "pool_port_conns",     # connections to well-known Stratum / mining-pool ports
    "miner_name_hits",     # processes whose name / cmdline matches miner signatures
    "blocklist_ip_conns",  # connections to IPs on the mining-pool blocklist
]

# Human-readable descriptions used by the explainer.
FEATURE_LABELS = {
    "cpu_percent": "CPU usage",
    "cpu_std": "CPU variability",
    "cpu_sustained": "time spent above 80% CPU",
    "top_proc_cpu": "CPU used by the top process",
    "top_proc_share": "share of CPU taken by a single process",
    "mem_percent": "memory usage",
    "num_procs": "number of processes",
    "new_procs": "newly spawned processes",
    "net_conns": "open network connections",
    "unique_remote_ips": "distinct remote IPs",
    "bytes_sent_kbps": "outbound traffic (kbps)",
    "bytes_recv_kbps": "inbound traffic (kbps)",
    "hour_sin": "time of day",
    "hour_cos": "time of day",
    "cpu_excess": "CPU not explained by the current workload",
}


def hour_encoding(hour_float: float) -> tuple[float, float]:
    """Encode an hour in [0, 24) on the unit circle."""
    angle = 2.0 * math.pi * (hour_float % 24.0) / 24.0
    return math.sin(angle), math.cos(angle)


def top_share(top_proc_cpu: float, cpu_percent: float) -> float:
    if cpu_percent <= 1e-6:
        return 0.0
    return float(min(1.0, top_proc_cpu / cpu_percent))
