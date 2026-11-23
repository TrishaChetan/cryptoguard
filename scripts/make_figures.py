#!/usr/bin/env python3
"""Build the paper figures from the CSVs written by `cryptoguard evaluate`.

    python scripts/make_figures.py --results results --out paper/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# reference palette (validated categorical order) + text tokens
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
SEQ = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.2,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": SURFACE,
})

ATTACK_LABEL = {"noisy": "Noisy miner", "stealthy": "Throttled miner", "container": "Container miner",
                "traffic_surge": "Traffic surge (benign)", "ci_build": "CI build (benign)"}
FEATURE_LABEL = {"cpu_percent": "CPU %", "cpu_std": "CPU variability", "cpu_sustained": "CPU sustained",
                 "top_proc_cpu": "Top-process CPU", "top_proc_share": "Top-process share",
                 "cpu_excess": "CPU excess", "hour_sin": "Time of day", "mem_percent": "Memory",
                 "num_procs": "Processes", "new_procs": "New processes", "net_conns": "Connections",
                 "unique_remote_ips": "Remote IPs", "bytes_sent_kbps": "Bytes out",
                 "bytes_recv_kbps": "Bytes in"}


def timeline(results: Path, out: Path) -> None:
    tr = pd.read_csv(results / "test_trace_seed0.csv", parse_dates=["timestamp"])
    info = json.loads((results / "run_info.json").read_text())
    ev = info["events_first_seed"]
    t = np.arange(len(tr)) / 60.0
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 2.6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    for e in ev:
        s, d = e["start"] / 60, e["duration"] / 60
        if e["malicious"]:
            a1.axvspan(s, s + d, color=ORANGE, alpha=0.16, lw=0)
        else:
            a1.axvspan(s, s + d, facecolor="none", edgecolor=INK2, hatch="////", alpha=0.35, lw=0)
    a1.plot(t, tr["cpu_percent"], color=BLUE, lw=0.6)
    a1.set_ylabel("CPU (%)")
    a1.set_ylim(0, 102)
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(color=ORANGE, alpha=0.3, label="cryptojacking attack"),
                       Patch(facecolor="none", edgecolor=INK2, hatch="////", label="benign heavy workload")],
              loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0, 1.18))
    rows = [("Static CPU alarm", tr["static_alert"].astype(bool), INK2),
            ("CryptoGuard", tr["alert"].astype(bool), BLUE)]
    for y, (name, mask, c) in enumerate(rows):
        a2.fill_between(t, y + 0.1, y + 0.9, where=mask.to_numpy(), color=c, step="mid", lw=0)
    a2.set_yticks([0.5, 1.5], [r[0] for r in rows])
    a2.set_ylim(0, 2)
    a2.grid(False)
    a2.set_xlabel("Time in test period (hours)")
    a2.set_xlim(0, t[-1])
    fig.savefig(out / "timeline.pdf")
    fig.savefig(out / "timeline.png")
    plt.close(fig)


def explanation_heatmap(results: Path, out: Path) -> None:
    prof = pd.read_csv(results / "explanation_profile.csv", index_col=0)
    cols = [c for c in FEATURE_LABEL if c in prof.columns]
    prof = prof[cols]
    prof.index = [ATTACK_LABEL.get(i, i) for i in prof.index]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    im = ax.imshow(prof.to_numpy(), cmap=cmap, vmin=0, vmax=max(0.4, prof.to_numpy().max()), aspect="auto")
    ax.set_xticks(range(len(cols)), [FEATURE_LABEL[c] for c in cols], rotation=60, ha="right")
    ax.set_yticks(range(len(prof)), prof.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[0, 0.1, 0.2, 0.3, 0.4])
    cb.set_label("share of SHAP attribution", fontsize=7)
    cb.outline.set_visible(False)
    fig.savefig(out / "explanation_profile.pdf")
    fig.savefig(out / "explanation_profile.png")
    plt.close(fig)


def gate_sensitivity(results: Path, out: Path) -> None:
    g = pd.read_csv(results / "gate_sensitivity_per_seed.csv")
    agg = g.groupby("gate").agg(fa=("false_alerts_per_day", "mean"), fa_s=("false_alerts_per_day", "std"),
                                ttd=("ttd_stealthy", "mean"), ttd_s=("ttd_stealthy", "std"),
                                det=("event_detection", "mean"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 1.6))
    x = agg.index.to_numpy()
    a1.errorbar(x, agg["fa"], yerr=agg["fa_s"], color=BLUE, marker="o", ms=4, capsize=2)
    a1.set_title("False alerts per day")
    a2.errorbar(x, agg["ttd"], yerr=agg["ttd_s"], color=BLUE, marker="o", ms=4, capsize=2)
    a2.set_title("Time to detect throttled miner (min)")
    for a in (a1, a2):
        a.set_xlabel("XAI gate $\\gamma$")
        a.set_xticks(x)
        a.set_ylim(bottom=0)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out / "gate_sensitivity.pdf")
    fig.savefig(out / "gate_sensitivity.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    timeline(res, out)
    explanation_heatmap(res, out)
    gate_sensitivity(res, out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
