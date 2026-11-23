#!/usr/bin/env python3
"""Turn the evaluation CSVs into LaTeX tables + number macros for the paper.

    python scripts/make_tables.py --results results --out paper/generated

Every number in the paper's results section comes from these files, so the
paper always matches the code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = ["Static CPU alarm", "Signature only", "OCSVM (behavioural)", "IForest (behavioural)",
        "CryptoGuard w/o maintenance window", "CryptoGuard (full)"]
ABL = ["CryptoGuard (full)", "CryptoGuard w/o CUSUM", "CryptoGuard w/o XAI gate",
       "CryptoGuard w/o threat intel", "CryptoGuard w/o persistence", "CryptoGuard w/o maintenance window"]
SHORT = {"Static CPU alarm": "Static CPU alarm ($>$90\\%, 5 min)",
         "Signature only": "Signatures / threat intel only",
         "OCSVM (behavioural)": "One-Class SVM",
         "IForest (behavioural)": "Isolation Forest",
         "CryptoGuard w/o maintenance window": "CryptoGuard (no maint.\\ window)",
         "CryptoGuard (full)": "\\textbf{CryptoGuard (full)}",
         "CryptoGuard w/o CUSUM": "\\quad -- CUSUM drift detector",
         "CryptoGuard w/o XAI gate": "\\quad -- XAI gate",
         "CryptoGuard w/o threat intel": "\\quad -- threat-intel fusion",
         "CryptoGuard w/o persistence": "\\quad -- persistence filter"}


def f(x, d=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def pm(m, s, d=2):
    return "--" if np.isnan(m) else f"{m:.{d}f}$\\pm${s:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/generated")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(res / "metrics_per_seed.csv")
    mean = d.groupby("method").mean(numeric_only=True)
    std = d.groupby("method").std(numeric_only=True)
    n_seeds = d["seed"].nunique()

    # ---------------- Table: comparison with baselines
    rows = []
    for m in MAIN:
        r, s = mean.loc[m], std.loc[m]
        rows.append(" & ".join([SHORT[m], f(r.precision), f(r.recall), pm(r.f1, s.f1),
                                f(r.event_detection), f(r.ttd_stealthy, 0),
                                pm(r.false_alerts_per_day, s.false_alerts_per_day),
                                f(r.hard_negative_fpr)]) + r" \\")
    (out / "table_main.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

    # ---------------- Table: ablation
    rows = []
    for m in ABL:
        r = mean.loc[m]
        rows.append(" & ".join([SHORT.get(m, m) if m != "CryptoGuard w/o maintenance window"
                                else "\\quad -- maintenance window",
                                f(r.f1), f(r.det_stealthy), f(r.event_detection),
                                f(r.ttd_noisy, 1), f(r.ttd_stealthy, 0), f(r.ttd_container, 1),
                                f(r.false_alerts_per_day), f(r.hard_negative_fpr)]) + r" \\")
    (out / "table_ablation.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

    # ---------------- Table: response
    resp = pd.read_csv(res / "response_per_seed.csv")
    rm, rs = resp.groupby("variant").mean(numeric_only=True), resp.groupby("variant").std(numeric_only=True)
    rows = []
    for v, label in (("full", "CryptoGuard (full)"), ("no_maintenance_window", "No maint.\\ window")):
        r, s = rm.loc[v], rs.loc[v]
        rows.append(" & ".join([label, f(r.containment_rate), pm(r.mean_ttc_min, s.mean_ttc_min, 1),
                                f(r.wrong_kills_per_day), f(r.wrong_isolations_per_day)]) + r" \\")
    (out / "table_response.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

    # ---------------- gate sensitivity table
    g = pd.read_csv(res / "gate_sensitivity_per_seed.csv").groupby("gate").mean(numeric_only=True)
    rows = [" & ".join([f(k, 2), f(r.f1), f(r.event_detection), f(r.ttd_stealthy, 0),
                        f(r.false_alerts_per_day), f(r.hard_negative_fpr)]) + r" \\" for k, r in g.iterrows()]
    (out / "table_gate.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

    # ---------------- number macros
    cg, nomw = mean.loc["CryptoGuard (full)"], mean.loc["CryptoGuard w/o maintenance window"]
    oc, st, sig, ifo = (mean.loc[k] for k in ("OCSVM (behavioural)", "Static CPU alarm", "Signature only",
                                              "IForest (behavioural)"))
    info = json.loads((res / "run_info.json").read_text())
    ex = json.loads((res / "explanation_examples.json").read_text())
    prof = pd.read_csv(res / "explanation_profile.csv", index_col=0)
    compute = ["cpu_percent", "cpu_std", "cpu_sustained", "top_proc_cpu", "top_proc_share", "cpu_excess",
               "hour_sin"]
    share = prof[[c for c in compute if c in prof]].sum(axis=1)
    macros = {
        "nSeeds": n_seeds, "trainWindows": info["train_windows"], "testWindows": info["test_windows"],
        "cgP": f(cg.precision), "cgR": f(cg.recall), "cgFone": f(cg.f1), "cgAUC": f(cg.auc),
        "cgDet": f(cg.event_detection * 100, 0), "cgDetStealthy": f(cg.det_stealthy * 100, 0),
        "cgTTDnoisy": f(cg.ttd_noisy, 1), "cgTTDstealthy": f(cg.ttd_stealthy, 0),
        "cgTTDcontainer": f(cg.ttd_container, 1), "cgFA": f(cg.false_alerts_per_day),
        "cgHN": f(cg.hard_negative_fpr * 100, 1),
        "nomwFone": f(nomw.f1), "nomwTTDstealthy": f(nomw.ttd_stealthy, 0), "nomwDetStealthy": f(nomw.det_stealthy * 100, 0),
        "ocFone": f(oc.f1), "ocFA": f(oc.false_alerts_per_day), "ocHN": f(oc.hard_negative_fpr * 100, 0),
        "ocDet": f(oc.event_detection * 100, 0),
        "stDet": f(st.event_detection * 100, 0), "stFone": f(st.f1), "stFA": f(st.false_alerts_per_day),
        "sigDet": f(sig.event_detection * 100, 0), "sigFone": f(sig.f1),
        "ifFone": f(ifo.f1), "ifFA": f(ifo.false_alerts_per_day), "ifHN": f(ifo.hard_negative_fpr * 100, 0),
        "respContain": f(rm.loc["full"].containment_rate * 100, 0),
        "respTTC": f(rm.loc["full"].mean_ttc_min, 1),
        "respWrongKill": f(rm.loc["full"].wrong_kills_per_day),
        "respWrongIso": f(rm.loc["full"].wrong_isolations_per_day),
        "shareMinerMin": f(share[["noisy", "stealthy", "container"]].min()),
        "shareSurge": f(share.get("traffic_surge", np.nan)), "shareCI": f(share.get("ci_build", np.nan)),
        "exStealthyMinute": ex.get("stealthy", {}).get("minute_into_attack", "--"),
    }
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    if "stealthy" in ex:
        lines.append("\\newcommand{\\exStealthySummary}{" +
                     ex["stealthy"]["summary"].replace("%", "\\%").replace("~", "$\\sim$") + "}")
    (out / "numbers.tex").write_text("\n".join(lines) + "\n")

    # ---------------- live test paragraph (only if scripts/live_demo.py was run)
    live_p = res / "live_demo.json"
    live = ""
    if live_p.exists():
        L = json.loads(live_p.read_text())
        sc = {x["name"]: x for x in L["scenarios"]}

        def t(x):
            return "not within the test" if x is None else f"{x:.1f}\\,s"
        n, s_ = sc.get("noisy", {}), sc.get("stealthy", {})
        live = (
            "\\subsection{Live Test}\n"
            f"CryptoGuard was also run end-to-end on a 2-vCPU Linux cloud container with active response "
            f"(process termination on, network isolation off). A baseline of {L['baseline_windows']} windows of "
            f"{L['window_seconds']:.0f}\\,s was recorded (mean CPU {L['baseline_cpu_mean']:.1f}\\%), the model was "
            f"trained on it, and the harmless miner simulator was then started twice. In the \\emph{{noisy}} run "
            f"(all cores, \\texttt{{xmrig}} in the command line, connection to a fake pool on port 3333), the first "
            f"alert came {t(n.get('detect'))} after the miner started and the process was terminated after "
            f"{t(n.get('kill'))}. In the \\emph{{stealthy}} run (one core at 50\\%, no signature, no pool "
            f"connection), detection relied on behaviour alone: the first alert came after {t(s_.get('detect'))} "
            f"and the process was terminated after {t(s_.get('kill'))}, once the drift confidence passed the kill threshold. No alerts were raised in the windows before either miner "
            f"started. Decisions are taken at the end of each window, so these latencies depend on when in a "
            f"window the miner starts. The live test shows that the pipeline works end to end on a real "
            f"host; it is not a measurement of accuracy, because the idle test host is far simpler than a "
            f"production server.\n")
        if s_.get("explanation"):
            live += ("\nThe first alert of the stealthy run, raised from behaviour alone, read: \\emph{"
                     + s_["explanation"].replace("%", "\\%").replace("_", "\\_") + "}\n")
    (out / "live.tex").write_text(live)
    if live_p.exists():
        with open(out / "numbers.tex", "a") as fh:
            fh.write(f"\\newcommand{{\\liveNoisyKill}}{{{f(sc['noisy'].get('kill'), 1)}}}\n")
            fh.write(f"\\newcommand{{\\liveStealthyKill}}{{{f(sc['stealthy'].get('kill'), 0)}}}\n")
    print(f"tables + macros written to {out}")


if __name__ == "__main__":
    main()
