# CryptoGuard

**Explainable, self-responding cryptojacking detection for cloud servers.**

CryptoGuard watches how a cloud server behaves (CPU usage, running processes, network connections) and learns what normal looks like for that server. When the server starts doing work that looks like hidden cryptocurrency mining, CryptoGuard explains in plain English why it is suspicious. If it is confident enough, it stops the threat automatically by killing the miner process or isolating the server, and it notifies a human to investigate.

Real output from the live test in this repo (`scripts/live_demo.py`). A throttled miner with no signature and no pool connection is caught from behaviour alone:

```
2026-11-23T15:30:43 cpu= 25.6% top=python3   score=0.692 conf=0.52 [ALERT]
   WHY: Looks like cryptojacking because the server is using 25 percentage points more CPU than
        its current traffic and process activity can explain (normally under 1); CPU usage is 26%,
        while this server normally runs at about 0% at 15:00; a single process is using 25% of
        total CPU (normal top process: ~0%).
   ...
2026-11-23T15:32:04 cpu= 25.3% top=python3   score=0.676 conf=0.76 [ALERT]
   ACTION: kill python3 -> terminated
```

---

## Why this problem matters

Cryptojacking means an attacker breaks into a cloud account or server and secretly runs a cryptocurrency miner on it. No data is stolen, so most security tools stay silent. The victim usually finds out only when a large, unexplained cloud bill arrives.

* Google Cloud found that 86% of 50 compromised Google Cloud instances were used for cryptocurrency mining (Threat Horizons report, Nov 2021).
* Sysdig estimated that victims pay about **$53 in cloud costs for every $1** the attacker mines (2022 Cloud-Native Threat Report).
* Attackers can easily evade signature checks: they rename the miner binary, tunnel mining traffic over TLS on port 443, or throttle the miner so it never trips a "CPU > 90%" alarm.

## What CryptoGuard does

| Stage | What happens |
|---|---|
| **1. Watch** | A lightweight agent (`psutil`) samples CPU every second and summarises each 10 s window: CPU level and variability, the hungriest process, process churn, connections, traffic, and time of day. |
| **2. Learn normal** | A ridge model learns how much CPU the server's workload *should* need. The difference from actual CPU is **CPU excess**. An **Isolation Forest** learns the joint normal profile of all features. No attack labels are needed. |
| **3. Detect** | Two time-scales: **point anomalies** catch abrupt miners within a few windows, and a **CUSUM** accumulator catches *throttled* miners that look only slightly unusual but stay that way for hours. |
| **4. Explain (XAI)** | **TreeSHAP** gives the exact contribution of each feature to every alert, and the alert text compares each value with the server's own normal range. |
| **5. XAI gate** | The explanation is also used *in the decision*. An anomaly counts as mining-like only if at least 75% of its SHAP attribution is on compute features. A marketing traffic surge or a CI build is unusual too, but it is explained by traffic or process churn, so CryptoGuard reports it without acting on it. |
| **6. Fuse threat intel** | Connections to Stratum mining ports, miner names or command lines (`xmrig`, `stratum+tcp`, …), and blocklisted pool IPs are combined with the behavioural score (noisy-OR). |
| **7. Respond** | Graduated response: **alert** (≥ 0.50) → **kill** the attributed process tree (≥ 0.75) → **isolate** the server with iptables or an AWS quarantine security group (≥ 0.90 with corroboration). Safety rails: dry-run by default, process allow-list, PID 1 and the agent itself are never touched, conservative culprit attribution, one incident per attack, and a JSON audit log. |

```
cloud server ──metrics──► collector ──x_t──► normal-profile model ──s_t──► decision layer ──c_t──► responder
                              │                      │                          ▲  ▲                 │   │
                              ▼                      ▼                          │  │                 │   └─► admin (alert + reasons)
                        threat intel ───────────────────── evidence ────────────┘  │                 │
                                              TreeSHAP explainer ── compute share ─┘                 └─► kill / isolate
```

## Results (simulated cloud workload, mean of 5 seeds)

| Method | Precision | Recall | F1 | Event det. | TTD throttled | False alerts/day | HN alerted |
|---|---|---|---|---|---|---|---|
| Static CPU alarm (>90 % for 5 min) | 0.93 | 0.31 | 0.46 | 61 % | never | 1.67 | 12 % |
| Signatures / threat intel only | 1.00 | 0.59 | 0.70 | 63 % | 1 min (only 53 % found) | 0.00 | 0 % |
| One-Class SVM | 0.85 | 0.99 | 0.91 | 100 % | 3 min | 2.47 | 96 % |
| Isolation Forest | 0.65 | 0.34 | 0.44 | 88 % | 179 min | 2.20 | 97 % |
| CryptoGuard, no maintenance window | 0.94 | 0.67 | 0.78 | 96 % | 134 min | 0.07 | 0 % |
| **CryptoGuard (full)** | **0.95** | **0.96** | **0.95** | **100 %** | **16 min** | **0.13** | **0 %** |

Automated response (replayed on the same streams): **100 % of attacks contained** (culprit killed or server isolated), on average 12.5 minutes after the attack began, with **0 wrong kills and 0 wrong isolations**.

The ablation study in the paper shows what each part contributes. Without the CUSUM drift detector, most throttled miners are missed. Without the XAI gate, every traffic surge and CI build becomes an alert (2.2 false alerts per day). Without the persistence filter, false alerts rise to 1.1 per day.

Notes:
* "Event det." is the share of attacks that raised at least one alert. "TTD throttled" is the minutes until a throttled miner is first detected. "HN alerted" is the share of *legitimate* heavy-load windows (traffic surges, CI builds) that raised an alert.
* Real labelled cryptojacking telemetry is not publicly available, so these results come from the workload simulator included in this repo (`src/cryptoguard/simulator.py`). Treat them as a controlled comparison of design choices, not as a production accuracy guarantee. The paper discusses this limitation.
* Every number is reproducible with `cryptoguard evaluate` (see below).

## Project structure

```
cryptoguard/
├── config/
│   ├── default.yaml            # all thresholds, response policy, notification settings
│   └── intel/                  # threat-intel lists (pool ports, miner signatures, pool domains, IP blocklist)
├── src/cryptoguard/
│   ├── collector.py            # live telemetry from the host (psutil)
│   ├── features.py             # feature definitions
│   ├── threat_intel.py         # mining-pool / miner-signature matching
│   ├── model.py                # CPU-excess regression + Isolation Forest + CUSUM calibration
│   ├── explainer.py            # TreeSHAP explanations + compute-share (XAI gate)
│   ├── detector.py             # point + drift detection, XAI gate, evidence fusion
│   ├── responder.py            # alert / kill / isolate (iptables, AWS), audit log
│   ├── monitor.py              # real-time loop
│   ├── simulator.py            # synthetic cloud-server telemetry with attacks
│   ├── evaluate.py             # experiments, baselines, ablations
│   ├── report.py               # HTML incident report
│   └── cli.py                  # `cryptoguard` command
├── scripts/
│   ├── fake_miner.py           # SAFE miner simulator for demos (does not mine)
│   ├── live_demo.py            # end-to-end live test on this machine
│   ├── make_figures.py         # paper figures from results/
│   └── make_tables.py          # paper tables from results/
├── tests/                      # pytest suite
├── results/                    # experiment outputs used in the paper
└── paper/                      # IEEE-format paper (LaTeX + PDF)
```

## Installation

Requires Python 3.9+ on Linux (macOS and Windows work for monitoring; iptables isolation is Linux only).

```bash
git clone https://github.com/<your-username>/cryptoguard.git
cd cryptoguard
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ,aws for AWS isolation: pip install -e ".[dev,aws]"
pytest                           # 19 tests
```

## Quick start (offline demo, about 1 minute)

```bash
cryptoguard simulate                                   # 7 days of normal data + 3 test days with attacks
cryptoguard train --data data/normal.csv --calib data/calib.csv
cryptoguard replay --data data/test.csv                # stream the test days through the full pipeline
cryptoguard report                                     # -> logs/report.html (incidents with explanations)
```

`replay` prints every incident with its explanation, the ground-truth scenario, and the action the responder *would* have taken. It always runs in dry-run mode.

## Running it on a real server

```bash
# 1. record what normal looks like on THIS server (360 x 10 s = 1 hour; longer is better)
sudo cryptoguard baseline --windows 360
cryptoguard train --data data/baseline.csv

# 2. monitor in dry-run mode first: alerts and explanations only, no actions
sudo cryptoguard monitor

# 3. when you trust it, turn on automatic response
sudo cryptoguard monitor --active                      # kill culprit processes
sudo cryptoguard monitor --active --isolation iptables # also isolate (keeps admin_cidr reachable!)
sudo cryptoguard release                               # undo iptables isolation
```

Root is needed to see every process's connections and to kill or isolate. Before you use `--isolation iptables`, set `response.admin_cidr` in your config so that you do not lock yourself out. For AWS, create an empty security group, put its ID in `response.aws.quarantine_security_group`, and give the instance role `ec2:DescribeInstances`, `ec2:ModifyInstanceAttribute` and `ec2:CreateTags`.

Register heavy scheduled jobs (backups, batch reports) as maintenance windows. They stay part of "normal", but they no longer delay detection of throttled miners:

```yaml
detector:
  maintenance_windows:
    - {days: [6], start: "04:00", end: "05:15"}   # 0 = Monday ... 6 = Sunday
```

Alerts go to `logs/alerts.jsonl`, and optionally to a Slack/Teams webhook (`response.notify.webhook_url`) or e-mail.

## Live demo with the safe miner simulator

`scripts/fake_miner.py` reproduces a miner's *observable behaviour* without mining anything: it burns CPU with SHA-256 hashing, can throttle itself, can add `xmrig` to its command line, and can connect to a local fake pool on port 3333. Run it only on a machine you own.

```bash
python scripts/live_demo.py --baseline-windows 120 --window 5
```

The script records a 10-minute baseline and trains on it (keep the machine otherwise quiet during this time, because anything you run becomes part of "normal"). It then launches a noisy miner and a throttled miner and measures how long CryptoGuard takes to detect and kill each one. Results go to `results/live_demo.json`. In our run on a 2-vCPU cloud container:

| Scenario | First alert | Process killed |
|---|---|---|
| Noisy: all cores, `xmrig` in the command line, fake pool on :3333 | 0.8 s | 0.8 s |
| Throttled: one core at 50 %, no signature, no pool connection | 10.8 s | 91.5 s |

To try it by hand, run these in two terminals:

```bash
cryptoguard monitor --active --window 5
python scripts/fake_miner.py --duration 300 --label xmrig --pool
```

## Reproducing the paper

```bash
cryptoguard evaluate --seeds 5                         # ~10 min, writes results/*.csv
python scripts/make_figures.py --results results --out paper/figures
python scripts/make_tables.py  --results results --out paper/generated
cd paper && pdflatex paper && pdflatex paper
```

## Configuration reference

All settings are in `config/default.yaml`. Pass your own file with `cryptoguard --config my.yaml <command>`. Only the keys you set are overridden.

| Key | Default | Meaning |
|---|---|---|
| `collector.window_seconds` | 10 | length of one observation window |
| `model.threshold_quantile` | 0.995 | point-anomaly threshold (quantile of normal scores) |
| `model.compute_gate` | 0.75 | XAI gate: minimum share of SHAP attribution on compute features |
| `model.cusum_k` | 1.0 | CUSUM allowance for low-and-slow miners |
| `detector.persistence_k / _n` | 3 / 5 | point anomalies must repeat in k of n windows |
| `response.dry_run` | true | log what would be done, do nothing |
| `response.alert/kill/isolate_threshold` | 0.5 / 0.75 / 0.9 | graduated response |
| `response.isolation_mode` | none | `none`, `iptables` or `aws` |
| `response.allowlist_processes` | sshd, systemd, nginx, postgres, … | never killed automatically |

## Limitations

* The quantitative evaluation uses simulated telemetry. Validate on your own servers in dry-run mode before you enable active response.
* An attacker who mines with only a few percent of CPU, or only inside a registered maintenance window, can stay below the detection threshold.
* GPU mining is not covered by the current features.
* The agent is assumed to be untampered. Run it with least privilege and protect its configuration.

## License

MIT. See [LICENSE](LICENSE).
