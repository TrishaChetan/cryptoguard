import subprocess
import sys
import time

import psutil

from cryptoguard.responder import Responder, iptables_isolation_commands

VERDICT_HIGH = {"confidence": 0.97, "intel_conf": 0.5}


def _window(pid, name="xmrig", cpu=95.0, reasons=("connected to 1.2.3.4:3333 (mining-pool port)",)):
    return {"timestamp": "t", "top_process": {"pid": pid, "name": name, "cpu": cpu},
            "suspicious_processes": [{"pid": pid, "name": name, "cpu": cpu, "reasons": list(reasons)}]}


def test_dry_run_does_not_kill(cfg):
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        r = Responder(cfg)
        actions = r.handle(_window(p.pid), VERDICT_HIGH, {"summary": "x"})
        kinds = [a["action"] for a in actions]
        assert "kill" in kinds and "notify" in kinds
        assert psutil.pid_exists(p.pid) and p.poll() is None
    finally:
        p.kill()


def test_active_kill(cfg):
    cfg["response"]["dry_run"] = False
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    r = Responder(cfg)
    actions = r.handle(_window(p.pid), VERDICT_HIGH, {"summary": "x"})
    kill = [a for a in actions if a["action"] == "kill"][0]
    assert kill["status"] == "terminated"
    for _ in range(20):
        if p.poll() is not None:
            break
        time.sleep(0.1)
    assert p.poll() is not None


def test_allowlist_and_pid1_protected(cfg):
    cfg["response"]["dry_run"] = False
    r = Responder(cfg)
    actions = r.handle(_window(1, name="systemd"), VERDICT_HIGH, {"summary": "x"})
    assert not [a for a in actions if a["action"] == "kill"]
    w = _window(999999, name="nginx", reasons=())
    w["suspicious_processes"] = []
    r2 = Responder(cfg)
    assert not [a for a in r2.handle(w, VERDICT_HIGH, None) if a["action"] == "kill"]


def test_graduated_policy(cfg):
    r = Responder(cfg)
    a = r.handle(_window(4242), {"confidence": 0.6, "intel_conf": 0.0}, None)
    assert [x["action"] for x in a] == ["notify"]          # alert only
    r = Responder(cfg)
    a = r.handle(_window(4242), {"confidence": 0.8, "intel_conf": 0.0}, None)
    assert "kill" in [x["action"] for x in a] and "isolate" not in [x["action"] for x in a]
    r = Responder(cfg)
    a = r.handle(_window(4242), {"confidence": 0.95, "intel_conf": 0.5}, None)
    assert "isolate" in [x["action"] for x in a]


def test_incident_dedup_and_close(cfg):
    r = Responder(cfg)
    first = r.handle(_window(4242), {"confidence": 0.6, "intel_conf": 0}, None)
    second = r.handle(_window(4242), {"confidence": 0.6, "intel_conf": 0}, None)
    assert len([a for a in first if a["action"] == "notify"]) == 1
    assert not [a for a in second if a["action"] == "notify"]
    closed = []
    for _ in range(6):
        closed += r.handle(_window(4242), {"confidence": 0.1, "intel_conf": 0}, None)
    assert closed and closed[-1]["action"] == "close_incident"


def test_iptables_rules_keep_admin_and_loopback():
    cmds = [" ".join(c) for c in iptables_isolation_commands("10.0.0.0/8")]
    assert any("-o lo -j RETURN" in c for c in cmds)
    assert any("-d 10.0.0.0/8 -j RETURN" in c for c in cmds)
    assert cmds[-1].startswith("iptables -I OUTPUT 1")
