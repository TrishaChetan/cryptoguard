"""Automated response: alert, terminate the culprit, isolate the server.

Graduated policy (thresholds from config):

    confidence >= alert_threshold    -> open incident, notify with explanation
    confidence >= kill_threshold     -> terminate the attributed process tree
    confidence >= isolate_threshold  -> isolate the server, but only when there
                                        is threat-intel evidence or the miner
                                        came back after being killed

Safety rails: ``dry_run`` (default on), a process allowlist, never touching
PID 1 / this process / its parent, one incident per attack (no alert storm),
and every action written to an audit log.
"""

from __future__ import annotations

import json
import os
import smtplib
import subprocess
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import psutil

QUARANTINE_IN = "CG_QUARANTINE_IN"
QUARANTINE_OUT = "CG_QUARANTINE_OUT"


@dataclass
class Incident:
    id: str
    opened: str
    max_confidence: float = 0.0
    actions: list[dict] = field(default_factory=list)
    killed_names: set[str] = field(default_factory=set)
    recurrences: int = 0
    quiet_windows: int = 0
    isolated: bool = False


class Responder:
    def __init__(self, cfg: dict):
        r = cfg["response"]
        self.cfg = r
        self.dry_run = bool(r.get("dry_run", True))
        self.alert_t = r["alert_threshold"]
        self.kill_t = r["kill_threshold"]
        self.isolate_t = r["isolate_threshold"]
        self.mode = r.get("isolation_mode", "none")
        self.allow = {a.lower() for a in r.get("allowlist_processes", [])}
        self.log_path = Path(r["notify"]["log_file"])
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.incident: Incident | None = None
        self.close_after = 6
        self._own = os.getpid()
        self._parent = os.getppid()

    # ============================================================== main hook
    def handle(self, window: dict, verdict: dict, explanation: dict | None) -> list[dict]:
        """Called once per window. Returns the actions taken (or simulated)."""
        conf = verdict["confidence"]
        actions: list[dict] = []
        if conf < self.alert_t:
            if self.incident:
                self.incident.quiet_windows += 1
                if self.incident.quiet_windows >= self.close_after:
                    actions.append(self._close_incident())
            return actions

        new = self.incident is None
        if new:
            self.incident = Incident(id=datetime.now().strftime("CG-%Y%m%d-%H%M%S"),
                                     opened=window.get("timestamp", ""))
        inc = self.incident
        inc.quiet_windows = 0
        inc.max_confidence = max(inc.max_confidence, conf)

        culprits = self._culprits(window, verdict)
        if inc.killed_names and any(c["name"] in inc.killed_names for c in culprits):
            inc.recurrences += 1

        if new:
            actions.append(self._notify("ALERT", window, verdict, explanation))

        if conf >= self.kill_t:
            for c in culprits:
                actions.append(self._kill(c))
                inc.killed_names.add(c["name"])

        intel_present = verdict.get("intel_conf", 0) > 0
        if conf >= self.isolate_t and not inc.isolated and (intel_present or inc.recurrences >= 2):
            actions.append(self._isolate())
            inc.isolated = True

        if actions and not new and any(a["action"] in ("kill", "isolate") for a in actions):
            actions.append(self._notify("ACTION", window, verdict, explanation, actions))
        inc.actions.extend(actions)
        return actions

    # ============================================================ attribution
    def _culprits(self, window: dict, verdict: dict | None = None) -> list[dict]:
        """Pick processes to terminate.

        1. processes flagged by threat intel (pool connection, miner signature);
        2. otherwise the top CPU process, but only if it can account for the
           unexplained CPU: it uses >= 20 % CPU, >= 40 % of all CPU in use, and
           at least half of the CPU excess. This stops CryptoGuard from killing
           a legitimate process that merely happens to be busiest."""
        out = [p for p in window.get("suspicious_processes", []) if self._killable(p)]
        if out:
            return out
        top = window.get("top_process")
        if not top or not self._killable(top):
            return []
        feats = window.get("features", {})
        cpu_total = float(feats.get("cpu_percent", top.get("cpu", 0)) or 0)
        share = top.get("cpu", 0) / cpu_total if cpu_total > 0 else 1.0
        excess = float((verdict or {}).get("cpu_excess", top.get("cpu", 0)))
        if top.get("cpu", 0) >= 20 and share >= 0.4 and excess >= 10 and top.get("cpu", 0) >= 0.5 * excess:
            return [top]
        return []

    def _killable(self, p: dict) -> bool:
        pid, name = int(p.get("pid", 0)), (p.get("name") or "").lower()
        if pid <= 1 or pid in (self._own, self._parent):
            return False
        return name not in self.allow

    # ================================================================ actions
    def _record(self, action: dict) -> dict:
        action.setdefault("time", datetime.now().isoformat(timespec="seconds"))
        action["dry_run"] = self.dry_run
        action["incident"] = self.incident.id if self.incident else None
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(action, default=str) + "\n")
        return action

    def _kill(self, proc: dict) -> dict:
        pid = int(proc["pid"])
        result = {"action": "kill", "pid": pid, "name": proc.get("name"), "status": "simulated"}
        if not self.dry_run:
            try:
                parent = psutil.Process(pid)
                victims = parent.children(recursive=True) + [parent]
                for v in victims:
                    v.terminate()
                gone, alive = psutil.wait_procs(victims, timeout=3)
                for v in alive:
                    v.kill()
                result["status"] = "terminated"
                result["processes"] = len(victims)
            except psutil.NoSuchProcess:
                result["status"] = "already_exited"
            except psutil.AccessDenied:
                result["status"] = "access_denied"
        return self._record(result)

    def _isolate(self) -> dict:
        result = {"action": "isolate", "mode": self.mode, "status": "simulated"}
        if self.mode == "iptables":
            cmds = iptables_isolation_commands(self.cfg.get("admin_cidr", "10.0.0.0/8"))
            result["commands"] = [" ".join(c) for c in cmds]
            if not self.dry_run:
                result["status"] = _run_all(cmds)
        elif self.mode == "aws":
            if not self.dry_run:
                result.update(aws_quarantine(self.cfg.get("aws", {})))
        else:
            result["status"] = "skipped (isolation_mode=none)"
        return self._record(result)

    def _close_incident(self) -> dict:
        inc = self.incident
        self.incident = None
        return self._record({"action": "close_incident", "id": inc.id,
                             "max_confidence": round(inc.max_confidence, 3)})

    def _notify(self, kind: str, window: dict, verdict: dict, explanation: dict | None,
                actions: list | None = None) -> dict:
        summary = (explanation or {}).get("summary", "Anomalous behaviour detected.")
        msg = (f"[CryptoGuard {kind}] host={os.uname().nodename if hasattr(os, 'uname') else 'host'} "
               f"confidence={verdict['confidence']:.2f}\n{summary}")
        if actions:
            msg += "\nActions: " + ", ".join(f"{a['action']}({a.get('name') or a.get('mode')}:"
                                              f"{a['status']})" for a in actions)
        payload = {"action": "notify", "kind": kind, "message": msg, "verdict": verdict,
                   "explanation": explanation, "window": window, "status": "logged"}
        hook = self.cfg["notify"].get("webhook_url")
        if hook and not self.dry_run:
            payload["webhook"] = _post_webhook(hook, msg)
        email = self.cfg["notify"].get("email", {})
        if email.get("enabled") and not self.dry_run:
            payload["email"] = _send_email(email, f"CryptoGuard {kind}", msg)
        return self._record(payload)


# ===================================================================== helpers
def iptables_isolation_commands(admin_cidr: str) -> list[list[str]]:
    """Drop all traffic except loopback and the admin network."""
    cmds = []
    for chain, hook, flag in ((QUARANTINE_IN, "INPUT", "-s"), (QUARANTINE_OUT, "OUTPUT", "-d")):
        cmds += [
            ["iptables", "-N", chain],
            ["iptables", "-A", chain, "-i" if hook == "INPUT" else "-o", "lo", "-j", "RETURN"],
            ["iptables", "-A", chain, flag, admin_cidr, "-j", "RETURN"],
            ["iptables", "-A", chain, "-j", "DROP"],
            ["iptables", "-I", hook, "1", "-j", chain],
        ]
    return cmds


def iptables_release_commands() -> list[list[str]]:
    cmds = []
    for chain, hook in ((QUARANTINE_IN, "INPUT"), (QUARANTINE_OUT, "OUTPUT")):
        cmds += [["iptables", "-D", hook, "-j", chain],
                 ["iptables", "-F", chain],
                 ["iptables", "-X", chain]]
    return cmds


def _run_all(cmds: list[list[str]]) -> str:
    for c in cmds:
        proc = subprocess.run(c, capture_output=True, text=True)
        if proc.returncode != 0 and "-N" not in c:
            return f"failed: {' '.join(c)}: {proc.stderr.strip()}"
    return "isolated"


def _imds_instance_id(timeout: float = 2.0) -> str:
    req = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                 headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
    token = urllib.request.urlopen(req, timeout=timeout).read().decode()
    req = urllib.request.Request("http://169.254.169.254/latest/meta-data/instance-id",
                                 headers={"X-aws-ec2-metadata-token": token})
    return urllib.request.urlopen(req, timeout=timeout).read().decode()


def aws_quarantine(aws_cfg: dict) -> dict:
    """Swap the instance's security groups for an empty 'quarantine' group."""
    try:
        import boto3  # optional dependency
    except ImportError:
        return {"status": "failed: boto3 not installed"}
    sg = aws_cfg.get("quarantine_security_group")
    if not sg:
        return {"status": "failed: no quarantine_security_group configured"}
    try:
        iid = aws_cfg.get("instance_id") or _imds_instance_id()
        ec2 = boto3.client("ec2", region_name=aws_cfg.get("region"))
        desc = ec2.describe_instances(InstanceIds=[iid])
        prev = [g["GroupId"] for g in desc["Reservations"][0]["Instances"][0]["SecurityGroups"]]
        ec2.modify_instance_attribute(InstanceId=iid, Groups=[sg])
        ec2.create_tags(Resources=[iid], Tags=[
            {"Key": "cryptoguard:quarantined", "Value": datetime.now().isoformat()},
            {"Key": "cryptoguard:previous-sgs", "Value": ",".join(prev)}])
        return {"status": "isolated", "instance_id": iid, "previous_groups": prev}
    except Exception as exc:   # pragma: no cover - needs AWS
        return {"status": f"failed: {exc}"}


def _post_webhook(url: str, text: str) -> str:
    try:
        data = json.dumps({"text": text}).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
        return "sent"
    except Exception as exc:
        return f"failed: {exc}"


def _send_email(cfg: dict, subject: str, body: str) -> str:
    try:
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = subject, cfg["from_addr"], ", ".join(cfg["to_addrs"])
        msg.set_content(body)
        with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"], timeout=10) as s:
            s.starttls()
            if cfg.get("username"):
                s.login(cfg["username"], os.environ.get(cfg.get("password_env", ""), ""))
            s.send_message(msg)
        return "sent"
    except Exception as exc:
        return f"failed: {exc}"
