"""Lightweight threat intelligence for cryptomining.

The lists live in plain text files under ``config/intel`` so they can be
updated without touching code. Lines starting with ``#`` are comments.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_INTEL_DIR = Path(__file__).resolve().parents[2] / "config" / "intel"


def _read_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            items.append(line.lower())
    return items


@dataclass
class ThreatIntel:
    pool_ports: set[int] = field(default_factory=set)
    miner_signatures: list[str] = field(default_factory=list)
    pool_domains: list[str] = field(default_factory=list)
    blocklist_ips: set[str] = field(default_factory=set)
    blocklist_nets: list = field(default_factory=list)

    @classmethod
    def load(cls, intel_dir: str | Path | None = None, resolve_domains: bool = False) -> "ThreatIntel":
        d = Path(intel_dir) if intel_dir else DEFAULT_INTEL_DIR
        ports = {int(p) for p in _read_list(d / "pool_ports.txt") if p.isdigit()}
        sigs = _read_list(d / "miner_signatures.txt")
        domains = _read_list(d / "pool_domains.txt")
        ips, nets = set(), []
        for entry in _read_list(d / "blocklist_ips.txt"):
            try:
                if "/" in entry:
                    nets.append(ipaddress.ip_network(entry, strict=False))
                else:
                    ips.add(str(ipaddress.ip_address(entry)))
            except ValueError:
                continue
        intel = cls(ports, sigs, domains, ips, nets)
        if resolve_domains:
            intel.resolve_pool_domains()
        return intel

    def resolve_pool_domains(self, timeout: float = 2.0) -> int:
        """Resolve known pool domains to IPs and add them to the blocklist."""
        old = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)
        added = 0
        try:
            for dom in self.pool_domains:
                try:
                    for info in socket.getaddrinfo(dom, None):
                        ip = info[4][0]
                        if ip not in self.blocklist_ips:
                            self.blocklist_ips.add(ip)
                            added += 1
                except OSError:
                    continue
        finally:
            socket.setdefaulttimeout(old)
        return added

    # ------------------------------------------------------------------ checks
    def is_pool_port(self, port: int | None) -> bool:
        return port is not None and int(port) in self.pool_ports

    def is_blocklisted(self, ip: str | None) -> bool:
        if not ip:
            return False
        if ip in self.blocklist_ips:
            return True
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        return any(addr in net for net in self.blocklist_nets)

    def matches_miner(self, name: str | None, cmdline: str | None = None) -> str | None:
        """Return the matched signature, or None."""
        hay = f"{name or ''} {cmdline or ''}".lower()
        for sig in self.miner_signatures:
            if sig and sig in hay:
                return sig
        return None
