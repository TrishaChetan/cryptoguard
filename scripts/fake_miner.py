#!/usr/bin/env python3
"""SAFE cryptojacking simulator for live demos and testing.

It does NOT mine anything. It only reproduces the *observable behaviour*
of a CPU miner so CryptoGuard has something to catch:

  * burns CPU on all (or some) cores with SHA-256 hashing in threads of a
    single process, like XMRig does
  * optionally throttles itself (``--throttle 0.3``) like stealthy miners
  * optionally opens a local fake "mining pool" on 127.0.0.1:3333 and keeps
    a connection to it (Stratum port), so the threat-intel check fires
  * optionally puts a miner-like word in its command line (``--label xmrig``)

Run it only on a machine you own, e.g. the VM where you test CryptoGuard:

    python scripts/fake_miner.py --duration 300 --label xmrig --pool
    python scripts/fake_miner.py --duration 900 --throttle 0.35      # stealthy
"""

from __future__ import annotations

import argparse
import hashlib
import os
import socket
import threading
import time


def burn(stop: threading.Event, throttle: float) -> None:
    buf = os.urandom(1 << 20)            # hashlib releases the GIL for large buffers
    period = 0.1
    while not stop.is_set():
        t0 = time.perf_counter()
        busy = period * throttle
        while time.perf_counter() - t0 < busy:
            hashlib.sha256(buf).digest()
        rest = period - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)


def fake_pool(port: int, stop: threading.Event) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    srv.settimeout(1.0)
    conns = []
    while not stop.is_set():
        try:
            c, _ = srv.accept()
            conns.append(c)
        except socket.timeout:
            continue
    for c in conns:
        c.close()
    srv.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=120, help="seconds to run")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--throttle", type=float, default=1.0, help="fraction of each core to use (0-1]")
    ap.add_argument("--pool", action="store_true", help="connect to a fake pool on 127.0.0.1:<port>")
    ap.add_argument("--port", type=int, default=3333)
    ap.add_argument("--label", default="", help="text added to the command line, e.g. 'xmrig'")
    args = ap.parse_args()

    stop = threading.Event()
    threads = [threading.Thread(target=burn, args=(stop, max(0.01, min(1.0, args.throttle))), daemon=True)
               for _ in range(args.threads)]
    client = None
    if args.pool:
        threading.Thread(target=fake_pool, args=(args.port, stop), daemon=True).start()
        time.sleep(0.3)
        client = socket.create_connection(("127.0.0.1", args.port))
    for t in threads:
        t.start()
    print(f"[fake_miner] pid={os.getpid()} threads={args.threads} throttle={args.throttle} "
          f"pool={'127.0.0.1:%d' % args.port if args.pool else 'none'}", flush=True)
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        pass
    stop.set()
    if client:
        client.close()
    print("[fake_miner] finished", flush=True)


if __name__ == "__main__":
    main()
