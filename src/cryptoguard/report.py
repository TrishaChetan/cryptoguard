"""Render the JSONL alert/audit log as a self-contained HTML report."""

from __future__ import annotations

import html
import json
from pathlib import Path


def _load(log_path: str | Path) -> list[dict]:
    p = Path(log_path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def render_report(log_path: str | Path, out_path: str | Path) -> Path:
    rows = _load(log_path)
    incidents: dict[str, dict] = {}
    for r in rows:
        inc = r.get("incident") or r.get("id") or "-"
        d = incidents.setdefault(inc, {"alerts": [], "actions": []})
        if r.get("action") == "notify":
            d["alerts"].append(r)
        elif r.get("action") in ("kill", "isolate"):
            d["actions"].append(r)

    cards = []
    for inc, d in incidents.items():
        if not d["alerts"]:
            continue
        first = d["alerts"][0]
        exp = first.get("explanation") or {}
        conf = first.get("verdict", {}).get("confidence", 0)
        reasons = "".join(
            f"<li><b>{html.escape(x['feature'])}</b> "
            f"<span class='w'>{x['weight'] * 100:.0f}%</span> {html.escape(x['text'])}</li>"
            for x in exp.get("reasons", []))
        intel = "".join(f"<li>{html.escape(t)}</li>" for t in exp.get("threat_intel", []))
        acts = "".join(
            f"<li>{html.escape(a['action'])} {html.escape(str(a.get('name') or a.get('mode') or ''))}: "
            f"{html.escape(str(a.get('status')))}{' (dry run)' if a.get('dry_run') else ''}</li>"
            for a in d["actions"]) or "<li>alert only</li>"
        cards.append(f"""
<section class="card">
  <header><h2>{html.escape(inc)}</h2><span class="conf">confidence {conf:.2f}</span></header>
  <p class="ts">{html.escape(str(first.get('time', '')))}</p>
  <p class="summary">{html.escape(exp.get('summary', ''))}</p>
  <div class="cols">
    <div><h3>Model explanation ({html.escape(exp.get('method', ''))})</h3><ul>{reasons}</ul></div>
    <div><h3>Threat intelligence</h3><ul>{intel or '<li>none</li>'}</ul></div>
    <div><h3>Response</h3><ul>{acts}</ul></div>
  </div>
</section>""")

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>CryptoGuard incident report</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:2rem auto;max-width:980px;color:#1d2433;background:#f6f7f9}}
h1{{font-size:1.6rem}} .card{{background:#fff;border:1px solid #e2e5ea;border-radius:10px;padding:1rem 1.25rem;margin:1rem 0}}
header{{display:flex;justify-content:space-between;align-items:baseline}} h2{{font-size:1.1rem;margin:0}}
.conf{{background:#fde8e8;color:#9b1c1c;padding:.15rem .6rem;border-radius:999px;font-size:.85rem}}
.ts{{color:#6b7280;font-size:.85rem;margin:.2rem 0}} .summary{{font-size:.95rem}}
.cols{{display:grid;grid-template-columns:1.4fr 1fr .8fr;gap:1rem}} h3{{font-size:.85rem;color:#4b5563;text-transform:uppercase}}
ul{{padding-left:1.1rem;font-size:.88rem}} .w{{color:#2563eb;font-variant-numeric:tabular-nums}}
</style></head><body>
<h1>CryptoGuard incident report</h1>
<p>{len(cards)} incident(s) from <code>{html.escape(str(log_path))}</code></p>
{''.join(cards) or '<p>No incidents.</p>'}
</body></html>"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
