#!/usr/bin/env python3
"""
composer_build_simple_symphony.py — assemble the SIMPLE 5-sleeve portfolio into
one importable Composer symphony, using the schema that actually imports:

  * a unique id on EVERY node (wrapper nodes get fresh UUIDs; fetched sub-trees
    already carry Composer's ids)
  * rebalance "none" + rebalance-corridor-width  (threshold rebalance)
  * weights as STRINGS ({"num":"20","den":"100"})

Equal weight (20% each). Read-only: fetches each sub-symphony's public score.
Writes data/simple_portfolio_symphony.json.
"""
from __future__ import annotations
import os, sys, json, uuid
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos as oos

# (name, id, weight%) — equal weight
SLEEVES = [
    ("Pop Bot (SPY vs BND) l BrianE l May 30th 2007",  "2cuimtTihBBpJgf7FSis", 20),
    ("Simple Regime Switching and Dip Buying",          "Dt37l1ceAggm8ggzBpRS", 20),
    ("Joseph Story Fund",                                "tEq3s5F3AzjqcxwwvVVJ", 20),
    ("Simple Dividends",                                 "0NukZC005nYIg0PZ7wET", 20),
    ("OG V 1bb | Gain Train DGAF | Deez",                "l4glDbmbbDFd3p1Mcjkx", 20),
]

ENDPOINTS = [
    "https://backtest-api.composer.trade/api/v1/public/symphonies/{id}/score",
    "https://backtest-api.composer.trade/api/v1/public/symphonies/{id}",
]


def tree(o):
    if isinstance(o, dict):
        if ("step" in o or ":step" in o) and ("children" in o or ":children" in o):
            return o
        for k in ("symphony", "score", "definition", "s"):
            if k in o:
                t = tree(o[k])
                if t: return t
        for v in o.values():
            t = tree(v)
            if t: return t
    return None


def fetch(sid):
    for tmpl in ENDPOINTS:
        try:
            r = requests.get(tmpl.format(id=sid), headers={"accept": "application/json"}, timeout=25)
        except Exception:
            continue
        if r.status_code == 200 and "json" in r.headers.get("content-type", ""):
            t = tree(r.json())
            if t:
                return t
    return None


def ensure_ids(node):
    """Guarantee every node carries an id (fill any that are missing)."""
    if isinstance(node, dict):
        if (node.get("step")) and not node.get("id"):
            node["id"] = str(uuid.uuid4())
        for c in node.get("children", []):
            ensure_ids(c)
    return node


def main():
    defs = {}
    for nm, sid, w in SLEEVES:
        t = fetch(sid)
        print(f"{'ok' if t else 'MISS'}  {nm[:40]} ({sid})")
        if t:
            defs[sid] = t
    if len(defs) < len(SLEEVES):
        print(f"\n{len(defs)}/{len(SLEEVES)} fetched — cannot assemble.", file=sys.stderr)
        sys.exit(1)

    groups = []
    for nm, sid, w in SLEEVES:
        sub = ensure_ids(defs[sid])
        groups.append({
            "id": str(uuid.uuid4()),
            "step": "group",
            "name": f"{nm} ({w}%)",
            "collapsed?": True,
            "weight": {"num": str(w), "den": "100"},   # STRING weights
            "children": sub.get("children", []),
        })
    root = {
        "id": str(uuid.uuid4()),
        "step": "root",
        "name": "Claude's Simple OOS Portfolio (5 eq-wt)",
        "collapsed?": True,
        "children": [{
            "id": str(uuid.uuid4()),
            "step": "wt-cash-specified",
            "name": "Weight",
            "suppress_incomplete_warnings": True,
            "children": groups,
        }],
        "description": ("Equal-weight blend of 5 simple, high-confidence, OOS-consistent "
                        "Discover strategies (avg ~23 nodes each). OOS 2023-06 -> 2026-07 "
                        "combined ~29% CAGR / -11% MaxDD / 1.52 Sharpe."),
        "rebalance": "none",
        "rebalance-corridor-width": 0.05,
    }

    # sanity: every node has an id
    missing = []
    def walk(n, path="root"):
        if isinstance(n, dict):
            if n.get("step") and not n.get("id"):
                missing.append(path)
            for i, c in enumerate(n.get("children", [])):
                walk(c, f"{path}/{c.get('step')}[{i}]")
    walk(root)
    os.makedirs("data", exist_ok=True)
    json.dump(root, open("data/simple_portfolio_symphony.json", "w"), indent=2, ensure_ascii=False)
    total = 0
    def count(n):
        nonlocal total
        if isinstance(n, dict) and n.get("step"):
            total += 1
            for c in n.get("children", []): count(c)
    count(root)
    print(f"\nwrote data/simple_portfolio_symphony.json  ({total} nodes, "
          f"{len(missing)} missing ids, rebalance=none, string weights)")


if __name__ == "__main__":
    main()
