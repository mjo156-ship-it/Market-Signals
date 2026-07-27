#!/usr/bin/env python3
"""
composer_build_symphony.py — assemble the 5-sleeve portfolio into ONE Composer
symphony (weighted, specified %), IF the sub-symphony definitions can be fetched.

Composer's symphony-definition ("score") lives behind PUBLIC endpoints that
usually need NO auth (auth headers can actually 404 them). This tries the public
score/definition endpoints — with and without auth — for the 5 portfolio ids.
If all 5 node-trees come back, it wraps them under a `wt-cash-specified` root at
the recommended inverse-vol weights and writes data/portfolio_symphony.json
(plus prints it). If they're gated, it says so clearly and writes nothing.

Read-only.
"""
from __future__ import annotations
import os, sys, json
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos as oos

# (name, id, weight%) — inverse-vol weighting from the OOS portfolio analysis
SLEEVES = [
    ("SPY S&P500 Avoid All Market Crashes", "F8cR1emy2BFsjiwVVwRI", 28),
    ("Pop Bot (SPY vs BND)",                "2cuimtTihBBpJgf7FSis", 25),
    ("Tech vs Utilities | Managed Risk",    "z3YepUFuvrpdV6QA2g2P", 20),
    ("Sector Selector Ninja Mode",          "A2ioLOOs13YguciW4Jhs", 15),
    ("NASDAQ-X DeETF ($Deez)",              "UcEdbLxz7BUGsBy0YRbf", 12),
]

ENDPOINTS = [
    "https://backtest-api.composer.trade/api/v1/public/symphonies/{id}/score",
    "https://backtest-api.composer.trade/api/v1/public/symphonies/{id}",
    "https://stagehand-api.composer.trade/api/v1/public/symphonies/{id}/score",
    "https://stagehand-api.composer.trade/api/v1/public/symphonies/{id}",
    "https://api.composer.trade/api/v1/public/symphonies/{id}/score",
    "https://api.composer.trade/api/v1/public/symphonies/{id}",
]


def _tree(obj):
    """Pull a Composer node-tree (step/children, either 'step' or ':step') out
    of an arbitrary response shape."""
    if isinstance(obj, dict):
        keys = set(obj.keys())
        if ("step" in keys or ":step" in keys) and ("children" in keys or ":children" in keys):
            return obj
        for k in ("symphony", "score", "definition", "spec", "symphony_score", "s"):
            if k in obj:
                t = _tree(obj[k])
                if t:
                    return t
        for v in obj.values():
            t = _tree(v)
            if t:
                return t
    return None


def fetch_def(sid):
    auth = oos._headers()
    for tmpl in ENDPOINTS:
        url = tmpl.format(id=sid)
        for label, hdr in (("noauth", {"accept": "application/json"}), ("auth", auth)):
            try:
                r = requests.get(url, headers=hdr, timeout=30)
            except Exception as e:
                continue
            if r.status_code == 200 and "json" in r.headers.get("content-type", ""):
                try:
                    t = _tree(r.json())
                except Exception:
                    t = None
                if t:
                    print(f"    got via [{label}] {url}")
                    return t
            print(f"    [{label}] {url} -> {r.status_code}")
    return None


def child_key(tree, base):
    """Return the key spelled to match the tree ('step' vs ':step' style)."""
    return (":" + base) if (":step" in tree) else base


def main():
    print("Fetching 5 symphony definitions ...")
    defs = {}
    for nm, sid, w in SLEEVES:
        print(f"  {nm} ({sid}):")
        t = fetch_def(sid)
        if t:
            defs[sid] = t
    if len(defs) < len(SLEEVES):
        print(f"\n{len(defs)}/{len(SLEEVES)} definitions retrieved — the rest are "
              f"gated (browser-session only). Cannot assemble a faithful single "
              f"symphony without them.")
        return

    # All 5 fetched — assemble under a wt-cash-specified root at the given weights.
    colon = ":step" in next(iter(defs.values()))
    K = (lambda b: ":" + b) if colon else (lambda b: b)
    def group_for(nm, sid, w, tree):
        kids = tree.get(K("children"), [])
        return {K("step"): "group", K("name"): f"{nm} ({w}%)",
                K("weight"): {K("num"): w, K("den"): 100}, K("children"): kids}
    root = {
        K("step"): "root",
        K("name"): "High-Confidence OOS Blend (5 symphonies)",
        K("description"): "Inverse-vol blend of 5 High-confidence, OOS-consistent Discover strategies.",
        K("rebalance"): "daily",
        K("children"): [{
            K("step"): "wt-cash-specified",
            K("children"): [group_for(nm, sid, w, defs[sid]) for nm, sid, w in SLEEVES],
        }],
    }
    os.makedirs("data", exist_ok=True)
    json.dump(root, open("data/portfolio_symphony.json", "w"), indent=2, ensure_ascii=False)
    print("\nAssembled single symphony -> data/portfolio_symphony.json")
    print(json.dumps(root, ensure_ascii=False)[:1200])


if __name__ == "__main__":
    main()
