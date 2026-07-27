#!/usr/bin/env python3
"""
composer_symphony_probe.py — read-only probe: can we retrieve a symphony's full
DEFINITION (node tree) from Composer? Needed to assemble a single backtestable
symphony that inlines the 5 portfolio sleeves under a weighted root.

Tries a few candidate read-only GET endpoints for the 5 portfolio symphony ids,
plus inspects the backtest POST response for an embedded definition. Saves what
it finds to data/symphony_defs.json and prints the structure. No state change.
"""
from __future__ import annotations
import os, sys, json
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos as oos

IDS = [
    ("SPY Avoid Crashes", "F8cR1emy2BFsjiwVVwRI"),
    ("Pop Bot SPY/BND",   "2cuimtTihBBpJgf7FSis"),
    ("Tech vs Utilities", "z3YepUFuvrpdV6QA2g2P"),
    ("Sector Selector",   "A2ioLOOs13YguciW4Jhs"),
    ("NASDAQ-X DeETF",    "UcEdbLxz7BUGsBy0YRbf"),
]

CANDIDATE_GETS = [
    "https://api.composer.trade/api/v0.1/symphonies/{id}",
    "https://api.composer.trade/api/v1/public/symphonies/{id}",
    "https://backtest-api.composer.trade/api/v1/public/symphonies/{id}",
    "https://backtest-api.composer.trade/api/v1/symphonies/{id}",
    "https://stagehand-api.composer.trade/api/v1/symphonies/{id}",
]


def _looks_like_tree(obj):
    """A Composer symphony definition tree has step/children (or :step) nodes."""
    s = json.dumps(obj)[:200000]
    return ('"step"' in s or '":step"' in s) and ('children' in s)


def _find_def(obj):
    """Return the embedded symphony-definition dict if present."""
    if isinstance(obj, dict):
        for k in ("symphony", "definition", "spec", "score", "symphony_score"):
            v = obj.get(k)
            if isinstance(v, dict) and _looks_like_tree(v):
                return v
        if _looks_like_tree(obj):
            return obj
        for v in obj.values():
            r = _find_def(v)
            if r:
                return r
    return None


def main():
    headers = oos._headers()
    defs = {}
    # 1) candidate GET endpoints (first id only, to find a working one)
    name0, id0 = IDS[0]
    working = None
    for tmpl in CANDIDATE_GETS:
        url = tmpl.format(id=id0)
        try:
            r = requests.get(url, headers=headers, timeout=30)
        except Exception as e:
            print(f"GET {url} -> ERR {type(e).__name__}: {str(e)[:60]}"); continue
        ct = r.headers.get("content-type", "").split(";")[0]
        print(f"GET {url} -> HTTP {r.status_code} ({ct}, {len(r.content)}b)")
        if r.status_code == 200 and "json" in ct:
            try:
                d = _find_def(r.json())
            except Exception:
                d = None
            if d:
                print(f"   ^ DEFINITION FOUND via this endpoint (root keys: {sorted(d.keys())[:12]})")
                working = tmpl
                break

    # 2) inspect the backtest POST response for an embedded definition
    if not working:
        print("\nInspecting backtest POST response for an embedded definition ...")
        try:
            body = {"capital": 100000, "apply_reg_fee": True, "apply_taf_fee": True,
                    "apply_cat_fee": True, "apply_subscription": "none",
                    "backtest_version": "v2", "slippage_percent": 0.0005,
                    "start_date": "2024-01-01", "end_date": "2024-02-01",
                    "broker": "ALPACA_WHITE_LABEL", "benchmark_tickers": ["SPY"]}
            d = oos._request("POST", f"/api/v0.1/symphonies/{id0}/backtest", json=body)
            print("   backtest response top-level keys:", sorted(d.keys())[:20])
            emb = _find_def(d)
            if emb:
                print("   ^ backtest response EMBEDS the definition (root keys:",
                      sorted(emb.keys())[:12], ")")
        except Exception as e:
            print("   backtest inspect failed:", type(e).__name__, str(e)[:80])

    # 3) if a working GET endpoint was found, pull all 5 definitions
    if working:
        print(f"\nFetching all 5 definitions via {working} ...")
        for nm, sid in IDS:
            try:
                r = requests.get(working.format(id=sid), headers=headers, timeout=30)
                d = _find_def(r.json())
                if d:
                    defs[sid] = {"name": nm, "definition": d}
                    print(f"  ok {nm}: {sid}")
                else:
                    print(f"  NO DEF {nm}: {sid}")
            except Exception as e:
                print(f"  fail {nm}: {type(e).__name__}")
    if defs:
        os.makedirs("data", exist_ok=True)
        json.dump(defs, open("data/symphony_defs.json", "w"), indent=1)
        print(f"\nwrote data/symphony_defs.json ({len(defs)} definitions)")
    else:
        print("\nNo definition endpoint reachable with the API keys.")


if __name__ == "__main__":
    main()
