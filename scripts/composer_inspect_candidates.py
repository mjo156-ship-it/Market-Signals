#!/usr/bin/env python3
"""
composer_inspect_candidates.py — inspect replacement-candidate symphonies to
prove they are ADAPTIVE rule engines (decision nodes / dip-buy & overbought
logic) rather than accidental static single-stock survivors.

For each candidate it prints:
  * structure: node / decision / selection-filter counts, static? flag
  * holdings: individual stocks vs ETFs (asset node has_marketcap)
  * decision logic: a human-readable line per `if` condition
    (lhs indicator + window, comparator, rhs value/indicator)
  * trade-days over the common OOS window (real turnover cadence)

Read-only. IDs come from CANDIDATES below. Prints a report.
"""
from __future__ import annotations
import os, sys, time
from datetime import date
import numpy as np
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos as oos

# (name, id)  — finalists from the OOS-ledger screen
CANDIDATES = [
    # Template A: SPY-simplicity / dip-buy / overbought-hedge (like "SIMPLICITY SPY")
    ("A: SPY Overbought Test (Publication)", "gv99mKXm2PaRO3CSoUPB"),
    ("A: SIMPLICITY, SPY 36/27",             "xN5Hi5Hv94gRHZynUTj5"),
    ("A: Simple S&P 500 & Nasdaq SAVE",      "R5UqpkTqw4DpEDHissw7"),
    ("A: Shy v. Spy: SPY (12d MDD, 12%)",    "lM8PIp0ipSjup6brD10c"),
    # Template B: defensive/income base + tech dip-buy
    ("B: Dip Buying Tech <10d RSI30, XLP cash", "98cACZSS00eDg8Kv5BBV"),
    ("B: Dividends (TECL-UVIX)",                "2NikM8NJOSadxphzzV6g"),
]
START = os.environ.get("PORTFOLIO_COMMON_START", "2023-06-14")
TRADE_THRESH = 0.02
ENDPOINTS = ["https://backtest-api.composer.trade/api/v1/public/symphonies/{id}/score",
             "https://backtest-api.composer.trade/api/v1/public/symphonies/{id}"]


def fetch(sid):
    def tr(o):
        if isinstance(o, dict):
            if ("step" in o) and ("children" in o): return o
            for k in ("symphony", "score", "definition", "s"):
                if k in o:
                    t = tr(o[k])
                    if t: return t
            for v in o.values():
                t = tr(v)
                if t: return t
        return None
    for tmpl in ENDPOINTS:
        try:
            r = requests.get(tmpl.format(id=sid), headers={"accept": "application/json"}, timeout=25)
        except Exception:
            continue
        if r.status_code == 200 and "json" in r.headers.get("content-type", ""):
            t = tr(r.json())
            if t: return t
    return None


def _cond_text(n):
    """Render an if-child condition node as a readable comparison."""
    def side(pfx):
        fn = n.get(f"{pfx}-fn") or n.get(f"{pfx}-fn-name")
        val = n.get(f"{pfx}-val") or n.get(f"{pfx}-value")
        win = n.get(f"{pfx}-window-days") or n.get(f"{pfx}-fn-params", {}).get("window") \
            if isinstance(n.get(f"{pfx}-fn-params"), dict) else n.get(f"{pfx}-window-days")
        fixed = n.get(f"{pfx}-fixed-value?")
        if fn:
            w = f"({win}d)" if win else ""
            return f"{fn}{w} {val or ''}".strip()
        if val is not None:
            return str(val)
        return "?"
    if n.get("is-else-condition?"): return "ELSE"
    lhs = side("lhs"); comp = n.get("comparator") or "?"; rhs = side("rhs")
    return f"{lhs} {comp} {rhs}"


def analyze(t):
    assets, nodes, decisions, selects, conds = {}, 0, 0, 0, []
    def walk(n, depth=0):
        nonlocal nodes, decisions, selects
        if not isinstance(n, dict): return
        st = n.get("step")
        if st: nodes += 1
        if st == "if": decisions += 1
        if st == "if-child" and (n.get("comparator") or n.get("lhs-fn")):
            conds.append(_cond_text(n))
        if st == "filter":
            selects += 1
            sel = n.get("select-fn") or n.get("sort-by-fn")
            num = n.get("select-n") or n.get("top-n")
            if sel: conds.append(f"SELECT {sel} top-{num}")
        if st == "asset" and n.get("ticker"):
            assets[n["ticker"]] = bool(n.get("has_marketcap"))
        for c in n.get("children", []): walk(c, depth+1)
    walk(t)
    return {"nodes": nodes, "decisions": decisions, "selects": selects,
            "stocks": sorted(k for k, v in assets.items() if v),
            "etfs": sorted(k for k, v in assets.items() if not v),
            "conds": conds, "static": decisions == 0 and selects == 0}


def backtest_tdvm(sid, start, end):
    body = {"capital": 100000, "apply_reg_fee": True, "apply_taf_fee": True,
            "apply_cat_fee": True, "apply_subscription": "none", "backtest_version": "v2",
            "slippage_percent": 0.0005, "start_date": start, "end_date": end,
            "broker": "ALPACA_WHITE_LABEL", "benchmark_tickers": ["SPY"]}
    for k in range(5):
        try:
            d = oos._request("POST", f"/api/v0.1/symphonies/{sid}/backtest", json=body); break
        except RuntimeError as e:
            if "rate limited" in str(e).lower() and k < 4: time.sleep(6 * (k + 1)); continue
            raise
    return d.get("tdvm_weights")


def _dkey(k):
    s = str(k).replace("-", "").replace(".", "")
    return float(s) if s.isdigit() else str(k)


def trade_days(tdvm):
    if not isinstance(tdvm, dict) or not tdvm: return None
    if not isinstance(next(iter(tdvm.values())), dict): return None
    tickers = sorted(tdvm.keys()); byday = {}
    for tk, series in tdvm.items():
        if isinstance(series, dict):
            for d, w in series.items(): byday.setdefault(d, {})[tk] = w
    days = sorted(byday.keys(), key=_dkey)
    if len(days) < 5: return None
    trades = 0; prev = None
    for dd in days:
        w = byday[dd]
        vec = np.array([float(w.get(k, 0) or 0) for k in tickers]); s = vec.sum()
        if s > 0: vec = vec / s
        if prev is not None and 0.5 * np.abs(vec - prev).sum() > TRADE_THRESH: trades += 1
        prev = vec
    return {"n": len(days) - 1, "trades": trades}


def main():
    print(f"window {START} -> {date.today().isoformat()}\n")
    for nm, sid in CANDIDATES:
        t = fetch(sid); time.sleep(0.4)
        if not t:
            print(f"### {nm} [{sid}]  -- FETCH FAILED\n"); continue
        info = analyze(t)
        try:
            cd = trade_days(backtest_tdvm(sid, START, date.today().isoformat()))
        except Exception as e:
            cd = None; print(f"  (bt fail {type(e).__name__})", file=sys.stderr)
        kind = ("STATIC single-stock basket" if info["static"] and info["stocks"] and not info["etfs"]
                else "static basket" if info["static"] else "ADAPTIVE (rules)")
        print(f"### {nm}  [{sid}]")
        print(f"    {info['nodes']} nodes, {info['decisions']} if-branches, "
              f"{info['selects']} selection-filters  -> {kind}")
        print(f"    stocks({len(info['stocks'])}): {info['stocks']}")
        print(f"    ETFs({len(info['etfs'])}): {info['etfs']}")
        if cd:
            tpy = round(cd['trades'] / (cd['n'] / 252)) if cd['n'] else None
            print(f"    trade-days: {cd['trades']}/{cd['n']} (~{tpy}/yr, {round(100*cd['trades']/cd['n'],1)}%)")
        if info["conds"]:
            print("    logic:")
            for c in info["conds"][:14]:
                print(f"      - {c}")
        print()


if __name__ == "__main__":
    main()
