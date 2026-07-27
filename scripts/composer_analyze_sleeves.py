#!/usr/bin/env python3
"""
composer_analyze_sleeves.py — dig into the 5 simple-portfolio sleeves:
  * holdings composition: individual stocks vs ETFs (asset node has_marketcap)
  * structure: decision nodes, selection/rotation vs STATIC fixed weights
    (static single-stock baskets carry survivorship/selection-bias risk)
  * trade-days: how many days each sleeve actually changes its target weights
    (a cleaner complexity measure than average turnover), + the portfolio union.

Read-only. Prints a report.
"""
from __future__ import annotations
import os, sys, time
from datetime import date
import numpy as np
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos as oos

SLEEVES = [
    ("Pop Bot (SPY vs BND)",              "2cuimtTihBBpJgf7FSis"),
    ("Simple Regime Switching + Dip Buy", "Dt37l1ceAggm8ggzBpRS"),
    ("Joseph Story Fund",                 "tEq3s5F3AzjqcxwwvVVJ"),
    ("Simple Dividends",                  "0NukZC005nYIg0PZ7wET"),
    ("OG Gain Train (DGAF)",              "l4glDbmbbDFd3p1Mcjkx"),
]
START = os.environ.get("PORTFOLIO_COMMON_START", "2023-06-14")
TRADE_THRESH = 0.02   # a day "trades" if one-way target-weight change > 2%

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


def analyze_def(t):
    assets, decisions, selects, nodes = {}, 0, 0, 0
    def walk(n):
        nonlocal decisions, selects, nodes
        if not isinstance(n, dict): return
        if n.get("step"): nodes += 1
        if n.get("step") in ("if", "filter"): decisions += 1
        if n.get("step") == "filter" and n.get("select?"): selects += 1
        if n.get("step") == "asset" and n.get("ticker"):
            assets[n["ticker"]] = bool(n.get("has_marketcap"))
        for c in n.get("children", []): walk(c)
    walk(t)
    stocks = sorted(k for k, v in assets.items() if v)
    etfs = sorted(k for k, v in assets.items() if not v)
    return {"nodes": nodes, "decisions": decisions, "selects": selects,
            "stocks": stocks, "etfs": etfs,
            "static": decisions == 0 and selects == 0}


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


def change_days(tdvm):
    """Count days the target weights actually change.

    Composer returns tdvm_weights as {ticker: {date: weight}}; transpose to
    {date: {ticker: weight}} before diffing consecutive days.
    """
    if not isinstance(tdvm, dict) or not tdvm: return None
    sample = next(iter(tdvm.values()))
    if not isinstance(sample, dict): return None          # unexpected shape
    tickers = sorted(tdvm.keys())
    byday = {}
    for tk, series in tdvm.items():
        if not isinstance(series, dict): continue
        for d, w in series.items():
            byday.setdefault(d, {})[tk] = w
    days = sorted(byday.keys(), key=_dkey)
    if len(days) < 5: return None
    flags = {}
    prev = None
    for dd in days:
        w = byday[dd]
        vec = np.array([float(w.get(k, 0) or 0) for k in tickers]); s = vec.sum()
        if s > 0: vec = vec / s
        if prev is not None:
            flags[dd] = 0.5 * np.abs(vec - prev).sum() > TRADE_THRESH
        prev = vec
    return {"days": days, "flags": flags, "n": len(flags),
            "trades": sum(flags.values())}


def main():
    asof = date.today().isoformat()
    print(f"window {START} -> {asof}\n")
    union_trade = {}
    total_days = 0
    for nm, sid in SLEEVES:
        info = analyze_def(fetch(sid)) if fetch else {}
        time.sleep(0.4)
        try:
            cd = change_days(backtest_tdvm(sid, START, asof))
        except Exception as e:
            cd = None
            print(f"{nm}: bt fail {type(e).__name__}", file=sys.stderr)
        yrs = (cd["n"] / 252) if cd else None
        tpy = round(cd["trades"] / yrs) if (cd and yrs) else None
        if cd:
            total_days = max(total_days, cd["n"])
            for dd, f in cd["flags"].items():
                union_trade[dd] = union_trade.get(dd, False) or f
        kind = ("STATIC single-stock basket" if info.get("static") and info.get("stocks") and not info.get("etfs")
                else "static basket" if info.get("static")
                else "adaptive (rotation/selection)")
        print(f"### {nm}  [{sid}]")
        print(f"    structure: {info.get('nodes')} nodes, {info.get('decisions')} decisions, "
              f"{info.get('selects')} selection-filters  -> {kind}")
        print(f"    stocks ({len(info.get('stocks', []))}): {info.get('stocks')}")
        print(f"    ETFs   ({len(info.get('etfs', []))}): {info.get('etfs')}")
        if cd:
            print(f"    trade-days: {cd['trades']}/{cd['n']} days  (~{tpy}/yr, "
                  f"{round(100*cd['trades']/cd['n'],1)}% of days)")
        print()
    if union_trade:
        ut = sum(union_trade.values())
        print(f"=== PORTFOLIO (union of sleeve trade-days) ===")
        print(f"    the basket wants to trade on {ut}/{total_days} days (~{round(ut/(total_days/252))}/yr, "
              f"{round(100*ut/total_days,1)}% of days) before the 5% corridor throttles it further")


if __name__ == "__main__":
    main()
