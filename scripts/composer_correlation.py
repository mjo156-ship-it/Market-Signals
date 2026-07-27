#!/usr/bin/env python3
"""
composer_correlation.py — daily-return correlation of candidate replacement
sleeves against the three sleeves being KEPT, over the common OOS window.

Also fetches each candidate's holdings (stocks vs ETFs) + decision count so we
can confirm the non-equity alternatives really are non-equity and adaptive.

Prints: holdings summary, full correlation matrix, and each candidate's average
correlation to the three kept equity sleeves.
Read-only.
"""
from __future__ import annotations
import os, sys, time
from datetime import date
import numpy as np
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos as oos

KEPT = [
    ("Pop Bot",              "2cuimtTihBBpJgf7FSis"),
    ("Regime+Dip",           "Dt37l1ceAggm8ggzBpRS"),
    ("OG Gain Train",        "l4glDbmbbDFd3p1Mcjkx"),
]
EQUITY_CAND = [
    ("SPY Overbought",       "gv99mKXm2PaRO3CSoUPB"),
    ("Shy v Spy",            "lM8PIp0ipSjup6brD10c"),
    ("TechDip/XLP",          "98cACZSS00eDg8Kv5BBV"),
]
NONEQ_CAND = [
    ("Gold&Dollar Diversify","0dKcj7cKmeHhafQKlrHM"),
    ("OG 2x AAA (SPY/TLT/UUP/GLD)","DLXJ2T0lIgBMGykzAf1U"),
    ("Stoic Inflation Hedge","c1bWgoNRaiff8ZZwBGD0"),
    ("Diversification at its Best","kULEdknQwUoefsGpQKsT"),
]
ALL = KEPT + EQUITY_CAND + NONEQ_CAND
START = os.environ.get("PORTFOLIO_COMMON_START", "2023-06-14")
END = date.today().isoformat()
ENDPOINTS = ["https://backtest-api.composer.trade/api/v1/public/symphonies/{id}/score",
             "https://backtest-api.composer.trade/api/v1/public/symphonies/{id}"]


def fetch(sid):
    def tr(o):
        if isinstance(o, dict):
            if ("step" in o) and ("children" in o): return o
            for k in ("symphony", "score", "definition", "s"):
                if k in o and (t := tr(o[k])): return t
            for v in o.values():
                if (t := tr(v)): return t
        return None
    for tmpl in ENDPOINTS:
        try:
            r = requests.get(tmpl.format(id=sid), headers={"accept": "application/json"}, timeout=25)
        except Exception:
            continue
        if r.status_code == 200 and "json" in r.headers.get("content-type", ""):
            if (t := tr(r.json())): return t
    return None


def holdings(t):
    assets, dec = {}, 0
    def walk(n):
        nonlocal dec
        if not isinstance(n, dict): return
        if n.get("step") in ("if", "filter"): dec += 1
        if n.get("step") == "asset" and n.get("ticker"):
            assets[n["ticker"]] = bool(n.get("has_marketcap"))
        for c in n.get("children", []): walk(c)
    walk(t or {})
    return dec, sorted(k for k, v in assets.items() if v), sorted(k for k, v in assets.items() if not v)


def equity(sid):
    body = {"capital": 100000, "apply_reg_fee": True, "apply_taf_fee": True,
            "apply_cat_fee": True, "apply_subscription": "none", "backtest_version": "v2",
            "slippage_percent": 0.0005, "start_date": START, "end_date": END,
            "broker": "ALPACA_WHITE_LABEL", "benchmark_tickers": ["SPY"]}
    for k in range(5):
        try:
            d = oos._request("POST", f"/api/v0.1/symphonies/{sid}/backtest", json=body); break
        except RuntimeError as e:
            if "rate limited" in str(e).lower() and k < 4: time.sleep(6 * (k + 1)); continue
            raise
    cap = d.get("dvm_capital") or {}
    # dvm_capital may be {date: val} or {sym_id: {date: val}}
    if cap and isinstance(next(iter(cap.values())), dict):
        cap = next(iter(cap.values()))
    return {str(k): float(v) for k, v in cap.items() if v is not None}


def main():
    print(f"window {START} -> {END}\n")
    curves, names = {}, []
    for nm, sid in ALL:
        t = fetch(sid); time.sleep(0.3)
        dec, st, etf = holdings(t)
        try:
            eq = equity(sid); time.sleep(0.3)
        except Exception as e:
            print(f"{nm}: bt fail {type(e).__name__}", file=sys.stderr); eq = {}
        curves[nm] = eq; names.append(nm)
        tag = "EQUITY" if any(x in etf for x in ("SPY","QQQ","XLK","UPRO","TQQQ","IWM","XLP")) and not any(
              x in etf for x in ("GLD","TLT","UUP","DBC","SLV","IEF","BND","KMLM","DBMF")) else "MIXED/NON-EQ"
        print(f"{nm:32s} dec={dec:>2} stocks={len(st)} etfs={etf} [{tag}]")

    # align on common dates
    common = None
    for nm in names:
        ks = set(curves[nm].keys())
        common = ks if common is None else (common & ks)
    common = sorted(common, key=lambda k: float(k) if k.replace(".","").isdigit() else k)
    print(f"\naligned trading days: {len(common)}")
    rets = {}
    for nm in names:
        v = np.array([curves[nm][d] for d in common], dtype=float)
        rets[nm] = np.diff(v) / v[:-1] if len(v) > 2 else np.array([])

    def corr(a, b):
        if len(a) != len(b) or len(a) < 20: return float("nan")
        if a.std() == 0 or b.std() == 0: return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    print("\n=== correlation matrix (daily returns) ===")
    hdr = "".join(f"{n[:9]:>10}" for n in names)
    print(f"{'':22}{hdr}")
    for i in names:
        row = "".join(f"{corr(rets[i], rets[j]):>10.2f}" for j in names)
        print(f"{i:22}{row}")

    keptn = [n for n, _ in KEPT]
    print("\n=== candidate avg |corr| to the 3 KEPT equity sleeves ===")
    for nm, _ in EQUITY_CAND + NONEQ_CAND:
        cs = [corr(rets[nm], rets[k]) for k in keptn]
        cs = [c for c in cs if c == c]
        avg = sum(cs)/len(cs) if cs else float("nan")
        mx = max(cs) if cs else float("nan")
        grp = "equity" if (nm, _) in [(n,i) for n,i in EQUITY_CAND] else "NON-EQ"
        print(f"  [{grp}] {nm:30s} avg={avg:>5.2f}  max={mx:>5.2f}  "
              f"(pop={corr(rets[nm],rets['Pop Bot']):.2f} reg={corr(rets[nm],rets['Regime+Dip']):.2f} og={corr(rets[nm],rets['OG Gain Train']):.2f})")


if __name__ == "__main__":
    main()
