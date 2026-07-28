#!/usr/bin/env python3
"""
composer_confidence_screen.py — surface 10 high-confidence Discover strategies
that look attractive across ALL the risk-adjusted ratios (Sharpe, Calmar,
Sortino, Ulcer, Martin), while excluding strategies that just pick from a small
list of individual stocks (survivorship/selection-bias traps).

Method:
  1. Pool = ledger rows with a long OOS window (>= MIN_OOS_DAYS) and all five
     ratios present + positive (uniformly attractive, not lopsided).
  2. Composite = mean percentile-rank within the pool across the 5 ratios
     (Ulcer inverted: lower is better). Rewards strategies strong on ALL of
     them, not one flashy metric.
  3. Fetch holdings for the top shortlist; DROP any strategy whose universe is
     predominantly individual stocks (has_marketcap) or a static single-stock
     basket. Keep rule-driven, ETF-based strategies.
  4. Print the top 10 survivors with every ratio + confidence tier + holdings.
Read-only.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import requests

MIN_OOS_DAYS = int(os.environ.get("MIN_OOS_DAYS", "504"))   # >= ~2y OOS ("Good"+)
SHORTLIST    = int(os.environ.get("SHORTLIST", "40"))
ENDPOINTS = ["https://backtest-api.composer.trade/api/v1/public/symphonies/{id}/score",
             "https://backtest-api.composer.trade/api/v1/public/symphonies/{id}"]


def load(fn):
    out=[]
    try:
        for l in open(fn):
            l=l.strip()
            if l:
                try: out.append(json.loads(l))
                except: pass
    except FileNotFoundError: pass
    return out


def conf_tier(d):
    if d>=756: return "High"   # >=3y
    if d>=504: return "Good"   # >=2y
    if d>=252: return "Fair"   # >=1y
    return "Low"


def tree(o):
    if isinstance(o, dict):
        if ("step" in o) and ("children" in o): return o
        for k in ("symphony","score","definition","s"):
            if k in o and (t:=tree(o[k])): return t
        for v in o.values():
            if (t:=tree(v)): return t
    return None


def fetch(sid):
    for tmpl in ENDPOINTS:
        try: r=requests.get(tmpl.format(id=sid),headers={"accept":"application/json"},timeout=25)
        except Exception: continue
        if r.status_code==200 and "json" in r.headers.get("content-type",""):
            if (t:=tree(r.json())): return t
    return None


def holdings(t):
    assets, dec, sel = {}, 0, 0
    def w(n):
        nonlocal dec, sel
        if not isinstance(n, dict): return
        if n.get("step") in ("if","filter"): dec+=1
        if n.get("step")=="filter" and n.get("select?"): sel+=1
        if n.get("step")=="asset" and n.get("ticker"): assets[n["ticker"]]=bool(n.get("has_marketcap"))
        for c in n.get("children",[]): w(c)
    w(t or {})
    stocks=sorted(k for k,v in assets.items() if v)
    etfs=sorted(k for k,v in assets.items() if not v)
    return dict(dec=dec, sel=sel, stocks=stocks, etfs=etfs,
                static=(dec==0 and sel==0))


def pctl(vals):
    """percentile rank (0..1) of each element within vals; ties share mid-rank."""
    a=np.asarray(vals,float); order=a.argsort()
    ranks=np.empty(len(a)); ranks[order]=np.arange(len(a))
    return ranks/(len(a)-1) if len(a)>1 else np.zeros(len(a))


def main():
    rows=load("data/composer_oos_discover.jsonl")+load("data/composer_oos_watchlist.jsonl")
    if not rows:
        print("no ledger data", file=sys.stderr); sys.exit(1)
    latest=max(r.get("date","") for r in rows)
    rows=[r for r in rows if r.get("date")==latest]
    seen={}
    for r in rows: seen[r.get("sym_id")]=r
    rows=list(seen.values())

    RAT=["oos_sharpe","oos_calmar","oos_sortino","oos_martin","oos_ulcer"]
    def ok(r):
        if (r.get("oos_days") or 0) < MIN_OOS_DAYS: return False
        for k in RAT:
            if r.get(k) is None: return False
        # uniformly attractive: positive return-based ratios (ulcer is a depth, >0 ok)
        if r["oos_sharpe"]<=0 or r["oos_calmar"]<=0 or r["oos_sortino"]<=0 or r["oos_martin"]<=0: return False
        if (r.get("oos_maxdd_pct") or -99) < -45: return False   # not a disaster
        return True
    pool=[r for r in rows if ok(r)]
    print(f"pool: {len(pool)} high-confidence (>= {MIN_OOS_DAYS} OOS days) with all 5 ratios positive\n")
    if not pool:
        print("empty pool", file=sys.stderr); sys.exit(0)

    # composite percentile across the 5 ratios (ulcer inverted)
    P={
      "oos_sharpe": pctl([r["oos_sharpe"] for r in pool]),
      "oos_calmar": pctl([r["oos_calmar"] for r in pool]),
      "oos_sortino":pctl([r["oos_sortino"] for r in pool]),
      "oos_martin": pctl([r["oos_martin"] for r in pool]),
      "oos_ulcer":  pctl([-r["oos_ulcer"] for r in pool]),   # lower ulcer -> higher rank
    }
    for i,r in enumerate(pool):
        r["_composite"]=float(np.mean([P[k][i] for k in P]))
    pool.sort(key=lambda r:-r["_composite"])

    # fetch holdings for the shortlist; drop stock-pickers
    shortlist=pool[:SHORTLIST]
    picks=[]; dropped=0
    for r in shortlist:
        h=holdings(fetch(r["sym_id"])); time.sleep(0.25)
        r["_h"]=h
        stock_picker = (h["stocks"] and (len(h["stocks"])>=len(h["etfs"]) or (h["static"] and h["etfs"]==[])))
        if stock_picker:
            dropped+=1; continue
        picks.append(r)
        if len(picks)>=10: break
    print(f"dropped {dropped} individual-stock-basket strategies from the shortlist\n")

    print("="*118)
    print(f"{'#':>2} {'OOSd':>5} {'Conf':>4} {'Shrp':>5} {'Calm':>5} {'Sort':>5} {'Ulcr':>5} {'Mart':>5} "
          f"{'CAGR':>6} {'MaxDD':>6} {'Comp':>4}  Strategy")
    print("="*118)
    for i,r in enumerate(picks,1):
        print(f"{i:>2} {r['oos_days']:>5} {conf_tier(r['oos_days']):>4} "
              f"{r['oos_sharpe']:>5.2f} {r['oos_calmar']:>5.2f} {r['oos_sortino']:>5.2f} "
              f"{r['oos_ulcer']:>5.1f} {r['oos_martin']:>5.2f} "
              f"{r['oos_cagr_pct']:>5.1f}% {r['oos_maxdd_pct']:>5.1f}% {r['_composite']:>4.2f}  "
              f"{(r.get('name') or '')[:46]}")
        h=r["_h"]
        print(f"     id={r['sym_id']}  dec={h['dec']}  ETFs={h['etfs'][:9]}"
              + (f"  +{len(h['etfs'])-9}more" if len(h['etfs'])>9 else "")
              + (f"  stocks={h['stocks']}" if h['stocks'] else ""))
    print("="*118)
    print("Ratios: Sharpe/Calmar/Sortino/Martin higher=better; Ulcer lower=better (RMS drawdown depth).")
    print("Comp = mean percentile-rank across all 5 ratios within the high-confidence pool.")


if __name__=="__main__":
    main()
