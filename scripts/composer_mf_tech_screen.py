#!/usr/bin/env python3
"""
composer_mf_tech_screen.py — find strategies that BLEND managed-futures / alt
ETFs with LEVERAGED-TECH dip buys, survived 2022, and ran well in the bull.

Pool = (active-OOS-2022) UNION (managed-futures/diversifier-named strategies whose
backtest spans 2022), rebuilt from the OOS ledgers. For each we fetch holdings and
keep only those holding >=1 managed-futures ETF AND >=1 leveraged-tech ETF. Those
matches are backtested over the 2022 bear and the 2023-06+ bull, and printed with a
genuine-OOS-2022 flag (split <= 2022-03) vs backtest-only.
Read-only.
"""
from __future__ import annotations
import os, sys, json, re, time
from datetime import date
import numpy as np
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos as oos

MF   = {"DBMF","KMLM","CTA","RSST","RSBT","HFMF","FMF","WTMF","MFUT","CTAG","RSSB","RSSY"}
LTECH= {"TQQQ","TECL","SOXL","QLD","FNGU","ROM","USD","BULZ","WEBL","TQQ","LABU","FNGO"}
END = date.today().isoformat()
BULL_START = "2023-06-14"
ENDPOINTS = ["https://backtest-api.composer.trade/api/v1/public/symphonies/{id}/score",
             "https://backtest-api.composer.trade/api/v1/public/symphonies/{id}"]


def load(fn):
    out=[]
    for l in open(fn):
        l=l.strip()
        if l:
            try: out.append(json.loads(l))
            except: pass
    return out


def build_pool():
    rows=load("data/composer_oos_discover.jsonl")+load("data/composer_oos_watchlist.jsonl")
    latest=max(r.get("date","") for r in rows)
    rows=[r for r in rows if r.get("date")==latest]
    seen={}
    for r in rows: seen[r["sym_id"]]=r
    rows=list(seen.values())
    def active(r): return (r.get("oos_date") or "9")<="2022-03-01" and (r.get("bt_end") or "")>="2022-12-31"
    def spans(r):  return (r.get("bt_start") or "9")<="2021-06-01" and (r.get("bt_end") or "")>="2022-12-31"
    mf=re.compile(r"managed|futures|\bdbmf\b|\bkmlm\b|\bcta\b|\brsst\b|\brsbt\b|return.?stack|trend|carry|diversif|all.?weather|risk.?parity|macro|alt", re.I)
    pool={}
    for r in rows:
        if active(r) or (spans(r) and mf.search(r.get("name") or "")): pool[r["sym_id"]]=r
    return pool


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
        try:
            r=requests.get(tmpl.format(id=sid), headers={"accept":"application/json"}, timeout=25)
        except Exception:
            continue
        if r.status_code==200 and "json" in r.headers.get("content-type",""):
            if (t:=tree(r.json())): return t
    return None


def holdings(t):
    assets, dec = {}, 0
    def w(n):
        nonlocal dec
        if not isinstance(n, dict): return
        if n.get("step") in ("if","filter"): dec+=1
        if n.get("step")=="asset" and n.get("ticker"): assets[n["ticker"]]=bool(n.get("has_marketcap"))
        for c in n.get("children",[]): w(c)
    w(t or {})
    tks=set(assets)
    return dict(dec=dec, etfs=sorted(k for k,v in assets.items() if not v),
                stocks=sorted(k for k,v in assets.items() if v),
                mf=sorted(tks&MF), ltech=sorted(tks&LTECH))


def equity(sid,s,e):
    body={"capital":100000,"apply_reg_fee":True,"apply_taf_fee":True,"apply_cat_fee":True,
          "apply_subscription":"none","backtest_version":"v2","slippage_percent":0.0005,
          "start_date":s,"end_date":e,"broker":"ALPACA_WHITE_LABEL","benchmark_tickers":["SPY"]}
    for k in range(5):
        try:
            d=oos._request("POST", f"/api/v0.1/symphonies/{sid}/backtest", json=body); break
        except RuntimeError as ex:
            if "rate limited" in str(ex).lower() and k<4: time.sleep(6*(k+1)); continue
            raise
    cap=d.get("dvm_capital") or {}
    if cap and isinstance(next(iter(cap.values())),dict): cap=next(iter(cap.values()))
    return {str(k):float(v) for k,v in cap.items() if v is not None}


def _dk(k):
    s=k.replace(".","").replace("-",""); return float(s) if s.isdigit() else k


def stats(cap):
    days=sorted(cap.keys(),key=_dk); v=np.array([cap[d] for d in days],float)
    if len(v)<5: return None
    r=np.diff(v)/v[:-1]; n=len(r); ann=252
    eq=np.cumprod(1+r); cagr=eq[-1]**(ann/n)-1
    vol=r.std(ddof=1)*np.sqrt(ann); sh=(r.mean()*ann)/vol if vol else float('nan')
    peak=np.maximum.accumulate(eq); mdd=(eq/peak-1).min()
    return dict(n=n,cagr=cagr,sharpe=sh,maxdd=mdd,cum=eq[-1]-1)


def main():
    pool=build_pool()
    print(f"pool: {len(pool)} candidates\n")
    matches=[]
    for sid,rec in pool.items():
        h=holdings(fetch(sid)); time.sleep(0.25)
        if h["mf"] and h["ltech"]:
            matches.append((sid,rec,h))
    print(f"MF + leveraged-tech blends: {len(matches)}\n")
    # cap backtests: prefer genuine-OOS-2022, then ledger OOS sharpe
    def pre(m):
        sid,rec,h=m
        oos22 = (rec.get("oos_date") or "9")<="2022-03-01"
        return (1 if oos22 else 0, rec.get("oos_sharpe") or 0)
    matches.sort(key=pre, reverse=True)
    cap=matches[:18]
    results=[]
    for sid,rec,h in cap:
        try:
            s22=stats(equity(sid,"2022-01-01","2022-12-31")); time.sleep(0.3)
            sb =stats(equity(sid,BULL_START,END)); time.sleep(0.3)
        except Exception as e:
            print(f"  bt fail {sid}: {type(e).__name__}", file=sys.stderr); continue
        if not (s22 and sb): continue
        results.append((sid,rec,h,s22,sb))
    # rank: survived 2022 (dd>-30, cagr>=-5) then bull cagr
    def survived(s22): return s22["maxdd"]>-0.30 and s22["cagr"]>=-0.08
    results.sort(key=lambda x:(survived(x[3]), x[4]["cagr"]), reverse=True)
    print("=== MF + LEV-TECH blends: 2022 survival vs 2023-06+ bull ===")
    print("    (OOS22 = genuinely out-of-sample in 2022; bt22 = backtest-only)\n")
    for sid,rec,h,s22,sb in results:
        flag="OOS22" if (rec.get("oos_date") or "9")<="2022-03-01" else "bt22 "
        surv="SURVIVED" if survived(s22) else "FRAGILE "
        print(f"# {rec.get('name','')[:50]}  [{sid}]  {flag} {surv}")
        print(f"    dec={h['dec']}  MF={h['mf']}  levtech={h['ltech']}  "
              f"otherETF={[x for x in h['etfs'] if x not in h['mf'] and x not in h['ltech']][:8]}")
        print(f"    2022 bear : CAGR {s22['cagr']*100:7.2f}%  MaxDD {s22['maxdd']*100:7.2f}%  Sharpe {s22['sharpe']:.2f}")
        print(f"    2023-06+  : CAGR {sb['cagr']*100:7.2f}%  MaxDD {sb['maxdd']*100:7.2f}%  Sharpe {sb['sharpe']:.2f}  Cum {sb['cum']*100:.0f}%")
        print()


if __name__ == "__main__":
    main()
