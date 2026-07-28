#!/usr/bin/env python3
"""
composer_2022_compare.py — what would config A (current 5) vs config C (5 + the
JEPI-DBMF-TQQQ managed-futures/tech blend) have delivered in the 2022 bear?

Backtests each sleeve over 2022-01-01..2022-12-31, reports per-sleeve 2022 stats
and DAY COUNT (to confirm each sleeve actually has full-2022 history), then blends
A (20% x5) and C (16.7% x6) with 5% corridor rebalancing over the aligned 2022 days.
Read-only.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos as oos

SLEEVES=[("Pop Bot","2cuimtTihBBpJgf7FSis"),
         ("Regime+Dip","Dt37l1ceAggm8ggzBpRS"),
         ("OG2xAAA","DLXJ2T0lIgBMGykzAf1U"),
         ("OG Gain Train","l4glDbmbbDFd3p1Mcjkx"),
         ("Gold&Dollar","0dKcj7cKmeHhafQKlrHM")]
JEFE=("JEPI-DBMF-TQQQ","KpCZaTavfhU2OoaSc3pG")
S,E="2022-01-01","2022-12-31"

def equity(sid,s,e):
    body={"capital":100000,"apply_reg_fee":True,"apply_taf_fee":True,"apply_cat_fee":True,
          "apply_subscription":"none","backtest_version":"v2","slippage_percent":0.0005,
          "start_date":s,"end_date":e,"broker":"ALPACA_WHITE_LABEL","benchmark_tickers":["SPY"]}
    for k in range(5):
        try: d=oos._request("POST",f"/api/v0.1/symphonies/{sid}/backtest",json=body); break
        except RuntimeError as ex:
            if "rate limited" in str(ex).lower() and k<4: time.sleep(6*(k+1)); continue
            raise
    cap=d.get("dvm_capital") or {}
    if cap and isinstance(next(iter(cap.values())),dict): cap=next(iter(cap.values()))
    return {str(k):float(v) for k,v in cap.items() if v is not None}

def _dk(k):
    s=k.replace(".","").replace("-",""); return float(s) if s.isdigit() else k

def stats(r):
    n=len(r); ann=252; eq=np.cumprod(1+r); cagr=eq[-1]**(ann/n)-1
    vol=r.std(ddof=1)*np.sqrt(ann); sh=(r.mean()*ann)/vol if vol else float('nan')
    dn=r[r<0].std(ddof=1)*np.sqrt(ann) if (r<0).any() else float('nan')
    sortino=(r.mean()*ann)/dn if dn else float('nan')
    peak=np.maximum.accumulate(eq); mdd=(eq/peak-1).min(); cal=cagr/abs(mdd) if mdd else float('nan')
    return dict(n=n,ret=eq[-1]-1,cagr=cagr,sharpe=sh,sortino=sortino,maxdd=mdd,calmar=cal)

def sim(R,w,cor):
    w0=np.array(w,float); cur=w0.copy(); out=[]
    for t in range(R.shape[0]):
        rt=R[t]; out.append(float(np.dot(cur,rt)))
        cur=cur*(1+rt); cur=cur/cur.sum()
        if cor<0 or np.max(np.abs(cur-w0))>cor: cur=w0.copy()
    return np.array(out)

def show(t,s):
    print(f"  {t:26s} 2022 return {s['ret']*100:7.2f}%   MaxDD {s['maxdd']*100:7.2f}%   "
          f"Sharpe {s['sharpe']:5.2f}   Sortino {s['sortino']:5.2f}   (days={s['n']+1})")

def main():
    allsl=SLEEVES+[JEFE]; cur={}
    print(f"backtest window {S} -> {E}\n=== per-sleeve 2022 ===")
    for nm,sid in allsl:
        cur[sid]=equity(sid,S,E); time.sleep(0.3)
        d=sorted(cur[sid],key=_dk); v=np.array([cur[sid][x] for x in d],float)
        show(nm, stats(np.diff(v)/v[:-1]))
    # align
    cm=None
    for _,sid in allsl:
        ks=set(cur[sid]); cm=ks if cm is None else cm&ks
    cm=sorted(cm,key=_dk)
    print(f"\naligned 2022 trading days: {len(cm)}")
    def R(idl):
        return np.column_stack([np.diff(np.array([cur[s][d] for d in cm],float))/np.array([cur[s][d] for d in cm],float)[:-1] for s in idl])
    A=[sid for _,sid in SLEEVES]; C=A+[JEFE[1]]
    print("\n=== 2022 COMBINED (5% corridor rebalance) ===")
    show("A) current 5 (20% ea)", stats(sim(R(A),[0.2]*5,0.05)))
    show("C) add MF/tech 6th (16.7%)", stats(sim(R(C),[1/6]*6,0.05)))
    print("\n=== 2022 COMBINED (daily rebalance, ref) ===")
    show("A) current 5", stats(sim(R(A),[0.2]*5,-1)))
    show("C) add MF/tech 6th", stats(sim(R(C),[1/6]*6,-1)))

if __name__=="__main__":
    main()
