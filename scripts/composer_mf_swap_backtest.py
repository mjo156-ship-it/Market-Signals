#!/usr/bin/env python3
"""
composer_mf_swap_backtest.py — test adding/swapping the one credible managed-
futures + leveraged-tech blend (JEPI-DBMF-TQQQ) into the revised 5-sleeve book.

Configs over the common window (5% corridor rebalance):
  A) current 5-sleeve (20% each)
  B) SWAP Gold&Dollar -> JEPI-DBMF-TQQQ (20% each)
  C) ADD JEPI-DBMF-TQQQ as 6th (16.7% each)

Writes importable JSONs for B and C. Read-only against Composer.
"""
from __future__ import annotations
import os, sys, json, uuid, time
from datetime import date
import numpy as np
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos as oos

POP="2cuimtTihBBpJgf7FSis"; REG="Dt37l1ceAggm8ggzBpRS"; AAA="DLXJ2T0lIgBMGykzAf1U"
OGG="l4glDbmbbDFd3p1Mcjkx"; GLD="0dKcj7cKmeHhafQKlrHM"; JEFE="KpCZaTavfhU2OoaSc3pG"
NAMES={POP:"Pop Bot (SPY vs BND) l BrianE l May 30th 2007",
       REG:"Simple Regime Switching and Dip Buying",
       AAA:"OG Adaptive Asset Allocation 2x (SPY/TLT/UUP/GLD)",
       OGG:"OG V 1bb | Gain Train DGAF | Deez",
       GLD:"Diversify with Gold & the Dollar",
       JEFE:"v2 JEPI-DBMF-TQQQ | Simple Monthly | Jefe"}
COMMON_START=os.environ.get("PORTFOLIO_COMMON_START","2023-06-14"); END=date.today().isoformat()
ENDPOINTS=["https://backtest-api.composer.trade/api/v1/public/symphonies/{id}/score",
           "https://backtest-api.composer.trade/api/v1/public/symphonies/{id}"]

def tree(o):
    if isinstance(o,dict):
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

def ensure_ids(n):
    if isinstance(n,dict):
        if n.get("step") and not n.get("id"): n["id"]=str(uuid.uuid4())
        for c in n.get("children",[]): ensure_ids(c)
    return n

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

def sim(R,w,cor):
    w0=np.array(w,float); cur=w0.copy(); out=[]
    for t in range(R.shape[0]):
        rt=R[t]; out.append(float(np.dot(cur,rt)))
        cur=cur*(1+rt); cur=cur/cur.sum()
        if cor<0 or np.max(np.abs(cur-w0))>cor: cur=w0.copy()
    return np.array(out)

def stats(r):
    n=len(r); ann=252; eq=np.cumprod(1+r); cagr=eq[-1]**(ann/n)-1
    vol=r.std(ddof=1)*np.sqrt(ann); sh=(r.mean()*ann)/vol if vol else float('nan')
    dn=r[r<0].std(ddof=1)*np.sqrt(ann) if (r<0).any() else float('nan')
    sortino=(r.mean()*ann)/dn if dn else float('nan')
    peak=np.maximum.accumulate(eq); mdd=(eq/peak-1).min(); cal=cagr/abs(mdd) if mdd else float('nan')
    return dict(n=n,cagr=cagr,sharpe=sh,sortino=sortino,maxdd=mdd,calmar=cal,cum=eq[-1]-1)

def show(t,s): print(f"  {t:32s} CAGR {s['cagr']*100:6.2f}%  Sharpe {s['sharpe']:4.2f}  Sortino {s['sortino']:4.2f}  "
                     f"MaxDD {s['maxdd']*100:7.2f}%  Calmar {s['calmar']:4.2f}  Cum {s['cum']*100:6.1f}%")

def build(defs,weights,fname,title):
    groups=[]
    for sid,w in weights:
        sub=ensure_ids(defs[sid])
        groups.append({"id":str(uuid.uuid4()),"step":"group","name":f"{NAMES[sid]} ({w['num']}/{w['den']})",
                       "collapsed?":True,"weight":w,"children":sub.get("children",[])})
    root={"id":str(uuid.uuid4()),"step":"root","name":title,"collapsed?":True,
          "children":[{"id":str(uuid.uuid4()),"step":"wt-cash-specified","name":"Weight",
                       "suppress_incomplete_warnings":True,"children":groups}],
          "description":"Adds managed-futures (DBMF) + leveraged-tech (TQQQ/SOXL) blend for crisis-alpha diversification.",
          "rebalance":"none","rebalance-corridor-width":0.05}
    os.makedirs("data",exist_ok=True); json.dump(root,open(f"data/{fname}","w"),indent=2,ensure_ascii=False)
    n=[0]
    def c(x):
        if isinstance(x,dict) and x.get("step"):
            n[0]+=1
            for k in x.get("children",[]): c(k)
    c(root); print(f"  wrote data/{fname} ({n[0]} nodes)")

def main():
    ids=[POP,REG,AAA,OGG,GLD,JEFE]; defs={}; cur={}
    for sid in ids:
        defs[sid]=fetch(sid); time.sleep(0.3)
        cur[sid]=equity(sid,COMMON_START,END); time.sleep(0.3)
    # JEFE regime standalone
    print("=== JEPI-DBMF-TQQQ standalone ===")
    def st_curve(cap):
        d=sorted(cap.keys(),key=_dk); v=np.array([cap[x] for x in d],float); return stats(np.diff(v)/v[:-1])
    show("2022 bear", st_curve(equity(JEFE,"2022-01-01","2022-12-31"))); time.sleep(0.3)
    show("2023-06+ bull", st_curve(cur[JEFE]))

    cm=None
    for sid in ids:
        ks=set(cur[sid]); cm=ks if cm is None else cm&ks
    cm=sorted(cm,key=_dk); print(f"\naligned common days: {len(cm)}")
    def R(idl):
        return np.column_stack([np.diff(np.array([cur[s][d] for d in cm],float))/np.array([cur[s][d] for d in cm],float)[:-1] for s in idl])
    A=[POP,REG,AAA,OGG,GLD]; B=[POP,REG,AAA,OGG,JEFE]; C=[POP,REG,AAA,OGG,GLD,JEFE]
    print("\n=== COMBINED (5% corridor) ===")
    show("A) current 5 (20% ea)", stats(sim(R(A),[0.2]*5,0.05)))
    show("B) swap Gold->Jefe (20% ea)", stats(sim(R(B),[0.2]*5,0.05)))
    show("C) add Jefe 6th (16.7% ea)", stats(sim(R(C),[1/6]*6,0.05)))

    print("\n=== importable symphonies ===")
    w20=[(s,{"num":"20","den":"100"}) for s in B]
    build(defs,w20,"swap_gold_to_mftech_symphony.json","Claude 5-sleeve (Gold->MF/Tech swap)")
    w6=[(s,{"num":"1","den":"6"}) for s in C]
    build(defs,w6,"add_mftech_6sleeve_symphony.json","Claude 6-sleeve (+MF/Tech)")

if __name__=="__main__":
    main()
