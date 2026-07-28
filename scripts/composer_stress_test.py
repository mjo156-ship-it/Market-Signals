#!/usr/bin/env python3
"""
composer_stress_test.py — structure-preserving bootstrap battery on the revised
5-sleeve book's daily returns vs SPY (per the strategy-stress-test skill).

Pipeline (all on the runner):
  1. backtest the 5 sleeves over max common history, blend 20% each @5% corridor
     -> strategy daily returns
  2. SPY daily returns from data/ohlcv/SPY.csv (Yahoo adj close), aligned
  3. three bootstraps, paired strat/bench, n=5000, seed=42:
       - block (L=5,10,20)
       - stationary / Politis-Romano (mean block=10,20)
       - regime-conditional (SPY drawdown + SMA21/210 taxonomy)
  4. per method: CAGR/Sharpe/MaxDD dists, paired edge (strat-SPY) 5/95 + %>0, win rate

Structure-preserving only. NEVER single-day shuffling.
"""
from __future__ import annotations
import os, sys, csv, time
from datetime import date, datetime, timezone, timedelta
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos as oos

SLEEVES=[("Pop Bot","2cuimtTihBBpJgf7FSis"),("Regime+Dip","Dt37l1ceAggm8ggzBpRS"),
         ("OG2xAAA","DLXJ2T0lIgBMGykzAf1U"),("OG Gain Train","l4glDbmbbDFd3p1Mcjkx"),
         ("Gold&Dollar","0dKcj7cKmeHhafQKlrHM")]
START=os.environ.get("STRESS_START","2016-01-01"); END=date.today().isoformat()
N_TRIALS=int(os.environ.get("STRESS_TRIALS","5000")); SEED=42; ANN=252

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
    return cap

def to_iso(k):
    s=str(k)
    if "-" in s and len(s)>=8: return s[:10]
    if s.replace(".","").isdigit():
        n=float(s)
        if n>1e12: return datetime.fromtimestamp(n/1000,tz=timezone.utc).date().isoformat()
        if n>1e9:  return datetime.fromtimestamp(n,tz=timezone.utc).date().isoformat()
        if n>10000: return (date(1970,1,1)+timedelta(days=int(n))).isoformat()
    return s

def load_spy():
    px={}
    with open("data/ohlcv/SPY.csv") as f:
        r=csv.DictReader(f)
        for row in r:
            try: px[row["Date"][:10]]=float(row["Adj Close"])
            except (KeyError,ValueError,TypeError): pass
    return px

def hist_stats(r):
    n=len(r); eq=np.cumprod(1+r); cagr=eq[-1]**(ANN/n)-1
    vol=r.std(ddof=1)*np.sqrt(ANN); sh=(r.mean()*ANN)/vol if vol else float('nan')
    peak=np.maximum.accumulate(eq); mdd=(eq/peak-1).min()
    return cagr,sh,mdd

# ---- vectorized metric on an index matrix (trials x N) ----
def metrics(R):
    cum=np.cumprod(1+R,axis=1); cagr=cum[:,-1]**(ANN/R.shape[1])-1
    vol=R.std(axis=1,ddof=1); sh=np.where(vol>0,R.mean(axis=1)*ANN/(vol*np.sqrt(ANN)),np.nan)
    peak=np.maximum.accumulate(cum,axis=1); mdd=(cum/peak-1).min(axis=1)
    return cagr,sh,mdd

def block_idx(rng,N,L,trials):
    nb=int(np.ceil(N/L)); starts=rng.integers(0,N,size=(trials,nb))
    idx=(starts[:,:,None]+np.arange(L)[None,None,:])%N
    return idx.reshape(trials,nb*L)[:,:N]

def stationary_idx(rng,N,L,trials):
    p=1.0/L; idx=np.empty((trials,N),dtype=np.int64)
    cur=rng.integers(0,N,size=trials); idx[:,0]=cur
    for t in range(1,N):
        jump=rng.random(trials)<p
        cur=np.where(jump,rng.integers(0,N,size=trials),(cur+1)%N); idx[:,t]=cur
    return idx

def regime_idx(rng,labels,trials):
    N=len(labels); pools={r:np.where(labels==r)[0] for r in np.unique(labels)}
    # contiguous segments preserve regime order + length
    segs=[]; s=0
    for i in range(1,N+1):
        if i==N or labels[i]!=labels[s]: segs.append((labels[s],i-s)); s=i
    cols=[]
    for lab,m in segs:
        pool=pools[lab]; cols.append(pool[rng.integers(0,len(pool),size=(trials,m))])
    return np.concatenate(cols,axis=1)

def summarize(name,rs,rb,idx):
    Rs=rs[idx]; Rb=rb[idx]
    cs,shs,mds=metrics(Rs); cb,shb,mdb=metrics(Rb)
    ec=cs-cb; es=shs-shb; em=mds-mdb
    win=((cs>cb)&(shs>shb)&(mds>mdb)).mean()*100
    def pctl(x): return (np.nanpercentile(x,5),np.nanpercentile(x,50),np.nanpercentile(x,95))
    ec5,ec50,ec95=pctl(ec); es5,es50,es95=pctl(es); em5,em50,em95=pctl(em)
    c5,c50,c95=pctl(cs)
    print(f"\n[{name}]  (n_trials={idx.shape[0]}, N={idx.shape[1]})")
    print(f"  strat CAGR   5/50/95: {c5*100:6.1f} /{c50*100:6.1f} /{c95*100:6.1f} %")
    print(f"  edge CAGR    5/50/95: {ec5*100:6.1f} /{ec50*100:6.1f} /{ec95*100:6.1f} %   P(edge>0)={(ec>0).mean()*100:4.0f}%")
    print(f"  edge Sharpe  5/50/95: {es5:6.2f} /{es50:6.2f} /{es95:6.2f}       P(edge>0)={(es>0).mean()*100:4.0f}%")
    print(f"  edge MaxDD   5/50/95: {em5*100:6.1f} /{em50*100:6.1f} /{em95*100:6.1f} %   P(edge>0)={(em>0).mean()*100:4.0f}%")
    print(f"  win rate (beats SPY on CAGR & Sharpe & MaxDD): {win:4.0f}%")
    return dict(name=name,edge_cagr_p5=ec5,win=win,edge_sharpe_p5=es5,edge_mdd_p5=em5)

def main():
    print(f"window request {START} -> {END}  (trials={N_TRIALS}, seed={SEED})\n")
    curves={}
    for nm,sid in SLEEVES:
        c=equity(sid,START,END); time.sleep(0.3)
        curves[sid]={to_iso(k):float(v) for k,v in c.items() if v is not None}
    spy=load_spy()
    common=None
    for sid,_ in [(s,n) for n,s in SLEEVES]:
        ks=set(curves[sid]); common=ks if common is None else common&ks
    common=sorted(common & set(spy))
    print(f"aligned days: {len(common)}  ({common[0]} -> {common[-1]})")

    # per-sleeve return vectors -> combined @5% corridor
    def rvec(d):
        v=np.array([d[x] for x in common],float); return np.diff(v)/v[:-1]
    Rmat=np.column_stack([rvec(curves[sid]) for _,sid in [(n,s) for n,s in SLEEVES]])
    # sim 5% corridor
    w0=np.array([0.2]*5); cur=w0.copy(); strat=[]
    for t in range(Rmat.shape[0]):
        rt=Rmat[t]; strat.append(float(np.dot(cur,rt)))
        cur=cur*(1+rt); cur=cur/cur.sum()
        if np.max(np.abs(cur-w0))>0.05: cur=w0.copy()
    strat=np.array(strat)
    spx=np.array([spy[x] for x in common],float); bench=np.diff(spx)/spx[:-1]
    dates=common[1:]; spx=spx[1:]   # align price levels to return dates

    # historical (actual) stats
    cs,shs,mds=hist_stats(strat); cb,shb,mdb=hist_stats(bench)
    print(f"\n=== ACTUAL (historical) over window, n={len(strat)} days ===")
    print(f"  5-sleeve : CAGR {cs*100:6.2f}%  Sharpe {shs:4.2f}  MaxDD {mds*100:6.2f}%")
    print(f"  SPY      : CAGR {cb*100:6.2f}%  Sharpe {shb:4.2f}  MaxDD {mdb*100:6.2f}%")

    # regime labels from SPY: crisis / bull / chop
    px=spx; peak=np.maximum.accumulate(px); ddp=px/peak-1
    def sma(a,w):
        out=np.full_like(a,np.nan)
        c=np.cumsum(np.insert(a,0,0)); out[w-1:]=(c[w:]-c[:-w])/w
        return out
    s21=sma(px,21); s210=sma(px,210)
    labels=np.empty(len(px),dtype=object)
    for i in range(len(px)):
        if ddp[i]<-0.10: labels[i]="crisis"
        elif not np.isnan(s210[i]) and px[i]>s210[i] and s21[i]>s210[i]: labels[i]="bull"
        else: labels[i]="chop"
    labels=labels.astype(str)
    uniq,cnts=np.unique(labels,return_counts=True)
    print("  regime mix:", dict(zip(uniq.tolist(),cnts.tolist())))

    rng=np.random.default_rng(SEED); N=len(strat); res=[]
    print("\n================ BOOTSTRAP BATTERY ================")
    for L in (5,10,20):
        res.append(summarize(f"block L={L}", strat,bench, block_idx(rng,N,L,N_TRIALS)))
    for L in (10,20):
        res.append(summarize(f"stationary mean={L}", strat,bench, stationary_idx(rng,N,L,N_TRIALS)))
    res.append(summarize("regime-conditional", strat,bench, regime_idx(rng,labels,N_TRIALS)))

    print("\n================ VERDICT ================")
    all_pos=all(r["edge_cagr_p5"]>0 for r in res)
    reg=[r for r in res if r["name"]=="regime-conditional"][0]
    block_pos=all(r["edge_cagr_p5"]>0 for r in res if r["name"].startswith(("block","stationary")))
    minwin=min(r["win"] for r in res)
    print(f"  edge CAGR 5%ile > 0 in ALL methods: {all_pos}")
    print(f"  edge CAGR 5%ile > 0 in block/stationary: {block_pos}   regime-cond 5%ile: {reg['edge_cagr_p5']*100:.1f}%")
    print(f"  min win-rate across methods: {minwin:.0f}%")
    if all_pos and minwin>80: verdict="ROBUST -> deploy at intended sizing"
    elif block_pos and reg["edge_cagr_p5"]<=0: verdict="WITHIN-REGIME edge, regime-fragile -> half sizing, monitor"
    elif reg["edge_cagr_p5"]<=0: verdict="REGIME-DEPENDENT -> not standalone; overlay only"
    else: verdict="MIXED -> size with judgment"
    print(f"  VERDICT: {verdict}")

if __name__=="__main__":
    main()
