#!/usr/bin/env python3
"""
composer_screen_corr_bootstrap.py — for the shortlisted candidates:
  1) daily-return CORRELATION matrix (flags near-duplicate pairs > 0.80), and
  2) structure-preserving BOOTSTRAP battery (block L=5/10/20, stationary mean
     10/20, regime-conditional) of each strategy's edge over SPY.

SPY from data/ohlcv/SPY.csv. Structure-preserving only (never single-day shuffle).
Read-only.
"""
from __future__ import annotations
import os, sys, csv, time
from datetime import date, datetime, timezone, timedelta
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos as oos

CANDS = [
    ("BullHedge",   "63C7ke1j2LCKrvdM1zIc"),
    ("2ndOpus",     "yUJOjT74qF9n4C10CU7Y"),
    ("HelloAlgo",   "UhxPRjDNHmEh9cQ2sUWR"),
    ("ScaledDip",   "n93mN7XnuHBISuvFKQuL"),
    ("Structures",  "GKVAk2VoOOBoqn3Ns2up"),
    ("Cocktail",    "yhhUROwdOFifzCH5f4SS"),
    ("NeoPops",     "6tZrN6aWaqy5zrfZkUs7"),
    ("SimplePortf", "MzhTJqzPLSeiGxy4ODYi"),
    ("PoppedGrind", "Y6ahkuEVTVioJYKfKArC"),
]
START=os.environ.get("STRESS_START","2016-01-01"); END=date.today().isoformat()
N_TRIALS=int(os.environ.get("STRESS_TRIALS","4000")); SEED=42; ANN=252


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
        for row in csv.DictReader(f):
            try: px[row["Date"][:10]]=float(row["Adj Close"])
            except (KeyError,ValueError,TypeError): pass
    return px


def metrics(R):
    cum=np.cumprod(1+R,axis=1); cagr=cum[:,-1]**(ANN/R.shape[1])-1
    vol=R.std(axis=1,ddof=1); sh=np.where(vol>0,R.mean(axis=1)*ANN/(vol*np.sqrt(ANN)),np.nan)
    peak=np.maximum.accumulate(cum,axis=1); mdd=(cum/peak-1).min(axis=1)
    return cagr,sh,mdd


def block_idx(rng,N,L,tr):
    nb=int(np.ceil(N/L)); st=rng.integers(0,N,size=(tr,nb))
    return ((st[:,:,None]+np.arange(L)[None,None,:])%N).reshape(tr,nb*L)[:,:N]


def stat_idx(rng,N,L,tr):
    p=1.0/L; idx=np.empty((tr,N),np.int64); cur=rng.integers(0,N,size=tr); idx[:,0]=cur
    for t in range(1,N):
        jump=rng.random(tr)<p
        cur=np.where(jump,rng.integers(0,N,size=tr),(cur+1)%N); idx[:,t]=cur
    return idx


def regime_idx(rng,labels,tr):
    N=len(labels); pools={r:np.where(labels==r)[0] for r in np.unique(labels)}
    segs=[]; s=0
    for i in range(1,N+1):
        if i==N or labels[i]!=labels[s]: segs.append((labels[s],i-s)); s=i
    cols=[pools[lab][rng.integers(0,len(pools[lab]),size=(tr,m))] for lab,m in segs]
    return np.concatenate(cols,axis=1)


def hist(r):
    n=len(r); eq=np.cumprod(1+r); cagr=eq[-1]**(ANN/n)-1
    vol=r.std(ddof=1)*np.sqrt(ANN); sh=(r.mean()*ANN)/vol if vol else float('nan')
    peak=np.maximum.accumulate(eq); mdd=(eq/peak-1).min()
    return cagr,sh,mdd


def battery(strat,bench,labels,rng):
    N=len(strat); methods=[]
    idxs=[("block5",block_idx(rng,N,5,N_TRIALS)),("block10",block_idx(rng,N,10,N_TRIALS)),
          ("block20",block_idx(rng,N,20,N_TRIALS)),("stat10",stat_idx(rng,N,10,N_TRIALS)),
          ("stat20",stat_idx(rng,N,20,N_TRIALS)),("regime",regime_idx(rng,labels,N_TRIALS))]
    out={}
    for nm,idx in idxs:
        cs,shs,mds=metrics(strat[idx]); cb,shb,mdb=metrics(bench[idx])
        ec=cs-cb; es=shs-shb; em=mds-mdb
        win=((cs>cb)&(shs>shb)&(mds>mdb)).mean()*100
        out[nm]=dict(ec5=np.nanpercentile(ec,5), es5=np.nanpercentile(es,5),
                     em5=np.nanpercentile(em,5), win=win, pcag=(ec>0).mean()*100)
    return out


def main():
    print(f"window {START} -> {END}  (trials={N_TRIALS})\n")
    curves={}
    for nm,sid in CANDS:
        try:
            curves[nm]={to_iso(k):v for k,v in equity(sid,START,END).items()}
        except Exception as e:
            print(f"{nm}: bt fail {type(e).__name__}", file=sys.stderr); curves[nm]={}
        time.sleep(0.3)
    spy=load_spy()
    names=[nm for nm,_ in CANDS if len(curves.get(nm,{}))>20]
    common=None
    for nm in names:
        ks=set(curves[nm]); common=ks if common is None else common&ks
    common=sorted(common & set(spy))
    print(f"strategies with data: {len(names)}   aligned days: {len(common)}  ({common[0]} -> {common[-1]})\n")

    def rv(d):
        v=np.array([d[x] for x in common],float); return np.diff(v)/v[:-1]
    R={nm:rv(curves[nm]) for nm in names}
    spx=np.array([spy[x] for x in common],float); bench=np.diff(spx)/spx[:-1]; spx=spx[1:]

    # ---- 1) CORRELATION ----
    print("="*100)
    print("=== 1) DAILY-RETURN CORRELATION AMONG CANDIDATES ===")
    M=np.vstack([R[nm] for nm in names]); C=np.corrcoef(M)
    hdr="".join(f"{n[:9]:>10}" for n in names)
    print(f"{'':12}{hdr}")
    for i,nm in enumerate(names):
        print(f"{nm:12}"+"".join(f"{C[i,j]:>10.2f}" for j in range(len(names))))
    iu=np.triu_indices(len(names),1)
    pairs=sorted(((C[i,j],names[i],names[j]) for i,j in zip(*iu)), reverse=True)
    print(f"\n  avg pairwise corr: {C[iu].mean():.2f}")
    print("  most-similar pairs (>0.80 = near-duplicate behavior):")
    for c,a,b in pairs:
        if c>0.80: print(f"     {c:.2f}  {a} ~ {b}")
    print("  least-correlated (diversifiers):")
    for c,a,b in pairs[-4:]: print(f"     {c:.2f}  {a} ~ {b}")

    # ---- regime labels for regime-conditional bootstrap ----
    peak=np.maximum.accumulate(spx); ddp=spx/peak-1
    def sma(a,w):
        out=np.full_like(a,np.nan); c=np.cumsum(np.insert(a,0,0)); out[w-1:]=(c[w:]-c[:-w])/w; return out
    s21=sma(spx,21); s210=sma(spx,210); lab=np.empty(len(spx),object)
    for i in range(len(spx)):
        lab[i]="crisis" if ddp[i]<-0.10 else ("bull" if (not np.isnan(s210[i]) and spx[i]>s210[i] and s21[i]>s210[i]) else "chop")
    lab=lab.astype(str)

    # ---- 2) BOOTSTRAP each vs SPY ----
    print("\n"+"="*100)
    print("=== 2) BOOTSTRAP EDGE OVER SPY (structure-preserving; 5%ile lower bounds) ===")
    cb,shb,mdb=hist(bench)
    print(f"    SPY actual over window: CAGR {cb*100:.1f}%  Sharpe {shb:.2f}  MaxDD {mdb*100:.1f}%\n")
    rng=np.random.default_rng(SEED)
    for nm in names:
        cs,shs,mds=hist(R[nm]); b=battery(R[nm],bench,lab,rng)
        min_ec5=min(b[m]["ec5"] for m in b); min_win=min(b[m]["win"] for m in b)
        reg=b["regime"]; block_ok=all(b[m]["ec5"]>0 for m in ("block5","block10","block20","stat10","stat20"))
        if min_ec5>0 and min_win>80: v="ROBUST"
        elif block_ok and reg["ec5"]<=0: v="WITHIN-REGIME (regime-fragile)"
        elif reg["ec5"]<=0: v="REGIME-DEPENDENT"
        else: v="MIXED"
        print(f"# {nm:12} actual CAGR {cs*100:5.1f}% Sh {shs:4.2f} MaxDD {mds*100:6.1f}%")
        print(f"    edge-CAGR 5%ile per method: " + " ".join(f"{m}={b[m]['ec5']*100:+.1f}" for m in
              ("block5","block10","block20","stat10","stat20","regime")))
        print(f"    edge-Sharpe 5%ile (regime) {reg['es5']:+.2f} · min win-rate {min_win:.0f}% · "
              f"regime edge-CAGR 5%ile {reg['ec5']*100:+.1f}% -> {v}\n")


if __name__=="__main__":
    main()
