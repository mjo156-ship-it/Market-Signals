#!/usr/bin/env python3
"""
composer_pair_portfolio.py — build & validate a small portfolio around the two
bootstrap-robust, low-correlation (0.27) strategies:
   BullHedge (63C7ke1j2LCKrvdM1zIc) + 2ndOpus (yUJOjT74qF9n4C10CU7Y)

  * backtest each + the blend (50/50, 60/40, 40/60) at 5% corridor
  * confirm the diversification benefit (combo Sharpe/DD vs each alone)
  * BOOTSTRAP the 50/50 combo's edge over SPY (block/stationary/regime)
  * write an importable Composer symphony for the 50/50 blend
Read-only vs Composer; writes one JSON.
"""
from __future__ import annotations
import os, sys, json, uuid, csv, time
from datetime import date, datetime, timezone, timedelta
import numpy as np
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos as oos

PAIR=[("BullHedge","SPY - Bull or Hedge","63C7ke1j2LCKrvdM1zIc"),
      ("2ndOpus","BE's 2nd Opus l HnL + Commodities","yUJOjT74qF9n4C10CU7Y")]
START=os.environ.get("STRESS_START","2016-01-01"); END=date.today().isoformat()
N_TRIALS=int(os.environ.get("STRESS_TRIALS","5000")); SEED=42; ANN=252
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

def sim(R,w,cor=0.05):
    w0=np.array(w,float); cur=w0.copy(); out=[]
    for t in range(R.shape[0]):
        rt=R[t]; out.append(float(np.dot(cur,rt)))
        cur=cur*(1+rt); cur=cur/cur.sum()
        if cor<0 or np.max(np.abs(cur-w0))>cor: cur=w0.copy()
    return np.array(out)

def stats(r):
    n=len(r); eq=np.cumprod(1+r); cagr=eq[-1]**(ANN/n)-1
    vol=r.std(ddof=1)*np.sqrt(ANN); sh=(r.mean()*ANN)/vol if vol else float('nan')
    dn=r[r<0].std(ddof=1)*np.sqrt(ANN) if (r<0).any() else float('nan')
    sortino=(r.mean()*ANN)/dn if dn else float('nan')
    peak=np.maximum.accumulate(eq); mdd=(eq/peak-1).min(); cal=cagr/abs(mdd) if mdd else float('nan')
    return dict(n=n,cagr=cagr,sharpe=sh,sortino=sortino,maxdd=mdd,calmar=cal,cum=eq[-1]-1)

def show(t,s): print(f"  {t:24s} CAGR {s['cagr']*100:6.2f}%  Sharpe {s['sharpe']:4.2f}  Sortino {s['sortino']:4.2f}  "
                     f"MaxDD {s['maxdd']*100:7.2f}%  Calmar {s['calmar']:4.2f}  Cum {s['cum']*100:6.0f}%")

# --- bootstrap (structure-preserving) ---
def metrics(R):
    cum=np.cumprod(1+R,axis=1); cagr=cum[:,-1]**(ANN/R.shape[1])-1
    vol=R.std(axis=1,ddof=1); sh=np.where(vol>0,R.mean(axis=1)*ANN/(vol*np.sqrt(ANN)),np.nan)
    peak=np.maximum.accumulate(cum,axis=1); mdd=(cum/peak-1).min(axis=1)
    return cagr,sh,mdd
def blk(rng,N,L,tr):
    nb=int(np.ceil(N/L)); st=rng.integers(0,N,size=(tr,nb))
    return ((st[:,:,None]+np.arange(L)[None,None,:])%N).reshape(tr,nb*L)[:,:N]
def stat(rng,N,L,tr):
    p=1/L; idx=np.empty((tr,N),np.int64); cur=rng.integers(0,N,size=tr); idx[:,0]=cur
    for t in range(1,N):
        j=rng.random(tr)<p; cur=np.where(j,rng.integers(0,N,size=tr),(cur+1)%N); idx[:,t]=cur
    return idx
def reg(rng,lab,tr):
    N=len(lab); pools={r:np.where(lab==r)[0] for r in np.unique(lab)}; segs=[]; s=0
    for i in range(1,N+1):
        if i==N or lab[i]!=lab[s]: segs.append((lab[s],i-s)); s=i
    return np.concatenate([pools[l][rng.integers(0,len(pools[l]),size=(tr,m))] for l,m in segs],axis=1)


def main():
    curves={}; defs={}
    for sh,nm,sid in PAIR:
        defs[sid]=fetch(sid); time.sleep(0.3)
        curves[sh]={to_iso(k):v for k,v in equity(sid,START,END).items()}; time.sleep(0.3)
    spy=load_spy()
    common=set(curves["BullHedge"]) & set(curves["2ndOpus"]) & set(spy)
    common=sorted(common)
    print(f"aligned window: {common[0]} -> {common[-1]}  ({len(common)} days)\n")
    def rv(d):
        v=np.array([d[x] for x in common],float); return np.diff(v)/v[:-1]
    rb=rv(curves["BullHedge"]); ro=rv(curves["2ndOpus"])
    spx=np.array([spy[x] for x in common],float); bench=np.diff(spx)/spx[:-1]; spx=spx[1:]
    R=np.column_stack([rb,ro])
    print(f"pair correlation over this window: {np.corrcoef(rb,ro)[0,1]:.2f}\n")

    print("=== standalone vs blends (5% corridor) ===")
    show("BullHedge only", stats(rb))
    show("2ndOpus only", stats(ro))
    show("50/50", stats(sim(R,[0.5,0.5])))
    show("60/40 (Bull/Opus)", stats(sim(R,[0.6,0.4])))
    show("40/60 (Bull/Opus)", stats(sim(R,[0.4,0.6])))
    show("SPY (benchmark)", stats(bench))

    # bootstrap the 50/50 combo vs SPY
    combo=sim(R,[0.5,0.5]); N=len(combo)
    peak=np.maximum.accumulate(spx); ddp=spx/peak-1
    def sma(a,w):
        out=np.full_like(a,np.nan); c=np.cumsum(np.insert(a,0,0)); out[w-1:]=(c[w:]-c[:-w])/w; return out
    s21=sma(spx,21); s210=sma(spx,210); lab=np.empty(len(spx),object)
    for i in range(len(spx)):
        lab[i]="crisis" if ddp[i]<-0.10 else ("bull" if (not np.isnan(s210[i]) and spx[i]>s210[i] and s21[i]>s210[i]) else "chop")
    lab=lab.astype(str); rng=np.random.default_rng(SEED)
    print("\n=== BOOTSTRAP: 50/50 combo edge over SPY (5%ile bounds) ===")
    allpos=True; minwin=100
    for nm,idx in [("block5",blk(rng,N,5,N_TRIALS)),("block10",blk(rng,N,10,N_TRIALS)),
                   ("block20",blk(rng,N,20,N_TRIALS)),("stat10",stat(rng,N,10,N_TRIALS)),
                   ("stat20",stat(rng,N,20,N_TRIALS)),("regime",reg(rng,lab,N_TRIALS))]:
        cs,shs,mds=metrics(combo[idx]); cb,shb,mdb=metrics(bench[idx])
        ec=cs-cb; es=shs-shb; em=mds-mdb; win=((cs>cb)&(shs>shb)&(mds>mdb)).mean()*100
        allpos = allpos and (np.nanpercentile(ec,5)>0); minwin=min(minwin,win)
        print(f"  {nm:8} edgeCAGR5%={np.nanpercentile(ec,5)*100:+5.1f}%  edgeSharpe5%={np.nanpercentile(es,5):+4.2f}  "
              f"edgeMaxDD5%={np.nanpercentile(em,5)*100:+5.1f}%  win={win:3.0f}%")
    print(f"  VERDICT: {'ROBUST -> deploy at intended sizing' if (allpos and minwin>80) else 'see per-method bounds'}")

    # importable 50/50 JSON
    groups=[]
    for sh,nm,sid in PAIR:
        sub=ensure_ids(defs[sid])
        groups.append({"id":str(uuid.uuid4()),"step":"group","name":f"{nm} (50%)","collapsed?":True,
                       "weight":{"num":"50","den":"100"},"children":sub.get("children",[])})
    root={"id":str(uuid.uuid4()),"step":"root","name":"Claude BullHedge + 2ndOpus (50/50)","collapsed?":True,
          "children":[{"id":str(uuid.uuid4()),"step":"wt-cash-specified","name":"Weight",
                       "suppress_incomplete_warnings":True,"children":groups}],
          "description":"50/50 blend of two bootstrap-robust, low-correlation (0.27) strategies: "
                        "SPY Bull-or-Hedge + BE's 2nd Opus (HnL + Commodities).",
          "rebalance":"none","rebalance-corridor-width":0.05}
    os.makedirs("data",exist_ok=True)
    json.dump(root,open("data/bullhedge_2ndopus_symphony.json","w"),indent=2,ensure_ascii=False)
    tot=[0]
    def c(x):
        if isinstance(x,dict) and x.get("step"):
            tot[0]+=1
            for k in x.get("children",[]): c(k)
    c(root); print(f"\nwrote data/bullhedge_2ndopus_symphony.json ({tot[0]} nodes)")


if __name__=="__main__":
    main()
