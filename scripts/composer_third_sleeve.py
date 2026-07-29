#!/usr/bin/env python3
"""
composer_third_sleeve.py — pick a diversifying THIRD sleeve for the
BullHedge + 2ndOpus pair. For each macro/multi-asset candidate: correlation to
the 50/50 pair, its own ratios, and the 3-way equal-weight blend's risk profile
(vs the pair alone, on the same window). Then build + bootstrap the best pick.

Winner = highest 3-way Calmar among genuine diversifiers (corr-to-pair < 0.6).
Read-only vs Composer; writes one JSON for the winner.
"""
from __future__ import annotations
import os, sys, json, uuid, csv, time
from datetime import date, datetime, timezone, timedelta
import numpy as np
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos as oos

BULL=("BullHedge","SPY - Bull or Hedge","63C7ke1j2LCKrvdM1zIc")
OPUS=("2ndOpus","BE's 2nd Opus l HnL + Commodities","yUJOjT74qF9n4C10CU7Y")
DIVS=[  # macro / multi-asset / managed-futures diversifiers (not equity-dip family)
    ("Gold&Dollar",     "Diversify with Gold & the Dollar",              "0dKcj7cKmeHhafQKlrHM"),
    ("OG2xAAA",         "OG Adaptive Asset Allocation 2x",               "DLXJ2T0lIgBMGykzAf1U"),
    ("StoicInflation",  "Stoic Finance Presents: Inflation Spiral Hedge","c1bWgoNRaiff8ZZwBGD0"),
    ("JEPI-DBMF-TQQQ",  "v2 JEPI-DBMF-TQQQ | Simple Monthly | Jefe",     "KpCZaTavfhU2OoaSc3pG"),
    ("DiversifyBest",   "Diversification at its Best",                   "kULEdknQwUoefsGpQKsT"),
]
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

def sim(cols,w,cor=0.05):
    R=np.column_stack(cols); w0=np.array(w,float); cur=w0.copy(); out=[]
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
    return dict(cagr=cagr,sharpe=sh,sortino=sortino,maxdd=mdd,calmar=cal)

def metrics(R):
    cum=np.cumprod(1+R,axis=1); cagr=cum[:,-1]**(ANN/R.shape[1])-1
    vol=R.std(axis=1,ddof=1); sh=np.where(vol>0,R.mean(axis=1)*ANN/(vol*np.sqrt(ANN)),np.nan)
    peak=np.maximum.accumulate(cum,axis=1); mdd=(cum/peak-1).min(axis=1); return cagr,sh,mdd
def blk(rng,N,L,tr):
    nb=int(np.ceil(N/L)); st=rng.integers(0,N,size=(tr,nb))
    return ((st[:,:,None]+np.arange(L)[None,None,:])%N).reshape(tr,nb*L)[:,:N]
def stat(rng,N,L,tr):
    p=1/L; idx=np.empty((tr,N),np.int64); cur=rng.integers(0,N,size=tr); idx[:,0]=cur
    for t in range(1,N):
        j=rng.random(tr)<p; cur=np.where(j,rng.integers(0,N,size=tr),(cur+1)%N); idx[:,t]=cur
    return idx
def regi(rng,lab,tr):
    N=len(lab); pools={r:np.where(lab==r)[0] for r in np.unique(lab)}; segs=[]; s=0
    for i in range(1,N+1):
        if i==N or lab[i]!=lab[s]: segs.append((lab[s],i-s)); s=i
    return np.concatenate([pools[l][rng.integers(0,len(pools[l]),size=(tr,m))] for l,m in segs],axis=1)


def main():
    ids={BULL[2],OPUS[2]} | {d[2] for d in DIVS}
    cur={}; defs={}
    for sid in ids:
        defs[sid]=fetch(sid); time.sleep(0.25)
        try: cur[sid]={to_iso(k):v for k,v in equity(sid,START,END).items()}
        except Exception as e: print(f"{sid}: bt fail {type(e).__name__}",file=sys.stderr); cur[sid]={}
        time.sleep(0.25)
    spy=load_spy()

    def rvec(sid,common):
        v=np.array([cur[sid][x] for x in common],float); return np.diff(v)/v[:-1]

    print("=== third-sleeve candidates (3-way equal-weight blend vs the pair) ===")
    print(f"    {'cand':16} {'win':>4} {'corr→pair':>9} | {'cand CAGR/Sh/DD':>20} | "
          f"{'3way CAGR':>9} {'Sh':>4} {'Sort':>4} {'DD':>7} {'Calm':>4}")
    rows=[]
    for sh,nm,sid in DIVS:
        if len(cur.get(sid,{}))<50: print(f"    {sh:16} — no data"); continue
        common=sorted(set(cur[BULL[2]])&set(cur[OPUS[2]])&set(cur[sid])&set(spy))
        rb=rvec(BULL[2],common); ro=rvec(OPUS[2],common); rc=rvec(sid,common)
        pair=sim([rb,ro],[0.5,0.5])
        corr=float(np.corrcoef(rc,pair)[0,1])
        cs=stats(rc); three=stats(sim([rb,ro,rc],[1/3,1/3,1/3]))
        rows.append((sh,sid,len(common),corr,cs,three,common))
        print(f"    {sh:16} {len(common):>4} {corr:>9.2f} | "
              f"{cs['cagr']*100:5.1f}%/{cs['sharpe']:.2f}/{cs['maxdd']*100:5.1f}% | "
              f"{three['cagr']*100:8.1f}% {three['sharpe']:4.2f} {three['sortino']:4.2f} "
              f"{three['maxdd']*100:6.1f}% {three['calmar']:4.2f}")

    # reference: pair alone over the FULL pair window
    pc=sorted(set(cur[BULL[2]])&set(cur[OPUS[2]])&set(spy))
    pair_full=stats(sim([rvec(BULL[2],pc),rvec(OPUS[2],pc)],[0.5,0.5]))
    print(f"\n    (reference) 50/50 pair alone: CAGR {pair_full['cagr']*100:.1f}%  Sharpe {pair_full['sharpe']:.2f}  "
          f"Sortino {pair_full['sortino']:.2f}  MaxDD {pair_full['maxdd']*100:.1f}%  Calmar {pair_full['calmar']:.2f}")

    # winner = best 3-way Calmar among genuine diversifiers (corr<0.6)
    elig=[r for r in rows if r[3]<0.6] or rows
    win=max(elig, key=lambda r: r[5]["calmar"])
    wsh,wsid,wn,wcorr,wcs,wthree,common=win
    print(f"\n>>> WINNER: {wsh}  (corr {wcorr:.2f} to pair; best 3-way Calmar {wthree['calmar']:.2f})")

    # bootstrap the 3-way vs SPY on the winner's window
    rb=rvec(BULL[2],common); ro=rvec(OPUS[2],common); rc=rvec(wsid,common)
    three=sim([rb,ro,rc],[1/3,1/3,1/3]); N=len(three)
    spx=np.array([spy[x] for x in common],float); bench=np.diff(spx)/spx[:-1]; spx=spx[1:]
    peak=np.maximum.accumulate(spx); ddp=spx/peak-1
    def sma(a,w):
        out=np.full_like(a,np.nan); c=np.cumsum(np.insert(a,0,0)); out[w-1:]=(c[w:]-c[:-w])/w; return out
    s21=sma(spx,21); s210=sma(spx,210); lab=np.empty(len(spx),object)
    for i in range(len(spx)):
        lab[i]="crisis" if ddp[i]<-0.10 else ("bull" if (not np.isnan(s210[i]) and spx[i]>s210[i] and s21[i]>s210[i]) else "chop")
    lab=lab.astype(str); rng=np.random.default_rng(SEED)
    print(f"\n=== BOOTSTRAP: 3-way (Bull/Opus/{wsh} equal) edge over SPY, window {common[0]}->{common[-1]} ({N+1}d) ===")
    allpos=True; mw=100
    for nm,idx in [("block5",blk(rng,N,5,N_TRIALS)),("block10",blk(rng,N,10,N_TRIALS)),
                   ("block20",blk(rng,N,20,N_TRIALS)),("stat10",stat(rng,N,10,N_TRIALS)),
                   ("stat20",stat(rng,N,20,N_TRIALS)),("regime",regi(rng,lab,N_TRIALS))]:
        cs,shs,mds=metrics(three[idx]); cb,shb,mdb=metrics(bench[idx])
        ec=cs-cb; es=shs-shb; em=mds-mdb; wr=((cs>cb)&(shs>shb)&(mds>mdb)).mean()*100
        allpos=allpos and np.nanpercentile(ec,5)>0; mw=min(mw,wr)
        print(f"  {nm:8} edgeCAGR5%={np.nanpercentile(ec,5)*100:+5.1f}% edgeSharpe5%={np.nanpercentile(es,5):+4.2f} "
              f"edgeMaxDD5%={np.nanpercentile(em,5)*100:+5.1f}% win={wr:3.0f}%")
    print(f"  VERDICT: {'ROBUST' if (allpos and mw>80) else 'see bounds'}")

    # build importable 3-sleeve JSON (equal thirds)
    trio=[BULL,OPUS,(wsh,win[0],wsid)]
    groups=[]
    for sh,nm,sid in trio:
        sub=ensure_ids(defs[sid])
        groups.append({"id":str(uuid.uuid4()),"step":"group","name":f"{nm} (1/3)","collapsed?":True,
                       "weight":{"num":"1","den":"3"},"children":sub.get("children",[])})
    root={"id":str(uuid.uuid4()),"step":"root","name":f"Claude Trio: BullHedge + 2ndOpus + {wsh}","collapsed?":True,
          "children":[{"id":str(uuid.uuid4()),"step":"wt-cash-specified","name":"Weight",
                       "suppress_incomplete_warnings":True,"children":groups}],
          "description":f"Equal-weight trio: two bootstrap-robust equity strategies + a diversifier "
                        f"({wsh}, corr {wcorr:.2f} to the pair).",
          "rebalance":"none","rebalance-corridor-width":0.05}
    os.makedirs("data",exist_ok=True)
    json.dump(root,open("data/trio_portfolio_symphony.json","w"),indent=2,ensure_ascii=False)
    tot=[0]
    def c(x):
        if isinstance(x,dict) and x.get("step"):
            tot[0]+=1
            for k in x.get("children",[]): c(k)
    c(root); print(f"\nwrote data/trio_portfolio_symphony.json ({tot[0]} nodes)")


if __name__=="__main__":
    main()
