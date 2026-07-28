#!/usr/bin/env python3
"""
composer_confidence_screen2.py — same high-confidence + all-5-ratios screen as v1,
but (a) EXCLUDES strategies already surfaced this session and exact duplicates,
and (b) prints each candidate's decision LOGIC (distinct conditions + indicator
palette) so complexity can be judged as defensible-or-not, not just by node count.
Read-only.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import requests

MIN_OOS_DAYS = int(os.environ.get("MIN_OOS_DAYS", "504"))
WANT = int(os.environ.get("WANT", "12"))
ENDPOINTS = ["https://backtest-api.composer.trade/api/v1/public/symphonies/{id}/score",
             "https://backtest-api.composer.trade/api/v1/public/symphonies/{id}"]

# Already shown to the user earlier this session (sleeves, template A/B, supercharge,
# MF/tech, and the two from screen-v1 that were flagged as previously featured).
EXCLUDE = {
    "2cuimtTihBBpJgf7FSis","Dt37l1ceAggm8ggzBpRS","DLXJ2T0lIgBMGykzAf1U","l4glDbmbbDFd3p1Mcjkx",
    "0dKcj7cKmeHhafQKlrHM","tEq3s5F3AzjqcxwwvVVJ","0NukZC005nYIg0PZ7wET","gv99mKXm2PaRO3CSoUPB",
    "xN5Hi5Hv94gRHZynUTj5","R5UqpkTqw4DpEDHissw7","lM8PIp0ipSjup6brD10c","98cACZSS00eDg8Kv5BBV",
    "2NikM8NJOSadxphzzV6g","cOF23d0HjdXFREkMfJnl","ltYJfAbjNyUjIrA2TQyS","KpCZaTavfhU2OoaSc3pG",
}


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
    return "High" if d>=756 else "Good" if d>=504 else "Fair" if d>=252 else "Low"


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


def cond_text(n):
    def side(p):
        fn=n.get(f"{p}-fn"); val=n.get(f"{p}-val") or n.get(f"{p}-value")
        win=n.get(f"{p}-window-days")
        if fn: return f"{fn}{('('+str(win)+'d)') if win else ''} {val or ''}".strip()
        return str(val) if val is not None else "?"
    if n.get("is-else-condition?"): return None
    if not (n.get("comparator") or n.get("lhs-fn")): return None
    return f"{side('lhs')} {n.get('comparator','?')} {side('rhs')}"


def analyze(t):
    assets, dec, sel = {}, 0, 0
    conds=set(); fns=set(); tks=set()
    def w(n):
        nonlocal dec, sel
        if not isinstance(n, dict): return
        st=n.get("step")
        if st in ("if","filter"): dec+=1
        if st=="filter" and n.get("select?"):
            sel+=1
            sf=n.get("sort-by-fn") or n.get("select-fn")
            if sf: conds.add(f"SELECT by {sf}"); fns.add(sf)
        if st=="if-child":
            c=cond_text(n)
            if c: conds.add(c)
            for p in ("lhs","rhs"):
                if n.get(f"{p}-fn"): fns.add(n[f"{p}-fn"])
                v=n.get(f"{p}-val")
                if v and not str(v).replace(".","").replace("-","").isdigit(): tks.add(str(v))
        if st=="asset" and n.get("ticker"): assets[n["ticker"]]=bool(n.get("has_marketcap"))
        for c in n.get("children",[]): w(c)
    w(t or {})
    return dict(dec=dec, sel=sel,
                stocks=sorted(k for k,v in assets.items() if v),
                etfs=sorted(k for k,v in assets.items() if not v),
                static=(dec==0 and sel==0),
                conds=sorted(conds), fns=sorted(fns), tks=sorted(tks))


def pctl(vals):
    a=np.asarray(vals,float); order=a.argsort(); ranks=np.empty(len(a)); ranks[order]=np.arange(len(a))
    return ranks/(len(a)-1) if len(a)>1 else np.zeros(len(a))


def main():
    rows=load("data/composer_oos_discover.jsonl")+load("data/composer_oos_watchlist.jsonl")
    latest=max(r.get("date","") for r in rows)
    rows=[r for r in rows if r.get("date")==latest]
    seen={}
    for r in rows: seen[r.get("sym_id")]=r
    rows=list(seen.values())
    RAT=["oos_sharpe","oos_calmar","oos_sortino","oos_martin","oos_ulcer"]
    def ok(r):
        if (r.get("oos_days") or 0) < MIN_OOS_DAYS: return False
        if any(r.get(k) is None for k in RAT): return False
        if r["oos_sharpe"]<=0 or r["oos_calmar"]<=0 or r["oos_sortino"]<=0 or r["oos_martin"]<=0: return False
        if (r.get("oos_maxdd_pct") or -99) < -45: return False
        if r.get("sym_id") in EXCLUDE: return False
        return True
    pool=[r for r in rows if ok(r)]
    P={k:pctl([r[k] for r in pool]) for k in ["oos_sharpe","oos_calmar","oos_sortino","oos_martin"]}
    P["oos_ulcer"]=pctl([-r["oos_ulcer"] for r in pool])
    for i,r in enumerate(pool): r["_c"]=float(np.mean([P[k][i] for k in P]))
    pool.sort(key=lambda r:-r["_c"])
    print(f"pool: {len(pool)} (excluded {sum(1 for r in rows if r.get('sym_id') in EXCLUDE)} already-shown)\n")

    picks=[]; seen_sig=set(); dropped_stock=0; dropped_dup=0
    for r in pool:
        if len(picks)>=WANT: break
        a=analyze(fetch(r["sym_id"])); time.sleep(0.25); r["_a"]=a
        # exact-duplicate collapse: same metrics + same holdings
        sig=(r["oos_days"], round(r["oos_sharpe"],2), round(r["oos_calmar"],2),
             tuple(a["etfs"]), tuple(a["stocks"]))
        if sig in seen_sig: dropped_dup+=1; continue
        seen_sig.add(sig)
        if a["stocks"] and (len(a["stocks"])>=len(a["etfs"]) or (a["static"] and not a["etfs"])):
            dropped_stock+=1; continue
        picks.append(r)
    print(f"dropped {dropped_dup} exact-duplicates, {dropped_stock} stock-baskets\n")

    for i,r in enumerate(picks,1):
        a=r["_a"]
        print("="*104)
        print(f"{i:>2}. {(r.get('name') or '')[:60]}   [{r['sym_id']}]")
        print(f"    {r['oos_days']}d {conf_tier(r['oos_days'])} · Sharpe {r['oos_sharpe']:.2f} · Calmar {r['oos_calmar']:.2f} · "
              f"Sortino {r['oos_sortino']:.2f} · Ulcer {r['oos_ulcer']:.1f} · Martin {r['oos_martin']:.2f} · "
              f"CAGR {r['oos_cagr_pct']:.1f}% · MaxDD {r['oos_maxdd_pct']:.1f}% · comp {r['_c']:.2f}")
        print(f"    structure: {a['dec']} decisions, {a['sel']} selection-filters · "
              f"{len(a['conds'])} distinct rules · indicators {a['fns']}")
        print(f"    ETFs({len(a['etfs'])}): {a['etfs'][:14]}" + (f" +{len(a['etfs'])-14}" if len(a['etfs'])>14 else ""))
        if a["stocks"]: print(f"    stocks: {a['stocks']}")
        if a["conds"]:
            print(f"    logic (distinct conditions):")
            for c in a["conds"][:12]: print(f"       - {c}")
            if len(a["conds"])>12: print(f"       … +{len(a['conds'])-12} more")


if __name__=="__main__":
    main()
