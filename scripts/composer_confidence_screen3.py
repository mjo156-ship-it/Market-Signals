#!/usr/bin/env python3
"""
composer_confidence_screen3.py — fresh high-confidence + all-5-ratios screen.
Adds on top of v2:
  * EXCLUDE now also covers everything already shown (portfolio work + screen-v1
    output), so results are genuinely new.
  * FAMILY dedup: collapse name-variants ("Simple Portfolio (UVXY) + v4 Pops",
    "... inv", "Copy of ...", version/date suffixes) to one representative.
  * Defensibility tag from logic shape: DEFENSIBLE (<=22 distinct rules AND <=5
    indicator types) vs COMPLEX (verify the thesis before trusting).
Read-only.
"""
from __future__ import annotations
import os, sys, json, time, re
import numpy as np
import requests

MIN_OOS_DAYS = int(os.environ.get("MIN_OOS_DAYS", "504"))
WANT = int(os.environ.get("WANT", "10"))
FETCH_MAX = int(os.environ.get("FETCH_MAX", "60"))
ENDPOINTS = ["https://backtest-api.composer.trade/api/v1/public/symphonies/{id}/score",
             "https://backtest-api.composer.trade/api/v1/public/symphonies/{id}"]

EXCLUDE = {
    # portfolio work
    "2cuimtTihBBpJgf7FSis","Dt37l1ceAggm8ggzBpRS","DLXJ2T0lIgBMGykzAf1U","l4glDbmbbDFd3p1Mcjkx",
    "0dKcj7cKmeHhafQKlrHM","tEq3s5F3AzjqcxwwvVVJ","0NukZC005nYIg0PZ7wET","gv99mKXm2PaRO3CSoUPB",
    "xN5Hi5Hv94gRHZynUTj5","R5UqpkTqw4DpEDHissw7","lM8PIp0ipSjup6brD10c","98cACZSS00eDg8Kv5BBV",
    "2NikM8NJOSadxphzzV6g","cOF23d0HjdXFREkMfJnl","ltYJfAbjNyUjIrA2TQyS","KpCZaTavfhU2OoaSc3pG",
    # screen-v1 list already shown to the user
    "F8cR1emy2BFsjiwVVwRI","5o9sNxjW2TWSC3CjWK7V","7JZ3CK7p5NlkzyKmdcav","QCvPrlwnYhtuUbvl8MK7",
    "KoHA3RobcRFhNatNmJ3q","dPfnPcfe4koAFcfkt9Ez","6idb9D5kCMKf7R3hJsp9","y03pNNw8lsvwsuSSL9KK",
}

STOP={'copy','of','mod','inv','edition','longer','backtest','period','the','and','or','not',
      'version','l','v','pops','pop','bot','bots'}


def famkey(name):
    s=re.sub(r'[^a-z0-9 ]',' ',(name or '').lower())
    toks=[t for t in s.split()
          if not re.fullmatch(r'v?\d+[a-z]?', t) and not re.fullmatch(r'\d{2,4}', t) and t not in STOP]
    return ' '.join(toks[:5])


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
        fn=n.get(f"{p}-fn"); val=n.get(f"{p}-val") or n.get(f"{p}-value"); win=n.get(f"{p}-window-days")
        if fn: return f"{fn}{('('+str(win)+'d)') if win else ''} {val or ''}".strip()
        return str(val) if val is not None else "?"
    if n.get("is-else-condition?"): return None
    if not (n.get("comparator") or n.get("lhs-fn")): return None
    return f"{side('lhs')} {n.get('comparator','?')} {side('rhs')}"


def analyze(t):
    assets, dec, sel = {}, 0, 0; conds=set(); fns=set()
    def w(n):
        nonlocal dec, sel
        if not isinstance(n, dict): return
        st=n.get("step")
        if st in ("if","filter"): dec+=1
        if st=="filter" and n.get("select?"):
            sel+=1; sf=n.get("sort-by-fn") or n.get("select-fn")
            if sf: conds.add(f"SELECT by {sf}"); fns.add(sf)
        if st=="if-child":
            c=cond_text(n)
            if c: conds.add(c)
            for p in ("lhs","rhs"):
                if n.get(f"{p}-fn"): fns.add(n[f"{p}-fn"])
        if st=="asset" and n.get("ticker"): assets[n["ticker"]]=bool(n.get("has_marketcap"))
        for c in n.get("children",[]): w(c)
    w(t or {})
    return dict(dec=dec, sel=sel,
                stocks=sorted(k for k,v in assets.items() if v),
                etfs=sorted(k for k,v in assets.items() if not v),
                static=(dec==0 and sel==0), conds=sorted(conds), fns=sorted(fns))


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
    print(f"pool: {len(pool)} (>= {MIN_OOS_DAYS} OOS days, all 5 ratios +, excluding already-shown)\n")

    picks=[]; fams=set(); d_fam=0; d_stock=0; fetched=0
    for r in pool:
        if len(picks)>=WANT or fetched>=FETCH_MAX: break
        fk=famkey(r.get("name"))
        if fk in fams: d_fam+=1; continue
        a=analyze(fetch(r["sym_id"])); fetched+=1; time.sleep(0.22); r["_a"]=a
        if a["stocks"] and (len(a["stocks"])>=len(a["etfs"]) or (a["static"] and not a["etfs"])):
            d_stock+=1; continue
        fams.add(fk); picks.append(r)
    print(f"dropped {d_fam} name-variant family dups, {d_stock} stock-baskets (fetched {fetched})\n")

    for i,r in enumerate(picks,1):
        a=r["_a"]; nr=len(a["conds"]); ni=len(a["fns"])
        tag = "DEFENSIBLE" if (nr<=22 and ni<=5) else "COMPLEX-verify"
        print("="*104)
        print(f"{i:>2}. {(r.get('name') or '')[:58]}   [{r['sym_id']}]   <{tag}>")
        print(f"    {r['oos_days']}d {conf_tier(r['oos_days'])} · Sharpe {r['oos_sharpe']:.2f} · Calmar {r['oos_calmar']:.2f} · "
              f"Sortino {r['oos_sortino']:.2f} · Ulcer {r['oos_ulcer']:.1f} · Martin {r['oos_martin']:.2f} · "
              f"CAGR {r['oos_cagr_pct']:.1f}% · MaxDD {r['oos_maxdd_pct']:.1f}% · comp {r['_c']:.2f}")
        print(f"    {a['dec']} decisions, {a['sel']} selects · {nr} distinct rules · {ni} indicators {a['fns']}")
        print(f"    ETFs({len(a['etfs'])}): {a['etfs'][:14]}" + (f" +{len(a['etfs'])-14}" if len(a['etfs'])>14 else ""))
        if a["stocks"]: print(f"    stocks: {a['stocks']}")
        for c in a["conds"][:10]: print(f"       - {c}")
        if len(a["conds"])>10: print(f"       … +{len(a['conds'])-10} more")


if __name__=="__main__":
    main()
