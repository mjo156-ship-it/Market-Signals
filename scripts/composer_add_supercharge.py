#!/usr/bin/env python3
"""
composer_add_supercharge.py

Evaluate a high-risk/high-reward 6th sleeve that was ACTIVE OUT-OF-SAMPLE in
2022 (survived a real bear), then add it to the revised 5-sleeve book.

For each candidate: structure (adaptive? holdings, leveraged?), standalone stats
over the 2022 bear (2022-01-01..2022-12-31) AND the common window
(2023-06-14..today). Then build the 6-sleeve combined book two ways
(equal-weight-6, and a supercharge tilt) with 5% corridor rebalancing, compare
to the 5-sleeve, and write an importable symphony for whichever tilt keeps
portfolio MaxDD under 20%.
"""
from __future__ import annotations
import os, sys, json, uuid, time
from datetime import date
import numpy as np
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos as oos

REVISED5 = [
    ("Pop Bot",       "2cuimtTihBBpJgf7FSis"),
    ("Regime+Dip",    "Dt37l1ceAggm8ggzBpRS"),
    ("OG2xAAA",       "DLXJ2T0lIgBMGykzAf1U"),
    ("OG Gain Train", "l4glDbmbbDFd3p1Mcjkx"),
    ("Gold&Dollar",   "0dKcj7cKmeHhafQKlrHM"),
]
CANDIDATES = [
    ("smarter TQQQ",          "cOF23d0HjdXFREkMfJnl"),
    ("Buy the Dips: Nasdaq100","ltYJfAbjNyUjIrA2TQyS"),
]
CHOICE = ("smarter TQQQ", "cOF23d0HjdXFREkMfJnl")   # supercharge pick

# display names for the importable JSON groups
NAMES = {
    "2cuimtTihBBpJgf7FSis": "Pop Bot (SPY vs BND) l BrianE l May 30th 2007",
    "Dt37l1ceAggm8ggzBpRS": "Simple Regime Switching and Dip Buying",
    "DLXJ2T0lIgBMGykzAf1U": "OG Adaptive Asset Allocation 2x (SPY/TLT/UUP/GLD)",
    "l4glDbmbbDFd3p1Mcjkx": "OG V 1bb | Gain Train DGAF | Deez",
    "0dKcj7cKmeHhafQKlrHM": "Diversify with Gold & the Dollar",
    "cOF23d0HjdXFREkMfJnl": "smarter TQQQ - longer backtest period",
}
COMMON_START = os.environ.get("PORTFOLIO_COMMON_START", "2023-06-14")
END = date.today().isoformat()
ENDPOINTS = ["https://backtest-api.composer.trade/api/v1/public/symphonies/{id}/score",
             "https://backtest-api.composer.trade/api/v1/public/symphonies/{id}"]
LEV = {"TQQQ","UPRO","SOXL","TECL","TMF","SPXL","UDOW","FNGU","LABU","QLD","SSO","SOXX","UVXY","UVIX","BOIL"}


def tree(o):
    if isinstance(o, dict):
        if ("step" in o) and ("children" in o): return o
        for k in ("symphony","score","definition","s"):
            if k in o and (t := tree(o[k])): return t
        for v in o.values():
            if (t := tree(v)): return t
    return None


def fetch(sid):
    for tmpl in ENDPOINTS:
        try:
            r = requests.get(tmpl.format(id=sid), headers={"accept":"application/json"}, timeout=25)
        except Exception:
            continue
        if r.status_code == 200 and "json" in r.headers.get("content-type",""):
            if (t := tree(r.json())): return t
    return None


def ensure_ids(node):
    if isinstance(node, dict):
        if node.get("step") and not node.get("id"): node["id"] = str(uuid.uuid4())
        for c in node.get("children", []): ensure_ids(c)
    return node


def analyze(t):
    assets, nodes, dec, conds = {}, 0, 0, []
    def cond(n):
        def side(p):
            fn=n.get(f"{p}-fn"); val=n.get(f"{p}-val") or n.get(f"{p}-value")
            win=n.get(f"{p}-window-days")
            return (f"{fn}({win}d) {val or ''}".strip() if fn else str(val) if val is not None else "?")
        if n.get("is-else-condition?"): return "ELSE"
        return f"{side('lhs')} {n.get('comparator','?')} {side('rhs')}"
    def walk(n):
        nonlocal nodes, dec
        if not isinstance(n, dict): return
        if n.get("step"): nodes += 1
        if n.get("step")=="if": dec += 1
        if n.get("step")=="if-child" and (n.get("comparator") or n.get("lhs-fn")):
            conds.append(cond(n))
        if n.get("step")=="asset" and n.get("ticker"):
            assets[n["ticker"]] = bool(n.get("has_marketcap"))
        for c in n.get("children", []): walk(c)
    walk(t or {})
    etfs=sorted(k for k,v in assets.items() if not v)
    return dict(nodes=nodes, dec=dec,
                stocks=sorted(k for k,v in assets.items() if v), etfs=etfs,
                lev=sorted(set(etfs)&LEV), conds=conds)


def equity(sid, start, end):
    body={"capital":100000,"apply_reg_fee":True,"apply_taf_fee":True,"apply_cat_fee":True,
          "apply_subscription":"none","backtest_version":"v2","slippage_percent":0.0005,
          "start_date":start,"end_date":end,"broker":"ALPACA_WHITE_LABEL","benchmark_tickers":["SPY"]}
    for k in range(5):
        try:
            d=oos._request("POST", f"/api/v0.1/symphonies/{sid}/backtest", json=body); break
        except RuntimeError as e:
            if "rate limited" in str(e).lower() and k<4: time.sleep(6*(k+1)); continue
            raise
    cap=d.get("dvm_capital") or {}
    if cap and isinstance(next(iter(cap.values())), dict): cap=next(iter(cap.values()))
    return {str(k):float(v) for k,v in cap.items() if v is not None}


def _dk(k):
    s=k.replace(".","").replace("-",""); return float(s) if s.isdigit() else k


def stats_from_curve(cap):
    days=sorted(cap.keys(), key=_dk)
    v=np.array([cap[d] for d in days], float)
    r=np.diff(v)/v[:-1]
    return stats(r)


def sim(R, w, corridor):
    w0=np.array(w,float); cur=w0.copy(); out=[]
    for t in range(R.shape[0]):
        rt=R[t]; out.append(float(np.dot(cur,rt)))
        cur=cur*(1+rt); cur=cur/cur.sum()
        if corridor<0 or np.max(np.abs(cur-w0))>corridor: cur=w0.copy()
    return np.array(out)


def stats(r):
    n=len(r); ann=252
    if n<5: return dict(n=n,cagr=float('nan'),sharpe=float('nan'),sortino=float('nan'),
                        maxdd=float('nan'),calmar=float('nan'),cum=float('nan'))
    eq=np.cumprod(1+r); cagr=eq[-1]**(ann/n)-1
    vol=r.std(ddof=1)*np.sqrt(ann); sharpe=(r.mean()*ann)/vol if vol else float('nan')
    dn=r[r<0].std(ddof=1)*np.sqrt(ann) if (r<0).any() else float('nan')
    sortino=(r.mean()*ann)/dn if dn else float('nan')
    peak=np.maximum.accumulate(eq); maxdd=(eq/peak-1).min()
    calmar=cagr/abs(maxdd) if maxdd else float('nan')
    return dict(n=n,cagr=cagr,sharpe=sharpe,sortino=sortino,maxdd=maxdd,calmar=calmar,cum=eq[-1]-1)


def show(tag,s):
    print(f"  {tag:30s} CAGR {s['cagr']*100:7.2f}%  Sharpe {s['sharpe']:5.2f}  "
          f"MaxDD {s['maxdd']*100:7.2f}%  Calmar {s['calmar']:5.2f}  Cum {s['cum']*100:7.1f}%  (n={s['n']})")


def build_json(defs, weights, fname, title):
    groups=[]
    for sid,w in weights:
        sub=ensure_ids(defs[sid])
        groups.append({"id":str(uuid.uuid4()),"step":"group",
            "name":f"{NAMES.get(sid,sid)} ({w['num']}/{w['den']})","collapsed?":True,
            "weight":w,"children":sub.get("children",[])})
    root={"id":str(uuid.uuid4()),"step":"root","name":title,"collapsed?":True,
        "children":[{"id":str(uuid.uuid4()),"step":"wt-cash-specified","name":"Weight",
                     "suppress_incomplete_warnings":True,"children":groups}],
        "description":("6-sleeve blend: revised diversified 5 + a leveraged-Nasdaq "
                       "supercharge sleeve that survived 2022 out-of-sample. All rule-driven, "
                       "no individual stocks."),
        "rebalance":"none","rebalance-corridor-width":0.05}
    os.makedirs("data",exist_ok=True)
    json.dump(root, open(f"data/{fname}","w"), indent=2, ensure_ascii=False)
    n=[0]
    def c(x):
        if isinstance(x,dict) and x.get("step"):
            n[0]+=1
            for k in x.get("children",[]): c(k)
    c(root); print(f"  wrote data/{fname} ({n[0]} nodes)")


def main():
    print(f"common window {COMMON_START} -> {END}\n")
    ids=[sid for _,sid in REVISED5]+[sid for _,sid in CANDIDATES]
    defs, common_curves = {}, {}
    for sid in ids:
        t=fetch(sid); time.sleep(0.3); defs[sid]=t
        common_curves[sid]=equity(sid, COMMON_START, END); time.sleep(0.3)

    print("=== CANDIDATE due-diligence (structure + 2022 bear + 2023-06+ standalone) ===")
    for nm,sid in CANDIDATES:
        a=analyze(defs[sid])
        kind="ADAPTIVE" if a["dec"]>0 else "STATIC"
        try:
            s22=stats_from_curve(equity(sid,"2022-01-01","2022-12-31")); time.sleep(0.3)
        except Exception as e:
            s22=None; print(f"  {nm}: 2022 bt fail {type(e).__name__}", file=sys.stderr)
        print(f"\n  # {nm} [{sid}]  {a['nodes']}n {a['dec']}if -> {kind}")
        print(f"    ETFs={a['etfs']}  leveraged={a['lev']}  stocks={a['stocks']}")
        if a["conds"]: print(f"    logic: {a['conds'][:6]}")
        if s22: show("2022 bear-year (OOS)", s22)
        show("2023-06+ standalone", stats_from_curve(common_curves[sid]))

    # align common curves
    allids=ids
    cm=None
    for sid in allids:
        ks=set(common_curves[sid].keys()); cm=ks if cm is None else (cm&ks)
    cm=sorted(cm, key=_dk)
    print(f"\naligned common days: {len(cm)}")
    def R(id_list):
        cols=[]
        for sid in id_list:
            v=np.array([common_curves[sid][d] for d in cm], float)
            cols.append(np.diff(v)/v[:-1])
        return np.column_stack(cols)

    five=[sid for _,sid in REVISED5]
    six=five+[CHOICE[1]]
    R5=R(five); R6=R(six)

    print("\n=== COMBINED (5% corridor rebalance) ===")
    show("5-sleeve revised (20% ea)", stats(sim(R5,[0.2]*5,0.05)))
    show("6-sleeve EQUAL (16.7% ea)", stats(sim(R6,[1/6]*6,0.05)))
    tilt=[0.14,0.14,0.14,0.14,0.14,0.30]   # existing 14% each, supercharge 30%
    s_tilt=stats(sim(R6,tilt,0.05))
    show("6-sleeve TILT (super=30%)", s_tilt)
    tilt2=[0.16,0.16,0.16,0.16,0.16,0.20]
    show("6-sleeve TILT (super=20%)", stats(sim(R6,tilt2,0.05)))

    # build importable JSON: EW6 always; tilt-30 only if it keeps MaxDD < 20%
    print("\n=== importable symphonies ===")
    ew6=[(sid,{"num":"1","den":"6"}) for sid in six]
    build_json(defs, ew6, "revised6_portfolio_symphony.json", "Claude 6-sleeve EW (supercharge)")
    if s_tilt["maxdd"] > -0.20:
        tw=[(sid,{"num":"14","den":"100"}) for sid in five]+[(CHOICE[1],{"num":"30","den":"100"})]
        build_json(defs, tw, "revised6_tilt_symphony.json", "Claude 6-sleeve tilt (supercharge 30%)")
        print("  (tilt-30 MaxDD stays < 20% -> built)")
    else:
        print(f"  (tilt-30 MaxDD {s_tilt['maxdd']*100:.1f}% breaches 20% -> tilt JSON skipped)")


if __name__ == "__main__":
    main()
