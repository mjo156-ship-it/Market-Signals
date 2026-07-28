#!/usr/bin/env python3
"""
composer_revise_and_backtest.py

(1) Assemble the REVISED equal-weight 5-sleeve portfolio into one importable
    Composer symphony (schema that imports: id on every node, string weights,
    rebalance "none" + 5% corridor). Writes data/revised_portfolio_symphony.json.

(2) Backtest the COMBINED book from each sleeve's own Composer backtest equity
    curve (fees + 5bps slippage already inside each sleeve), blended 20% each
    with 5% threshold-corridor rebalancing. Also backtests the OLD 5-sleeve
    version (with Joseph Story + Simple Dividends) for a side-by-side.

Read-only against Composer; writes one JSON. Prints a tearsheet.
"""
from __future__ import annotations
import os, sys, json, uuid, time
from datetime import date
import numpy as np
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos as oos

# REVISED portfolio (equal weight 20%)
NEW = [
    ("Pop Bot (SPY vs BND) l BrianE l May 30th 2007",   "2cuimtTihBBpJgf7FSis", 20),
    ("Simple Regime Switching and Dip Buying",           "Dt37l1ceAggm8ggzBpRS", 20),
    ("OG Adaptive Asset Allocation 2x (SPY/TLT/UUP/GLD)", "DLXJ2T0lIgBMGykzAf1U", 20),
    ("OG V 1bb | Gain Train DGAF | Deez",                "l4glDbmbbDFd3p1Mcjkx", 20),
    ("Diversify with Gold & the Dollar",                 "0dKcj7cKmeHhafQKlrHM", 20),
]
# OLD portfolio sleeves that were swapped out (for comparison only)
OLD_OUT = [
    ("Joseph Story Fund", "tEq3s5F3AzjqcxwwvVVJ"),
    ("Simple Dividends",  "0NukZC005nYIg0PZ7wET"),
]
START = os.environ.get("PORTFOLIO_COMMON_START", "2023-06-14")
END = date.today().isoformat()
ENDPOINTS = ["https://backtest-api.composer.trade/api/v1/public/symphonies/{id}/score",
             "https://backtest-api.composer.trade/api/v1/public/symphonies/{id}"]


def tree(o):
    if isinstance(o, dict):
        if ("step" in o) and ("children" in o): return o
        for k in ("symphony", "score", "definition", "s"):
            if k in o and (t := tree(o[k])): return t
        for v in o.values():
            if (t := tree(v)): return t
    return None


def fetch(sid):
    for tmpl in ENDPOINTS:
        try:
            r = requests.get(tmpl.format(id=sid), headers={"accept": "application/json"}, timeout=25)
        except Exception:
            continue
        if r.status_code == 200 and "json" in r.headers.get("content-type", ""):
            if (t := tree(r.json())): return t
    return None


def ensure_ids(node):
    if isinstance(node, dict):
        if node.get("step") and not node.get("id"):
            node["id"] = str(uuid.uuid4())
        for c in node.get("children", []):
            ensure_ids(c)
    return node


def build(defs):
    groups = []
    for nm, sid, w in NEW:
        sub = ensure_ids(defs[sid])
        groups.append({
            "id": str(uuid.uuid4()), "step": "group", "name": f"{nm} ({w}%)",
            "collapsed?": True, "weight": {"num": str(w), "den": "100"},
            "children": sub.get("children", []),
        })
    root = {
        "id": str(uuid.uuid4()), "step": "root",
        "name": "Claude's Revised OOS Portfolio (5 eq-wt, diversified)",
        "collapsed?": True,
        "children": [{
            "id": str(uuid.uuid4()), "step": "wt-cash-specified", "name": "Weight",
            "suppress_incomplete_warnings": True, "children": groups,
        }],
        "description": ("Equal-weight blend of 5 adaptive, OOS-consistent Composer "
                        "strategies. Survivor-biased single-stock sleeves (Joseph Story, "
                        "Simple Dividends) replaced with non-equity diversifiers "
                        "(OG 2x Adaptive Asset Allocation; Gold & Dollar). No individual "
                        "stocks; all sleeves rule-driven."),
        "rebalance": "none", "rebalance-corridor-width": 0.05,
    }
    return root


def equity(sid):
    body = {"capital": 100000, "apply_reg_fee": True, "apply_taf_fee": True,
            "apply_cat_fee": True, "apply_subscription": "none", "backtest_version": "v2",
            "slippage_percent": 0.0005, "start_date": START, "end_date": END,
            "broker": "ALPACA_WHITE_LABEL", "benchmark_tickers": ["SPY"]}
    for k in range(5):
        try:
            d = oos._request("POST", f"/api/v0.1/symphonies/{sid}/backtest", json=body); break
        except RuntimeError as e:
            if "rate limited" in str(e).lower() and k < 4: time.sleep(6 * (k + 1)); continue
            raise
    cap = d.get("dvm_capital") or {}
    if cap and isinstance(next(iter(cap.values())), dict):
        cap = next(iter(cap.values()))
    return {str(k): float(v) for k, v in cap.items() if v is not None}


def _dk(k):
    s = k.replace(".", "").replace("-", "")
    return float(s) if s.isdigit() else k


def sim(R, w_target, corridor):
    """R: (T,k) daily returns; blend with threshold-corridor rebalancing.
    corridor<0 => rebalance every day (daily)."""
    w0 = np.array(w_target, float); cur = w0.copy(); out = []
    for t in range(R.shape[0]):
        rt = R[t]
        out.append(float(np.dot(cur, rt)))
        cur = cur * (1 + rt); cur = cur / cur.sum()
        if corridor < 0 or np.max(np.abs(cur - w0)) > corridor:
            cur = w0.copy()
    return np.array(out)


def stats(r):
    n = len(r); ann = 252
    eq = np.cumprod(1 + r)
    cagr = eq[-1] ** (ann / n) - 1
    vol = r.std(ddof=1) * np.sqrt(ann)
    sharpe = (r.mean() * ann) / vol if vol else float("nan")
    downside = r[r < 0].std(ddof=1) * np.sqrt(ann) if (r < 0).any() else float("nan")
    sortino = (r.mean() * ann) / downside if downside else float("nan")
    peak = np.maximum.accumulate(eq); dd = eq / peak - 1; maxdd = dd.min()
    calmar = cagr / abs(maxdd) if maxdd else float("nan")
    return dict(n=n, cagr=cagr, vol=vol, sharpe=sharpe, sortino=sortino,
                maxdd=maxdd, calmar=calmar, cum=eq[-1] - 1)


def show(tag, s):
    print(f"  {tag:26s} CAGR {s['cagr']*100:6.2f}%  Sharpe {s['sharpe']:4.2f}  "
          f"Sortino {s['sortino']:4.2f}  MaxDD {s['maxdd']*100:6.2f}%  "
          f"Calmar {s['calmar']:4.2f}  Cum {s['cum']*100:6.1f}%  (n={s['n']})")


def main():
    ids = {sid for _, sid, _ in NEW} | {sid for _, sid in OLD_OUT}
    curves, defs = {}, {}
    for sid in ids:
        t = fetch(sid); time.sleep(0.3)
        if t: defs[sid] = t
        try:
            curves[sid] = equity(sid); time.sleep(0.3)
        except Exception as e:
            print(f"bt fail {sid}: {type(e).__name__}", file=sys.stderr); curves[sid] = {}

    # ---- build importable JSON (revised) ----
    if all(sid in defs for _, sid, _ in NEW):
        root = build(defs)
        os.makedirs("data", exist_ok=True)
        json.dump(root, open("data/revised_portfolio_symphony.json", "w"), indent=2, ensure_ascii=False)
        tot = [0]
        def cnt(n):
            if isinstance(n, dict) and n.get("step"):
                tot[0] += 1
                for c in n.get("children", []): cnt(c)
        cnt(root)
        print(f"wrote data/revised_portfolio_symphony.json ({tot[0]} nodes)\n")
    else:
        print("WARN: could not fetch all NEW sleeve defs; JSON not written\n", file=sys.stderr)

    # ---- align curves on common dates ----
    common = None
    for sid in ids:
        ks = set(curves[sid].keys())
        common = ks if common is None else (common & ks)
    common = sorted(common, key=_dk)
    print(f"window {START} -> {END}   aligned days: {len(common)}\n")

    def rets(sid):
        v = np.array([curves[sid][d] for d in common], float)
        return np.diff(v) / v[:-1]

    new_ids = [sid for _, sid, _ in NEW]
    old_ids = ["2cuimtTihBBpJgf7FSis", "Dt37l1ceAggm8ggzBpRS",
               "tEq3s5F3AzjqcxwwvVVJ", "0NukZC005nYIg0PZ7wET", "l4glDbmbbDFd3p1Mcjkx"]
    Rn = np.column_stack([rets(s) for s in new_ids])
    Ro = np.column_stack([rets(s) for s in old_ids])
    w = [0.2] * 5

    print("=== COMBINED PORTFOLIO (equal weight 20% each) ===")
    print("  -- 5% threshold-corridor rebalance (matches the symphony) --")
    show("REVISED (new sleeves)", stats(sim(Rn, w, 0.05)))
    show("OLD (JS + SimpleDiv)",  stats(sim(Ro, w, 0.05)))
    print("  -- daily rebalance (reference bound) --")
    show("REVISED (new sleeves)", stats(sim(Rn, w, -1)))
    show("OLD (JS + SimpleDiv)",  stats(sim(Ro, w, -1)))

    # revised combined vs SPY-like? show pairwise corr of the 5 new sleeves
    print("\n=== revised sleeve intercorrelation (daily) ===")
    labels = ["PopBot", "Regime", "OG2xAAA", "OGGain", "Gold&$"]
    C = np.corrcoef(Rn.T)
    print("           " + "".join(f"{l:>9}" for l in labels))
    for i, l in enumerate(labels):
        print(f"  {l:8s}" + "".join(f"{C[i,j]:9.2f}" for j in range(5)))
    iu = np.triu_indices(5, 1)
    print(f"  avg pairwise corr: {C[iu].mean():.2f}   max: {C[iu].max():.2f}")


if __name__ == "__main__":
    main()
