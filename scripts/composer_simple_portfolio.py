#!/usr/bin/env python3
"""
composer_simple_portfolio.py — a SIMPLE version of the OOS-confidence portfolio.

Adds two hard simplicity gates the first version ignored:
  * node count   — total nodes in the symphony's definition (fewer = simpler logic)
  * daily churn  — average one-way turnover from the backtest's daily target
                   weights (how often holdings actually change)

Pipeline:
  1. Screen the Discover ledger for High-confidence, OOS-consistent candidates.
  2. For each: fetch its definition (public score endpoint) -> node count; backtest
     over the common OOS window -> equity curve + tdvm_weights -> daily turnover +
     OOS stats.
  3. Keep the simplest / lowest-churn names that still clear quality gates.
  4. Correlation + combo search over 5-7 (EQUAL weight — simplest scheme); keep
     combos with portfolio MaxDD < 20% and genuine diversification; rank by Calmar.

Read-only. Writes data/simple_portfolio_analysis.json and prints a report.
"""
from __future__ import annotations
import os, sys, json, time, itertools
from datetime import date
import numpy as np
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos as oos

LEDGER_URL = os.environ.get("DISCOVER_LEDGER",
    "https://raw.githubusercontent.com/mjo156-ship-it/Market-Signals/main/data/composer_oos_discover.jsonl")
OUT = os.environ.get("SIMPLE_OUT", "data/simple_portfolio_analysis.json")
COMMON_START = os.environ.get("PORTFOLIO_COMMON_START", "2023-06-14")
MAXDD_LIMIT = float(os.environ.get("PORTFOLIO_MAXDD_LIMIT", "20"))
POOL_MAX = int(os.environ.get("POOL_MAX", "45"))          # cap candidates enriched
SIMPLE_KEEP = int(os.environ.get("SIMPLE_KEEP", "14"))    # simplest N -> optimizer
PORT_MIN, PORT_MAX = 5, 7
MAX_PAIR_CORR = float(os.environ.get("MAX_PAIR_CORR", "0.85"))
MAX_AVG_CORR = float(os.environ.get("MAX_AVG_CORR", "0.60"))

SCORE_ENDPOINTS = [
    "https://backtest-api.composer.trade/api/v1/public/symphonies/{id}/score",
    "https://backtest-api.composer.trade/api/v1/public/symphonies/{id}",
]


def fetch_nodecount(sid):
    def tree(o):
        if isinstance(o, dict):
            if ("step" in o or ":step" in o) and ("children" in o or ":children" in o):
                return o
            for k in ("symphony", "score", "definition", "s"):
                if k in o:
                    t = tree(o[k])
                    if t: return t
            for v in o.values():
                t = tree(v)
                if t: return t
        return None
    def count(t):
        if not isinstance(t, dict): return 0, 0
        n = 1 if (t.get("step") or t.get(":step")) else 0
        dec = 1 if (t.get("step") in ("if", "filter")) else 0
        for c in (t.get("children") or t.get(":children") or []):
            cn, cd = count(c); n += cn; dec += cd
        return n, dec
    for tmpl in SCORE_ENDPOINTS:
        try:
            r = requests.get(tmpl.format(id=sid), headers={"accept": "application/json"}, timeout=25)
        except Exception:
            continue
        if r.status_code == 200 and "json" in r.headers.get("content-type", ""):
            t = tree(r.json())
            if t:
                return count(t)
    return (None, None)


def backtest_full(sid, start, end):
    """Full backtest response so we get both the curve and tdvm_weights."""
    body = {"capital": 100000, "apply_reg_fee": True, "apply_taf_fee": True,
            "apply_cat_fee": True, "apply_subscription": "none", "backtest_version": "v2",
            "slippage_percent": 0.0005, "start_date": start, "end_date": end,
            "broker": "ALPACA_WHITE_LABEL", "benchmark_tickers": ["SPY"]}
    for k in range(5):
        try:
            d = oos._request("POST", f"/api/v0.1/symphonies/{sid}/backtest", json=body)
            break
        except RuntimeError as e:
            if "rate limited" in str(e).lower() and k < 4:
                time.sleep(6 * (k + 1)); continue
            raise
    dvm = d.get("dvm_capital") or {}
    curve = dvm.get(sid) or (next(iter(dvm.values()), {}) if dvm else {})
    curve = {oos._to_date(kk): float(v) for kk, v in curve.items()}
    return curve, d.get("tdvm_weights")


def turnover(tdvm):
    """Average one-way daily turnover (0..1) from date->{asset:weight}."""
    try:
        if not isinstance(tdvm, dict) or len(tdvm) < 5:
            return None
        days = sorted(tdvm.keys(), key=lambda k: float(k) if str(k).replace(".", "").isdigit() else 0)
        keys = set()
        for dd in days:
            w = tdvm[dd]
            if isinstance(w, dict): keys |= set(w.keys())
        keys = sorted(keys)
        if not keys: return None
        prev, tos = None, []
        for dd in days:
            w = tdvm[dd]
            if not isinstance(w, dict): return None
            vec = np.array([float(w.get(k, 0) or 0) for k in keys])
            s = vec.sum()
            if s > 0: vec = vec / s
            if prev is not None:
                tos.append(0.5 * np.abs(vec - prev).sum())
            prev = vec
        return round(float(np.mean(tos)), 3) if tos else None
    except Exception:
        return None


def _consistency(r):
    iss, oss = r.get("is_sharpe"), r.get("oos_sharpe")
    gap, isc = r.get("cagr_gap_pct"), r.get("is_cagr_pct")
    if None in (iss, oss, isc) or iss <= 0: return None
    ret = oss / iss
    cs = ret / 0.7 * 0.6 if ret < 0.7 else (1.0 if ret <= 1.3 else max(0.6, 1.3 / ret))
    cg = max(0.0, 1 - (abs(gap) / max(abs(isc), 15)) * 0.6) if gap is not None else 0.5
    return round(0.6 * cs + 0.4 * cg, 3)


def _metrics(pr):
    eq = np.cumprod(1 + pr); n = len(pr); cum = eq[-1] - 1; yrs = n / 252
    cagr = ((1 + cum) ** (1 / yrs) - 1) * 100 if yrs > 0 and (1 + cum) > 0 else float("nan")
    sd = pr.std(ddof=1)
    sharpe = (pr.mean() * 252) / (sd * 252 ** 0.5) if sd > 0 else float("nan")
    dn = pr[pr < 0]; dd_dev = np.sqrt((dn ** 2).sum() / n) if n else 0
    sortino = (pr.mean() * 252) / (dd_dev * 252 ** 0.5) if dd_dev > 0 else float("nan")
    rm = np.maximum.accumulate(eq); mdd = (eq / rm - 1).min() * 100
    return {"cagr_pct": round(cagr, 1), "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2), "maxdd_pct": round(mdd, 1),
            "vol_pct": round(sd * (252 ** 0.5) * 100, 1)}


def main():
    asof = date.today().isoformat()
    rows = [json.loads(l) for l in requests.get(LEDGER_URL, timeout=30).text.splitlines() if l.strip()]
    # screen: High conf, consistent, quality
    pool = []
    for r in rows:
        if r.get("oos_conf") != "High": continue
        c = _consistency(r)
        if c is None or c < 0.9: continue
        if (r.get("oos_sharpe") or 0) < 1.0 or (r.get("oos_cagr_pct") or 0) < 8 or (r.get("oos_maxdd_pct") or -99) < -25:
            continue
        pool.append({**r, "consistency": c})
    pool.sort(key=lambda r: r["consistency"], reverse=True)
    pool = pool[:POOL_MAX]
    print(f"[simple] screened pool: {len(pool)} High-conf consistent candidates")

    # enrich with node count + turnover + curve
    enr = []
    for i, r in enumerate(pool, 1):
        time.sleep(0.3)
        nodes, dec = fetch_nodecount(r["sym_id"])
        try:
            curve, tdvm = backtest_full(r["sym_id"], COMMON_START, asof)
        except Exception as e:
            print(f"  [{i}] {r['name'][:30]}: bt FAIL {type(e).__name__}", file=sys.stderr); continue
        curve = {d: v for d, v in curve.items() if d >= COMMON_START}
        if len(curve) < 60 or nodes is None:
            print(f"  [{i}] {r['name'][:30]}: skip (nodes={nodes}, days={len(curve)})", file=sys.stderr); continue
        r.update({"nodes": nodes, "decisions": dec, "turnover": turnover(tdvm), "_curve": curve})
        enr.append(r)
        print(f"  [{i}] nodes={nodes:>3} dec={dec:>2} turn={r['turnover']} "
              f"oosSh={r['oos_sharpe']} cons={r['consistency']}  {r['name'][:34]}")

    print(f"\n[simple] enriched {len(enr)} candidates")
    # pick the simplest / lowest-churn that clear quality (already gated), rank by nodes then turnover
    enr.sort(key=lambda r: (r["nodes"], r["turnover"] if r["turnover"] is not None else 1.0))
    simple = enr[:SIMPLE_KEEP]
    print(f"[simple] SIMPLE pool (fewest nodes, lowest churn) -> {len(simple)}:")
    for r in simple:
        print(f"    nodes={r['nodes']:>3} turn={r['turnover']} oosSh={r['oos_sharpe']} "
              f"maxDD={r['oos_maxdd_pct']}  {r['name'][:40]}")

    # align + correlation
    ids = [r["sym_id"] for r in simple]; names = [r["name"] for r in simple]
    common = sorted(set.intersection(*[set(r["_curve"]) for r in simple]))
    px = np.array([[r["_curve"][d] for r in simple] for d in common], dtype=float)
    rets = px[1:] / px[:-1] - 1
    corr = np.corrcoef(rets, rowvar=False)
    print(f"[simple] aligned {len(ids)} strategies on {len(common)} days")

    # EQUAL-weight combos (simplest scheme)
    N = len(ids); feas = []
    for k in range(PORT_MIN, PORT_MAX + 1):
        for combo in itertools.combinations(range(N), k):
            c = list(combo); sub = corr[np.ix_(c, c)]; off = sub[np.triu_indices(k, 1)]
            if off.max() > MAX_PAIR_CORR or off.mean() > MAX_AVG_CORR: continue
            port = rets[:, c].mean(axis=1)          # equal weight
            m = _metrics(port)
            if m["maxdd_pct"] < -MAXDD_LIMIT: continue
            calmar = m["cagr_pct"] / abs(m["maxdd_pct"]) if m["maxdd_pct"] < 0 else 0
            feas.append({"idx": c, "names": [names[j] for j in c],
                         "nodes": [simple[j]["nodes"] for j in c],
                         "turnover": [simple[j]["turnover"] for j in c],
                         "avg_nodes": round(float(np.mean([simple[j]["nodes"] for j in c])), 1),
                         "avg_corr": round(float(off.mean()), 2), "calmar": round(calmar, 2), **m})
    feas.sort(key=lambda p: (p["calmar"], p["sharpe"]), reverse=True)
    print(f"[simple] {len(feas)} feasible equal-weight portfolios (maxDD<-{MAXDD_LIMIT:.0f}%)")

    result = {"as_of": asof, "common_start": COMMON_START, "common_days": len(common),
              "weighting": "equal", "simple_pool": [{"id": r["sym_id"], "name": r["name"],
                "nodes": r["nodes"], "decisions": r["decisions"], "turnover": r["turnover"],
                "oos_sharpe": r["oos_sharpe"], "oos_cagr_pct": r["oos_cagr_pct"],
                "oos_maxdd_pct": r["oos_maxdd_pct"], "consistency": r["consistency"]} for r in simple],
              "corr_labels": names, "corr": [[round(float(corr[a][b]), 2) for b in range(N)] for a in range(N)],
              "top_portfolios": [{kk: vv for kk, vv in p.items() if kk != "idx"} for p in feas[:12]]}
    os.makedirs("data", exist_ok=True)
    json.dump(result, open(OUT, "w"), indent=2, ensure_ascii=False)

    print("\n=== TOP 6 SIMPLE PORTFOLIOS (equal weight, maxDD < {:.0f}%) ===".format(MAXDD_LIMIT))
    for p in feas[:6]:
        print(f"\nCAGR {p['cagr_pct']}%  MaxDD {p['maxdd_pct']}%  Sharpe {p['sharpe']}  "
              f"Sortino {p['sortino']}  Calmar {p['calmar']}  avgNodes {p['avg_nodes']}  (n={len(p['names'])})")
        for nm, nd, tu in zip(p["names"], p["nodes"], p["turnover"]):
            print(f"    {nd:>3} nodes  turn {tu}  {nm[:46]}")
    print(f"\n[simple] wrote {OUT}")


if __name__ == "__main__":
    main()
