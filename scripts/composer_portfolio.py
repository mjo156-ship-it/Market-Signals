#!/usr/bin/env python3
"""
composer_portfolio.py — build a diversified 5-7 symphony portfolio from the
high-confidence, OOS-consistent Discover candidates.

Inputs a shortlist (data/portfolio_shortlist.json) of candidate symphonies that
already passed the consistency screen (High OOS confidence, OOS Sharpe holding vs
in-sample, shallow drawdown). This script does the part that needs actual equity
curves — portfolio drawdown depends on how the strategies' drawdowns overlap:

  1. Backtest each candidate over a COMMON out-of-sample window (default the
     latest freeze date among the shortlist, so the window is genuinely OOS for
     every strategy), producing daily equity curves.
  2. Align on common trading days -> daily-returns matrix; compute the pairwise
     correlation matrix (the real diversification signal, since Composer's asset-
     class labels are too coarse).
  3. Search every combo of PORT_MIN..PORT_MAX names, inverse-volatility weighted,
     and compute portfolio CAGR / annualized vol / Sharpe / Sortino / MaxDD.
  4. Keep only combos that satisfy the hard constraints — portfolio MaxDD shallower
     than MAXDD_LIMIT and genuine diversification (bounded average & pairwise
     correlation) — then rank by Calmar (CAGR / |MaxDD|), i.e. return per unit of
     drawdown, the "attractive return while keeping MaxDD low" objective.

Read-only vs Composer: uses composer_oos.backtest (the one permitted POST).
Writes data/portfolio_analysis.json and prints a ranked report.
"""
from __future__ import annotations
import os, sys, json, time, itertools
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos as oos

SHORTLIST = os.environ.get("PORTFOLIO_SHORTLIST", "data/portfolio_shortlist.json")
OUT = os.environ.get("PORTFOLIO_OUT", "data/portfolio_analysis.json")
MAXDD_LIMIT = float(os.environ.get("PORTFOLIO_MAXDD_LIMIT", "20"))     # percent, portfolio
PORT_MIN = int(os.environ.get("PORTFOLIO_MIN", "5"))
PORT_MAX = int(os.environ.get("PORTFOLIO_MAX", "7"))
MAX_PAIR_CORR = float(os.environ.get("PORTFOLIO_MAX_PAIR_CORR", "0.85"))  # no two near-duplicates
MAX_AVG_CORR = float(os.environ.get("PORTFOLIO_MAX_AVG_CORR", "0.60"))    # basket must be diversified


def _backtest_resilient(sid, start, end, tries=5):
    for k in range(tries):
        try:
            return oos.backtest(sid, start, end)
        except RuntimeError as e:
            if "rate limited" in str(e).lower() and k < tries - 1:
                time.sleep(6 * (k + 1)); continue
            raise


def _metrics(port_ret):
    """CAGR%, ann vol%, Sharpe, Sortino, MaxDD% from a daily return series."""
    eq = np.cumprod(1.0 + port_ret)
    n = len(port_ret)
    cum = eq[-1] / 1.0 - 1.0
    yrs = n / 252.0
    cagr = ((1 + cum) ** (1 / yrs) - 1) * 100 if (yrs > 0 and (1 + cum) > 0) else float("nan")
    vol = port_ret.std(ddof=1) * (252 ** 0.5) * 100
    mean = port_ret.mean()
    sharpe = (mean * 252) / (port_ret.std(ddof=1) * (252 ** 0.5)) if port_ret.std(ddof=1) > 0 else float("nan")
    downside = port_ret[port_ret < 0]
    dd_dev = (np.sqrt((downside ** 2).sum() / n)) if n else 0.0
    sortino = (mean * 252) / (dd_dev * (252 ** 0.5)) if dd_dev > 0 else float("nan")
    runmax = np.maximum.accumulate(eq)
    maxdd = (eq / runmax - 1.0).min() * 100
    return {"cagr_pct": round(cagr, 1), "vol_pct": round(vol, 1),
            "sharpe": round(sharpe, 2), "sortino": round(sortino, 2),
            "maxdd_pct": round(maxdd, 1)}


def main():
    asof = date.today().isoformat()
    shortlist = json.load(open(SHORTLIST))
    common_start = os.environ.get("PORTFOLIO_COMMON_START") or max(s["oos_date"] for s in shortlist)
    print(f"[portfolio] {len(shortlist)} candidates · common OOS window {common_start} -> {asof}")

    # 1) backtest each candidate over the common window
    curves, names, ids, keep = {}, [], [], []
    for i, s in enumerate(shortlist, 1):
        time.sleep(0.4)
        try:
            curve, _ = _backtest_resilient(s["id"], common_start, asof)
        except Exception as e:
            print(f"  [{i}] {s['name'][:32]}: FAIL {type(e).__name__}", file=sys.stderr); continue
        curve = {d: v for d, v in curve.items() if d >= common_start}
        if len(curve) < 60:
            print(f"  [{i}] {s['name'][:32]}: too short ({len(curve)}d)", file=sys.stderr); continue
        curves[s["id"]] = curve
        keep.append(s); ids.append(s["id"]); names.append(s["name"])
        print(f"  [{i}] {s['name'][:38]}: {len(curve)}d")

    if len(keep) < PORT_MIN:
        print("[portfolio] not enough candidates backtested", file=sys.stderr); sys.exit(1)

    # 2) align on common trading days -> returns matrix (assets in columns)
    common_days = sorted(set.intersection(*[set(curves[i]) for i in ids]))
    print(f"[portfolio] {len(ids)} strategies aligned on {len(common_days)} common trading days")
    px = np.array([[curves[i][d] for i in ids] for d in common_days], dtype=float)  # (T, N)
    rets = px[1:] / px[:-1] - 1.0                                                    # (T-1, N)
    vols = rets.std(axis=0, ddof=1)
    corr = np.corrcoef(rets, rowvar=False)

    # 3) search combos, inverse-vol weighted
    N = len(ids)
    idx = list(range(N))
    feasible = []
    for k in range(PORT_MIN, PORT_MAX + 1):
        for combo in itertools.combinations(idx, k):
            c = list(combo)
            sub = corr[np.ix_(c, c)]
            off = sub[np.triu_indices(k, 1)]
            avg_c, max_c = float(off.mean()), float(off.max())
            if max_c > MAX_PAIR_CORR or avg_c > MAX_AVG_CORR:
                continue
            w = 1.0 / vols[c]
            w = w / w.sum()
            port = rets[:, c] @ w
            m = _metrics(port)
            if m["maxdd_pct"] < -MAXDD_LIMIT:               # drawdown too deep
                continue
            calmar = m["cagr_pct"] / abs(m["maxdd_pct"]) if m["maxdd_pct"] < 0 else float("nan")
            feasible.append({"ids": [ids[j] for j in c], "names": [names[j] for j in c],
                             "weights": [round(float(x), 3) for x in w],
                             "avg_corr": round(avg_c, 2), "max_corr": round(max_c, 2),
                             "calmar": round(calmar, 2), **m})

    feasible.sort(key=lambda p: (p["calmar"], p["sharpe"]), reverse=True)
    print(f"[portfolio] {len(feasible)} feasible portfolios (maxDD>-{MAXDD_LIMIT:.0f}%, "
          f"avgCorr<={MAX_AVG_CORR}, maxPairCorr<={MAX_PAIR_CORR})")

    result = {
        "as_of": asof, "common_start": common_start, "common_days": len(common_days),
        "maxdd_limit_pct": MAXDD_LIMIT,
        "candidates": [{"id": s["id"], "name": s["name"], "oos_date": s["oos_date"],
                        "vol_pct": round(float(vols[ids.index(s["id"])] * (252 ** 0.5) * 100), 1)}
                       for s in keep],
        "corr_labels": names,
        "corr": [[round(float(corr[a][b]), 2) for b in range(N)] for a in range(N)],
        "top_portfolios": feasible[:15],
    }
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(result, open(OUT, "w"), indent=2)

    print("\n=== TOP 8 PORTFOLIOS (by Calmar, maxDD < {:.0f}%) ===".format(MAXDD_LIMIT))
    for p in feasible[:8]:
        print(f"\nCAGR {p['cagr_pct']}%  MaxDD {p['maxdd_pct']}%  Sharpe {p['sharpe']}  "
              f"Sortino {p['sortino']}  Calmar {p['calmar']}  avgCorr {p['avg_corr']}  (n={len(p['ids'])})")
        for nm, w in zip(p["names"], p["weights"]):
            print(f"    {w*100:4.0f}%  {nm[:52]}")
    print(f"\n[portfolio] wrote {OUT}")


if __name__ == "__main__":
    main()
