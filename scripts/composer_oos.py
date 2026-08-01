#!/usr/bin/env python3
"""
composer_oos.py — matched live-vs-backtest capture for Composer symphonies.

WHY THIS EXISTS
    composer_ledger.py records since-funded TWR only, and its symphony-level history
    starts 2026-05-30. Neither supports "how has this symphony done out-of-sample over
    the last 12 months versus its own backtest." This module pulls both curves from
    Composer directly and stores them on a matched window.

READ-ONLY BY CONSTRUCTION
    GETs are whitelisted by prefix. Exactly one POST is permitted: the backtest
    simulation endpoint. Deploy / trading / dry-run paths raise. Do not relax.

USAGE
    python composer_oos.py sync                     # matched to each invested_since
    python composer_oos.py sync --start 2025-07-24  # force a common 12-month window
    python composer_oos.py sync --dry-run --limit 1 # smoke test, writes nothing
    python composer_oos.py report --min-days 120
"""
from __future__ import annotations
import os, sys, json, time, csv, argparse
from datetime import datetime, date, timezone, timedelta

import requests

BASE = "https://api.composer.trade"
OOS_PATH = os.environ.get("COMPOSER_OOS_PATH", "data/composer_oos.jsonl")
# SPY benchmark for the OOS tabs: adjusted-close buy-and-hold from the repo's
# committed OHLCV (same source as yfinance Adj Close, no slippage — a cleaner
# beta reference than a fee-laden Composer SPY backtest). Loaded once per run.
SPY_BENCH_PATH = os.environ.get("COMPOSER_OOS_SPY_CSV", "data/ohlcv/SPY.csv")

# A window must be at least this long to publish an ANNUALIZED figure (CAGR,
# Calmar, Martin). Below it, annualizing turns a 37-day run into "1,387,327%".
MIN_ANNUALIZE_YEARS = 1.0
# A window must be at least this many trading days to publish a Sharpe ratio.
MIN_SHARPE_DAYS = 60

# Backtest cost assumptions — match the house standard, not Composer's 1bp default.
SLIPPAGE = float(os.environ.get("COMPOSER_OOS_SLIPPAGE", 0.0005))   # 5 bps
CAPITAL = 100_000.0
THROTTLE = 1.05          # seconds; most endpoints are 1 req/sec
BACKTEST_THROTTLE = 0.05  # backtest endpoint allows 500 req/sec

_FORBIDDEN = ("/deploy/", "/trading/", "/dry-run")
_POST_ALLOWED = "/backtest"


def _headers():
    # Accept either the brief's names (COMPOSER_API_KEY_ID / COMPOSER_API_SECRET) or
    # this repo's existing convention (COMPOSER_KEY_ID / COMPOSER_KEY_SECRET, already
    # wired into signal_monitor.yml and used by composer_dry_run.py). Never hardcode.
    kid = os.environ.get("COMPOSER_API_KEY_ID") or os.environ.get("COMPOSER_KEY_ID")
    sec = os.environ.get("COMPOSER_API_SECRET") or os.environ.get("COMPOSER_KEY_SECRET")
    if not kid or not sec:
        raise RuntimeError(
            "Set COMPOSER_API_KEY_ID/COMPOSER_API_SECRET (or COMPOSER_KEY_ID/"
            "COMPOSER_KEY_SECRET). Generate at app.composer.trade -> Settings -> API "
            "Access. NOTE: only one key pair is active at a time; generating a new one "
            "revokes the existing key and will break anything already using it."
        )
    return {"x-api-key-id": kid, "authorization": f"Bearer {sec}",
            "accept": "application/json", "content-type": "application/json"}


def _request(method: str, path: str, **kw):
    if any(f in path for f in _FORBIDDEN):
        raise PermissionError(f"refusing state-changing path: {path}")
    if method == "POST" and not path.endswith(_POST_ALLOWED):
        raise PermissionError(f"POST only permitted to backtest, got: {path}")
    if method not in ("GET", "POST"):
        raise PermissionError(f"method not permitted: {method}")

    for attempt in range(4):
        r = requests.request(method, BASE + path, headers=_headers(), timeout=45, **kw)
        if r.status_code == 429:              # rate limited — back off
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"rate limited after retries: {path}")


def _get(path, params=None):
    time.sleep(THROTTLE)
    return _request("GET", path, params=params or {})


# ──────────────────────────────────────────────────────────────────────
# Epoch decoding.  Live series uses epoch MILLISECONDS; backtest
# dvm_capital keys are epoch DAYS. Auto-detect by magnitude.
# ──────────────────────────────────────────────────────────────────────
def _to_date(v) -> str:
    n = int(v)
    if n > 10_000_000_000:                       # milliseconds
        return datetime.fromtimestamp(n / 1000, tz=timezone.utc).date().isoformat()
    if n > 100_000_000:                          # seconds
        return datetime.fromtimestamp(n, tz=timezone.utc).date().isoformat()
    return (date(1970, 1, 1) + timedelta(days=n)).isoformat()   # days


# ──────────────────────────────────────────────────────────────────────
# API surface
# ──────────────────────────────────────────────────────────────────────
def list_accounts() -> list[dict]:
    return _get("/api/v0.1/accounts/list").get("accounts", [])


def symphony_stats(account_uuid: str) -> list[dict]:
    """Per-symphony live stats: value, sharpe_ratio, max_drawdown, invested_since,
    time_weighted_return, annualized_rate_of_return, name, tickers."""
    path = f"/api/v0.1/portfolio/accounts/{account_uuid}/symphony-stats-meta"
    return _get(path).get("symphonies", [])


def live_series(account_uuid: str, sym_id: str) -> dict[str, float]:
    """date -> deposit-adjusted value. Deposit-adjusted is the flow-neutral series;
    use it, not `series`, or contributions will masquerade as returns."""
    path = f"/api/v0.1/portfolio/accounts/{account_uuid}/symphonies/{sym_id}"
    d = _get(path)
    ms = d.get("epoch_ms") or []
    vals = d.get("deposit_adjusted_series") or d.get("series") or []
    return {_to_date(t): float(v) for t, v in zip(ms, vals) if v is not None}


def backtest(sym_id: str, start: str, end: str) -> tuple[dict[str, float], dict]:
    """Returns (date -> equity, stats). Backtest endpoint allows 500 req/sec."""
    time.sleep(BACKTEST_THROTTLE)
    body = {
        "capital": CAPITAL,
        "apply_reg_fee": True,
        "apply_taf_fee": True,
        "apply_cat_fee": True,
        "apply_subscription": "none",
        "backtest_version": "v2",
        "slippage_percent": SLIPPAGE,   # NOT percent: 0.0005 == 5 bps
        "start_date": start,
        "end_date": end,
        "broker": "ALPACA_WHITE_LABEL",
        "benchmark_tickers": ["SPY"],
    }
    d = _request("POST", f"/api/v0.1/symphonies/{sym_id}/backtest", json=body)
    dvm = d.get("dvm_capital") or {}
    curve = dvm.get(sym_id) or (next(iter(dvm.values()), {}) if dvm else {})
    return ({_to_date(k): float(v) for k, v in curve.items()}, d.get("stats") or {})


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────
def _curve_stats(curve: dict[str, float]) -> dict:
    """Cumulative return, annualized, Sharpe (arithmetic, rf=0, sqrt(252)), max DD."""
    if len(curve) < 2:
        return {}
    ds = sorted(curve)
    vals = [curve[d] for d in ds]
    rets = [vals[i] / vals[i - 1] - 1 for i in range(1, len(vals)) if vals[i - 1]]
    if not rets:
        return {}
    cum = vals[-1] / vals[0] - 1
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1) if n > 1 else 0.0
    sd = var ** 0.5
    peak, mdd = vals[0], 0.0
    for v in vals:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    yrs = n / 252
    return {
        "n_days": n,
        "cum_pct": round(cum * 100, 2),
        # Annualize only over >= 1 year; suppress Sharpe on windows < ~60 days.
        # Below those thresholds these are noise dressed up as results.
        "cagr_pct": round(((1 + cum) ** (1 / yrs) - 1) * 100, 2)
                    if (yrs >= MIN_ANNUALIZE_YEARS and (1 + cum) > 0) else None,
        "sharpe": round((mean * 252) / (sd * 252 ** 0.5), 2)
                  if (sd > 0 and n >= MIN_SHARPE_DAYS) else None,
        "maxdd_pct": round(mdd * 100, 2),
        "start": ds[0], "end": ds[-1],
    }


# ──────────────────────────────────────────────────────────────────────
# OOS confidence tier from the OOS window length. Shared by the watchlist
# and discover ledgers so both carry the same schema. The SE of an
# annualized Sharpe is ~ sqrt(252/n_days): short windows can't separate
# skill from noise. High >=3y, Good >=2y, Fair >=1y, Low >=6mo, Thin <6mo.
# ──────────────────────────────────────────────────────────────────────
def _confidence(oos_days):
    d = oos_days or 0
    se = round((252.0 / d) ** 0.5, 3) if d else None
    if d >= 756:
        return "High", 4, se
    if d >= 504:
        return "Good", 3, se
    if d >= 252:
        return "Fair", 2, se
    if d >= 126:
        return "Low", 1, se
    return "Thin", 0, se


# ──────────────────────────────────────────────────────────────────────
# SPY benchmark + OOS uncertainty fields (P0-1 / P0-2 / P0-3).
# Kept here, in one place, so the watchlist and discover row dicts stay
# schema-identical — the dashboard treats the two ledgers uniformly.
# ──────────────────────────────────────────────────────────────────────
_SPY_CACHE: dict[str, float] | None = None


def spy_series(path: str = SPY_BENCH_PATH) -> dict[str, float]:
    """date(iso) -> SPY adjusted close, loaded once and cached for the run."""
    global _SPY_CACHE
    if _SPY_CACHE is not None:
        return _SPY_CACHE
    out: dict[str, float] = {}
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                d = (row.get("Date") or "")[:10]
                v = row.get("Adj Close") or row.get("Close")
                if not d or v in (None, ""):
                    continue
                try:
                    out[d] = float(v)
                except ValueError:
                    pass
    except FileNotFoundError:
        print(f"[oos] SPY benchmark file not found: {path} — benchmark fields "
              f"will be null", file=sys.stderr)
    _SPY_CACHE = out
    return out


def _daily_returns(curve: dict[str, float]) -> dict[str, float]:
    """date -> simple return vs the prior available date in the curve."""
    ds = sorted(curve)
    out = {}
    for i in range(1, len(ds)):
        prev = curve[ds[i - 1]]
        if prev:
            out[ds[i]] = curve[ds[i]] / prev - 1
    return out


def _corr_beta(sym_curve: dict[str, float], bench_curve: dict[str, float]):
    """Pearson corr + OLS beta of symphony daily returns vs SPY, on the trading
    days both share. Returns (corr, beta, n_overlap); None,None if too few days."""
    sr, br = _daily_returns(sym_curve), _daily_returns(bench_curve)
    days = sorted(set(sr) & set(br))
    n = len(days)
    if n < 20:
        return None, None, n
    x = [br[d] for d in days]   # SPY (independent)
    y = [sr[d] for d in days]   # symphony (dependent)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    corr = sxy / (sxx * syy) ** 0.5 if (sxx > 0 and syy > 0) else None
    beta = sxy / sxx if sxx > 0 else None
    return (round(corr, 3) if corr is not None else None,
            round(beta, 3) if beta is not None else None, n)


def _raw_annualized(curve: dict[str, float]):
    """UNGATED annualized CAGR% and Sharpe from a curve — used only to size the
    CAGR standard error / low-power flag, where the whole point is to quantify
    short, high-vol windows that _curve_stats deliberately suppresses for display."""
    ds = sorted(curve)
    vals = [curve[d] for d in ds]
    rets = [vals[i] / vals[i - 1] - 1 for i in range(1, len(vals)) if vals[i - 1]]
    if len(rets) < 2:
        return None, None, len(rets)
    n = len(rets)
    mean = sum(rets) / n
    sd = (sum((r - mean) ** 2 for r in rets) / (n - 1)) ** 0.5
    cum = vals[-1] / vals[0] - 1
    yrs = n / 252
    cagr = ((1 + cum) ** (1 / yrs) - 1) * 100 if (yrs > 0 and (1 + cum) > 0) else None
    sharpe = (mean * 252) / (sd * 252 ** 0.5) if sd > 0 else None
    return cagr, sharpe, n


def oos_extra_fields(oos_curve: dict[str, float], out_stats: dict,
                     is_cagr_pct, cagr_gap_pct, spy: dict | None = None) -> dict:
    """All P0 fields for one symphony's OOS window, from its equity curve:
        P0-1 benchmark   spy_oos_cagr_pct, spy_oos_maxdd_pct, oos_alpha_pct,
                         oos_excess_sharpe
        P0-2 corr/beta   spy_corr_oos, spy_beta_oos, stream_bucket
        P0-3 z-score     oos_vol_pct, cagr_se_pct, gap_z, low_power
        P2-7 ranking     sharpe_lcb (OOS Sharpe shrunk by 1 SE for sample size)
    `out_stats` is _curve_stats(oos_curve) (already computed by the caller)."""
    spy = spy if spy is not None else spy_series()
    f = {"spy_oos_cagr_pct": None, "spy_oos_maxdd_pct": None,
         "oos_alpha_pct": None, "oos_excess_sharpe": None,
         "spy_corr_oos": None, "spy_beta_oos": None, "stream_bucket": None,
         "oos_vol_pct": None, "cagr_se_pct": None, "gap_z": None, "low_power": None,
         "sharpe_lcb": None}
    if not oos_curve:
        return f
    ds = sorted(oos_curve)
    w0, w1 = ds[0], ds[-1]

    # ── P0-1: SPY over the identical [oos_start, oos_end] window ──
    bench = {d: v for d, v in spy.items() if w0 <= d <= w1}
    bstat = _curve_stats(bench) if len(bench) >= 2 else {}
    f["spy_oos_cagr_pct"] = bstat.get("cagr_pct")
    f["spy_oos_maxdd_pct"] = bstat.get("maxdd_pct")
    oc, sc = out_stats.get("cagr_pct"), bstat.get("cagr_pct")
    if oc is not None and sc is not None:
        f["oos_alpha_pct"] = round(oc - sc, 2)
    osh, bsh = out_stats.get("sharpe"), bstat.get("sharpe")
    if osh is not None and bsh is not None:
        f["oos_excess_sharpe"] = round(osh - bsh, 2)

    # ── P0-2: correlation + beta vs SPY daily returns ──
    corr, beta, _ = _corr_beta(oos_curve, bench)
    f["spy_corr_oos"], f["spy_beta_oos"] = corr, beta
    if corr is not None:
        a = abs(corr)
        f["stream_bucket"] = ("equity" if a > 0.60 else
                              "mixed" if a >= 0.30 else "diversifier")

    # ── P0-3: gap z-score + low-power flag ──
    # SE uses the UNGATED annualized figures so short/high-vol windows are
    # flagged (low_power) rather than silently dropped. gap_z, however, uses the
    # published (gated) gap, which is None below 1y — you cannot annualize a gap
    # over a sub-year window without it exploding.
    raw_cagr, raw_sharpe, _ = _raw_annualized(oos_curve)
    if raw_cagr is not None and raw_sharpe and raw_sharpe > 0:
        vol = raw_cagr / raw_sharpe
        f["oos_vol_pct"] = round(vol, 2)
        nd = out_stats.get("n_days") or 0
        if nd > 0:
            se = vol / (nd / 252.0) ** 0.5
            f["cagr_se_pct"] = round(se, 2)
            if cagr_gap_pct is not None and se > 0:
                f["gap_z"] = round(cagr_gap_pct / se, 2)
            if is_cagr_pct is not None:
                f["low_power"] = bool(se > abs(is_cagr_pct))

    # ── P2-7: sample-size-shrunk Sharpe for ranking ──
    # The SE of an annualized Sharpe is ≈ √(252/n). Subtracting one SE gives a
    # lower-confidence bound that penalizes short OOS windows: a 1.45-Sharpe /
    # 900d strategy (lcb ≈ 0.92) outranks a 1.60-Sharpe / 200d one (lcb ≈ 0.48).
    sh = out_stats.get("sharpe")
    nd = out_stats.get("n_days") or 0
    if sh is not None and nd > 0:
        f["sharpe_lcb"] = round(sh - (252.0 / nd) ** 0.5, 2)
    return f


# ──────────────────────────────────────────────────────────────────────
# Sync
# ──────────────────────────────────────────────────────────────────────
def sync(path: str = OOS_PATH, start: str | None = None,
         dry_run: bool = False, limit: int | None = None) -> int:
    asof = date.today().isoformat()
    now = datetime.now(timezone.utc).isoformat()
    end = asof
    rows, seen = [], 0

    for acct in list_accounts():
        auuid = acct["account_uuid"]
        atype = acct.get("account_type") or "UNKNOWN"
        for s in symphony_stats(auuid):
            if limit is not None and seen >= limit:
                break
            seen += 1
            sid, nm = s.get("id"), s.get("name")
            if not sid:
                continue
            win_start = start or s.get("invested_since") or "2020-01-01"
            try:
                live = live_series(auuid, sid)
                bt, btstats = backtest(sid, win_start, end)
            except Exception as e:
                print(f"[oos] {nm}: {type(e).__name__}: {e}", file=sys.stderr)
                continue

            # Restrict both curves to their overlapping trading days.
            common = sorted(set(live) & set(bt))
            common = [d for d in common if d >= win_start]
            if len(common) < 2:
                print(f"[oos] {nm}: only {len(common)} overlapping days, skipping",
                      file=sys.stderr)
                continue
            lv = {d: live[d] for d in common}
            bv = {d: bt[d] for d in common}
            ls, bs = _curve_stats(lv), _curve_stats(bv)

            rows.append({
                "date": asof, "asof_utc": now, "scope": "oos_summary",
                "account": atype, "sym_id": sid, "name": nm,
                "window_start": common[0], "window_end": common[-1],
                "n_days": ls.get("n_days"),
                "live_cum_pct": ls.get("cum_pct"), "bt_cum_pct": bs.get("cum_pct"),
                "gap_pct": (round(ls["cum_pct"] - bs["cum_pct"], 2)
                            if ls.get("cum_pct") is not None
                            and bs.get("cum_pct") is not None else None),
                "live_sharpe": ls.get("sharpe"), "bt_sharpe": bs.get("sharpe"),
                "live_maxdd_pct": ls.get("maxdd_pct"), "bt_maxdd_pct": bs.get("maxdd_pct"),
                "bt_ann_pct": round((btstats.get("annualized_rate_of_return") or 0) * 100, 2),
                "bt_win_rate": btstats.get("win_rate"),
                "invested_since": s.get("invested_since"),
                "value": s.get("value"),
            })
            for d in common:
                rows.append({"date": d, "asof_utc": now, "scope": "oos_daily",
                             "sym_id": sid, "live": round(lv[d], 4), "bt": round(bv[d], 4)})

    if dry_run:
        for r in rows:
            if r["scope"] == "oos_summary":
                print(json.dumps(r, indent=2))
        return 0

    # Idempotent: drop any prior rows for today's summary + these sym_ids, then append.
    kill_ids = {r["sym_id"] for r in rows if r["scope"] == "oos_summary"}
    keep = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("scope") == "oos_summary" and r.get("date") == asof \
                        and r.get("sym_id") in kill_ids:
                    continue
                if r.get("scope") == "oos_daily" and r.get("sym_id") in kill_ids:
                    continue
                keep.append(r)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in keep + rows:
            f.write(json.dumps(r) + "\n")
    print(f"[oos] wrote {len(rows)} rows ({len(kill_ids)} symphonies) -> {path}")
    return len(rows)


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────
def report(path: str = OOS_PATH, min_days: int = 0) -> list[dict]:
    if not os.path.exists(path):
        return []
    latest: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("scope") != "oos_summary":
                continue
            if (r.get("n_days") or 0) < min_days:
                continue
            k = r["sym_id"]
            if k not in latest or r["date"] >= latest[k]["date"]:
                latest[k] = r
    out = sorted(latest.values(), key=lambda r: (r.get("gap_pct") is None,
                                                 r.get("gap_pct") or 0))
    hdr = f"{'n':>4} {'live%':>8} {'bt%':>8} {'gap':>8} {'lSh':>5} {'bSh':>5}  name"
    print(hdr); print("-" * len(hdr))
    for r in out:
        print(f"{r.get('n_days') or 0:4d} {r.get('live_cum_pct') or 0:8.1f} "
              f"{r.get('bt_cum_pct') or 0:8.1f} {r.get('gap_pct') or 0:8.1f} "
              f"{r.get('live_sharpe') or 0:5.2f} {r.get('bt_sharpe') or 0:5.2f}  "
              f"{(r.get('name') or '')[:44]}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sync", "report"])
    ap.add_argument("--start", default=None, help="force common window, e.g. 2025-07-24")
    ap.add_argument("--path", default=OOS_PATH)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-days", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "sync":
        sync(a.path, start=a.start, dry_run=a.dry_run, limit=a.limit)
    else:
        report(a.path, min_days=a.min_days)
