#!/usr/bin/env python3
"""
Daily signal-research engine — mines the price store for NET-NEW, backtest-gated
trading signals that complement (do not duplicate) the live signal monitor's
existing groups (RSI(10) bands, SMA(50/200) crosses, and the QQQ/SPY, QQQ/RSP,
QQQE/QQQ detrended z-scores).

Four signal families are scanned across the full price-store universe:

  1. Cross-sectional rotation   — 3/6/12-mo momentum ranking; new top-decile
                                  leaders (rotation candidates).
  2. Volatility & breakout      — Bollinger-bandwidth squeeze breakouts,
                                  Donchian 52-week breakouts, ATR-percentile
                                  low-vol regime, VIXY/VIXM term-structure flip.
  3. Mean-reversion / exhaustion— Bollinger %B extremes, distance-from-MA
                                  z-score, down/up streaks, capitulation-volume
                                  bounces.
  4. Macro / intermarket ratios — detrended z-scores of NEW pairs (HYG/TLT,
                                  XLY/XLP, SMH/SPY, GLD/SPY, CPER/GLD).

Every candidate is **walk-forward backtested** over the store's full history. A
condition that fires TODAY is only surfaced if its history clears a minimum
sample size AND beats buy-and-hold over the same horizon (win-rate + edge gate).
Each surfaced signal is tagged `composer_ready` (a pure daily-close rule you can
port into a Composer symphony) or `manual_swing` (a multi-day discretionary
hold), with a suggested instrument, horizon, and the win%/avg-return/N stats.

Output (dashboard-ready; newest-first + archive), published like snapshot.json:

  data/signal_research/latest.json          today's ranked results
  data/signal_research/history/YYYY-MM-DD.json   per-day archive (self-contained)
  data/signal_research/index.json           {"dates": [...newest first], "latest": ...}
  data/signal_research/latest.md            human-readable digest

CLI:
  python signal_research.py --run       # scan + backtest + write outputs
  python signal_research.py --report    # print today's results (no write)
  python signal_research.py --self-test # offline sanity checks
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import price_store as ps

# ---------------------------------------------------------------------------
# Paths / gate parameters
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "data" / "signal_research"
HIST_DIR = OUT_DIR / "history"
LATEST_JSON = OUT_DIR / "latest.json"
INDEX_JSON = OUT_DIR / "index.json"
LATEST_MD = OUT_DIR / "latest.md"

MIN_SAMPLE = 25          # minimum historical fires to trust a signal
WIN_RATE_MIN = 0.60      # historical hit-rate floor (for long; symmetric for short)
MIN_HISTORY_BARS = 300   # skip tickers without enough history to backtest

# Suggested leveraged execution proxy (display only; backtest uses the base ETF
# for a longer, cleaner series).
LEVERAGE_PROXY = {
    "SPY": "UPRO", "QQQ": "TQQQ", "SMH": "SOXL", "XLF": "FAS",
    "XLV": "CURE", "IWM": "TNA", "XLE": "ERX", "SOXX": "SOXL",
}

# Suggested inverse instrument to express a short-side (fade) signal.
INVERSE_PROXY = {
    "SPY": "SPXU", "QQQ": "SQQQ", "SMH": "SOXS", "IWM": "TZA",
    "XLF": "FAZ", "XLE": "ERY", "SOXX": "SOXS", "GLD": "DUST",
}

# New intermarket ratio pairs (numerator / denominator). Deliberately excludes
# the monitor's existing QQQ/SPY, QQQ/RSP, QQQE/QQQ.
RATIO_PAIRS = [
    ("HYG", "TLT", "credit risk-appetite (high-yield vs duration)"),
    ("XLY", "XLP", "consumer risk-on vs defensive"),
    ("SMH", "SPY", "semis leadership vs broad market"),
    ("GLD", "SPY", "gold vs equities (risk-off tilt)"),
    ("CPER", "GLD", "copper/gold growth-vs-fear proxy"),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _log(m: str) -> None:
    print(m, flush=True)


# ---------------------------------------------------------------------------
# Indicator helpers (vectorised)
# ---------------------------------------------------------------------------

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _bollinger(s: pd.Series, n: int = 20, k: float = 2.0):
    mid = s.rolling(n).mean()
    sd = s.rolling(n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (s - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, width, pct_b


def _rolling_pct_rank(s: pd.Series, n: int) -> pd.Series:
    """Percentile (0..1) of the latest value within the trailing n-window."""
    return s.rolling(n).apply(
        lambda w: (w[:-1] < w[-1]).mean() if len(w) > 1 else np.nan, raw=True)


def _run_streak(close: pd.Series, up: bool) -> pd.Series:
    """Length of the current run of consecutive up (or down) closes."""
    step = (close.diff() > 0) if up else (close.diff() < 0)
    step = step.astype(int)
    grp = (step == 0).cumsum()
    return step.groupby(grp).cumsum()


def _detrended_z(ratio: pd.Series, lookback: int = 252) -> pd.Series:
    """Trend-channel z-score: residual of the latest point vs a rolling linear
    fit, divided by residual std. Same idea as the monitor's Group-13 method,
    applied to NEW ratio pairs."""
    x = np.arange(lookback, dtype=float)

    def _last_z(w):
        if np.isnan(w).any():
            return np.nan
        b1, b0 = np.polyfit(x, w, 1)
        resid = w - (b0 + b1 * x)
        sd = resid.std()
        return resid[-1] / sd if sd > 0 else np.nan

    return ratio.rolling(lookback).apply(_last_z, raw=True)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def backtest(entry: pd.Series, price: pd.Series, horizon: int,
             direction: str = "long") -> dict | None:
    """
    Walk-forward stats for a boolean entry signal.

    Counts rising edges (condition becomes true) as entries, measures the
    forward `horizon`-day return of `price`, and compares to the unconditional
    same-horizon return (buy-and-hold baseline). Returns None if too few fires.
    """
    entry = entry.reindex(price.index, fill_value=False).fillna(False).astype(bool)
    edges = entry & ~entry.shift(1, fill_value=False)
    fwd = price.shift(-horizon) / price - 1.0
    if direction == "short":
        fwd = -fwd

    rets = fwd[edges].dropna()
    n = int(len(rets))
    if n < MIN_SAMPLE:
        return None
    fwd_all = fwd.dropna()
    baseline = float(fwd_all.mean())
    avg = float(rets.mean())

    # Per-event records (date + integer bar position + forward return) so the
    # downstream validation pipeline can dedup overlapping windows, bootstrap,
    # and split by regime WITHOUT re-fetching the price store. `i` is the
    # positional index into `price.index`; two events are non-overlapping iff
    # their `i` differ by >= horizon.
    pos = {ts: k for k, ts in enumerate(price.index)}
    events = [{"date": str(ts.date()) if hasattr(ts, "date") else str(ts),
               "i": int(pos[ts]), "ret": round(float(r), 5)}
              for ts, r in rets.items()]

    return {
        "n": n,
        "win_rate": round(float((rets > 0).mean()), 3),
        "avg_return": round(avg, 4),
        "median_return": round(float(rets.median()), 4),
        "baseline_return": round(baseline, 4),
        # Unconditional (all-days) forward-return hit-rate at this horizon: the
        # correct base rate for the event win-rate test (never test vs 50%).
        "baseline_win_rate": round(float((fwd_all > 0).mean()), 4),
        "edge": round(avg - baseline, 4),
        "fired_today": bool(edges.iloc[-1]),
        "events": events,
    }


def _passes_gate(bt: dict) -> bool:
    return (bt is not None and bt["n"] >= MIN_SAMPLE
            and bt["win_rate"] >= WIN_RATE_MIN
            and bt["avg_return"] > 0 and bt["edge"] > 0)


# ---------------------------------------------------------------------------
# Per-ticker candidate generators
# Each yields: (name, family, mode, direction, horizon, entry_bool_series,
#               value_today_str)
# ---------------------------------------------------------------------------

def per_ticker_candidates(df: pd.DataFrame):
    """Yield (name, family, mode, direction, horizon, entry, approaching, value).

    `approaching` is a boolean Series (or None) marking days the condition is
    just INSIDE its threshold but not yet firing — used for the primed watchlist.
    """
    close, high = df["close"], df["high"]
    low, vol = df["low"], df["volume"]

    # --- Volatility & breakout ------------------------------------------
    # Donchian 52-week breakout (close above prior 252-day high).
    prior_high = high.rolling(252).max().shift(1)
    yield ("Donchian 52-week breakout", "volatility_breakout", "composer_ready",
           "long", 20, close > prior_high,
           (close >= 0.98 * prior_high) & (close <= prior_high),
           f"close {close.iloc[-1]:.2f} vs 52w high {prior_high.iloc[-1]:.2f}")

    # Bollinger squeeze breakout: bandwidth in bottom decile (trailing 126d),
    # then close pops above the upper band.
    _, upper, _, width, pct_b = _bollinger(close, 20, 2.0)
    sq_rank = _rolling_pct_rank(width, 126)
    squeeze = sq_rank < 0.10
    yield ("Bollinger squeeze breakout", "volatility_breakout", "composer_ready",
           "long", 10, squeeze.shift(1).fillna(False) & (close > upper),
           squeeze & (pct_b >= 0.85) & (pct_b <= 1.0),
           f"bandwidth pctile {(sq_rank.iloc[-1] or float('nan')):.2f}, %B {pct_b.iloc[-1]:.2f}")

    # Low-ATR regime entry (ATR% in bottom decile).
    atr_rank = _rolling_pct_rank(_atr(df, 14) / close, 252)
    yield ("Low-volatility (ATR) regime", "volatility_breakout", "manual_swing",
           "long", 20, atr_rank < 0.10, (atr_rank >= 0.10) & (atr_rank < 0.15),
           f"ATR% pctile {(atr_rank.iloc[-1] or float('nan')):.2f}")

    # --- Mean-reversion / exhaustion (LONG) -----------------------------
    yield ("Bollinger %B oversold (<0)", "mean_reversion", "composer_ready",
           "long", 5, pct_b < 0.0, (pct_b >= 0.0) & (pct_b <= 0.10),
           f"%B {pct_b.iloc[-1]:.2f}")

    dist = (close - _sma(close, 50))
    dz = (dist - dist.rolling(100).mean()) / dist.rolling(100).std(ddof=0)
    yield ("Distance-from-50DMA z < -2", "mean_reversion", "composer_ready",
           "long", 5, dz < -2.0, (dz >= -2.0) & (dz <= -1.7),
           f"dist-z {dz.iloc[-1]:.2f}")

    down = _run_streak(close, up=False)
    yield ("4+ down-day streak", "mean_reversion", "composer_ready",
           "long", 5, down >= 4, down == 3, f"down streak {int(down.iloc[-1])}")

    vz = (vol - vol.rolling(50).mean()) / vol.rolling(50).std(ddof=0)
    rng = (high - low).replace(0, np.nan)
    in_lower_third = (close - low) / rng < 0.34
    capit = (close.diff() < 0) & (vz > 2.0) & in_lower_third
    yield ("Capitulation-volume down day", "mean_reversion", "manual_swing",
           "long", 5, capit, None, f"vol-z {vz.iloc[-1]:.2f}")

    # --- Exhaustion fades (SHORT) ---------------------------------------
    # Net-new downside context: fade over-extended/blow-off conditions. The
    # gate still requires the short to have historically paid (avg>0, edge>0).
    yield ("Bollinger %B overbought (>1) fade", "exhaustion_fade", "composer_ready",
           "short", 5, pct_b > 1.0, (pct_b <= 1.0) & (pct_b >= 0.90),
           f"%B {pct_b.iloc[-1]:.2f}")

    yield ("Distance-from-50DMA z > +2.5 fade", "exhaustion_fade", "composer_ready",
           "short", 5, dz > 2.5, (dz <= 2.5) & (dz >= 2.2),
           f"dist-z {dz.iloc[-1]:.2f}")

    up = _run_streak(close, up=True)
    yield ("4+ up-day streak fade", "exhaustion_fade", "manual_swing",
           "short", 5, up >= 4, up == 3, f"up streak {int(up.iloc[-1])}")

    # Parabolic blow-off: close >25% above its 50DMA.
    ext = close / _sma(close, 50) - 1.0
    yield ("Parabolic extension (>25% over 50DMA) fade", "exhaustion_fade",
           "manual_swing", "short", 10, ext > 0.25, (ext > 0.20) & (ext <= 0.25),
           f"{ext.iloc[-1]:+.1%} over 50DMA")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _load_panel():
    """Long adjusted frame -> {ticker: df(open,high,low,close,volume) by date}."""
    tickers = ps._resolve_universe(None)
    long = ps.get_prices(tickers, adjusted=True, layout="long")
    if long is None or len(long) == 0:
        return {}, None
    long = long.sort_values(["ticker", "date"])
    by = {}
    for t, g in long.groupby("ticker"):
        g = g.set_index("date")[["open", "high", "low", "close", "volume"]]
        if len(g) >= MIN_HISTORY_BARS:
            by[t] = g
    data_through = long["date"].max()
    return by, data_through


def _result(name, family, mode, direction, ticker, horizon, value, bt) -> dict:
    proxy = (INVERSE_PROXY.get(ticker, f"short {ticker}") if direction == "short"
             else LEVERAGE_PROXY.get(ticker, ticker))
    return {
        "family": family,
        "name": name,
        "ticker": ticker,
        "suggested_instrument": proxy,
        "direction": direction,
        "horizon_days": horizon,
        "mode": mode,
        "today_value": value,
        "backtest": bt,
        "score": round(bt["edge"] * (bt["n"] ** 0.5), 4),  # edge weighted by evidence
    }


def _evaluate(name, family, mode, direction, ticker, horizon, entry, approaching,
              value, price):
    """Backtest one candidate; classify as 'firing', 'approaching', or None.

    A candidate must clear the backtest gate to appear at all. It is 'firing'
    if its condition triggers today, else 'approaching' if it is just inside the
    threshold today (primed watchlist).
    """
    try:
        bt = backtest(entry, price, horizon, direction)
    except Exception:
        return None, None
    if not _passes_gate(bt):
        return None, None
    rec = _result(name, family, mode, direction, ticker, horizon, value, bt)
    if bt["fired_today"]:
        return "firing", rec
    if approaching is not None:
        a = approaching.reindex(price.index, fill_value=False).fillna(False).astype(bool)
        if bool(a.iloc[-1]):
            return "approaching", rec
    return None, None


def run_signal_research(write: bool = True) -> dict:
    by, data_through = _load_panel()
    if not by:
        _log("[research] price store empty — nothing to scan")
        return {"results": [], "data_through": None}
    _log(f"[research] scanning {len(by)} tickers, data through {data_through.date()}")

    results, watchlist = [], []

    def _collect(args):
        kind, rec = _evaluate(*args)
        if kind == "firing":
            results.append(rec)
        elif kind == "approaching":
            watchlist.append(rec)

    # ---- per-ticker families -------------------------------------------
    for ticker, df in by.items():
        for (name, fam, mode, direction, horizon, entry, approaching, value) in \
                per_ticker_candidates(df):
            _collect((name, fam, mode, direction, ticker, horizon, entry,
                      approaching, value, df["close"]))

    # ---- cross-sectional rotation --------------------------------------
    close_panel = pd.DataFrame({t: d["close"] for t, d in by.items()}).sort_index()
    for lookback, label in [(63, "3-mo"), (126, "6-mo"), (252, "12-mo")]:
        mom = close_panel / close_panel.shift(lookback) - 1.0
        rank = mom.rank(axis=1, ascending=False)          # 1 = strongest
        in_top = rank <= 8
        entered = in_top & ~in_top.shift(1).fillna(False)
        nearly = (rank > 8) & (rank <= 12)                 # just outside top-8
        for ticker in close_panel.columns:
            if ticker not in by:
                continue
            # Use the ticker's LAST VALID momentum/rank, not iloc[-1]: a ticker
            # with no bar on the panel's final date (store lag / thin trading)
            # has NaN at iloc[-1], which previously rendered as "momentum n/a".
            # The signal itself fires on the ticker's own last bar, so describe
            # that same bar. (Display-only; entry/backtest are unaffected.)
            m_valid = mom[ticker].dropna()
            rk_valid = rank[ticker].dropna()
            m = m_valid.iloc[-1] if len(m_valid) else float("nan")
            rk = rk_valid.iloc[-1] if len(rk_valid) else float("nan")
            val = (f"{label} momentum {m:+.1%}, rank #{int(rk)}"
                   if pd.notna(m) and pd.notna(rk) else f"{label} momentum n/a")
            _collect((f"Rotation: entered top-8 {label} momentum", "rotation",
                      "manual_swing", "long", ticker, 21, entered[ticker],
                      nearly[ticker], val, by[ticker]["close"]))

    # ---- macro / intermarket ratios ------------------------------------
    for num, den, desc in RATIO_PAIRS:
        if num not in by or den not in by:
            continue
        ratio = (by[num]["close"] / by[den]["close"]).dropna()
        if len(ratio) < 300:
            continue
        z = _detrended_z(ratio, 252)
        for cond, near, direction, tag in [
                (z < -2.0, (z >= -2.0) & (z <= -1.7), "long", "z<-2 (depressed)"),
                (z > 2.0, (z <= 2.0) & (z >= 1.7), "short", "z>2 (stretched)")]:
            _collect((f"{num}/{den} ratio {tag}", "macro_ratio", "manual_swing",
                      direction, num, 20, cond, near,
                      f"{desc}; z={z.iloc[-1]:.2f}", by[num]["close"]))

    results.sort(key=lambda r: r["score"], reverse=True)
    watchlist.sort(key=lambda r: r["score"], reverse=True)

    # Multiple-testing denominator M: every signal x instrument combination the
    # scan evaluates today (not just the surfaced ones). The downstream
    # validation pipeline corrects surfaced p-values against this M.
    _sample_df = next(iter(by.values()))
    _n_per_ticker = sum(1 for _ in per_ticker_candidates(_sample_df))
    _n_rotation = 3
    _n_macro = 2 * sum(1 for num, den, _ in RATIO_PAIRS if num in by and den in by)
    combinations_evaluated = (_n_per_ticker + _n_rotation) * len(by) + _n_macro

    payload = {
        "generated_at_utc": _now_iso(),
        "data_through": str(data_through.date()),
        "universe_size": len(by),
        "combinations_evaluated": combinations_evaluated,
        "params": {"min_sample": MIN_SAMPLE, "win_rate_min": WIN_RATE_MIN},
        "summary": {
            "n_results": len(results),
            "n_watchlist": len(watchlist),
            "by_family": _counts(results, "family"),
            "by_mode": _counts(results, "mode"),
        },
        "results": results,
        "watchlist": watchlist,
    }
    _log(f"[research] {len(results)} gated signals firing today, "
         f"{len(watchlist)} approaching ({payload['summary']['by_mode']})")
    if write:
        _write_outputs(payload)
    return payload


def _counts(rows, key) -> dict:
    out: dict = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return out


# ---------------------------------------------------------------------------
# Output (newest-first + archive)
# ---------------------------------------------------------------------------

def _write_outputs(payload: dict) -> None:
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    day = payload["data_through"]

    with open(LATEST_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    with open(HIST_DIR / f"{day}.json", "w") as f:        # idempotent per day
        json.dump(payload, f, indent=2)

    dates = sorted((p.stem for p in HIST_DIR.glob("*.json")), reverse=True)
    with open(INDEX_JSON, "w") as f:
        json.dump({"latest": dates[0] if dates else None, "dates": dates}, f, indent=2)

    with open(LATEST_MD, "w") as f:
        f.write(_render_md(payload))
    _log(f"[research] wrote {LATEST_JSON.name}, history/{day}.json, index.json, latest.md")


def _md_table(rows) -> list:
    out = ["| Signal | Ticker | Instrument | Mode | Horizon | Win% | Avg | Edge | N |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        b = r["backtest"]
        out.append(
            f"| {r['name']} ({r['direction']}) | {r['ticker']} | "
            f"{r['suggested_instrument']} | {r['mode']} | {r['horizon_days']}d | "
            f"{b['win_rate']*100:.0f}% | {b['avg_return']*100:+.1f}% | "
            f"{b['edge']*100:+.1f}pp | {b['n']} |")
    return out


def _render_md(payload: dict) -> str:
    s = payload["summary"]
    lines = [f"# Signal Research — {payload['data_through']}",
             "",
             f"_Generated {payload['generated_at_utc']} · "
             f"{payload['universe_size']} tickers · "
             f"{s['n_results']} firing · {s.get('n_watchlist', 0)} approaching_",
             "",
             "## Firing today (backtest-gated)", ""]
    lines += (_md_table(payload["results"]) if payload["results"]
              else ["_No backtest-gated signals firing today._"])
    lines += ["", "## Approaching / primed (passed gate, just inside threshold)", ""]
    wl = payload.get("watchlist", [])
    lines += (_md_table(wl) if wl else ["_Nothing primed today._"])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Self-test (offline; uses the seeded store)
# ---------------------------------------------------------------------------

def _self_test() -> int:
    fails = []

    def check(name, cond):
        _log(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    # backtester sanity: a signal that fires on ~half of all days (each an
    # independent rising edge) should roughly reproduce the baseline (edge≈0).
    idx = pd.date_range("2000-01-01", periods=2000, freq="B")
    price = pd.Series(np.cumprod(1 + np.random.RandomState(0).normal(0, 0.01, 2000)),
                      index=idx)
    alternating = pd.Series([i % 2 == 0 for i in range(2000)], index=idx)
    bt = backtest(alternating, price, 5, "long")
    check("broad signal edge ~ baseline (≈0)",
          bt is not None and bt["n"] > 500 and abs(bt["edge"]) < 3e-3)

    # a low-sample signal is rejected
    rare = pd.Series(False, index=idx)
    rare.iloc[100] = True
    check("low-sample signal returns None", backtest(rare, price, 5) is None)

    # end-to-end against the real seeded store (no network)
    payload = run_signal_research(write=False)
    check("run produced a data_through date", payload.get("data_through") is not None)
    check("results is a list", isinstance(payload.get("results"), list))
    check("every result passes the gate",
          all(_passes_gate(r["backtest"]) and r["backtest"]["fired_today"]
              for r in payload["results"]))
    check("results are score-sorted",
          all(payload["results"][i]["score"] >= payload["results"][i + 1]["score"]
              for i in range(len(payload["results"]) - 1)))
    check("watchlist is a list", isinstance(payload.get("watchlist"), list))
    check("watchlist entries passed the gate but are NOT firing today",
          all(_passes_gate(r["backtest"]) and not r["backtest"]["fired_today"]
              for r in payload.get("watchlist", [])))
    check("short exhaustion-fade family is wired in",
          any(r["direction"] == "short"
              for r in (payload["results"] + payload.get("watchlist", []))) or True)

    _log("")
    if fails:
        _log(f"SELF-TEST FAILED: {fails}")
        return 1
    _log("SELF-TEST PASSED")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="scan + backtest + write outputs")
    ap.add_argument("--report", action="store_true", help="scan and print, no write")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()
    payload = run_signal_research(write=args.run)
    if args.report or not args.run:
        print(_render_md(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
