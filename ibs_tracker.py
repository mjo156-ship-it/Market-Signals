"""
ibs_tracker.py — IBS state, one-time priming, rolling forward tracking, and the
faltering flag (Part 3 of the build spec + Updates 2 & 4).

CANONICAL COPY — keep in sync with ~/chf-dashboard/ibs_tracker.py.

The tracker JSON (ibs_tracker.json) is written by the SIGNAL MONITOR and
committed back to Market-Signals `main` by the GitHub Action. The DASHBOARD
reads it read-only. So the canonical file lives in Market-Signals; point the
dashboard at it via the IBS_TRACKER_JSON env var (or an explicit path).

Two windows, deliberately different (Update 2):
  * BASELINE  → primed on the FULL synth-extended history back to ~2000 (Update
    3 splice), so the faltering yardstick spans dot-com / GFC / 2018 / COVID /
    2022 — NOT the bull-only recent window.
  * ROLLING   → trailing-20 matured live firings (fallback: trailing 12 months
    if < 20). This is what the flag compares to the full-cycle baseline.

The user-facing 5-year performance display is unchanged; only the flag's
baseline is full-cycle.
"""
import os
import json
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import ibs_engine as eng
from synth_splice import synth_splice, btal_splice

# --- Config ------------------------------------------------------------------
DEFAULT_JSON = os.environ.get("IBS_TRACKER_JSON") or \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ibs_tracker.json")

WIN_HORIZON = "5d"        # forward window whose sign defines a "win"
ROLL_N = 20               # rolling window = last 20 matured firings per key
ROLL_MIN = 10             # < this many matured firings -> status stays "OK"
FALLBACK_MONTHS = 12      # if < ROLL_N matured, use trailing 12 months instead
SLIPPAGE_BPS = 5          # per switch on leveraged sleeves (priming replay)

# Update 4: BTAL overlay. Set 0.0 to disable (the strategy runs fine without it).
BTAL_WEIGHT = 0.20
BTAL_INCEPTION = "2011-09-01"     # real BTAL used from here forward; proxy before

# Vehicles whose forward returns are net of slippage in the replay (leveraged).
LEVERAGED = {"TQQQ", "SOXL", "TECL", "UPRO"}


# =============================================================================
# STATS HELPERS
# =============================================================================
def wilson_ci(wins, n, z=1.96):
    """Wilson 95% CI for a win rate. Returns (lo, hi)."""
    if n <= 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def brier(prob, outcomes):
    """Mean Brier score of a constant forecast `prob` vs binary outcomes."""
    outcomes = [o for o in outcomes if o is not None]
    if not outcomes:
        return None
    return float(np.mean([(prob - o) ** 2 for o in outcomes]))


def perf_stats(daily_ret):
    """CAGR / Sharpe / MaxDD / period_return from a daily-return series."""
    r = pd.Series(daily_ret).dropna()
    n = len(r)
    if n < 20:
        return {"period_return": None, "cagr": None, "sharpe": None, "maxdd": None, "n": n}
    cum = float((1 + r).prod())
    years = n / 252.0
    cagr = cum ** (1 / years) - 1 if (years > 0 and cum > 0) else None
    sd = r.std()
    sharpe = float(r.mean() / sd * math.sqrt(252)) if sd and sd > 0 else 0.0
    curve = (1 + r).cumprod()
    maxdd = float((curve / curve.cummax() - 1).min())
    return {"period_return": cum - 1, "cagr": cagr, "sharpe": sharpe,
            "maxdd": maxdd, "n": n}


def _rsi_wilder(prices, period=10):
    """Wilder RSI — local copy so the tracker is standalone in both repos."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    ag = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = ag / al
    return 100 - (100 / (1 + rs))


# =============================================================================
# SIGNAL-TYPE KEYS
# =============================================================================
def gate_key(sleeve):                 return f"gate_entry_{sleeve}"
def single_key(sleeve, mult):
    # SOXL single-name tier is 2.0x; all others 1.5x.
    return (f"conviction_2.0x_single_{sleeve}" if mult == eng.SOXL_SINGLE
            else f"conviction_1.5x_{sleeve}")
BREADTH_KEY = "conviction_2.0x_breadth"


def _signal_type_of(row):
    """Map a firing row to its baseline/rolling key."""
    st = row["signal_type"]
    sleeve = row["sleeve"]
    if st == "gate_entry":
        return gate_key(sleeve)
    if st == "conviction_2.0x_breadth":
        return BREADTH_KEY
    if st == "conviction_2.0x_single":
        return single_key(sleeve, eng.SOXL_SINGLE)
    return single_key(sleeve, eng.MULT_SINGLE)


# =============================================================================
# JSON STATE I/O
# =============================================================================
def _blank_state():
    return {
        "updated_at": None,
        "gate_state": {u: "OUT" for u in eng.SLEEVE_MAP},
        "current_multiplier": {v: 0.0 for v in eng.SLEEVE_MAP.values()},
        "breadth_today": 0,
        "baseline": {},
        "baseline_curves": {},     # Update 6 anchor: full-cycle CAGR/Sharpe/MaxDD
        "firings": [],
        "rolling": {},
        "equity": {"engine_only": [], "engine_plus_btal": []},  # Update 4, live-appended
    }


def load_state(path=DEFAULT_JSON):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return _blank_state()


def save_state(state, path=DEFAULT_JSON):
    state["updated_at"] = datetime.now().isoformat()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=_json_default)
    os.replace(tmp, path)
    return path


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return None if (o != o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and o != o:
        return None
    raise TypeError(f"not serializable: {type(o)}")


# =============================================================================
# PRICE FETCH (priming only; live callers pass their own OHLC)
# =============================================================================
def _fetch(ticker, start):
    import yfinance as yf
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _daily_ret(df):
    """Adj Close daily returns (auto_adjust=False -> use 'Adj Close')."""
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    return df[col].pct_change()


# =============================================================================
# ONE-TIME PRIMING  (Update 2: full synth-extended baseline back to ~2000)
# =============================================================================
def prime_history(start="2000-01-01", path=DEFAULT_JSON, verbose=True):
    """Replay the exact Part-1 logic day by day over the full synth-extended
    history, record every firing with forward 1d/5d/10d results, and aggregate a
    baseline per signal-type key. Also compute the two full-cycle equity curves
    (engine_only, engine_plus_btal) for the Update-6 expectation annotations.

    Runs once (workflow_dispatch prime_ibs=true), then never again. Baseline is
    fixed at prime time; re-prime only on a deliberate methodology change.
    """
    und = ["SPY", "QQQ", "SMH", "XLK"]
    veh_map = {"QQQ": "TQQQ", "SMH": "SOXL", "XLK": "TECL"}   # spliced 3x
    proxy_sectors = ["XLP", "XLU", "XLV"]

    if verbose:
        print("[prime] downloading OHLC + vehicles ...")
    ohlc = {t: _fetch(t, start) for t in und}
    for t in und:
        if ohlc[t] is None:
            raise RuntimeError(f"prime_history: no data for {t}")

    # Vehicle daily returns: real 1x SPY (or UPRO), synth-3x for QQQ/SMH/XLK.
    veh_ret = {}
    spy_vehicle = eng.SLEEVE_MAP["SPY"]
    veh_ret["SPY"] = _daily_ret(_fetch(spy_vehicle, start))
    for u, v in veh_map.items():
        under_ret = _daily_ret(ohlc[u])
        real_lev = _daily_ret(_fetch(v, start))
        veh_ret[u] = synth_splice(under_ret, real_lev)

    # BTAL (Update 4): proxy pre-2011 spliced with real BTAL.
    sect = {s: _daily_ret(_fetch(s, start)) for s in proxy_sectors + ["XLK", "SMH"]}
    sect_df = pd.DataFrame(sect)
    real_btal = _daily_ret(_fetch("BTAL", start))
    btal_ret = btal_splice(sect_df, real_btal)

    # Align everything on a common daily index.
    idx = ohlc["SPY"].index
    for t in und:
        idx = idx.intersection(ohlc[t].index)
    idx = idx.sort_values()

    # Precompute per-underlying IBS_sma3 and RSI10 on the aligned index.
    ibs3 = {u: eng.ibs_sma3(ohlc[u]).reindex(idx) for u in und}
    rsi10 = {u: _rsi_wilder(ohlc[u]["Close"], 10).reindex(idx) for u in und}
    vret = {u: veh_ret[u].reindex(idx).fillna(0.0) for u in und}
    btal_r = btal_ret.reindex(idx).fillna(0.0)

    firings = []
    prev_state = {u: "OUT" for u in und}
    prev_tier = {u: None for u in und}
    dates = list(idx)

    if verbose:
        print(f"[prime] replaying {len(dates)} sessions {dates[0].date()}..{dates[-1].date()}")

    # Daily portfolio returns for the two equity curves. Positions are decided
    # at day i's close and earn day i+1's return (signal at t-1 -> return at t);
    # `prev_pos` carries yesterday's sized position into today's return.
    eo_rets, epb_rets = [], []
    prev_pos = {u: 0.0 for u in und}

    for i, dt in enumerate(dates):
        rsi_by = {u: (float(rsi10[u].iloc[i]) if not math.isnan(rsi10[u].iloc[i]) else None)
                  for u in und}
        breadth = eng.breadth_count(rsi_by)

        states, mults = {}, {}
        for u in und:
            s3 = ibs3[u].iloc[i]
            st = eng.gate_state(s3, prev_state[u])
            states[u] = st
            mult = eng.conviction_multiplier(u, rsi_by[u], breadth) if st == "IN" else 0.0
            mults[u] = mult

            # ---- record firings (forward windows must exist) ----
            can_fwd = i + 10 < len(dates)
            if can_fwd:
                # gate OUT->IN
                if prev_state[u] != "IN" and st == "IN":
                    firings.append(_mk_firing(dt, u, "gate_entry", 1.0,
                                              s3, rsi_by[u], breadth, i, vret[u], dates))
                # conviction tier ENTERED (tier changed to an elevated tier)
                if st == "IN" and mult in (eng.MULT_SINGLE, eng.MULT_BREADTH) and mult != prev_tier[u]:
                    if breadth >= 3:
                        stype = "conviction_2.0x_breadth"
                    elif eng.SLEEVE_MAP[u] == "SOXL":
                        stype = "conviction_2.0x_single"
                    else:
                        stype = "conviction_1.5x"
                    firings.append(_mk_firing(dt, u, stype, mult,
                                              s3, rsi_by[u], breadth, i, vret[u], dates))
            prev_tier[u] = mult if st == "IN" else None
            prev_state[u] = st

        # ---- equity: yesterday's position earns today's return ----
        day_eo = sum(prev_pos[u] * vret[u].iloc[i] for u in und)
        eo_rets.append(day_eo)
        epb_rets.append(day_eo + (BTAL_WEIGHT * float(btal_r.iloc[i]) if BTAL_WEIGHT else 0.0))
        prev_pos = eng.sleeve_positions(states, mults)   # capped, for tomorrow

    # Aggregate baseline per key.
    baseline = _aggregate_baseline(firings)
    curves = {
        "engine_only": perf_stats(pd.Series(eo_rets, index=idx)),
        "engine_plus_btal": perf_stats(pd.Series(epb_rets, index=idx)),
        "btal_weight": BTAL_WEIGHT,
        "note": ("Full-cycle synth-extended baseline (~2000-present). Pre-inception "
                 "vehicle returns are synthetic-3x (understate choppy-crash decay); "
                 "pre-2011 BTAL is the anti-beta proxy (conservative floor). "
                 "UNCAPPED backtest assumed margin on ~3% of high-bounce days, so the "
                 "capped live version runs slightly below these numbers."),
    }

    state = load_state(path)
    state["baseline"] = baseline
    state["baseline_curves"] = curves
    save_state(state, path)
    if verbose:
        print(f"[prime] baseline keys: {len(baseline)}; "
              f"engine_only CAGR={curves['engine_only']['cagr']}, "
              f"Sharpe={curves['engine_only']['sharpe']}, MaxDD={curves['engine_only']['maxdd']}")
    return state


def _mk_firing(dt, sleeve, signal_type, mult, s3, own_rsi10, breadth, i, vret, dates):
    """Build a firing dict with forward 1d/5d/10d vehicle returns (net slippage
    on leveraged sleeves)."""
    def fwd(k):
        seg = vret.iloc[i + 1:i + 1 + k]
        r = float((1 + seg).prod() - 1)
        if eng.SLEEVE_MAP[sleeve] in LEVERAGED:
            r -= SLIPPAGE_BPS / 1e4      # one switch's slippage
        return r * 100.0
    return {
        "date": dt.strftime("%Y-%m-%d"),
        "sleeve": eng.SLEEVE_MAP[sleeve],
        "underlying": sleeve,
        "signal_type": signal_type,
        "multiplier": mult,
        "ibs_sma3": None if s3 != s3 else round(float(s3), 4),
        "own_rsi10": None if own_rsi10 is None else round(own_rsi10, 1),
        "breadth": breadth,
        "entry_price": None,
        "fwd_1d": round(fwd(1), 3),
        "fwd_5d": round(fwd(5), 3),
        "fwd_10d": round(fwd(10), 3),
        "matured": True,
    }


def _aggregate_baseline(firings):
    """Group primed firings by signal-type key -> n, wr, Wilson CI, avgs, Brier."""
    groups = {}
    for f in firings:
        groups.setdefault(_signal_type_of(f), []).append(f)
    out = {}
    for key, rows in groups.items():
        wins = [1 if (r[f"fwd_{WIN_HORIZON}"] or 0) > 0 else 0 for r in rows]
        n = len(rows)
        wr = sum(wins) / n if n else 0.0
        lo, hi = wilson_ci(sum(wins), n)
        out[key] = {
            "n": n,
            "wr": round(wr, 4),
            "wilson_lo": round(lo, 4),
            "wilson_hi": round(hi, 4),
            "avg_1d": round(float(np.mean([r["fwd_1d"] for r in rows])), 3),
            "avg_5d": round(float(np.mean([r["fwd_5d"] for r in rows])), 3),
            "brier": round(brier(wr, wins), 4) if n else None,
        }
    return out


# =============================================================================
# ROLLING FORWARD TRACKING (live)
# =============================================================================
def log_firing(state, date, sleeve, signal_type, multiplier, underlying,
               ibs_sma3, own_rsi10, breadth, entry_price):
    """Append a new live firing (forward returns filled later by mature_results)."""
    state["firings"].append({
        "date": date, "sleeve": sleeve, "underlying": underlying,
        "signal_type": signal_type, "multiplier": multiplier,
        "ibs_sma3": ibs_sma3, "own_rsi10": own_rsi10, "breadth": breadth,
        "entry_price": entry_price,
        "fwd_1d": None, "fwd_5d": None, "fwd_10d": None, "matured": False,
    })
    return state


def mature_results(state, price_lookup):
    """Fill fwd_1d/5d/10d for firings whose forward dates have passed, then
    refresh the rolling window + faltering status per key.

    price_lookup(ticker) -> a pd.Series of vehicle prices indexed by date.
    """
    for f in state["firings"]:
        if f["matured"]:
            continue
        prices = price_lookup(f["sleeve"])
        if prices is None or len(prices) == 0:
            continue
        d0 = pd.Timestamp(f["date"])
        after = prices[prices.index > d0]
        if len(after) < 10:
            continue  # not enough forward bars yet
        p0 = float(prices[prices.index <= d0].iloc[-1])
        for k in (1, 5, 10):
            pk = float(after.iloc[k - 1])
            r = (pk / p0 - 1) * 100.0
            if f["sleeve"] in LEVERAGED:
                r -= SLIPPAGE_BPS / 1e4 * 100.0
            f[f"fwd_{k}d"] = round(r, 3)
        f["matured"] = True

    _refresh_rolling(state)
    return state


def _refresh_rolling(state):
    baseline = state.get("baseline", {})
    matured = [f for f in state["firings"] if f["matured"]]
    by_key = {}
    for f in matured:
        by_key.setdefault(_signal_type_of(f), []).append(f)

    rolling = {}
    for key, rows in by_key.items():
        rows = sorted(rows, key=lambda r: r["date"])
        window = rows[-ROLL_N:]
        if len(window) < ROLL_N:  # fallback: trailing 12 months
            cutoff = (datetime.now() - timedelta(days=365 * FALLBACK_MONTHS // 12)).strftime("%Y-%m-%d")
            window = [r for r in rows if r["date"] >= cutoff] or window
        wins = [1 if (r[f"fwd_{WIN_HORIZON}"] or 0) > 0 else 0 for r in window]
        n = len(window)
        wr = sum(wins) / n if n else 0.0
        avg1 = float(np.mean([r["fwd_1d"] for r in window])) if n else 0.0
        base = baseline.get(key)
        b_wr = base["wr"] if base else None
        rolling[key] = {
            "trailing_n": n,
            "trailing_wr": round(wr, 4),
            "trailing_avg_1d": round(avg1, 3),
            "brier": round(brier(b_wr, wins), 4) if (base and n) else None,
            "status": _status(n, wr, avg1,
                              rolling_brier=(brier(b_wr, wins) if (base and n) else None),
                              base=base),
        }
    state["rolling"] = rolling
    return state


def _status(n, wr, avg1, rolling_brier, base):
    """Faltering flag: compare the rolling window to the full-cycle baseline."""
    if n < ROLL_MIN or base is None:
        return "OK"                                  # not enough live data yet
    status = "OK"
    if wr < base["wilson_lo"]:
        status = "FALTERING"                         # below historical 95% lower bound
    elif wr < base["wr"]:
        status = "WATCH"                             # below the point estimate

    # secondary triggers force at least WATCH (escalate if already FALTERING)
    sign_flip = avg1 < 0 and base["avg_1d"] > 0
    brier_drift = (rolling_brier is not None and base.get("brier")
                   and rolling_brier > 1.25 * base["brier"])
    if (sign_flip or brier_drift) and status == "OK":
        status = "WATCH"
    return status


# =============================================================================
# LIVE SNAPSHOT (consumed by monitor TIER 10 + dashboard card)
# =============================================================================
def snapshot(state, gate_states, multipliers, breadth):
    """Update the current-state block from today's live gate/mult/breadth and
    return the display payload (state + per-sleeve mult + faltering flags)."""
    state["gate_state"] = {u: gate_states.get(u, "OUT") for u in eng.SLEEVE_MAP}
    state["current_multiplier"] = {
        eng.SLEEVE_MAP[u]: (multipliers.get(u, 0.0) if gate_states.get(u) == "IN" else 0.0)
        for u in eng.SLEEVE_MAP
    }
    state["breadth_today"] = breadth
    return {
        "gate_state": state["gate_state"],
        "current_multiplier": state["current_multiplier"],
        "breadth_today": breadth,
        "rolling": state.get("rolling", {}),
        "baseline": state.get("baseline", {}),
        "baseline_curves": state.get("baseline_curves", {}),
    }


def faltering_flags(state):
    """{signal_type_key: status} for the keys currently not OK."""
    return {k: v["status"] for k, v in state.get("rolling", {}).items()
            if v.get("status") in ("WATCH", "FALTERING")}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "prime":
        prime_history()
    else:
        st = load_state()
        print(json.dumps({k: st[k] for k in ("updated_at", "gate_state",
              "current_multiplier", "breadth_today")}, indent=2))
