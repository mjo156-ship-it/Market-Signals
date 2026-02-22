#!/usr/bin/env python3
"""
Intraday Signal Analyzer — Polygon Edition
============================================
Uses hourly and 5-minute bars from Polygon to analyze:
1. Which HOUR of the day concentrates the edge for each signal
2. Optimal entry timing (open vs. first 30min vs. VWAP)
3. Intraday RSI scalps (extreme readings → same-day reversal)
4. Overnight vs. intraday decomposition with REAL data (not daily proxies)

REQUIRES:
  Polygon data downloaded via polygon_downloader.py
  Daily CSVs in /mnt/project/ (for signal identification)

USAGE:
  python polygon_analyzer.py                    # Run all analyses
  python polygon_analyzer.py --signal uvxy_qqq  # Specific signal only
  python polygon_analyzer.py --list             # List available signals
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# =============================================================================
# CONFIGURATION
# =============================================================================
POLYGON_DIR = Path("data/polygon")
DAILY_DIR = Path("/mnt/project")  # Daily CSVs from project files

# =============================================================================
# HELPERS
# =============================================================================
def calculate_rsi_wilder(prices, period=10):
    """Wilder's RSI matching our signal monitor."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def load_daily(ticker, use_adj_close=False):
    """Load daily data from project CSVs."""
    # Handle special tickers
    filename_map = {
        "BTC-USD": "BTCUSD",
        "X:BTCUSD": "BTCUSD",
    }
    fname = filename_map.get(ticker, ticker)

    # Try multiple filename patterns
    for pattern in [f"{fname}.csv", f"{fname}2.csv"]:
        path = DAILY_DIR / pattern
        if path.exists():
            df = pd.read_csv(path, parse_dates=["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            col = "Adj Close" if use_adj_close and "Adj Close" in df.columns else "Close"
            df["rsi10"] = calculate_rsi_wilder(df[col], 10)
            df["rsi50"] = calculate_rsi_wilder(df[col], 50)
            df["sma200"] = df[col].rolling(200).mean()
            df["sma50"] = df[col].rolling(50).mean()
            return df
    return None


def load_intraday(ticker, resolution="60m"):
    """Load intraday data from Polygon CSVs."""
    filename_map = {
        "BTC-USD": "BTCUSD",
        "X:BTCUSD": "BTCUSD",
    }
    fname = filename_map.get(ticker, ticker)
    path = POLYGON_DIR / f"{fname}_{resolution}.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_signal_dates(signal_name):
    """Identify dates when a signal was active using daily data."""
    signals = {
        "uvxy_qqq79": _signal_qqq_overbought,
        "upro_spy21": _signal_spy_oversold,
        "fas_short85": _signal_fas_overbought,
        "upro_short85": _signal_spy_overbought,
        "tqqq_double": _signal_gld_usdu_double,
        "soxs_dollar": _signal_soxs_dollar_squeeze,
        "cure_dip": _signal_cure_oversold,
        "defense_rotation": _signal_defensive_rotation,
    }

    if signal_name not in signals:
        print(f"Unknown signal: {signal_name}")
        print(f"Available: {', '.join(signals.keys())}")
        return []

    return signals[signal_name]()


# =============================================================================
# SIGNAL DEFINITIONS (using daily data)
# =============================================================================
def _signal_qqq_overbought():
    """QQQ RSI(10) > 79 → trade UVXY next day."""
    df = load_daily("QQQ")
    if df is None:
        return []
    mask = df["rsi10"] > 79
    # Signal fires at close, trade next day
    signal_dates = df.loc[mask, "Date"].tolist()
    trade_dates = []
    for d in signal_dates:
        next_idx = df[df["Date"] > d].index
        if len(next_idx) > 0:
            trade_dates.append(df.loc[next_idx[0], "Date"])
    return trade_dates


def _signal_spy_oversold():
    """SPY RSI(10) < 21 → trade UPRO next day."""
    df = load_daily("SPY")
    if df is None:
        return []
    # Use SPY2.csv for longer history
    df2 = load_daily("SPY")
    if df2 is not None and len(df2) > len(df):
        df = df2
    mask = df["rsi10"] < 21
    signal_dates = df.loc[mask, "Date"].tolist()
    trade_dates = []
    for d in signal_dates:
        next_idx = df[df["Date"] > d].index
        if len(next_idx) > 0:
            trade_dates.append(df.loc[next_idx[0], "Date"])
    return trade_dates


def _signal_fas_overbought():
    """FAS RSI(10) > 85 → short/fade FAS next day."""
    df = load_daily("FAS")
    if df is None:
        return []
    mask = df["rsi10"] > 85
    signal_dates = df.loc[mask, "Date"].tolist()
    trade_dates = []
    for d in signal_dates:
        next_idx = df[df["Date"] > d].index
        if len(next_idx) > 0:
            trade_dates.append(df.loc[next_idx[0], "Date"])
    return trade_dates


def _signal_spy_overbought():
    """SPY RSI(10) > 85 → short/fade UPRO next day."""
    df = load_daily("SPY")
    if df is None:
        return []
    mask = df["rsi10"] > 85
    signal_dates = df.loc[mask, "Date"].tolist()
    trade_dates = []
    for d in signal_dates:
        next_idx = df[df["Date"] > d].index
        if len(next_idx) > 0:
            trade_dates.append(df.loc[next_idx[0], "Date"])
    return trade_dates


def _signal_gld_usdu_double():
    """GLD RSI(10) > 79 AND USDU RSI(10) < 25 → trade TQQQ next day."""
    gld = load_daily("GLD")
    usdu = load_daily("USDU")
    if gld is None or usdu is None:
        return []

    merged = gld[["Date", "rsi10"]].merge(
        usdu[["Date", "rsi10"]], on="Date", suffixes=("_gld", "_usdu")
    )
    mask = (merged["rsi10_gld"] > 79) & (merged["rsi10_usdu"] < 25)
    signal_dates = merged.loc[mask, "Date"].tolist()

    spy = load_daily("SPY")  # Use SPY for trading day reference
    if spy is None:
        return signal_dates

    trade_dates = []
    for d in signal_dates:
        next_idx = spy[spy["Date"] > d].index
        if len(next_idx) > 0:
            trade_dates.append(spy.loc[next_idx[0], "Date"])
    return trade_dates


def _signal_soxs_dollar_squeeze():
    """SMH RSI(10) > 79 AND USDU RSI(10) > 70 → trade SOXS next day."""
    smh = load_daily("SMH")
    usdu = load_daily("USDU")
    if smh is None or usdu is None:
        return []

    merged = smh[["Date", "rsi10"]].merge(
        usdu[["Date", "rsi10"]], on="Date", suffixes=("_smh", "_usdu")
    )
    mask = (merged["rsi10_smh"] > 79) & (merged["rsi10_usdu"] > 70)
    signal_dates = merged.loc[mask, "Date"].tolist()

    trade_dates = []
    for d in signal_dates:
        next_idx = smh[smh["Date"] > d].index
        if len(next_idx) > 0:
            trade_dates.append(smh.loc[next_idx[0], "Date"])
    return trade_dates


def _signal_cure_oversold():
    """CURE RSI(10) < 21 → trade CURE next day."""
    df = load_daily("CURE")
    if df is None:
        return []
    mask = df["rsi10"] < 21
    signal_dates = df.loc[mask, "Date"].tolist()
    trade_dates = []
    for d in signal_dates:
        next_idx = df[df["Date"] > d].index
        if len(next_idx) > 0:
            trade_dates.append(df.loc[next_idx[0], "Date"])
    return trade_dates


def _signal_defensive_rotation():
    """XLP/XLU/XLV RSI > 79 AND SPY/QQQ < 79 → trade TQQQ next day."""
    xlp = load_daily("XLP")
    xlu = load_daily("XLU")
    xlv = load_daily("XLV")
    spy = load_daily("SPY")
    qqq = load_daily("QQQ")

    if any(d is None for d in [xlp, xlu, xlv, spy, qqq]):
        return []

    # Merge all on date
    merged = xlp[["Date", "rsi10"]].rename(columns={"rsi10": "xlp_rsi"})
    for df, name in [(xlu, "xlu_rsi"), (xlv, "xlv_rsi"),
                     (spy, "spy_rsi"), (qqq, "qqq_rsi")]:
        merged = merged.merge(df[["Date", "rsi10"]].rename(
            columns={"rsi10": name}), on="Date", how="inner")

    def_ob = (merged["xlp_rsi"] > 79) | (merged["xlu_rsi"] > 79) | (merged["xlv_rsi"] > 79)
    idx_not_ob = (merged["spy_rsi"] < 79) & (merged["qqq_rsi"] < 79)
    mask = def_ob & idx_not_ob

    signal_dates = merged.loc[mask, "Date"].tolist()
    trade_dates = []
    for d in signal_dates:
        next_idx = spy[spy["Date"] > d].index
        if len(next_idx) > 0:
            trade_dates.append(spy.loc[next_idx[0], "Date"])
    return trade_dates


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def analyze_hourly_profile(trade_ticker, signal_name, trade_dates, direction="long"):
    """
    Analysis 1: Which hour of the day concentrates the edge?

    For each signal date, compute return for each hourly bar.
    Show average return by hour to find when the move happens.
    """
    print(f"\n{'='*70}")
    print(f"HOURLY RETURN PROFILE: {trade_ticker} on [{signal_name}]")
    print(f"Direction: {direction.upper()} | Signal dates: {len(trade_dates)}")
    print(f"{'='*70}")

    intraday = load_intraday(trade_ticker, "60m")
    if intraday is None:
        print(f"  No intraday data for {trade_ticker}. Run polygon_downloader.py first.")
        return None

    # Normalize trade dates
    trade_dates_norm = set(pd.to_datetime(d).date() for d in trade_dates)

    # Filter intraday to signal dates only
    intraday["date_key"] = intraday["date"].dt.date
    signal_bars = intraday[intraday["date_key"].isin(trade_dates_norm)].copy()

    if signal_bars.empty:
        print(f"  No matching intraday bars found. Check date overlap.")
        return None

    # Group by date and compute hourly returns
    results = []
    for dt, group in signal_bars.groupby("date_key"):
        group = group.sort_values("time")
        if len(group) < 4:  # Need at least a few bars
            continue

        day_open = group.iloc[0]["open"]
        prev_close = group.iloc[0]["open"]  # Approximation; actual prev close from daily

        for _, bar in group.iterrows():
            bar_return = (bar["close"] / bar["open"] - 1) * 100
            if direction == "short":
                bar_return = -bar_return

            cum_from_open = (bar["close"] / day_open - 1) * 100
            if direction == "short":
                cum_from_open = -cum_from_open

            results.append({
                "date": dt,
                "hour": bar["time"],
                "bar_return_pct": bar_return,
                "cum_from_open_pct": cum_from_open,
                "volume": bar["volume"],
            })

    if not results:
        print("  No valid trading days found.")
        return None

    df_results = pd.DataFrame(results)
    n_days = df_results["date"].nunique()

    # Aggregate by hour
    hourly = df_results.groupby("hour").agg(
        avg_bar_return=("bar_return_pct", "mean"),
        win_rate=("bar_return_pct", lambda x: (x > 0).mean() * 100),
        avg_cum_return=("cum_from_open_pct", "mean"),
        avg_volume=("volume", "mean"),
        n_days=("date", "nunique"),
    ).round(3)

    print(f"\n  Matching days with intraday data: {n_days}")
    print(f"\n  {'Hour':<8} {'Bar Ret%':>10} {'Win%':>8} {'Cum Ret%':>10} {'Avg Vol':>12} {'Days':>6}")
    print(f"  {'-'*58}")

    for hour, row in hourly.iterrows():
        bar_r = f"{row['avg_bar_return']:+.3f}%"
        win = f"{row['win_rate']:.0f}%"
        cum_r = f"{row['avg_cum_return']:+.3f}%"
        vol = f"{row['avg_volume']:,.0f}"
        print(f"  {hour:<8} {bar_r:>10} {win:>8} {cum_r:>10} {vol:>12} {int(row['n_days']):>6}")

    # Identify the best hour
    best_hour = hourly["avg_bar_return"].idxmax()
    best_return = hourly.loc[best_hour, "avg_bar_return"]
    print(f"\n  ★ Best hour: {best_hour} ({best_return:+.3f}% avg)")

    # First hour vs rest of day
    if "09:30" in hourly.index or "10:00" in hourly.index:
        first_hour_key = "09:30" if "09:30" in hourly.index else "10:00"
        first_hour = hourly.loc[first_hour_key, "avg_bar_return"]
        rest_of_day = hourly.loc[hourly.index != first_hour_key, "avg_bar_return"].mean()
        print(f"  First hour: {first_hour:+.3f}% | Rest of day avg: {rest_of_day:+.3f}%")

    return df_results


def analyze_5min_entry(trade_ticker, signal_name, trade_dates, direction="long"):
    """
    Analysis 2: Optimal entry within first 30 minutes (5-min resolution).

    Shows whether buying at open, 9:35, 9:40... 10:00 gives best results.
    """
    print(f"\n{'='*70}")
    print(f"5-MIN ENTRY ANALYSIS: {trade_ticker} on [{signal_name}]")
    print(f"{'='*70}")

    intraday = load_intraday(trade_ticker, "5m")
    if intraday is None:
        print(f"  No 5-min data for {trade_ticker}. Run: python polygon_downloader.py --backfill --tickers {trade_ticker} --resolution 5m")
        return None

    trade_dates_norm = set(pd.to_datetime(d).date() for d in trade_dates)
    intraday["date_key"] = intraday["date"].dt.date
    signal_bars = intraday[intraday["date_key"].isin(trade_dates_norm)].copy()

    if signal_bars.empty:
        print("  No matching data.")
        return None

    # For each day, compute return from each entry time to close (3:55 PM)
    entry_times = ["09:30", "09:35", "09:40", "09:45", "09:50", "09:55", "10:00",
                   "10:15", "10:30", "11:00"]
    results = []

    for dt, group in signal_bars.groupby("date_key"):
        group = group.sort_values("time")
        # Get the day's closing price (last bar near 15:55)
        late_bars = group[group["time"] >= "15:50"]
        if late_bars.empty:
            late_bars = group[group["time"] >= "15:00"]
        if late_bars.empty:
            continue
        day_close = late_bars.iloc[-1]["close"]

        for entry_time in entry_times:
            entry_bar = group[group["time"] == entry_time]
            if entry_bar.empty:
                # Try closest bar after entry time
                entry_bar = group[group["time"] >= entry_time].head(1)
            if entry_bar.empty:
                continue

            entry_price = entry_bar.iloc[0]["open"]
            ret = (day_close / entry_price - 1) * 100
            if direction == "short":
                ret = -ret

            results.append({
                "date": dt,
                "entry_time": entry_time,
                "return_pct": ret,
            })

    if not results:
        print("  No valid entries found.")
        return None

    df = pd.DataFrame(results)

    print(f"\n  {'Entry Time':<12} {'Avg Ret%':>10} {'Win%':>8} {'n':>6}")
    print(f"  {'-'*40}")

    for et in entry_times:
        subset = df[df["entry_time"] == et]
        if len(subset) < 3:
            continue
        avg_ret = subset["return_pct"].mean()
        win = (subset["return_pct"] > 0).mean() * 100
        n = len(subset)
        print(f"  {et:<12} {avg_ret:>+9.3f}% {win:>7.0f}% {n:>6}")

    # Best entry
    by_time = df.groupby("entry_time")["return_pct"].mean()
    best = by_time.idxmax()
    print(f"\n  ★ Best entry time: {best} ({by_time[best]:+.3f}% avg)")

    return df


def analyze_overnight_vs_intraday(trade_ticker, signal_name, trade_dates,
                                   direction="long"):
    """
    Analysis 3: TRUE overnight vs intraday decomposition.

    Uses actual previous close → open (overnight) and open → close (intraday)
    with real data instead of daily OHLC approximations.
    """
    print(f"\n{'='*70}")
    print(f"OVERNIGHT vs INTRADAY: {trade_ticker} on [{signal_name}]")
    print(f"{'='*70}")

    # Load daily data for the traded ticker to get actual close prices
    daily = load_daily(trade_ticker)
    intraday = load_intraday(trade_ticker, "60m")

    if daily is None or intraday is None:
        print(f"  Missing data for {trade_ticker}.")
        return None

    trade_dates_norm = sorted(set(pd.to_datetime(d).date() for d in trade_dates))

    results = []
    daily["date_key"] = daily["Date"].dt.date

    for td in trade_dates_norm:
        # Get today's open (from intraday)
        intraday_today = intraday[intraday["date"].dt.date == td]
        if intraday_today.empty:
            continue

        today_open = intraday_today.sort_values("time").iloc[0]["open"]
        today_close = intraday_today.sort_values("time").iloc[-1]["close"]

        # Get previous close (from daily data)
        prev_rows = daily[daily["date_key"] < td].tail(1)
        if prev_rows.empty:
            continue
        prev_close = float(prev_rows.iloc[0]["Close"])

        overnight = (today_open / prev_close - 1) * 100
        intraday_ret = (today_close / today_open - 1) * 100
        total = (today_close / prev_close - 1) * 100

        if direction == "short":
            overnight = -overnight
            intraday_ret = -intraday_ret
            total = -total

        results.append({
            "date": td,
            "overnight_pct": overnight,
            "intraday_pct": intraday_ret,
            "total_pct": total,
        })

    if not results:
        print("  No matching dates with complete data.")
        return None

    df = pd.DataFrame(results)
    n = len(df)

    avg_on = df["overnight_pct"].mean()
    avg_id = df["intraday_pct"].mean()
    avg_total = df["total_pct"].mean()

    win_on = (df["overnight_pct"] > 0).mean() * 100
    win_id = (df["intraday_pct"] > 0).mean() * 100
    win_total = (df["total_pct"] > 0).mean() * 100

    # Contribution
    if avg_total != 0:
        on_contrib = avg_on / avg_total * 100
        id_contrib = avg_id / avg_total * 100
    else:
        on_contrib = id_contrib = 50

    print(f"\n  Signal days with data: {n}")
    print(f"\n  {'Component':<15} {'Avg Ret%':>10} {'Win%':>8} {'Contribution':>14}")
    print(f"  {'-'*50}")
    print(f"  {'Overnight':<15} {avg_on:>+9.3f}% {win_on:>7.0f}% {on_contrib:>12.0f}%")
    print(f"  {'Intraday':<15} {avg_id:>+9.3f}% {win_id:>7.0f}% {id_contrib:>12.0f}%")
    print(f"  {'TOTAL':<15} {avg_total:>+9.3f}% {win_total:>7.0f}%")

    # Day-trade verdict
    print(f"\n  DAY TRADE VERDICT: ", end="")
    if id_contrib >= 60:
        print(f"✅ GOOD — {id_contrib:.0f}% of edge is intraday")
    elif id_contrib >= 40:
        print(f"⚠️  MIXED — {id_contrib:.0f}% intraday, {on_contrib:.0f}% overnight")
    else:
        print(f"❌ NOT DAY-TRADEABLE — only {id_contrib:.0f}% intraday")

    # Multi-day analysis (hold 1-5 days)
    print(f"\n  Multi-day decomposition (cumulative):")
    print(f"  {'Days':<6} {'Overnight%':>12} {'Intraday%':>12} {'Total%':>10} {'ID Contrib':>12}")
    print(f"  {'-'*55}")

    daily_dates = daily["date_key"].tolist()
    for hold_days in [1, 2, 3, 5]:
        on_cum = []
        id_cum = []
        tot_cum = []

        for _, row in df.iterrows():
            td = row["date"]
            if td not in daily_dates:
                continue
            td_idx = daily_dates.index(td)

            cum_on = 0
            cum_id = 0
            for d in range(hold_days):
                idx = td_idx + d
                if idx >= len(daily_dates):
                    break
                check_date = daily_dates[idx]

                id_bars = intraday[intraday["date"].dt.date == check_date]
                if id_bars.empty:
                    continue

                day_open = id_bars.sort_values("time").iloc[0]["open"]
                day_close = id_bars.sort_values("time").iloc[-1]["close"]

                if idx > 0:
                    p_close = float(daily[daily["date_key"] == daily_dates[idx-1]].iloc[0]["Close"]) if idx > 0 else day_open
                else:
                    p_close = day_open

                cum_on += (day_open / p_close - 1) * 100 * (1 if direction == "long" else -1)
                cum_id += (day_close / day_open - 1) * 100 * (1 if direction == "long" else -1)

            on_cum.append(cum_on)
            id_cum.append(cum_id)
            tot_cum.append(cum_on + cum_id)

        if on_cum:
            avg_on_c = np.mean(on_cum)
            avg_id_c = np.mean(id_cum)
            avg_tot_c = np.mean(tot_cum)
            id_c = avg_id_c / avg_tot_c * 100 if avg_tot_c != 0 else 50
            print(f"  {hold_days:<6} {avg_on_c:>+11.3f}% {avg_id_c:>+11.3f}% {avg_tot_c:>+9.3f}% {id_c:>10.0f}%")

    return df


def analyze_gap_patterns(trade_ticker, signal_name, trade_dates, direction="long"):
    """
    Analysis 4: Gap pattern classification.

    Classifies each signal day into:
    - Gap Down → Reverse Up (intraday buy opportunity)
    - Gap Down → Continue Down
    - Gap Up → Continue Up
    - Gap Up → Reverse Down
    """
    print(f"\n{'='*70}")
    print(f"GAP PATTERN ANALYSIS: {trade_ticker} on [{signal_name}]")
    print(f"{'='*70}")

    daily = load_daily(trade_ticker)
    intraday = load_intraday(trade_ticker, "60m")

    if daily is None or intraday is None:
        print(f"  Missing data.")
        return None

    trade_dates_norm = sorted(set(pd.to_datetime(d).date() for d in trade_dates))
    daily["date_key"] = daily["Date"].dt.date

    patterns = []
    for td in trade_dates_norm:
        id_today = intraday[intraday["date"].dt.date == td]
        if id_today.empty:
            continue

        today_open = id_today.sort_values("time").iloc[0]["open"]
        today_close = id_today.sort_values("time").iloc[-1]["close"]

        prev_rows = daily[daily["date_key"] < td].tail(1)
        if prev_rows.empty:
            continue
        prev_close = float(prev_rows.iloc[0]["Close"])

        gap_pct = (today_open / prev_close - 1) * 100
        intraday_pct = (today_close / today_open - 1) * 100

        if gap_pct < 0 and intraday_pct > 0:
            pattern = "Gap Down → Reverse Up"
        elif gap_pct < 0 and intraday_pct <= 0:
            pattern = "Gap Down → Continue Down"
        elif gap_pct >= 0 and intraday_pct > 0:
            pattern = "Gap Up → Continue Up"
        else:
            pattern = "Gap Up → Reverse Down"

        total = (today_close / prev_close - 1) * 100

        patterns.append({
            "date": td,
            "pattern": pattern,
            "gap_pct": gap_pct,
            "intraday_pct": intraday_pct,
            "total_pct": total,
        })

    if not patterns:
        print("  No pattern data.")
        return None

    df = pd.DataFrame(patterns)

    # Summary by pattern
    print(f"\n  Signal days analyzed: {len(df)}")
    print(f"\n  {'Pattern':<30} {'Count':>6} {'%':>6} {'Avg Gap':>10} {'Avg ID':>10} {'Avg Total':>10}")
    print(f"  {'-'*75}")

    for pattern in ["Gap Down → Reverse Up", "Gap Down → Continue Down",
                    "Gap Up → Continue Up", "Gap Up → Reverse Down"]:
        subset = df[df["pattern"] == pattern]
        if len(subset) == 0:
            continue
        pct = len(subset) / len(df) * 100
        print(f"  {pattern:<30} {len(subset):>6} {pct:>5.0f}% "
              f"{subset['gap_pct'].mean():>+9.2f}% "
              f"{subset['intraday_pct'].mean():>+9.2f}% "
              f"{subset['total_pct'].mean():>+9.2f}%")

    # Best entry pattern for direction
    if direction == "long":
        best_pattern = "Gap Down → Reverse Up"
    else:
        best_pattern = "Gap Up → Reverse Down"

    subset = df[df["pattern"] == best_pattern]
    if len(subset) > 0:
        print(f"\n  ★ Primary day-trade setup: {best_pattern}")
        print(f"    Frequency: {len(subset)}/{len(df)} ({len(subset)/len(df)*100:.0f}%)")
        print(f"    Avg intraday return: {subset['intraday_pct'].mean():+.2f}%")

    return df


# =============================================================================
# SIGNAL ANALYSIS CONFIGURATIONS
# =============================================================================
SIGNAL_CONFIGS = {
    "uvxy_qqq79": {
        "name": "UVXY when QQQ RSI > 79",
        "trade_ticker": "UVXY",
        "direction": "long",
        "description": "Buy UVXY at open when QQQ closed RSI > 79 previous day",
    },
    "upro_spy21": {
        "name": "UPRO when SPY RSI < 21",
        "trade_ticker": "UPRO",
        "direction": "long",
        "description": "Buy UPRO at open when SPY closed RSI < 21 previous day",
    },
    "fas_short85": {
        "name": "Short FAS when RSI > 85",
        "trade_ticker": "FAS",
        "direction": "short",
        "description": "Short FAS at open when FAS closed RSI > 85 previous day",
    },
    "upro_short85": {
        "name": "Short UPRO when SPY RSI > 85",
        "trade_ticker": "UPRO",
        "direction": "short",
        "description": "Short UPRO at open when SPY closed RSI > 85 previous day",
    },
    "tqqq_double": {
        "name": "TQQQ on Double Signal",
        "trade_ticker": "TQQQ",
        "direction": "long",
        "description": "Buy TQQQ when GLD RSI > 79 AND USDU RSI < 25",
    },
    "soxs_dollar": {
        "name": "SOXS Dollar Squeeze",
        "trade_ticker": "SOXS",
        "direction": "long",
        "description": "Buy SOXS when SMH RSI > 79 AND USDU RSI > 70",
    },
    "cure_dip": {
        "name": "CURE Dip Buy",
        "trade_ticker": "CURE",
        "direction": "long",
        "description": "Buy CURE when CURE RSI < 21",
    },
    "defense_rotation": {
        "name": "Defensive Rotation → TQQQ",
        "trade_ticker": "TQQQ",
        "direction": "long",
        "description": "Buy TQQQ when XLP/XLU/XLV RSI > 79 and SPY/QQQ < 79",
    },
}


# =============================================================================
# MAIN
# =============================================================================
def run_full_analysis(signal_key):
    """Run all analyses for a single signal."""
    config = SIGNAL_CONFIGS[signal_key]
    trade_ticker = config["trade_ticker"]
    direction = config["direction"]

    print(f"\n{'#'*70}")
    print(f"# SIGNAL: {config['name']}")
    print(f"# {config['description']}")
    print(f"{'#'*70}")

    # Get signal dates
    trade_dates = get_signal_dates(signal_key)
    if not trade_dates:
        print(f"\n  No signal dates found. Check daily data in {DAILY_DIR}")
        return

    print(f"\n  Total signal dates from daily data: {len(trade_dates)}")
    print(f"  Date range: {min(trade_dates).strftime('%Y-%m-%d')} to {max(trade_dates).strftime('%Y-%m-%d')}")

    # Run all analyses
    analyze_overnight_vs_intraday(trade_ticker, signal_key, trade_dates, direction)
    analyze_hourly_profile(trade_ticker, signal_key, trade_dates, direction)
    analyze_gap_patterns(trade_ticker, signal_key, trade_dates, direction)
    analyze_5min_entry(trade_ticker, signal_key, trade_dates, direction)


def main():
    parser = argparse.ArgumentParser(description="Analyze intraday signals using Polygon data")
    parser.add_argument("--signal", help="Specific signal to analyze")
    parser.add_argument("--list", action="store_true", help="List available signals")
    parser.add_argument("--check", action="store_true", help="Check data availability")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable signals:")
        for key, config in SIGNAL_CONFIGS.items():
            print(f"  {key:<20} {config['name']}")
        return

    if args.check:
        print(f"\nChecking data availability...")
        print(f"\nDaily data ({DAILY_DIR}):")
        for ticker in ["SPY", "QQQ", "SMH", "GLD", "USDU", "XLP", "XLU", "XLV",
                        "UVXY", "TQQQ", "UPRO", "FAS", "SOXS", "CURE", "NAIL"]:
            daily = load_daily(ticker)
            status = f"{len(daily):,} rows" if daily is not None else "MISSING"
            print(f"  {ticker:<8} {status}")

        print(f"\nIntraday data ({POLYGON_DIR}):")
        for ticker in ["UVXY", "TQQQ", "UPRO", "FAS", "SOXS", "SPY", "QQQ",
                        "SMH", "GLD", "CURE", "SOXL"]:
            for res in ["60m", "5m"]:
                data = load_intraday(ticker, res)
                if data is not None:
                    first = data["date"].min()
                    last = data["date"].max()
                    print(f"  {ticker:<8} {res:<4} {len(data):>10,} bars  ({first} → {last})")
                else:
                    print(f"  {ticker:<8} {res:<4} {'MISSING':>10}")
        return

    if args.signal:
        if args.signal not in SIGNAL_CONFIGS:
            print(f"Unknown signal: {args.signal}")
            print(f"Available: {', '.join(SIGNAL_CONFIGS.keys())}")
            return
        run_full_analysis(args.signal)
    else:
        # Run top 4 day-tradeable signals
        priority = ["uvxy_qqq79", "upro_spy21", "fas_short85", "upro_short85"]
        for sig in priority:
            run_full_analysis(sig)

        # Also run the swing signals to confirm they're NOT day-tradeable
        print(f"\n\n{'='*70}")
        print("SWING TRADE SIGNALS (confirming overnight vs intraday split)")
        print(f"{'='*70}")
        for sig in ["tqqq_double", "cure_dip", "defense_rotation"]:
            run_full_analysis(sig)


if __name__ == "__main__":
    main()
