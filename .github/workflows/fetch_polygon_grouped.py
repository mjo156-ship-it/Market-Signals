#!/usr/bin/env python3
"""
Polygon.io Grouped Daily Bars Fetcher
======================================
Single API call returns OHLCV for ~8,000+ US equities.
Appends daily breadth metrics to breadth_daily.csv.

Usage:
    POLYGON_API_KEY=xxx python fetch_polygon_grouped.py [YYYY-MM-DD]

If no date given, fetches previous trading day.
Free tier: 5 calls/minute.

Setup:
    Add POLYGON_API_KEY to GitHub Actions secrets.
"""

import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
DATA_DIR = os.environ.get("BREADTH_DATA_DIR", "./data/breadth")
DAILY_CSV = os.path.join(DATA_DIR, "breadth_daily.csv")


def get_previous_trading_day():
    """Get the most recent trading day (skip weekends)."""
    d = datetime.now() - timedelta(days=1)
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def fetch_grouped_bars(date_str):
    """Fetch grouped daily bars from Polygon.io."""
    if not POLYGON_API_KEY:
        print("ERROR: POLYGON_API_KEY not set")
        return None

    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date_str}"
    params = {"adjusted": "true", "apiKey": POLYGON_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("resultsCount", 0) == 0:
            print(f"No results for {date_str} (market closed?)")
            return None

        results = data.get("results", [])
        print(f"Fetched {len(results)} tickers for {date_str}")
        return results

    except Exception as e:
        print(f"Polygon API error: {e}")
        return None


def compute_breadth(results, date_str):
    """Compute breadth metrics from grouped bars."""
    if not results:
        return None

    advancing = 0
    declining = 0
    unchanged = 0
    total = 0
    above_prev = 0  # crude proxy using open vs close

    for r in results:
        ticker = r.get("T", "")
        # Skip non-equity tickers (options, warrants, etc.)
        if len(ticker) > 5 or "." in ticker or "/" in ticker:
            continue

        o = r.get("o", 0)
        c = r.get("c", 0)
        if o == 0 or c == 0:
            continue

        total += 1
        if c > o:
            advancing += 1
        elif c < o:
            declining += 1
        else:
            unchanged += 1

    if total == 0:
        return None

    ratio = advancing / (advancing + declining) if (advancing + declining) > 0 else 0.5
    net = advancing - declining

    return {
        "date": date_str,
        "advancing": advancing,
        "declining": declining,
        "unchanged": unchanged,
        "total": total,
        "ratio": round(ratio, 4),
        "net": net,
    }


def append_to_daily(row):
    """Append breadth row to daily CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DAILY_CSV):
        df = pd.read_csv(DAILY_CSV)
        # Don't duplicate
        if row["date"] in df["date"].values:
            print(f"Date {row['date']} already exists in {DAILY_CSV}, skipping")
            return df
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(DAILY_CSV, index=False)
    print(f"Appended {row['date']} to {DAILY_CSV} ({len(df)} total rows)")
    return df


def compute_indicators(df):
    """Compute ZBT, McClellan, and breadth quality from daily CSV."""
    if len(df) < 2:
        return {}

    df = df.sort_values("date").reset_index(drop=True)
    ratios = df["ratio"].astype(float)
    nets = df["net"].astype(float)

    # ZBT: 10-day EMA of advance/decline ratio
    zbt_ema = ratios.ewm(alpha=0.1, adjust=False).mean()
    zbt_val = float(zbt_ema.iloc[-1])

    # Check for ZBT thrust: < 0.40 to > 0.615 within last 10 days
    zbt_thrust = False
    if len(zbt_ema) >= 10:
        recent = zbt_ema.tail(10)
        if recent.min() < 0.40 and zbt_val >= 0.615:
            zbt_thrust = True

    # McClellan Oscillator: 19d EMA - 39d EMA of net advances
    ema19 = nets.ewm(span=19, adjust=False).mean()
    ema39 = nets.ewm(span=39, adjust=False).mean()
    mcclellan = ema19 - ema39
    mcl_val = float(mcclellan.iloc[-1])
    mcl_prev = float(mcclellan.iloc[-2]) if len(mcclellan) >= 2 else mcl_val

    # McClellan Summation Index (cumulative)
    summation = float(mcclellan.sum())

    return {
        "date": df["date"].iloc[-1],
        "zbt_ema": round(zbt_val, 4),
        "zbt_zone": "OVERSOLD" if zbt_val < 0.40 else "THRUST" if zbt_val >= 0.615 else "NEUTRAL",
        "zbt_thrust": zbt_thrust,
        "mcclellan": round(mcl_val, 1),
        "mcl_ema19": round(float(ema19.iloc[-1]), 1),
        "mcl_ema39": round(float(ema39.iloc[-1]), 1),
        "mcl_direction": "RISING" if mcl_val > mcl_prev else "FALLING",
        "mcl_zone": "OVERSOLD" if mcl_val < -100 else "OVERBOUGHT" if mcl_val > 100 else "POSITIVE" if mcl_val > 0 else "NEGATIVE",
        "mcl_summation": round(summation, 0),
        "advancing": int(df["advancing"].iloc[-1]),
        "declining": int(df["declining"].iloc[-1]),
        "total": int(df["total"].iloc[-1]),
        "ratio": float(df["ratio"].iloc[-1]),
    }


def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else get_previous_trading_day()
    print(f"Fetching Polygon grouped bars for {date_str}")

    results = fetch_grouped_bars(date_str)
    if results is None:
        print("No data fetched. Exiting.")
        return

    row = compute_breadth(results, date_str)
    if row is None:
        print("No breadth computed. Exiting.")
        return

    print(f"  Adv: {row['advancing']} | Dec: {row['declining']} | Ratio: {row['ratio']}")

    df = append_to_daily(row)
    indicators = compute_indicators(df)

    if indicators:
        print(f"\n  ZBT:       {indicators['zbt_ema']} ({indicators['zbt_zone']})")
        print(f"  McClellan: {indicators['mcclellan']:+.1f} ({indicators['mcl_zone']}, {indicators['mcl_direction']})")
        print(f"  Summation: {indicators['mcl_summation']}")
        if indicators["zbt_thrust"]:
            print("  *** ZBT THRUST SIGNAL ***")

    # Save indicators as JSON for signal monitor to pick up
    ind_path = os.path.join(DATA_DIR, "latest_indicators.json")
    with open(ind_path, "w") as f:
        json.dump(indicators, f, indent=2)
    print(f"\n  Indicators saved to {ind_path}")


if __name__ == "__main__":
    main()
