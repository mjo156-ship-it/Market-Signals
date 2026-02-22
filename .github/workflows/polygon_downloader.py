#!/usr/bin/env python3
"""
Polygon.io Intraday Data Downloader
====================================
Downloads historical intraday bars for all monitored tickers.

USAGE:
  # Initial backfill (run once - takes ~30-60 min for all tickers)
  python polygon_downloader.py --backfill

  # Daily update (append new data - run via GitHub Actions)
  python polygon_downloader.py --update

  # Download specific tickers only
  python polygon_downloader.py --backfill --tickers UVXY TQQQ SPY

  # Specific resolution
  python polygon_downloader.py --backfill --resolution 5min

REQUIRES:
  pip install polygon-api-client pandas
  Environment variable: POLYGON_API_KEY

DATA STRUCTURE:
  data/polygon/TICKER_60m.csv   - 60-minute bars (all tickers)
  data/polygon/TICKER_5m.csv    - 5-minute bars (day-trade candidates)
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path

try:
    from polygon import RESTClient
except ImportError:
    print("ERROR: polygon-api-client not installed.")
    print("Run: pip install polygon-api-client")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data/polygon")

# API key from environment
API_KEY = os.environ.get("POLYGON_API_KEY", "")

# Ticker definitions with start dates (when each ETF launched)
# Using conservative start dates to avoid requesting data before inception
TICKER_CONFIG = {
    # Core Indices
    "SPY":  {"start": "2005-01-01", "tiers": ["60m", "5m"]},
    "QQQ":  {"start": "2005-01-01", "tiers": ["60m", "5m"]},
    "SMH":  {"start": "2005-01-01", "tiers": ["60m", "5m"]},
    "IWM":  {"start": "2005-01-01", "tiers": ["60m"]},

    # Volatility - TOP DAY TRADE CANDIDATES
    "UVXY": {"start": "2011-10-01", "tiers": ["60m", "5m"]},
    "SVXY": {"start": "2011-10-01", "tiers": ["60m"]},
    "VIXY": {"start": "2011-01-01", "tiers": ["60m"]},

    # 3x Leveraged - DAY TRADE CANDIDATES
    "TQQQ": {"start": "2010-02-01", "tiers": ["60m", "5m"]},
    "UPRO": {"start": "2009-06-01", "tiers": ["60m", "5m"]},
    "SOXL": {"start": "2010-03-01", "tiers": ["60m", "5m"]},
    "SOXS": {"start": "2010-03-01", "tiers": ["60m", "5m"]},
    "TECL": {"start": "2008-12-01", "tiers": ["60m"]},
    "FAS":  {"start": "2008-11-01", "tiers": ["60m", "5m"]},
    "CURE": {"start": "2011-06-01", "tiers": ["60m"]},
    "LABU": {"start": "2015-05-01", "tiers": ["60m"]},
    "NAIL": {"start": "2015-08-01", "tiers": ["60m"]},
    "FNGO": {"start": "2018-01-01", "tiers": ["60m"]},
    "HIBL": {"start": "2020-04-01", "tiers": ["60m"]},

    # Defensive Sectors (signal triggers)
    "XLP":  {"start": "2005-01-01", "tiers": ["60m"]},
    "XLU":  {"start": "2005-01-01", "tiers": ["60m"]},
    "XLV":  {"start": "2005-01-01", "tiers": ["60m"]},
    "XLF":  {"start": "2005-01-01", "tiers": ["60m"]},
    "XLE":  {"start": "2005-01-01", "tiers": ["60m"]},

    # Safe Havens & Macro
    "GLD":  {"start": "2005-01-01", "tiers": ["60m", "5m"]},
    "TLT":  {"start": "2005-01-01", "tiers": ["60m"]},
    "HYG":  {"start": "2007-04-01", "tiers": ["60m"]},
    "USDU": {"start": "2013-12-01", "tiers": ["60m"]},
    "TMV":  {"start": "2009-04-01", "tiers": ["60m"]},

    # Individual Stocks
    "AMD":  {"start": "2005-01-01", "tiers": ["60m"]},
    "NVDA": {"start": "2005-01-01", "tiers": ["60m"]},

    # Managed Futures / Alternatives
    "KMLM": {"start": "2020-12-01", "tiers": ["60m"]},
    "DBMF": {"start": "2019-05-01", "tiers": ["60m"]},
    "CTA":  {"start": "2022-02-01", "tiers": ["60m"]},
    "BTAL": {"start": "2011-09-01", "tiers": ["60m"]},

    # Style/Factor
    "VOOV": {"start": "2010-09-01", "tiers": ["60m"]},
    "VOOG": {"start": "2010-09-01", "tiers": ["60m"]},

    # Crypto (Polygon uses X: prefix for crypto)
    "X:BTCUSD": {"start": "2014-01-01", "tiers": ["60m"], "filename": "BTCUSD"},
}

# Resolution mapping
RESOLUTION_MAP = {
    "60m": {"multiplier": 1, "timespan": "hour"},
    "5m":  {"multiplier": 5, "timespan": "minute"},
    "1m":  {"multiplier": 1, "timespan": "minute"},
}

# Maximum date range per API request to avoid timeouts
# Polygon handles pagination internally but large ranges can be slow
CHUNK_DAYS = {
    "60m": 365,      # 1 year chunks for hourly
    "5m":  30,        # 1 month chunks for 5-min
    "1m":  7,         # 1 week chunks for 1-min
}


# =============================================================================
# DOWNLOADER
# =============================================================================
class PolygonDownloader:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "POLYGON_API_KEY not set. Export it or pass via --api-key.\n"
                "  export POLYGON_API_KEY=your_key_here"
            )
        self.client = RESTClient(api_key=api_key)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def get_filename(self, ticker: str, resolution: str) -> Path:
        """Get CSV filename for a ticker/resolution combo."""
        # Use custom filename for crypto tickers (X:BTCUSD -> BTCUSD)
        config = TICKER_CONFIG.get(ticker, {})
        name = config.get("filename", ticker.replace(":", "_"))
        return DATA_DIR / f"{name}_{resolution}.csv"

    def download_bars(self, ticker: str, resolution: str,
                      start_date: str, end_date: str) -> pd.DataFrame:
        """Download aggregate bars from Polygon for a date range."""
        res_config = RESOLUTION_MAP[resolution]
        chunk_days = CHUNK_DAYS[resolution]

        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        all_bars = []
        current_start = start

        while current_start < end:
            current_end = min(current_start + timedelta(days=chunk_days), end)

            try:
                bars = []
                for a in self.client.list_aggs(
                    ticker=ticker,
                    multiplier=res_config["multiplier"],
                    timespan=res_config["timespan"],
                    from_=current_start.strftime("%Y-%m-%d"),
                    to=current_end.strftime("%Y-%m-%d"),
                    adjusted=True,
                    sort="asc",
                    limit=50000,
                ):
                    bars.append({
                        "timestamp": a.timestamp,
                        "open": a.open,
                        "high": a.high,
                        "low": a.low,
                        "close": a.close,
                        "volume": a.volume,
                        "vwap": getattr(a, "vwap", None),
                        "transactions": getattr(a, "transactions", None),
                    })

                all_bars.extend(bars)

                period_str = f"{current_start} to {current_end}"
                print(f"    {period_str}: {len(bars):,} bars")

            except Exception as e:
                print(f"    ERROR {current_start} to {current_end}: {e}")
                # Brief pause on error, then continue
                time.sleep(2)

            current_start = current_end + timedelta(days=1)

        if not all_bars:
            return pd.DataFrame()

        df = pd.DataFrame(all_bars)

        # Convert timestamp (Unix ms) to datetime
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["datetime"] = df["datetime"].dt.tz_convert("US/Eastern")

        # Create clean date/time columns
        df["date"] = df["datetime"].dt.date
        df["time"] = df["datetime"].dt.strftime("%H:%M")

        # Sort and deduplicate
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

        # Reorder columns
        cols = ["datetime", "date", "time", "open", "high", "low", "close",
                "volume", "vwap", "transactions", "timestamp"]
        df = df[[c for c in cols if c in df.columns]]

        return df

    def backfill_ticker(self, ticker: str, resolution: str,
                        force: bool = False) -> int:
        """Backfill full history for a single ticker/resolution."""
        filepath = self.get_filename(ticker, resolution)
        config = TICKER_CONFIG.get(ticker, {})
        start_date = config.get("start", "2010-01-01")
        end_date = date.today().strftime("%Y-%m-%d")

        # If file exists and not forcing, skip
        if filepath.exists() and not force:
            existing = pd.read_csv(filepath)
            print(f"  {ticker} {resolution}: EXISTS ({len(existing):,} rows) - skipping")
            print(f"    Use --force to re-download")
            return len(existing)

        print(f"  {ticker} {resolution}: Downloading {start_date} to {end_date}...")
        df = self.download_bars(ticker, resolution, start_date, end_date)

        if df.empty:
            print(f"  {ticker} {resolution}: No data returned")
            return 0

        # Filter to regular trading hours for equities (9:30 AM - 4:00 PM ET)
        # Crypto trades 24/7 so don't filter
        if not ticker.startswith("X:"):
            df_rth = df[
                (df["time"] >= "09:30") & (df["time"] < "16:00")
            ].copy()
            excluded = len(df) - len(df_rth)
            if excluded > 0:
                print(f"    Filtered {excluded:,} pre/post-market bars")
            df = df_rth

        df.to_csv(filepath, index=False)
        print(f"  {ticker} {resolution}: SAVED {len(df):,} bars → {filepath}")
        return len(df)

    def update_ticker(self, ticker: str, resolution: str) -> int:
        """Incrementally update a ticker with new data since last bar."""
        filepath = self.get_filename(ticker, resolution)

        if not filepath.exists():
            print(f"  {ticker} {resolution}: No existing file, running backfill...")
            return self.backfill_ticker(ticker, resolution)

        # Read existing data to find last timestamp
        existing = pd.read_csv(filepath)
        if existing.empty:
            return self.backfill_ticker(ticker, resolution, force=True)

        last_date = pd.to_datetime(existing["date"]).max()
        start_date = (last_date - timedelta(days=2)).strftime("%Y-%m-%d")  # overlap for safety
        end_date = date.today().strftime("%Y-%m-%d")

        if start_date >= end_date:
            print(f"  {ticker} {resolution}: Already up to date")
            return len(existing)

        print(f"  {ticker} {resolution}: Updating from {start_date}...")
        new_df = self.download_bars(ticker, resolution, start_date, end_date)

        if new_df.empty:
            print(f"  {ticker} {resolution}: No new data")
            return len(existing)

        # Filter RTH for equities
        if not ticker.startswith("X:"):
            new_df = new_df[
                (new_df["time"] >= "09:30") & (new_df["time"] < "16:00")
            ].copy()

        # Combine and deduplicate
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        combined = combined.sort_values("timestamp").reset_index(drop=True)

        new_bars = len(combined) - len(existing)
        combined.to_csv(filepath, index=False)
        print(f"  {ticker} {resolution}: +{new_bars} new bars (total: {len(combined):,})")
        return len(combined)

    def run_backfill(self, tickers: list = None, resolutions: list = None,
                     force: bool = False):
        """Run full backfill for specified or all tickers."""
        tickers = tickers or list(TICKER_CONFIG.keys())
        total_bars = 0
        total_files = 0

        print(f"\n{'='*60}")
        print(f"POLYGON BACKFILL - {len(tickers)} tickers")
        print(f"{'='*60}\n")

        for ticker in tickers:
            config = TICKER_CONFIG.get(ticker, {"tiers": ["60m"]})
            ticker_resolutions = resolutions or config.get("tiers", ["60m"])

            print(f"\n[{ticker}]")
            for res in ticker_resolutions:
                if res not in RESOLUTION_MAP:
                    print(f"  Skipping unknown resolution: {res}")
                    continue
                bars = self.backfill_ticker(ticker, res, force=force)
                total_bars += bars
                if bars > 0:
                    total_files += 1

        print(f"\n{'='*60}")
        print(f"BACKFILL COMPLETE")
        print(f"  Files: {total_files}")
        print(f"  Total bars: {total_bars:,}")
        print(f"  Location: {DATA_DIR.resolve()}")
        print(f"{'='*60}\n")

    def run_update(self, tickers: list = None, resolutions: list = None):
        """Run incremental update for all tickers."""
        tickers = tickers or list(TICKER_CONFIG.keys())
        new_bars_total = 0

        print(f"\n{'='*60}")
        print(f"POLYGON UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}\n")

        for ticker in tickers:
            config = TICKER_CONFIG.get(ticker, {"tiers": ["60m"]})
            ticker_resolutions = resolutions or config.get("tiers", ["60m"])

            for res in ticker_resolutions:
                if res not in RESOLUTION_MAP:
                    continue
                bars = self.update_ticker(ticker, res)

        print(f"\n{'='*60}")
        print(f"UPDATE COMPLETE")
        print(f"{'='*60}\n")

    def show_status(self):
        """Show current data status for all tickers."""
        print(f"\n{'='*70}")
        print(f"POLYGON DATA STATUS")
        print(f"{'='*70}")
        print(f"{'Ticker':<12} {'Res':<6} {'Bars':>10} {'First Date':>12} {'Last Date':>12} {'File Size':>10}")
        print(f"{'-'*70}")

        total_bars = 0
        total_size = 0

        for ticker in TICKER_CONFIG:
            config = TICKER_CONFIG[ticker]
            for res in config.get("tiers", ["60m"]):
                filepath = self.get_filename(ticker, res)
                if filepath.exists():
                    df = pd.read_csv(filepath, nrows=1)  # Just check structure
                    df_full = pd.read_csv(filepath)
                    bars = len(df_full)
                    first = df_full["date"].iloc[0] if bars > 0 else "N/A"
                    last = df_full["date"].iloc[-1] if bars > 0 else "N/A"
                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    total_bars += bars
                    total_size += size_mb
                    name = config.get("filename", ticker.replace(":", "_"))
                    print(f"{name:<12} {res:<6} {bars:>10,} {str(first):>12} {str(last):>12} {size_mb:>8.1f} MB")
                else:
                    name = config.get("filename", ticker.replace(":", "_"))
                    print(f"{name:<12} {res:<6} {'—':>10} {'—':>12} {'—':>12} {'—':>10}")

        print(f"{'-'*70}")
        print(f"{'TOTAL':<12} {'':<6} {total_bars:>10,} {'':<12} {'':<12} {total_size:>8.1f} MB")
        print()


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Download intraday data from Polygon.io",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full backfill (first time setup)
  python polygon_downloader.py --backfill

  # Daily update (run via cron/GitHub Actions)
  python polygon_downloader.py --update

  # Backfill specific tickers
  python polygon_downloader.py --backfill --tickers UVXY TQQQ SPY

  # Only 5-minute data
  python polygon_downloader.py --backfill --resolution 5m

  # Check what data you have
  python polygon_downloader.py --status

  # Re-download everything (overwrite existing)
  python polygon_downloader.py --backfill --force
        """,
    )

    parser.add_argument("--backfill", action="store_true",
                        help="Download full history for all tickers")
    parser.add_argument("--update", action="store_true",
                        help="Incrementally update existing data")
    parser.add_argument("--status", action="store_true",
                        help="Show current data status")
    parser.add_argument("--tickers", nargs="+",
                        help="Specific tickers to download (default: all)")
    parser.add_argument("--resolution", choices=["60m", "5m", "1m"],
                        help="Specific resolution (default: per-ticker config)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing files on backfill")
    parser.add_argument("--api-key",
                        help="Polygon API key (default: POLYGON_API_KEY env var)")

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or API_KEY
    if not api_key:
        print("ERROR: No API key provided.")
        print("  Set POLYGON_API_KEY environment variable or use --api-key")
        sys.exit(1)

    downloader = PolygonDownloader(api_key)

    # Parse resolution
    resolutions = [args.resolution] if args.resolution else None

    if args.status:
        downloader.show_status()
    elif args.backfill:
        downloader.run_backfill(
            tickers=args.tickers,
            resolutions=resolutions,
            force=args.force,
        )
    elif args.update:
        downloader.run_update(
            tickers=args.tickers,
            resolutions=resolutions,
        )
    else:
        parser.print_help()
        print("\nTip: Start with --backfill for initial download, then --update daily.")


if __name__ == "__main__":
    main()
