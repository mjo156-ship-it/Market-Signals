#!/usr/bin/env python3
"""
Daily OHLCV data updater.

Reads `data/ohlcv/tickers.txt` and maintains one CSV per ticker at
`data/ohlcv/{TICKER}.csv` in the standard yfinance format:
    Date,Open,High,Low,Close,Adj Close,Volume

Modes:
  --mode=append   (default) Fetch from last-stored-date+1 through today.
                  If file doesn't exist, fetch full history from 1990-01-01.
  --mode=rewrite  Fetch full history for every ticker, overwriting files.
                  Used weekly to capture retroactive Adj Close adjustments
                  (splits, dividends).
  --mode=backfill Like append, but tolerates skipped runs (runner outage,
                  weekends, holidays) — same code path as append today.

Per-ticker errors are logged and skipped; the script exits 0 on partial
failure so the workflow's commit step still runs. Exit 1 only if every
ticker fails (network outage).

Writes `data/ohlcv/_last_run.json` summarizing the run.
"""

from __future__ import annotations  # tolerate `str | None` on Python 3.7-3.9

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


REPO_ROOT = Path(__file__).resolve().parent.parent
OHLCV_DIR = REPO_ROOT / 'data' / 'ohlcv'
TICKERS_FILE = OHLCV_DIR / 'tickers.txt'
MANIFEST_FILE = OHLCV_DIR / '_last_run.json'
SEED_START = '1990-01-01'
SNAPSHOT_URL = ('https://raw.githubusercontent.com/mjo156-ship-it/'
                'Market-Signals/refs/heads/main/data/snapshot.json')
SANITY_PCT_THRESHOLD = 0.5  # warn if yfinance close differs from snapshot by >0.5%


def ticker_to_path(ticker: str) -> Path:
    """Map ticker symbol to CSV filename. Hyphens kept (BTC-USD), slashes → underscore."""
    safe = ticker.replace('/', '_')
    return OHLCV_DIR / f'{safe}.csv'


def read_tickers() -> list:
    if not TICKERS_FILE.exists():
        raise FileNotFoundError(f'ticker list missing: {TICKERS_FILE}')
    with open(TICKERS_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def fetch_snapshot_prices() -> dict:
    """Load live snapshot.json from main; return {ticker: price} for sanity check."""
    try:
        r = requests.get(SNAPSHOT_URL, timeout=15)
        r.raise_for_status()
        snap = r.json()
        out = {}
        for tkr, ind in (snap.get('indicators') or {}).items():
            p = ind.get('price')
            if p is not None:
                out[tkr] = float(p)
        return out
    except Exception as e:
        print(f'  [snapshot] WARN: could not fetch snapshot for sanity check: {e}')
        return {}


def yf_fetch(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download from yfinance, flatten MultiIndex, return DF with Date as a column."""
    kwargs = {'start': start, 'auto_adjust': False, 'progress': False}
    if end is not None:
        kwargs['end'] = end
    df = yf.download(ticker, **kwargs)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Standardize column order
    expected = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    keep = [c for c in expected if c in df.columns]
    df = df[keep].copy()
    df = df.reset_index()  # Date becomes a column
    # Date column may be named 'Date' or 'index' depending on yfinance version
    if 'Date' not in df.columns:
        first = df.columns[0]
        df = df.rename(columns={first: 'Date'})
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV in the canonical format (no index, Date first)."""
    cols = ['Date'] + [c for c in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                       if c in df.columns]
    df = df[cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def last_date_in_csv(path: Path) -> str | None:
    """Return the last Date string in an existing CSV, or None."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=['Date'])
        if len(df) == 0:
            return None
        return str(df['Date'].iloc[-1])
    except Exception:
        return None


def update_ticker(ticker: str, mode: str) -> tuple:
    """
    Update one ticker. Returns (status, info_dict) where status is one of
    'ok', 'skip', 'fail'. info_dict carries error / sanity-warn details.
    """
    path = ticker_to_path(ticker)

    if mode == 'rewrite' or not path.exists():
        df = yf_fetch(ticker, start=SEED_START)
        if df.empty:
            return 'fail', {'error': 'yfinance returned no data (full history)'}
        write_csv(df, path)
        return 'ok', {'rows': len(df), 'last_date': df['Date'].iloc[-1]}

    # append / backfill: fetch from (last_date + 1) through today
    last = last_date_in_csv(path)
    if last is None:
        # File exists but is unreadable / empty — treat as full reseed
        df = yf_fetch(ticker, start=SEED_START)
        if df.empty:
            return 'fail', {'error': 'CSV unreadable AND yfinance returned empty'}
        write_csv(df, path)
        return 'ok', {'rows': len(df), 'last_date': df['Date'].iloc[-1]}

    start_dt = (datetime.strptime(last, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    if start_dt > today:
        return 'skip', {'reason': f'already up to date (last={last})'}

    new = yf_fetch(ticker, start=start_dt)
    if new.empty:
        # Likely a weekend / holiday gap — not an error.
        return 'skip', {'reason': f'no new rows since {last}'}

    # Append to existing file, dedup on Date (in case of overlap)
    existing = pd.read_csv(path)
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=['Date'], keep='last').sort_values('Date')
    write_csv(combined, path)
    return 'ok', {'rows_added': len(new), 'total_rows': len(combined),
                  'last_date': combined['Date'].iloc[-1]}


def sanity_check(ticker: str, snapshot_prices: dict) -> dict | None:
    """Compare today's close in the CSV against the live snapshot price.
    Returns a warning dict if divergent, else None. Skips if ticker not in snapshot."""
    if ticker not in snapshot_prices:
        return None
    path = ticker_to_path(ticker)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if len(df) == 0:
            return None
        yf_close = float(df['Close'].iloc[-1])
        snap_price = snapshot_prices[ticker]
        if snap_price == 0:
            return None
        pct_diff = abs(yf_close - snap_price) / snap_price * 100
        if pct_diff > SANITY_PCT_THRESHOLD:
            return {
                'ticker': ticker,
                'yf_close': round(yf_close, 4),
                'snapshot_price': round(snap_price, 4),
                'pct_diff': round(pct_diff, 3),
            }
    except Exception:
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['append', 'rewrite', 'backfill'],
                        default='append', help='update mode (default: append)')
    args = parser.parse_args()

    tickers = read_tickers()
    print(f'Loaded {len(tickers)} tickers from {TICKERS_FILE}')
    print(f'Mode: {args.mode}')

    snapshot_prices = {}
    if args.mode in ('append', 'backfill'):
        print('Fetching live snapshot for sanity check...')
        snapshot_prices = fetch_snapshot_prices()
        print(f'  snapshot has prices for {len(snapshot_prices)} tickers')

    succeeded = []
    failed = []
    skipped = []
    warnings = []

    for ticker in tickers:
        try:
            status, info = update_ticker(ticker, args.mode)
            if status == 'ok':
                succeeded.append(ticker)
                rows = info.get('rows') or info.get('rows_added') or 0
                last = info.get('last_date', '?')
                print(f'  ✓ {ticker:<10}  rows={rows:<6}  last={last}')
            elif status == 'skip':
                skipped.append(ticker)
                print(f'  - {ticker:<10}  {info.get("reason", "skipped")}')
            else:
                failed.append({'ticker': ticker, 'error': info.get('error', 'unknown')})
                print(f'  ✗ {ticker:<10}  ERROR: {info.get("error", "unknown")}')
                continue

            # Sanity check (only meaningful in append/backfill mode AFTER write)
            if args.mode in ('append', 'backfill') and status == 'ok':
                warn = sanity_check(ticker, snapshot_prices)
                if warn is not None:
                    warnings.append(warn)
                    print(f'    ⚠  WARNING: yf_close={warn["yf_close"]} vs snapshot={warn["snapshot_price"]} '
                          f'(diff {warn["pct_diff"]}%)')
        except Exception as e:
            failed.append({'ticker': ticker, 'error': f'{type(e).__name__}: {e}'})
            print(f'  ✗ {ticker:<10}  UNCAUGHT: {type(e).__name__}: {e}')

    # Summary
    n_total = len(tickers)
    print('')
    print(f'Summary: {len(succeeded)} ok, {len(skipped)} skipped, {len(failed)} failed (of {n_total})')
    if failed:
        print('Failed tickers:')
        for f in failed:
            print(f'  {f["ticker"]:<10}  {f["error"]}')
    if warnings:
        print(f'Sanity warnings ({len(warnings)}):')
        for w in warnings:
            print(f'  {w["ticker"]:<10}  yf={w["yf_close"]} snap={w["snapshot_price"]} '
                  f'diff={w["pct_diff"]}%')

    # Manifest
    manifest = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'mode': args.mode,
        'succeeded': succeeded,
        'skipped': skipped,
        'failed': failed,
        'warnings': warnings,
        'counts': {'total': n_total, 'ok': len(succeeded),
                   'skipped': len(skipped), 'failed': len(failed)},
    }
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'Manifest: {MANIFEST_FILE}')

    # Exit 1 only if EVERY ticker failed (likely a systemic outage)
    if failed and not succeeded and not skipped:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
