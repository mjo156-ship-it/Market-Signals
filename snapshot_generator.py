#!/usr/bin/env python3
"""
Real-Time Signal Snapshot Generator
=====================================
Generates a JSON snapshot of all signal conditions for consumption by Claude
and other tools. Matches signal_monitor_complete.py calculations exactly.

USAGE:
  python snapshot_generator.py              # Full snapshot
  python snapshot_generator.py --compact    # Minimal output (smaller file)

OUTPUT:
  data/snapshot.json — Full signal state
  
SCHEDULE:
  Every 15 minutes during market hours via GitHub Actions
  9:30 AM – 4:15 PM ET, weekdays
"""

import os
import sys
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "snapshot.json"

# Complete ticker list — union of signal_monitor + polygon_downloader + dashboard extras
TICKERS = [
    # Core Indices
    'SPY', 'QQQ', 'SMH', 'IWM',
    # Defensive Sectors
    'XLP', 'XLU', 'XLV', 'XLF', 'XLE',
    # Safe Havens & Macro
    'GLD', 'TLT', 'HYG', 'LQD', 'TMV', 'USDU', 'BND',
    # Commodities
    'UCO', 'BOIL', 'DBC',
    # Volatility
    'UVXY', 'SVXY', 'VIXY', 'VIXM',
    # 3x Leveraged ETFs
    'TQQQ', 'SOXL', 'SOXS', 'TECL', 'FAS', 'UPRO',
    'NAIL', 'CURE', 'LABU', 'DRN', 'FNGO', 'HIBL',
    # International
    'EDC', 'YINN', 'KORU', 'EURL', 'INDL',
    # Crypto
    'BTC-USD',
    # Individual Stocks
    'AMD', 'NVDA',
    # Style/Factor
    'VOOV', 'VOOG', 'VTV', 'QQQE',
    # Managed Futures / Alternatives
    'KMLM', 'DBMF', 'CTA', 'BTAL',
]

# Playbook signal thresholds
PLAYBOOK = {
    'GLD_RSI_ABOVE': 79,
    'USDU_RSI_BELOW': 25,
    'XLP_RSI_ABOVE_TRIPLE': 65,
    'XLP_RSI_ABOVE_CASCADE': 75,
    'SPY_RSI_ABOVE': 79,
    'QQQ_RSI_ABOVE': 79,
    'SMH_RSI_ABOVE': 79,
    'XLF_RSI_ABOVE': 70,
    'UVXY_RSI_ABOVE': 82,
    'VIXM_RSI_BELOW': 25,
}

# SMH / SOXL levels
SMH_LEVELS = {
    'trim': 30,    # % above SMA200 → trim
    'warn': 35,    # → warning
    'sell': 40,    # → exit
}

# =============================================================================
# CALCULATIONS — exact match to signal_monitor_complete.py
# =============================================================================
def calculate_rsi_wilder(prices, period):
    """Wilder's RSI — matches signal monitor exactly."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def safe_float(value):
    """Safely convert to float, handling Series/arrays/NaN."""
    if isinstance(value, pd.Series):
        return float(value.iloc[-1]) if len(value) > 0 else None
    elif isinstance(value, np.ndarray):
        return float(value[-1]) if len(value) > 0 else None
    elif pd.isna(value):
        return None
    else:
        return float(value)


def compute_indicators(df):
    """Compute all indicators for a single ticker."""
    if len(df) < 200:
        return None

    try:
        close = df['Close']
        price = safe_float(close.iloc[-1])
        prev_close = safe_float(close.iloc[-2]) if len(close) > 1 else price

        rsi10 = safe_float(calculate_rsi_wilder(close, 10).iloc[-1])

        sma50 = safe_float(close.rolling(50).mean().iloc[-1])
        sma200 = safe_float(close.rolling(200).mean().iloc[-1])

        ema9 = safe_float(close.ewm(span=9, adjust=False).mean().iloc[-1])
        ema20 = safe_float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = safe_float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        ema200 = safe_float(close.ewm(span=200, adjust=False).mean().iloc[-1])

        # Returns
        ret_1d = (price / prev_close - 1) * 100 if prev_close else None
        ret_5d = safe_float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) > 6 else None
        ret_10d = safe_float((close.iloc[-1] / close.iloc[-11] - 1) * 100) if len(close) > 11 else None
        ret_20d = safe_float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) > 21 else None

        # Derived
        vs_sma200 = ((price / sma200) - 1) * 100 if sma200 and sma200 > 0 else None
        vs_sma50 = ((price / sma50) - 1) * 100 if sma50 and sma50 > 0 else None
        vs_ema9 = ((price / ema9) - 1) * 100 if ema9 and ema9 > 0 else None
        vs_ema20 = ((price / ema20) - 1) * 100 if ema20 and ema20 > 0 else None

        # EMA trend
        ema_bull = ema9 > ema20 if (ema9 and ema20) else None

        return {
            'price': round(price, 2) if price else None,
            'change_pct': round(ret_1d, 2) if ret_1d else None,
            'rsi10': round(rsi10, 1) if rsi10 else None,
            'ema9': round(ema9, 2) if ema9 else None,
            'ema20': round(ema20, 2) if ema20 else None,
            'ema50': round(ema50, 2) if ema50 else None,
            'ema200': round(ema200, 2) if ema200 else None,
            'sma50': round(sma50, 2) if sma50 else None,
            'sma200': round(sma200, 2) if sma200 else None,
            'ema_cross': 'BULL' if ema_bull else ('BEAR' if ema_bull is not None else None),
            'above_sma200': price > sma200 if (price and sma200) else None,
            'vs_sma200': round(vs_sma200, 1) if vs_sma200 is not None else None,
            'vs_sma50': round(vs_sma50, 1) if vs_sma50 is not None else None,
            'vs_ema9': round(vs_ema9, 2) if vs_ema9 is not None else None,
            'vs_ema20': round(vs_ema20, 2) if vs_ema20 is not None else None,
            'ret_1d': round(ret_1d, 2) if ret_1d is not None else None,
            'ret_5d': round(ret_5d, 2) if ret_5d is not None else None,
            'ret_10d': round(ret_10d, 2) if ret_10d is not None else None,
            'ret_20d': round(ret_20d, 2) if ret_20d is not None else None,
            # EMA detail for SPY-style breakdown
            'above_ema9': price > ema9 if (price and ema9) else None,
            'above_ema20': price > ema20 if (price and ema20) else None,
            'above_ema50': price > ema50 if (price and ema50) else None,
        }
    except Exception as e:
        print(f"  Error computing indicators: {e}")
        return None


# =============================================================================
# SIGNAL EVALUATION
# =============================================================================
def evaluate_signals(indicators):
    """Evaluate all playbook signals and return structured state."""
    signals = {}

    def rsi(ticker):
        return indicators.get(ticker, {}).get('rsi10')

    # --- Playbook conditions ---
    gld_rsi = rsi('GLD')
    usdu_rsi = rsi('USDU')
    xlp_rsi = rsi('XLP')
    spy_rsi = rsi('SPY')
    qqq_rsi = rsi('QQQ')
    smh_rsi = rsi('SMH')
    xlf_rsi = rsi('XLF')
    uvxy_rsi = rsi('UVXY')
    vixm_rsi = rsi('VIXM')

    signals['playbook'] = {
        'GLD_RSI_gt_79': {'value': gld_rsi, 'threshold': 79, 'active': gld_rsi > 79 if gld_rsi else False},
        'USDU_RSI_lt_25': {'value': usdu_rsi, 'threshold': 25, 'active': usdu_rsi < 25 if usdu_rsi else False},
        'XLP_RSI_gt_65': {'value': xlp_rsi, 'threshold': 65, 'active': xlp_rsi > 65 if xlp_rsi else False},
        'XLP_RSI_gt_75': {'value': xlp_rsi, 'threshold': 75, 'active': xlp_rsi > 75 if xlp_rsi else False},
        'SPY_RSI_gt_79': {'value': spy_rsi, 'threshold': 79, 'active': spy_rsi > 79 if spy_rsi else False},
        'QQQ_RSI_gt_79': {'value': qqq_rsi, 'threshold': 79, 'active': qqq_rsi > 79 if qqq_rsi else False},
        'SMH_RSI_gt_79': {'value': smh_rsi, 'threshold': 79, 'active': smh_rsi > 79 if smh_rsi else False},
        'XLF_RSI_gt_70': {'value': xlf_rsi, 'threshold': 70, 'active': xlf_rsi > 70 if xlf_rsi else False},
        'UVXY_RSI_gt_82': {'value': uvxy_rsi, 'threshold': 82, 'active': uvxy_rsi > 82 if uvxy_rsi else False},
        'VIXM_RSI_lt_25': {'value': vixm_rsi, 'threshold': 25, 'active': vixm_rsi < 25 if vixm_rsi else False},
    }

    # --- Combo signals ---
    gld_active = gld_rsi and gld_rsi > 79
    usdu_active = usdu_rsi and usdu_rsi < 25
    xlp_65_active = xlp_rsi and xlp_rsi > 65
    xlp_75_active = xlp_rsi and xlp_rsi > 75

    signals['combos'] = {
        'double_signal': {
            'active': bool(gld_active and usdu_active),
            'description': 'GLD RSI>79 + USDU RSI<25 → TQQQ buy',
            'components': {'GLD': gld_active, 'USDU': usdu_active},
        },
        'triple_signal': {
            'active': bool(gld_active and usdu_active and xlp_65_active),
            'description': 'Double + XLP RSI>65 → TQQQ high conviction',
            'components': {'GLD': gld_active, 'USDU': usdu_active, 'XLP': xlp_65_active},
        },
        'xlp_cascade': {
            'active': bool(xlp_75_active),
            'description': 'XLP RSI>75 → UVXY 1-day hold in Composer',
        },
    }

    # --- Bond momentum ---
    tlt_ind = indicators.get('TLT', {})
    bnd_ind = indicators.get('BND', {})
    tlt_ret10 = tlt_ind.get('ret_10d')
    bnd_ret10 = bnd_ind.get('ret_10d')
    bonds_rising = tlt_ret10 > 0 if tlt_ret10 is not None else None

    # UVXY conviction based on bond momentum
    uvxy_conviction = None
    if bonds_rising is True:
        uvxy_conviction = 'MODERATE'
    elif bonds_rising is False:
        uvxy_conviction = 'HIGH'

    signals['bond_momentum'] = {
        'direction': 'RISING' if bonds_rising else ('FALLING' if bonds_rising is False else 'UNKNOWN'),
        'tlt_ret_10d': round(tlt_ret10, 2) if tlt_ret10 is not None else None,
        'bnd_ret_10d': round(bnd_ret10, 2) if bnd_ret10 is not None else None,
        'uvxy_conviction': uvxy_conviction,
    }

    # --- SMH / SOXL levels ---
    smh_ind = indicators.get('SMH', {})
    smh_vs200 = smh_ind.get('vs_sma200')
    smh_sma200 = smh_ind.get('sma200')
    smh_price = smh_ind.get('price')

    signals['smh_levels'] = {
        'price': smh_price,
        'sma200': smh_sma200,
        'pct_above': round(smh_vs200, 1) if smh_vs200 is not None else None,
        'trim_level': round(smh_sma200 * (1 + SMH_LEVELS['trim'] / 100), 2) if smh_sma200 else None,
        'warn_level': round(smh_sma200 * (1 + SMH_LEVELS['warn'] / 100), 2) if smh_sma200 else None,
        'sell_level': round(smh_sma200 * (1 + SMH_LEVELS['sell'] / 100), 2) if smh_sma200 else None,
    }

    # --- Contrarian weakness setups ---
    contrarian = {}
    for ticker in ['FAS', 'TECL', 'FNGO', 'LABU', 'NAIL']:
        ind = indicators.get(ticker, {})
        r = ind.get('rsi10')
        below_200 = ind.get('above_sma200') is False
        bear_ema = ind.get('ema_cross') == 'BEAR'
        
        if r is not None and below_200 and r < 40:
            status = 'ACTIVE'
        elif r is not None and below_200 and r < 50:
            status = 'WATCH'
        else:
            status = 'INACTIVE'
        
        contrarian[ticker] = {
            'rsi10': r,
            'below_sma200': below_200,
            'bear_ema': bear_ema,
            'status': status,
        }
    signals['contrarian'] = contrarian

    # --- Extended / overbought warnings ---
    extended = {}
    for ticker in ['SOXL', 'KORU', 'EDC', 'HIBL', 'LABU', 'SMH']:
        ind = indicators.get(ticker, {})
        vs200 = ind.get('vs_sma200')
        r = ind.get('rsi10')
        if vs200 is not None and vs200 > 50:
            extended[ticker] = {
                'vs_sma200': vs200,
                'rsi10': r,
                'warning': 'EXTENDED' if vs200 > 100 else 'ELEVATED',
            }
    signals['extended'] = extended

    # --- Active alerts summary ---
    active_alerts = []
    if signals['combos']['double_signal']['active']:
        active_alerts.append('🟢🔥 DOUBLE SIGNAL: GLD/USDU → TQQQ buy')
    if signals['combos']['triple_signal']['active']:
        active_alerts.append('🟢🔥🔥 TRIPLE SIGNAL: GLD/USDU/XLP → TQQQ high conviction')
    if signals['combos']['xlp_cascade']['active']:
        active_alerts.append('🟡 XLP CASCADE: RSI>75 → UVXY 1-day hold')
    for ticker, c in contrarian.items():
        if c['status'] == 'ACTIVE':
            active_alerts.append(f'🟢 {ticker} CONTRARIAN: RSI<40 + below SMA200')
        elif c['status'] == 'WATCH':
            active_alerts.append(f'⚠️ {ticker} WATCH: below SMA200, approaching oversold')
    for ticker, e in extended.items():
        active_alerts.append(f'🔴 {ticker} {e["warning"]}: {e["vs_sma200"]:+.0f}% above SMA200')

    signals['active_alerts'] = active_alerts

    return signals


# =============================================================================
# MAIN
# =============================================================================
def main():
    compact = '--compact' in sys.argv

    print(f"Generating signal snapshot at {datetime.now()}")
    print(f"Downloading data for {len(TICKERS)} tickers...")

    # Download all data
    data = {}
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, period='2y', progress=False)
            if len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
        except Exception as e:
            print(f"  Error downloading {ticker}: {e}")
    
    print(f"Downloaded data for {len(data)} tickers")

    # Compute indicators
    indicators = {}
    for ticker, df in data.items():
        ind = compute_indicators(df)
        if ind:
            indicators[ticker] = ind

    print(f"Computed indicators for {len(indicators)} tickers")

    # Evaluate signals
    signals = evaluate_signals(indicators)

    # Build snapshot
    now_utc = datetime.now(timezone.utc)
    # Eastern time offset (simplified — doesn't handle DST perfectly but close enough)
    et_offset = timedelta(hours=-5)
    now_et = now_utc + et_offset

    snapshot = {
        'meta': {
            'generated_utc': now_utc.isoformat(),
            'generated_et': now_et.strftime('%Y-%m-%d %H:%M:%S ET'),
            'ticker_count': len(indicators),
            'version': '1.0',
        },
        'signals': signals,
    }

    if not compact:
        snapshot['indicators'] = indicators
    else:
        # Compact: only include key tickers + any with active signals
        key_set = {'SPY', 'QQQ', 'SMH', 'GLD', 'USDU', 'XLP', 'TLT', 'UVXY',
                    'SVXY', 'VIXM', 'TQQQ', 'SOXL', 'UPRO', 'FAS', 'TECL',
                    'FNGO', 'KMLM', 'BTAL', 'BND', 'BTC-USD'}
        snapshot['indicators'] = {k: v for k, v in indicators.items() if k in key_set}

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(snapshot, f, indent=2, default=str)

    print(f"\nSnapshot written to {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size:,} bytes")

    # Summary
    print(f"\n{'='*60}")
    print(f"SNAPSHOT SUMMARY — {now_et.strftime('%Y-%m-%d %H:%M ET')}")
    print(f"{'='*60}")
    
    if signals['active_alerts']:
        for alert in signals['active_alerts']:
            print(f"  {alert}")
    else:
        print("  No active signals")

    pb = signals['playbook']
    print(f"\n  Playbook Conditions:")
    for key, val in pb.items():
        status = '🟢' if val['active'] else '○'
        print(f"    {status} {key}: {val['value']}")

    bm = signals['bond_momentum']
    print(f"\n  Bond Momentum: {bm['direction']} (TLT 10d: {bm['tlt_ret_10d']}%)")
    print(f"  UVXY Conviction: {bm['uvxy_conviction']}")


if __name__ == '__main__':
    main()
