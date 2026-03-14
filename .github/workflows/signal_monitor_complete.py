#!/usr/bin/env python3
"""
Comprehensive Market Signal Monitor v4.0
========================================
Monitors all backtested trading signals and sends alerts.
NEW IN v4: Rolling Brier Score calibration tracking for all signals.

SCHEDULE: Two emails daily (weekdays)
- 3:15 PM ET: Pre-close preview
- 4:05 PM ET: Market close confirmation

BRIER SCORES — WHAT THEY MEAN
==============================
Every time the signal monitor says "SPY RSI<30 → UPRO: 69% win rate," it's
making a prediction: "there is a 69% chance UPRO will be higher in 5 days."

The Brier Score measures how accurate that prediction is over time.

  Brier Score = average of (predicted_probability - actual_outcome)²

  - Predicted probability = the signal's historical win rate (e.g., 0.69)
  - Actual outcome = 1 if the trade won, 0 if it lost

  Perfect predictions:  Brier = 0.00 (every 69% call lands correctly)
  Random coin flip:     Brier = 0.25
  Always wrong:         Brier = 1.00

The Brier Skill Score (BSS) compares the signal to a naive baseline that
just uses the unconditional win rate (how often the asset goes up in any
5-day window, regardless of signal):

  BSS = 1 - (Signal Brier / Baseline Brier)

  BSS > 0:  Signal ADDS value vs just guessing the base rate
  BSS = 0:  Signal is no better than guessing
  BSS < 0:  Signal is WORSE than guessing — actively harmful

DEGRADATION ALERTS
==================
The monitor tracks a rolling window of the last 20 signal occurrences.
When a signal's trailing win rate drops below 50% (for signals with
historical WR > 65%), it fires a degradation warning. This is the early
warning system for structural market changes — like the 0DTE compression
that killed UVXY multi-day holds post-2020.

Degradation alert thresholds:
  🔴 CRITICAL:  Trailing WR < 40% (signal is actively losing money)
  🟡 WARNING:   Trailing WR < 50% (signal has lost its edge)
  🟢 HEALTHY:   Trailing WR within 15% of historical WR
"""

import os
import json
import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from pathlib import Path
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================
SENDER_EMAIL = os.environ.get('SENDER_EMAIL', '')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD', '')
RECIPIENT_EMAIL = os.environ.get('RECIPIENT_EMAIL', '')
PHONE_EMAIL = os.environ.get('PHONE_EMAIL', '')

IS_PRECLOSE = len(sys.argv) > 1 and sys.argv[1] == 'preclose'

# Path for persistent Brier score data
BRIER_DATA_PATH = Path(os.environ.get('BRIER_DATA_PATH',
                        Path(__file__).parent / 'brier_score_history.json'))

# =============================================================================
# CALCULATIONS
# =============================================================================
def calculate_rsi_wilder(prices, period):
    """Calculate Wilder's RSI"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def safe_float(value):
    """Safely convert a value to float, handling Series and arrays"""
    if isinstance(value, pd.Series):
        return float(value.iloc[-1]) if len(value) > 0 else 0.0
    elif isinstance(value, np.ndarray):
        return float(value[-1]) if len(value) > 0 else 0.0
    elif pd.isna(value):
        return 0.0
    else:
        return float(value)

def download_data(tickers, period='2y'):
    """Download data for multiple tickers"""
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False)
            if len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    return data

# =============================================================================
# BRIER SCORE SIGNAL REGISTRY
# =============================================================================
# Each signal definition includes:
#   - id: unique identifier
#   - name: human-readable name
#   - condition: function(indicators, data) -> bool (is signal active today?)
#   - target: the ETF whose forward return determines win/loss
#   - hold_days: how many days forward to measure
#   - direction: 'long' (win = positive return) or 'short' (win = negative return)
#   - historical_wr: the backtested win rate used as the prediction
#   - tier: signal confidence tier (1=robust, 2=moderate, 3=low-n)
#   - min_n_for_alert: minimum signal count before Brier alerts activate

SIGNAL_REGISTRY = [
    # --- TIER 1: Statistically Robust ---
    {
        'id': 'spy_rsi_lt21_upro',
        'name': 'SPY RSI<21 → UPRO',
        'condition': lambda ind, data: ind.get('SPY', {}).get('rsi10', 50) < 21,
        'target': 'UPRO',
        'hold_days': 5,
        'direction': 'long',
        'historical_wr': 0.87,
        'tier': 1,
        'min_n_for_alert': 10,
    },
    {
        'id': 'spy_rsi_lt30_upro',
        'name': 'SPY RSI<30 → UPRO',
        'condition': lambda ind, data: 21 <= ind.get('SPY', {}).get('rsi10', 50) < 30,
        'target': 'UPRO',
        'hold_days': 5,
        'direction': 'long',
        'historical_wr': 0.69,
        'tier': 1,
        'min_n_for_alert': 15,
    },
    {
        'id': 'qqq_rsi_lt20_tqqq',
        'name': 'QQQ RSI<20 → TQQQ',
        'condition': lambda ind, data: ind.get('QQQ', {}).get('rsi10', 50) < 20,
        'target': 'TQQQ',
        'hold_days': 5,
        'direction': 'long',
        'historical_wr': 1.00,
        'tier': 1,
        'min_n_for_alert': 5,
    },
    {
        'id': 'cure_rsi_lt21',
        'name': 'CURE RSI<21 → CURE',
        'condition': lambda ind, data: ind.get('CURE', {}).get('rsi10', 50) < 21,
        'target': 'CURE',
        'hold_days': 5,
        'direction': 'long',
        'historical_wr': 0.85,
        'tier': 1,
        'min_n_for_alert': 10,
    },
    {
        'id': 'cure_rsi_lt25',
        'name': 'CURE RSI<25 → CURE',
        'condition': lambda ind, data: 21 <= ind.get('CURE', {}).get('rsi10', 50) < 25,
        'target': 'CURE',
        'hold_days': 5,
        'direction': 'long',
        'historical_wr': 0.81,
        'tier': 1,
        'min_n_for_alert': 10,
    },
    # --- TIER 1: Exit/Hedge Signals ---
    {
        'id': 'spy_rsi_gt79_uvxy',
        'name': 'SPY RSI>79 → UVXY',
        'condition': lambda ind, data: ind.get('SPY', {}).get('rsi10', 50) > 79,
        'target': 'UVXY',
        'hold_days': 1,
        'direction': 'long',
        'historical_wr': 0.686,
        'tier': 1,
        'min_n_for_alert': 15,
    },
    {
        'id': 'qqq_rsi_gt79_uvxy',
        'name': 'QQQ RSI>79 → UVXY',
        'condition': lambda ind, data: ind.get('QQQ', {}).get('rsi10', 50) > 79,
        'target': 'UVXY',
        'hold_days': 5,
        'direction': 'long',
        'historical_wr': 0.67,
        'tier': 1,
        'min_n_for_alert': 15,
    },
    {
        'id': 'spy_rsi_gt85_exit',
        'name': 'SPY RSI>85 → Exit UPRO',
        'condition': lambda ind, data: ind.get('SPY', {}).get('rsi10', 50) > 85,
        'target': 'UPRO',
        'hold_days': 5,
        'direction': 'short',  # win = UPRO goes DOWN (confirming exit was right)
        'historical_wr': 0.64,  # 64% chance of decline = 36% WR for longs
        'tier': 1,
        'min_n_for_alert': 5,
    },
    # --- TIER 1: Oil/Bond Signal ---
    {
        'id': 'uco_rsi_gt75_tmv',
        'name': 'UCO RSI>75 → TMV',
        'condition': lambda ind, data: ind.get('UCO', {}).get('rsi10', 50) > 75,
        'target': 'TMV',
        'hold_days': 5,
        'direction': 'long',
        'historical_wr': 0.65,
        'tier': 1,
        'min_n_for_alert': 15,
    },
    # --- TIER 2: Combo Signals ---
    {
        'id': 'double_signal_tqqq',
        'name': 'GLD>79 + USDU<25 → TQQQ',
        'condition': lambda ind, data: (
            ind.get('GLD', {}).get('rsi10', 50) > 79 and
            ind.get('USDU', {}).get('rsi10', 50) < 25
        ),
        'target': 'TQQQ',
        'hold_days': 5,
        'direction': 'long',
        'historical_wr': 0.88,
        'tier': 2,
        'min_n_for_alert': 5,
    },
    {
        'id': 'soxs_dollar_squeeze',
        'name': 'SMH>79 + USDU>70 → SOXS',
        'condition': lambda ind, data: (
            ind.get('SMH', {}).get('rsi10', 50) > 79 and
            ind.get('USDU', {}).get('rsi10', 50) > 70
        ),
        'target': 'SOXS',
        'hold_days': 5,
        'direction': 'long',
        'historical_wr': 1.00,
        'tier': 2,
        'min_n_for_alert': 5,
    },
    {
        'id': 'gld_overbought_tqqq',
        'name': 'GLD RSI>79 → TQQQ',
        'condition': lambda ind, data: (
            ind.get('GLD', {}).get('rsi10', 50) > 79 and
            ind.get('USDU', {}).get('rsi10', 50) >= 25  # NOT double signal
        ),
        'target': 'TQQQ',
        'hold_days': 5,
        'direction': 'long',
        'historical_wr': 0.72,
        'tier': 2,
        'min_n_for_alert': 10,
    },
    # --- TIER 2: Leveraged ETF Signals ---
    {
        'id': 'fas_rsi_lt30',
        'name': 'FAS RSI<30 → FAS',
        'condition': lambda ind, data: ind.get('FAS', {}).get('rsi10', 50) < 30,
        'target': 'FAS',
        'hold_days': 5,
        'direction': 'long',
        'historical_wr': 0.63,
        'tier': 2,
        'min_n_for_alert': 15,
    },
    {
        'id': 'labu_rsi_lt25',
        'name': 'LABU RSI<25 → LABU',
        'condition': lambda ind, data: ind.get('LABU', {}).get('rsi10', 50) < 25,
        'target': 'LABU',
        'hold_days': 5,
        'direction': 'long',
        'historical_wr': 0.66,
        'tier': 2,
        'min_n_for_alert': 10,
    },
    {
        'id': 'fas_rsi_gt85_exit',
        'name': 'FAS RSI>85 → Exit',
        'condition': lambda ind, data: ind.get('FAS', {}).get('rsi10', 50) > 85,
        'target': 'FAS',
        'hold_days': 5,
        'direction': 'short',
        'historical_wr': 0.92,  # 92% chance of decline
        'tier': 2,
        'min_n_for_alert': 5,
    },
    {
        'id': 'cure_rsi_gt79_exit',
        'name': 'CURE RSI>79 → Exit',
        'condition': lambda ind, data: ind.get('CURE', {}).get('rsi10', 50) > 79,
        'target': 'CURE',
        'hold_days': 5,
        'direction': 'short',
        'historical_wr': 0.60,  # 60% decline = 40% WR for longs
        'tier': 2,
        'min_n_for_alert': 10,
    },
    # --- TIER 2: Volatility/Regime ---
    {
        'id': 'uvxy_gt82_soxl',
        'name': 'UVXY RSI>82 → SOXL (B1)',
        'condition': lambda ind, data: ind.get('UVXY', {}).get('rsi10', 50) > 82,
        'target': 'SOXL',
        'hold_days': 1,
        'direction': 'long',
        'historical_wr': 0.81,
        'tier': 2,
        'min_n_for_alert': 8,
    },
    # --- TIER 2: Defensive Rotation ---
    {
        'id': 'defensive_rotation_tqqq',
        'name': 'Defensive OB → TQQQ',
        'condition': lambda ind, data: (
            any(ind.get(t, {}).get('rsi10', 50) > 79 for t in ['XLP', 'XLU', 'XLV']) and
            ind.get('SPY', {}).get('rsi10', 50) < 79 and
            ind.get('QQQ', {}).get('rsi10', 50) < 79
        ),
        'target': 'TQQQ',
        'hold_days': 20,
        'direction': 'long',
        'historical_wr': 0.70,
        'tier': 2,
        'min_n_for_alert': 10,
    },
]


# =============================================================================
# BRIER SCORE ENGINE
# =============================================================================
def load_brier_history():
    """Load persistent Brier score tracking data"""
    if BRIER_DATA_PATH.exists():
        try:
            with open(BRIER_DATA_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}

def save_brier_history(history):
    """Save Brier score tracking data"""
    try:
        with open(BRIER_DATA_PATH, 'w') as f:
            json.dump(history, f, indent=2, default=str)
    except IOError as e:
        print(f"Warning: Could not save Brier history: {e}")

def compute_brier_scores_historical(data, signal_def):
    """
    Scan historical data to find all past occurrences of a signal,
    then compute the Brier score over those occurrences.
    
    Returns dict with:
      - episodes: list of (date, rsi_value, forward_return, outcome)
      - brier_score: overall Brier score
      - brier_skill: BSS vs unconditional baseline
      - trailing_20_wr: win rate of last 20 signals
      - trailing_20_brier: Brier score of last 20 signals
      - total_n: total signal count
    """
    target_ticker = signal_def['target']
    hold_days = signal_def['hold_days']
    direction = signal_def['direction']
    hist_wr = signal_def['historical_wr']
    
    if target_ticker not in data:
        return None
    
    target_df = data[target_ticker]
    target_close = target_df['Close']
    
    # Calculate forward returns for the target
    fwd_ret = target_close.pct_change(hold_days).shift(-hold_days)
    
    # Calculate indicators needed for signal conditions
    # Build a daily indicator dict for each date
    all_indicators = {}
    for ticker, df in data.items():
        if len(df) < 200:
            continue
        close = df['Close']
        rsi10 = calculate_rsi_wilder(close, 10)
        sma200 = close.rolling(200).mean()
        all_indicators[ticker] = {
            'rsi10': rsi10,
            'sma200': sma200,
            'close': close,
        }
    
    # Scan each date to check if signal was active
    episodes = []
    common_dates = target_close.index
    
    for i in range(200, len(common_dates)):
        date = common_dates[i]
        
        # Build indicator snapshot for this date
        ind_snapshot = {}
        for ticker, ticker_data in all_indicators.items():
            rsi_val = ticker_data['rsi10'].get(date, np.nan) if date in ticker_data['rsi10'].index else np.nan
            if isinstance(rsi_val, pd.Series):
                rsi_val = rsi_val.iloc[0] if len(rsi_val) > 0 else np.nan
            if pd.isna(rsi_val):
                continue
            ind_snapshot[ticker] = {'rsi10': float(rsi_val)}
        
        # Check if signal fires
        try:
            signal_active = signal_def['condition'](ind_snapshot, data)
        except (KeyError, TypeError, IndexError):
            continue
        
        if not signal_active:
            continue
        
        # Get forward return
        fr = fwd_ret.get(date, np.nan)
        if isinstance(fr, pd.Series):
            fr = fr.iloc[0] if len(fr) > 0 else np.nan
        if pd.isna(fr):
            continue
        
        # Determine outcome
        if direction == 'long':
            outcome = 1 if float(fr) > 0 else 0
        else:  # short
            outcome = 1 if float(fr) < 0 else 0
        
        episodes.append({
            'date': date.strftime('%Y-%m-%d'),
            'fwd_return': round(float(fr) * 100, 2),
            'outcome': outcome,
        })
    
    if len(episodes) == 0:
        return None
    
    # Compute Brier scores
    outcomes = [e['outcome'] for e in episodes]
    predicted = hist_wr  # We use the historical WR as our "prediction" each time
    
    # Overall Brier score
    brier = np.mean([(predicted - o) ** 2 for o in outcomes])
    
    # Baseline Brier (using unconditional WR — just how often target goes up)
    unconditional_wr = np.mean(outcomes)
    brier_baseline = np.mean([(unconditional_wr - o) ** 2 for o in outcomes])
    
    # Brier Skill Score
    bss = 1 - (brier / brier_baseline) if brier_baseline > 0 else 0
    
    # Trailing 20 metrics
    recent = outcomes[-20:] if len(outcomes) >= 20 else outcomes
    trailing_wr = np.mean(recent)
    trailing_brier = np.mean([(predicted - o) ** 2 for o in recent])
    
    # Actual overall WR
    actual_wr = np.mean(outcomes)
    
    return {
        'signal_id': signal_def['id'],
        'signal_name': signal_def['name'],
        'total_n': len(episodes),
        'historical_wr': hist_wr,
        'actual_wr': round(actual_wr, 3),
        'brier_score': round(brier, 4),
        'brier_baseline': round(brier_baseline, 4),
        'brier_skill': round(bss, 4),
        'trailing_20_n': len(recent),
        'trailing_20_wr': round(trailing_wr, 3),
        'trailing_20_brier': round(trailing_brier, 4),
        'last_10_episodes': episodes[-10:],
        'tier': signal_def['tier'],
    }


def assess_signal_health(result, signal_def):
    """
    Evaluate whether a signal is healthy, degrading, or broken.
    Returns (status, message) where status is 'healthy', 'warning', or 'critical'
    """
    if result is None:
        return ('unknown', 'Insufficient data')
    
    hist_wr = signal_def['historical_wr']
    trailing_wr = result['trailing_20_wr']
    actual_wr = result['actual_wr']
    n = result['total_n']
    bss = result['brier_skill']
    
    if n < signal_def['min_n_for_alert']:
        return ('insufficient', f'Only {n} occurrences (need {signal_def["min_n_for_alert"]})')
    
    # Critical: trailing WR below 40%
    if trailing_wr < 0.40 and hist_wr > 0.60:
        return ('critical', 
                f'Trailing WR {trailing_wr:.0%} vs historical {hist_wr:.0%} — '
                f'signal may be BROKEN (n={result["trailing_20_n"]})')
    
    # Warning: trailing WR below 50% for signals that should be >65%
    if trailing_wr < 0.50 and hist_wr > 0.65:
        return ('warning',
                f'Trailing WR {trailing_wr:.0%} vs historical {hist_wr:.0%} — '
                f'signal DEGRADING (n={result["trailing_20_n"]})')
    
    # Warning: BSS negative (signal worse than base rate)
    if bss < -0.05 and n >= 20:
        return ('warning',
                f'Brier Skill Score {bss:+.3f} — signal WORSE than base rate')
    
    # Warning: actual WR more than 15% below historical
    if actual_wr < hist_wr - 0.15 and n >= 15:
        return ('warning',
                f'Actual WR {actual_wr:.0%} vs historical {hist_wr:.0%} — '
                f'persistent underperformance over {n} signals')
    
    # Healthy
    return ('healthy', f'WR {actual_wr:.0%} (hist {hist_wr:.0%}), BSS {bss:+.3f}')


# =============================================================================
# SIGNAL CHECKS (unchanged from v3, but calls Brier engine)
# =============================================================================
def check_signals(data):
    """Check all signals and return alerts"""
    alerts = []
    status = {}
    
    # Calculate indicators for all tickers
    indicators = {}
    for ticker, df in data.items():
        if len(df) < 200:
            continue
        try:
            close = df['Close']
            price = safe_float(close.iloc[-1])
            rsi10 = safe_float(calculate_rsi_wilder(close, 10).iloc[-1])
            rsi50 = safe_float(calculate_rsi_wilder(close, 50).iloc[-1])
            sma200 = safe_float(close.rolling(window=200).mean().iloc[-1])
            sma50 = safe_float(close.rolling(window=50).mean().iloc[-1])
            ema21 = safe_float(close.ewm(span=21, adjust=False).mean().iloc[-1])
            indicators[ticker] = {
                'price': price, 'rsi10': rsi10, 'rsi50': rsi50,
                'sma200': sma200, 'sma50': sma50, 'ema21': ema21,
            }
            if sma200 > 0:
                indicators[ticker]['pct_above_sma200'] = (price / sma200 - 1) * 100
            else:
                indicators[ticker]['pct_above_sma200'] = 0
        except Exception as e:
            print(f"Error calculating indicators for {ticker}: {e}")
            continue
    
    status['indicators'] = indicators
    
    # =========================================================================
    # SIGNAL GROUP 1: SOXL/SMH Long-Term Signals
    # =========================================================================
    if 'SMH' in indicators:
        smh = indicators['SMH']
        if smh['pct_above_sma200'] >= 40:
            alerts.append(('🔴 SOXL EXIT', f"SMH {smh['pct_above_sma200']:.1f}% above SMA(200) - SELL SOXL", 'exit'))
        elif smh['pct_above_sma200'] >= 35:
            alerts.append(('🟡 SOXL WARNING', f"SMH {smh['pct_above_sma200']:.1f}% above SMA(200) - Approaching sell zone", 'warning'))
        elif smh['pct_above_sma200'] >= 30:
            alerts.append(('🟡 SOXL TRIM', f"SMH {smh['pct_above_sma200']:.1f}% above SMA(200) - Consider trimming 25-50%", 'warning'))
        if smh['sma50'] < smh['sma200'] and smh['sma200'] > 0:
            alerts.append(('🔴 DEATH CROSS', f"SMH SMA(50) below SMA(200) - Bearish trend", 'exit'))
        if 'SMH' in data:
            smh_df = data['SMH']
            close = smh_df['Close']
            sma200_series = close.rolling(window=200).mean()
            days_below = 0
            for i in range(len(close)-1, max(len(close)-500, 199), -1):
                try:
                    c = safe_float(close.iloc[i])
                    s = safe_float(sma200_series.iloc[i])
                    if s > 0 and c < s:
                        days_below += 1
                    else:
                        break
                except:
                    break
            if days_below >= 100:
                if smh['rsi50'] < 45:
                    alerts.append(('🟢 SOXL STRONG BUY', f"SMH {days_below} days below SMA(200) + RSI(50)={smh['rsi50']:.1f} < 45 | 97% win, +81% avg", 'buy'))
                else:
                    alerts.append(('🟢 SOXL ACCUMULATE', f"SMH {days_below} days below SMA(200) | 85% win, +54% avg", 'buy'))
            status['smh_days_below_sma200'] = days_below
    
    # =========================================================================
    # SIGNAL GROUP 2: GLD/USDU Combo Signals
    # =========================================================================
    if 'GLD' in indicators and 'USDU' in indicators:
        gld = indicators['GLD']
        usdu = indicators['USDU']
        if gld['rsi10'] > 79 and usdu['rsi10'] < 25:
            alerts.append(('🟢🔥 DOUBLE SIGNAL ACTIVE',
                f"GLD RSI={gld['rsi10']:.1f} > 79 AND USDU RSI={usdu['rsi10']:.1f} < 25\n"
                f"   → Long TQQQ: 88% win, +7% avg (5d)\n"
                f"   → Long UPRO: 85% win, +5.2% avg (5d)\n"
                f"   → AMD/NVDA: 86% win, +5-8% avg (5d)", 'buy'))
            if 'XLP' in indicators and indicators['XLP']['rsi10'] > 65:
                xlp = indicators['XLP']
                alerts.append(('🟢🔥🔥 TRIPLE SIGNAL ACTIVE',
                    f"GLD RSI={gld['rsi10']:.1f} + USDU RSI={usdu['rsi10']:.1f} + XLP RSI={xlp['rsi10']:.1f}\n"
                    f"   → Long TQQQ: 100% win, +11.6% avg (5d) - RARE!", 'buy'))
        elif gld['rsi10'] > 79:
            alerts.append(('🟢 GLD OVERBOUGHT',
                f"GLD RSI={gld['rsi10']:.1f} > 79 → Long TQQQ: 72% win, +3.2% avg (5d)", 'buy'))
    
    # =========================================================================
    # SIGNAL GROUP 3: Defensive Rotation
    # =========================================================================
    defensive_ob = any(indicators.get(t, {}).get('rsi10', 0) > 79 for t in ['XLP', 'XLU', 'XLV'])
    if defensive_ob:
        spy_ob = indicators.get('SPY', {}).get('rsi10', 0) > 79
        qqq_ob = indicators.get('QQQ', {}).get('rsi10', 0) > 79
        if not spy_ob and not qqq_ob:
            alerts.append(('🟢 DEFENSIVE ROTATION',
                f"Defensive sector overbought, SPY/QQQ not → Long TQQQ 20d: 70% win, +5% avg", 'buy'))
    
    # =========================================================================
    # SIGNAL GROUP 4: Volatility Hedge Signals
    # =========================================================================
    if 'QQQ' in indicators:
        qqq = indicators['QQQ']
        if qqq['rsi10'] > 79:
            alerts.append(('🟡 VOL HEDGE',
                f"QQQ RSI={qqq['rsi10']:.1f} > 79 → Long UVXY 5d: 67% win, +33% CAGR", 'hedge'))
        if qqq['rsi10'] < 20:
            alerts.append(('🟢 QQQ DIP BUY',
                f"QQQ RSI={qqq['rsi10']:.1f} < 20 → Long TQQQ 5d: 100% win, n=12", 'buy'))
    
    # =========================================================================
    # SIGNAL GROUP 5: SOXS Short Signals
    # =========================================================================
    if 'SMH' in indicators and 'USDU' in indicators:
        smh = indicators['SMH']
        usdu = indicators['USDU']
        if smh['rsi10'] > 79 and usdu['rsi10'] > 70:
            alerts.append(('🔴 SOXS SIGNAL',
                f"SMH RSI={smh['rsi10']:.1f} > 79 AND USDU RSI={usdu['rsi10']:.1f} > 70\n"
                f"   → Long SOXS 5d: 100% win, +9.5% avg", 'short'))
        if 'IWM' in indicators and smh['rsi10'] > 79 and indicators['IWM']['rsi10'] < 50:
            alerts.append(('🔴 SOXS DIVERGENCE',
                f"SMH RSI={smh['rsi10']:.1f} > 79 AND IWM RSI={indicators['IWM']['rsi10']:.1f} < 50\n"
                f"   → Long SOXS 5d: 86% win, +6.9% avg", 'short'))
    
    # =========================================================================
    # SIGNAL GROUP 6: BTC Signals
    # =========================================================================
    if 'BTC-USD' in indicators:
        btc = indicators['BTC-USD']
        if btc['rsi10'] > 79:
            alerts.append(('🟢 BTC MOMENTUM',
                f"BTC RSI={btc['rsi10']:.1f} > 79 → Hold/Add BTC: 67% win, +5.2% avg (5d)", 'buy'))
        if btc['rsi10'] < 30:
            uvxy_low = indicators.get('UVXY', {}).get('rsi10', 50) < 40
            if uvxy_low:
                alerts.append(('🟢 BTC DIP BUY',
                    f"BTC RSI={btc['rsi10']:.1f} < 30 AND UVXY < 40 → Buy BTC: 77% win, +4.1% avg (5d)", 'buy'))
            else:
                alerts.append(('🟡 BTC OVERSOLD',
                    f"BTC RSI={btc['rsi10']:.1f} < 30 (wait for UVXY < 40 for better signal)", 'watch'))
    
    # =========================================================================
    # SIGNAL GROUP 7: UPRO Entry/Exit Signals
    # =========================================================================
    if 'SPY' in indicators:
        spy = indicators['SPY']
        if spy['rsi10'] > 85:
            alerts.append(('🔴 UPRO EXIT',
                f"SPY RSI={spy['rsi10']:.1f} > 85 → Trim/Exit UPRO: Only 36% win, -3.5% avg (5d)", 'exit'))
        elif spy['rsi10'] > 82:
            alerts.append(('🟡 UPRO CAUTION',
                f"SPY RSI={spy['rsi10']:.1f} > 82 → Watch UPRO: 49% win at 5d", 'warning'))
        if spy['rsi10'] < 21:
            alerts.append(('🟢 UPRO STRONG BUY',
                f"SPY RSI={spy['rsi10']:.1f} < 21 → Add UPRO: 94% win, +8.9% avg (5d)", 'buy'))
        elif spy['rsi10'] < 25:
            alerts.append(('🟢 UPRO BUY',
                f"SPY RSI={spy['rsi10']:.1f} < 25 → Add UPRO: 74% win, +3.9% avg (5d)", 'buy'))
        elif spy['rsi10'] < 30:
            alerts.append(('🟢 UPRO CONSIDER',
                f"SPY RSI={spy['rsi10']:.1f} < 30 → Consider UPRO: 69% win, +4.3% avg (5d)", 'buy'))
    
    # =========================================================================
    # SIGNAL GROUP 8: AMD/NVDA Specific
    # =========================================================================
    for ticker in ['AMD', 'NVDA']:
        if ticker in indicators and indicators[ticker]['rsi10'] > 85:
            alerts.append((f'🟡 {ticker} EXTENDED',
                f"{ticker} RSI={indicators[ticker]['rsi10']:.1f} > 85 → Consider taking profits", 'warning'))
    
    # =========================================================================
    # SIGNAL GROUP 9-12: NAIL, CURE, FAS, LABU (same as v3)
    # =========================================================================
    if 'NAIL' in indicators:
        nail = indicators['NAIL']
        if 'GLD' in indicators and 'USDU' in indicators and 'XLF' in indicators:
            gld = indicators['GLD']
            usdu = indicators['USDU']
            xlf = indicators['XLF']
            if gld['rsi10'] > 79 and usdu['rsi10'] < 25 and xlf['rsi10'] < 70:
                alerts.append(('🟢 NAIL SIGNAL',
                    f"GLD>{gld['rsi10']:.0f} + USDU<{usdu['rsi10']:.0f} + XLF<{xlf['rsi10']:.0f}\n"
                    f"   → Long NAIL: 90% win, +4.9% avg (5d), +14.4% avg (10d) | n=10", 'buy'))
            if xlf['rsi10'] > 70 and usdu['rsi10'] < 25:
                alerts.append(('🔴 NAIL DANGER',
                    f"XLF RSI={xlf['rsi10']:.1f} > 70 + USDU < 25 = Historically BAD for NAIL\n"
                    f"   → 11% win, -11.5% avg (5d) | Consider exit", 'exit'))
        if nail['rsi10'] > 79:
            alerts.append(('🔴 NAIL OVERBOUGHT',
                f"NAIL RSI={nail['rsi10']:.1f} > 79 → Consider exit", 'warning'))
    
    if 'CURE' in indicators:
        cure = indicators['CURE']
        if cure['rsi10'] < 21:
            alerts.append(('🟢 CURE STRONG BUY',
                f"CURE RSI={cure['rsi10']:.1f} < 21 → Buy CURE: 85% win, +7.3% avg (5d) | n=33", 'buy'))
        elif cure['rsi10'] < 25:
            alerts.append(('🟢 CURE BUY',
                f"CURE RSI={cure['rsi10']:.1f} < 25 → Buy CURE: 81% win, +5.4% avg (5d) | n=70", 'buy'))
        if cure['rsi10'] > 85:
            alerts.append(('🔴 CURE SELL',
                f"CURE RSI={cure['rsi10']:.1f} > 85 → Sell CURE: Only 33% win (5d) | n=15", 'exit'))
        elif cure['rsi10'] > 79:
            alerts.append(('🔴 CURE OVERBOUGHT',
                f"CURE RSI={cure['rsi10']:.1f} > 79 → Exit CURE: Only 40% win (5d) | n=95", 'exit'))
    
    if 'FAS' in indicators:
        fas = indicators['FAS']
        if 'GLD' in indicators and 'USDU' in indicators:
            gld = indicators['GLD']
            usdu = indicators['USDU']
            if gld['rsi10'] > 79 and usdu['rsi10'] < 25:
                alerts.append(('🟢 FAS SIGNAL',
                    f"GLD>{gld['rsi10']:.0f} + USDU<{usdu['rsi10']:.0f}\n"
                    f"   → Long FAS 10d: 92% win, +5.8% avg | n=13", 'buy'))
        if fas['rsi10'] < 30:
            alerts.append(('🟢 FAS BUY',
                f"FAS RSI={fas['rsi10']:.1f} < 30 → Buy FAS: 63% win, +3.3% avg (5d) | n=195", 'buy'))
        if fas['rsi10'] > 85:
            alerts.append(('🔴 FAS SELL',
                f"FAS RSI={fas['rsi10']:.1f} > 85 → Sell FAS: Only 8% win! (5d) | n=12", 'exit'))
        elif fas['rsi10'] > 82:
            alerts.append(('🔴 FAS OVERBOUGHT',
                f"FAS RSI={fas['rsi10']:.1f} > 82 → Exit FAS: Only 38% win (5d) | n=40", 'exit'))
    
    if 'LABU' in indicators:
        labu = indicators['LABU']
        if labu['rsi10'] < 21:
            alerts.append(('🟢 LABU STRONG BUY',
                f"LABU RSI={labu['rsi10']:.1f} < 21 → Buy LABU: 73% win, +11.2% avg (5d) | n=11", 'buy'))
        elif labu['rsi10'] < 25:
            alerts.append(('🟢 LABU BUY',
                f"LABU RSI={labu['rsi10']:.1f} < 25 → Buy LABU: 66% win, +5.7% avg (5d) | n=59", 'buy'))
        if labu['rsi10'] > 70:
            alerts.append(('🟡 LABU EXTENDED',
                f"LABU RSI={labu['rsi10']:.1f} > 70 → Caution: 42% win (5d) | n=180", 'warning'))
        if labu.get('pct_above_sma200', 0) > 80:
            alerts.append(('🟡 LABU EXTREME',
                f"LABU {labu['pct_above_sma200']:.0f}% above SMA(200) → Very extended", 'warning'))
    
    return alerts, status


# =============================================================================
# EMAIL FORMATTING
# =============================================================================
def format_email(alerts, status, brier_results, is_preclose=False):
    """Format the email body including Brier score section"""
    now = datetime.now()
    timing = "PRE-CLOSE PREVIEW (3:15 PM)" if is_preclose else "MARKET CLOSE CONFIRMATION (4:05 PM)"
    
    body = f"""
{'='*70}
MARKET SIGNAL MONITOR v4.0 — {timing}
{now.strftime('%Y-%m-%d %H:%M')} ET
{'='*70}

"""
    # --- Active Alerts ---
    if alerts:
        buy_alerts = [a for a in alerts if a[2] == 'buy']
        exit_alerts = [a for a in alerts if a[2] in ['exit', 'short']]
        warning_alerts = [a for a in alerts if a[2] in ['warning', 'hedge', 'watch']]
        
        if buy_alerts:
            body += "🟢 BUY SIGNALS:\n" + "-"*50 + "\n"
            for title, msg, _ in buy_alerts:
                body += f"{title}\n{msg}\n\n"
        if exit_alerts:
            body += "🔴 EXIT/SHORT SIGNALS:\n" + "-"*50 + "\n"
            for title, msg, _ in exit_alerts:
                body += f"{title}\n{msg}\n\n"
        if warning_alerts:
            body += "🟡 WARNINGS/WATCH:\n" + "-"*50 + "\n"
            for title, msg, _ in warning_alerts:
                body += f"{title}\n{msg}\n\n"
    else:
        body += "No signals triggered today.\n\n"
    
    # --- Indicator Status ---
    body += f"""
{'='*70}
CURRENT INDICATOR STATUS
{'='*70}

"""
    indicators = status.get('indicators', {})
    key_tickers = ['SPY', 'QQQ', 'SMH', 'GLD', 'USDU', 'XLP', 'TLT', 'HYG',
                   'XLF', 'UVXY', 'UCO', 'BTC-USD', 'AMD', 'NVDA']
    body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}\n"
    body += "-"*50 + "\n"
    for ticker in key_tickers:
        if ticker in indicators:
            ind = indicators[ticker]
            price = f"${ind['price']:.2f}" if ind['price'] < 1000 else f"${ind['price']:,.0f}"
            rsi = f"{ind['rsi10']:.1f}"
            pct = f"{ind.get('pct_above_sma200', 0):+.1f}%"
            body += f"{ticker:<10} {price:>12} {rsi:>10} {pct:>12}\n"
    
    # --- 3x Leveraged ETFs ---
    body += f"\n{'='*70}\n3x LEVERAGED ETFs\n{'='*70}\n"
    body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}  Signal\n"
    body += "-"*65 + "\n"
    for ticker in ['NAIL', 'CURE', 'FAS', 'LABU', 'TQQQ', 'SOXL', 'TECL', 'DRN']:
        if ticker in indicators:
            ind = indicators[ticker]
            price = f"${ind['price']:.2f}"
            rsi = f"{ind['rsi10']:.1f}"
            pct = f"{ind.get('pct_above_sma200', 0):+.1f}%"
            rsi_val = ind['rsi10']
            if rsi_val < 21: signal = "🟢 OVERSOLD"
            elif rsi_val < 30: signal = "🟢 Watch"
            elif rsi_val > 85: signal = "🔴 OVERBOUGHT"
            elif rsi_val > 79: signal = "🟡 Extended"
            else: signal = ""
            body += f"{ticker:<10} {price:>12} {rsi:>10} {pct:>12}  {signal}\n"
    
    # --- SMH/SOXL Levels ---
    if 'SMH' in indicators:
        smh = indicators['SMH']
        sma200 = smh['sma200']
        body += f"""
{'='*70}
SMH/SOXL LEVELS
{'='*70}
Current Price:    ${smh['price']:.2f}
SMA(200):         ${sma200:.2f}
% Above SMA200:   {smh['pct_above_sma200']:+.1f}%
Days Below SMA:   {status.get('smh_days_below_sma200', 0)}

Key Levels:
  30% (Trim):     ${sma200 * 1.30:.2f}
  35% (Warning):  ${sma200 * 1.35:.2f}
  40% (Sell):     ${sma200 * 1.40:.2f}
"""
    
    # =========================================================================
    # BRIER SCORE CALIBRATION REPORT
    # =========================================================================
    body += f"""
{'='*70}
SIGNAL CALIBRATION — BRIER SCORES
{'='*70}

What this means:
  Brier Score: 0.00 = perfect, 0.25 = coin flip, lower is better
  BSS (Brier Skill Score): positive = signal adds value vs base rate
  Trailing WR: win rate of last 20 signal firings
  Status: 🟢 healthy / 🟡 degrading / 🔴 broken / ⚪ insufficient data

"""
    
    # Separate into degradation alerts and healthy signals
    degradation_alerts = []
    healthy_signals = []
    insufficient_signals = []
    
    for result in brier_results:
        if result is None:
            continue
        sig_def = next((s for s in SIGNAL_REGISTRY if s['id'] == result['signal_id']), None)
        if sig_def is None:
            continue
        
        health_status, health_msg = assess_signal_health(result, sig_def)
        
        if health_status in ('critical', 'warning'):
            degradation_alerts.append((result, health_status, health_msg))
        elif health_status == 'insufficient':
            insufficient_signals.append((result, health_msg))
        else:
            healthy_signals.append((result, health_msg))
    
    # Print degradation alerts first (these are the actionable items)
    if degradation_alerts:
        body += "⚠️  DEGRADATION ALERTS — REQUIRES ATTENTION\n"
        body += "-"*60 + "\n"
        for result, health_status, health_msg in degradation_alerts:
            icon = "🔴" if health_status == 'critical' else "🟡"
            body += f"  {icon} {result['signal_name']}\n"
            body += f"     {health_msg}\n"
            body += f"     Brier: {result['brier_score']:.3f} | BSS: {result['brier_skill']:+.3f} | n={result['total_n']}\n"
            # Show last 5 episodes
            recent = result['last_10_episodes'][-5:]
            if recent:
                body += f"     Recent: "
                body += " ".join([f"{'W' if e['outcome'] else 'L'}({e['fwd_return']:+.1f}%)" for e in recent])
                body += "\n"
            body += "\n"
    
    # Compact table for healthy signals
    body += f"\n{'Signal':<30} {'Hist':>5} {'Actual':>7} {'Trail20':>8} {'Brier':>7} {'BSS':>7} {'n':>5}  Status\n"
    body += "-"*85 + "\n"
    
    for result, health_msg in healthy_signals:
        body += (f"  {result['signal_name']:<28} "
                 f"{result['historical_wr']:>4.0%} "
                 f"{result['actual_wr']:>6.0%} "
                 f"{result['trailing_20_wr']:>7.0%} "
                 f"{result['brier_score']:>7.3f} "
                 f"{result['brier_skill']:>+6.3f} "
                 f"{result['total_n']:>5}  🟢\n")
    
    for result, health_msg in insufficient_signals:
        body += (f"  {result['signal_name']:<28} "
                 f"{result['historical_wr']:>4.0%} "
                 f"{result['actual_wr']:>6.0%} "
                 f"{'—':>8} "
                 f"{result['brier_score']:>7.3f} "
                 f"{'—':>7} "
                 f"{result['total_n']:>5}  ⚪ ({health_msg})\n")
    
    body += f"""
{'─'*70}
Reading guide:
  Hist     = backtested historical win rate (our prediction each time)
  Actual   = realized win rate across ALL occurrences in data
  Trail20  = win rate of the LAST 20 firings (most recent behavior)
  Brier    = calibration score (0=perfect, 0.25=coin flip)
  BSS      = skill vs naive base rate (positive=signal adds value)

  If Trail20 drops well below Hist, the signal may be degrading.
  If BSS goes negative, the signal is WORSE than just guessing the
  unconditional win rate — consider suspending or reducing position size.
"""
    
    if is_preclose:
        body += f"\n{'='*70}\nNOTE: PRE-CLOSE preview. Signals may change by market close.\n{'='*70}\n"
    
    return body


def send_email(subject, body):
    """Send email alert"""
    if not SENDER_EMAIL or not SENDER_PASSWORD or not RECIPIENT_EMAIL:
        print("Email not configured - printing to console:")
        print(f"Subject: {subject}")
        print(body)
        return False
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"Email sent successfully to {RECIPIENT_EMAIL}")
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"Running signal check at {datetime.now()}")
    print(f"Mode: {'PRE-CLOSE (3:15 PM)' if IS_PRECLOSE else 'MARKET CLOSE (4:05 PM)'}")
    
    tickers = [
        'SMH', 'SPY', 'QQQ', 'IWM',
        'XLP', 'XLU', 'XLV',
        'GLD', 'TLT', 'HYG', 'LQD', 'TMV',
        'USDU', 'UCO', 'BOIL',
        'UVXY',
        'EDC', 'YINN', 'KORU', 'EURL', 'INDL',
        'BTC-USD',
        'AMD', 'NVDA',
        'NAIL', 'CURE', 'FAS', 'LABU',
        'TQQQ', 'SOXL', 'SOXS', 'TECL', 'DRN',
        'UPRO',
        'VOOV', 'VOOG', 'VTV', 'QQQE',
        'XLE', 'XLF',
    ]
    
    print("Downloading market data...")
    data = download_data(tickers)
    print(f"Downloaded data for {len(data)} tickers")
    
    # --- Run signal checks ---
    alerts, status = check_signals(data)
    
    # --- Compute Brier scores ---
    print("Computing Brier scores for signal calibration...")
    brier_results = []
    for sig_def in SIGNAL_REGISTRY:
        try:
            result = compute_brier_scores_historical(data, sig_def)
            if result:
                brier_results.append(result)
                health, msg = assess_signal_health(result, sig_def)
                status_icon = {'healthy': '🟢', 'warning': '🟡', 'critical': '🔴',
                               'insufficient': '⚪', 'unknown': '?'}
                print(f"  {status_icon.get(health, '?')} {sig_def['name']}: "
                      f"WR={result['actual_wr']:.0%} (hist {sig_def['historical_wr']:.0%}), "
                      f"Brier={result['brier_score']:.3f}, BSS={result['brier_skill']:+.3f}, "
                      f"n={result['total_n']}")
        except Exception as e:
            print(f"  Error computing Brier for {sig_def['name']}: {e}")
    
    # Save Brier history
    history = load_brier_history()
    today = datetime.now().strftime('%Y-%m-%d')
    history[today] = {r['signal_id']: {
        'actual_wr': r['actual_wr'],
        'brier_score': r['brier_score'],
        'brier_skill': r['brier_skill'],
        'trailing_20_wr': r['trailing_20_wr'],
        'total_n': r['total_n'],
    } for r in brier_results}
    save_brier_history(history)
    
    # --- Check for degradation alerts to add to main alerts ---
    for result in brier_results:
        sig_def = next((s for s in SIGNAL_REGISTRY if s['id'] == result['signal_id']), None)
        if sig_def is None:
            continue
        health, msg = assess_signal_health(result, sig_def)
        if health == 'critical':
            alerts.append(('🔴 SIGNAL DEGRADATION',
                f"{result['signal_name']}: {msg}", 'warning'))
        elif health == 'warning':
            alerts.append(('🟡 SIGNAL CALIBRATION',
                f"{result['signal_name']}: {msg}", 'watch'))
    
    # --- Format and send email ---
    if alerts:
        buy_count = len([a for a in alerts if a[2] == 'buy'])
        exit_count = len([a for a in alerts if a[2] in ['exit', 'short']])
        if exit_count > 0:
            emoji, urgency = "🔴", "EXIT SIGNALS"
        elif buy_count > 0:
            emoji, urgency = "🟢", "BUY SIGNALS"
        else:
            emoji, urgency = "🟡", "WATCH"
        timing = "PRE-CLOSE" if IS_PRECLOSE else "CLOSE"
        subject = f"{emoji} [{timing}] Market Signals: {len(alerts)} Alert(s) - {urgency}"
    else:
        timing = "PRE-CLOSE" if IS_PRECLOSE else "CLOSE"
        subject = f"📊 [{timing}] Market Signals: No Alerts"
    
    body = format_email(alerts, status, brier_results, IS_PRECLOSE)
    send_email(subject, body)
    
    print(f"\n{len(alerts)} signal(s) detected")
    for title, msg, _ in alerts:
        print(f"  {title}")


if __name__ == "__main__":
    main()
