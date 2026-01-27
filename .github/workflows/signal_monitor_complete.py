#!/usr/bin/env python3
"""
Comprehensive Market Signal Monitor v3.0
========================================
Monitors all backtested trading signals and sends alerts.
NEW: Includes BOIL/KOLD signals with weather forecast integration.

SCHEDULE: Two emails daily (weekdays)
- 3:15 PM ET: Pre-close preview
- 4:05 PM ET: Market close confirmation

NEW IN v3.0:
- BOIL/KOLD natural gas signals with forward weather forecast
- Automated NOAA CPC temperature outlook data pull
- EIA natural gas inventory tracking
- Weather override logic for fade signals
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import sys
import requests
import re

# =============================================================================
# CONFIGURATION
# =============================================================================
SENDER_EMAIL = os.environ.get('SENDER_EMAIL', '')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD', '')
RECIPIENT_EMAIL = os.environ.get('RECIPIENT_EMAIL', '')
PHONE_EMAIL = os.environ.get('PHONE_EMAIL', '')

IS_PRECLOSE = len(sys.argv) > 1 and sys.argv[1] == 'preclose'

# =============================================================================
# WEATHER DATA FUNCTIONS
# =============================================================================
def fetch_cpc_outlook():
    """
    Fetch 6-10 day temperature outlook from NOAA CPC prognostic discussion.
    Returns dict with outlook info.
    """
    outlook = {
        'fetched': False,
        'source': 'NOAA CPC',
        '6_10_day': {
            'eastern_us': 'Unknown',
            'trend': 'Unknown'
        },
        '8_14_day': {
            'eastern_us': 'Unknown'
        },
        'raw_text': '',
        'error': None
    }
    
    try:
        # 6-10 day prognostic discussion text
        url = "https://www.cpc.ncep.noaa.gov/products/predictions/610day/fxus06.html"
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; WeatherBot/1.0)'}
        response = requests.get(url, timeout=15, headers=headers)
        
        if response.status_code == 200:
            text = response.text.lower()
            outlook['fetched'] = True
            
            # Extract key phrases for Eastern US temperature outlook
            # Look for temperature-related keywords
            below_count = text.count('below normal') + text.count('below-normal')
            above_count = text.count('above normal') + text.count('above-normal')
            
            # Check for specific regional mentions
            east_cold = any(phrase in text for phrase in [
                'eastern' in text and 'below' in text,
                'east coast' in text and 'cold' in text,
                'northeast' in text and 'below' in text,
                'much below' in text
            ])
            
            if 'much below normal' in text or 'well below normal' in text:
                outlook['6_10_day']['eastern_us'] = 'Much Below Normal'
                outlook['6_10_day']['cold_intensity'] = 'HIGH'
            elif below_count > above_count:
                outlook['6_10_day']['eastern_us'] = 'Below Normal'
                outlook['6_10_day']['cold_intensity'] = 'MODERATE'
            elif above_count > below_count:
                outlook['6_10_day']['eastern_us'] = 'Above Normal'
                outlook['6_10_day']['cold_intensity'] = 'LOW'
            else:
                outlook['6_10_day']['eastern_us'] = 'Near Normal'
                outlook['6_10_day']['cold_intensity'] = 'LOW'
            
            # Store snippet for debugging
            outlook['raw_text'] = text[:500]
            
    except Exception as e:
        outlook['error'] = str(e)
        print(f"Weather fetch error: {e}")
    
    return outlook

def fetch_weather_hazards():
    """
    Fetch weather hazards outlook from NOAA CPC.
    Includes cold waves, heat waves, etc.
    """
    hazards = {
        'fetched': False,
        'cold_risk': 'Unknown',
        'source': 'NOAA CPC Hazards',
        'error': None
    }
    
    try:
        url = "https://www.cpc.ncep.noaa.gov/products/predictions/threats/threats.php"
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; WeatherBot/1.0)'}
        response = requests.get(url, timeout=15, headers=headers)
        
        if response.status_code == 200:
            text = response.text.lower()
            hazards['fetched'] = True
            
            # Check for cold-related hazard keywords
            if any(phrase in text for phrase in ['dangerously cold', 'arctic', 'extreme cold', 'much below']):
                hazards['cold_risk'] = 'HIGH'
            elif any(phrase in text for phrase in ['below normal', 'below-normal', 'cold']):
                hazards['cold_risk'] = 'MODERATE'
            else:
                hazards['cold_risk'] = 'LOW'
                
    except Exception as e:
        hazards['error'] = str(e)
        print(f"Hazards fetch error: {e}")
    
    return hazards

def fetch_eia_storage():
    """
    Fetch natural gas storage data from EIA.
    Returns dict with inventory info.
    """
    inventory = {
        'fetched': False,
        'source': 'EIA Weekly Natural Gas Storage',
        'current_level': 'N/A',
        'vs_5yr_avg': 'N/A',
        'weekly_change': 'N/A',
        'report_date': 'N/A',
        'next_report': 'Thursday 10:30 AM ET',
        'error': None
    }
    
    try:
        # EIA natural gas storage summary page
        url = "https://ir.eia.gov/ngs/ngs.html"
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; EIABot/1.0)'}
        response = requests.get(url, timeout=15, headers=headers)
        
        if response.status_code == 200:
            text = response.text
            inventory['fetched'] = True
            
            # Try to extract numbers using regex
            # Look for patterns like "3,065 Bcf" or storage numbers
            bcf_match = re.search(r'(\d{1,2},?\d{3})\s*Bcf', text)
            if bcf_match:
                inventory['current_level'] = bcf_match.group(1) + ' Bcf'
            
            # Look for percentage vs 5-year average
            pct_match = re.search(r'([+-]?\d+\.?\d*)%?\s*(above|below)\s*5-year', text, re.IGNORECASE)
            if pct_match:
                sign = '+' if 'above' in pct_match.group(2).lower() else '-'
                inventory['vs_5yr_avg'] = f"{sign}{pct_match.group(1)}%"
                
    except Exception as e:
        inventory['error'] = str(e)
        print(f"EIA fetch error: {e}")
    
    return inventory

def get_all_weather_data():
    """
    Fetch all weather and inventory data.
    Returns combined dict.
    """
    print("Fetching weather and inventory data...")
    
    weather_data = {
        'outlook': fetch_cpc_outlook(),
        'hazards': fetch_weather_hazards(),
        'inventory': fetch_eia_storage(),
        'fetch_time': datetime.now().strftime('%Y-%m-%d %H:%M ET')
    }
    
    print(f"  CPC Outlook fetched: {weather_data['outlook'].get('fetched', False)}")
    print(f"  Hazards fetched: {weather_data['hazards'].get('fetched', False)}")
    print(f"  EIA Storage fetched: {weather_data['inventory'].get('fetched', False)}")
    
    return weather_data

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
                # Flatten multi-index columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    return data

# =============================================================================
# BOIL/KOLD SIGNAL LOGIC
# =============================================================================
def check_boil_kold_signals(data, weather_data):
    """
    Check BOIL/KOLD signals incorporating weather forecasts and backtested patterns.
    
    Key signals from backtest:
    - 5d rally >40%: 84% KOLD win rate (n=19)
    - 5d rally >50%: 90% KOLD win rate (n=10)
    - RSI > 79: 56% KOLD win rate (n=48)
    - RSI < 21: 69% BOIL win rate at 20d (n=62)
    - Winter spikes reverse MORE than summer (12.5% vs 53.8% continuation)
    
    Weather override: If cold forecast continues, delay fade signal.
    """
    alerts = []
    boil_status = {
        'signal': 'N/A',
        'action': 'N/A',
        'signal_score': 0,
        'reasoning': [],
        'weather_override': False
    }
    
    if 'BOIL' not in data:
        print("BOIL data not available")
        return alerts, boil_status
    
    boil_df = data['BOIL']
    if len(boil_df) < 50:
        print("Insufficient BOIL data")
        return alerts, boil_status
    
    close = boil_df['Close']
    
    # Calculate indicators
    price = safe_float(close.iloc[-1])
    rsi10 = safe_float(calculate_rsi_wilder(close, 10).iloc[-1])
    ema9 = safe_float(close.ewm(span=9, adjust=False).mean().iloc[-1])
    ema20 = safe_float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    
    # Calculate returns
    daily_ret = safe_float(close.pct_change().iloc[-1]) * 100
    
    if len(close) >= 6:
        ret_5d = (price / safe_float(close.iloc[-6]) - 1) * 100
    else:
        ret_5d = 0
    
    if len(close) >= 11:
        ret_10d = (price / safe_float(close.iloc[-11]) - 1) * 100
    else:
        ret_10d = 0
    
    # Store status
    boil_status.update({
        'price': price,
        'rsi10': rsi10,
        'ret_5d': ret_5d,
        'ret_10d': ret_10d,
        'daily_ret': daily_ret,
        'ema9': ema9,
        'ema20': ema20,
        'above_ema9': price > ema9,
        'above_ema20': price > ema20
    })
    
    # ==========================================================================
    # SIGNAL SCORING
    # ==========================================================================
    signal_score = 0
    reasoning = []
    
    # --- TECHNICAL SIGNALS ---
    
    # RSI signals
    if rsi10 > 79:
        signal_score -= 2
        reasoning.append(f"RSI {rsi10:.1f} > 79 = OVERBOUGHT (exit signal)")
    elif rsi10 > 70:
        signal_score -= 1
        reasoning.append(f"RSI {rsi10:.1f} approaching overbought")
    elif rsi10 < 21:
        signal_score += 2
        reasoning.append(f"RSI {rsi10:.1f} < 21 = OVERSOLD (69% win at 20d)")
    elif rsi10 < 30:
        signal_score += 1
        reasoning.append(f"RSI {rsi10:.1f} approaching oversold")
    
    # --- BACKTEST SIGNALS (5-day rally fade) ---
    
    if ret_5d > 50:
        signal_score -= 2
        reasoning.append(f"5d rally {ret_5d:+.1f}% > 50% → 90% KOLD win (n=10)")
    elif ret_5d > 40:
        signal_score -= 1.5
        reasoning.append(f"5d rally {ret_5d:+.1f}% > 40% → 84% KOLD win (n=19)")
    elif ret_5d > 30:
        signal_score -= 1
        reasoning.append(f"5d rally {ret_5d:+.1f}% > 30% → 69% KOLD win (n=65)")
    elif ret_5d > 20:
        signal_score -= 0.5
        reasoning.append(f"5d rally {ret_5d:+.1f}% > 20% → 62% KOLD win (n=196)")
    
    # --- WEATHER SIGNALS ---
    
    cold_risk = weather_data.get('hazards', {}).get('cold_risk', 'Unknown')
    outlook = weather_data.get('outlook', {}).get('6_10_day', {}).get('eastern_us', 'Unknown')
    
    weather_bullish = False
    weather_bearish = False
    
    if cold_risk == 'HIGH':
        signal_score += 1.5
        reasoning.append(f"NOAA Hazards: HIGH cold risk → supports BOIL")
        weather_bullish = True
    elif cold_risk == 'MODERATE':
        signal_score += 0.5
        reasoning.append(f"NOAA Hazards: MODERATE cold risk")
        weather_bullish = True
    elif cold_risk == 'LOW':
        reasoning.append(f"NOAA Hazards: LOW cold risk")
    
    if 'Above Normal' in outlook:
        signal_score -= 1
        reasoning.append(f"6-10 Day Outlook: {outlook} → bearish BOIL")
        weather_bearish = True
    elif 'Below Normal' in outlook:
        signal_score += 0.5
        reasoning.append(f"6-10 Day Outlook: {outlook} → supports BOIL")
        weather_bullish = True
    
    # --- WEATHER OVERRIDE LOGIC ---
    # If backtest says fade but weather says cold continues, WAIT
    
    backtest_says_fade = ret_5d > 30 or rsi10 > 79
    weather_says_cold = weather_bullish and not weather_bearish
    
    weather_override = False
    if backtest_says_fade and weather_says_cold and rsi10 < 79:
        weather_override = True
        reasoning.append("⚠️ WEATHER OVERRIDE: Fade blocked - cold forecast continues")
    
    boil_status['weather_override'] = weather_override
    boil_status['signal_score'] = signal_score
    boil_status['reasoning'] = reasoning
    
    # ==========================================================================
    # DETERMINE FINAL SIGNAL
    # ==========================================================================
    
    if rsi10 > 79:
        # RSI overbought overrides everything
        signal = "🔴 EXIT BOIL"
        action = "RSI > 79 - Exit regardless of weather"
        signal_type = 'natgas_exit'
    elif weather_override:
        signal = "🟡 HOLD - WEATHER OVERRIDE"
        action = "Backtest says fade, but cold continues. WAIT."
        signal_type = 'natgas_hold'
    elif signal_score >= 2:
        signal = "🟢 BUY BOIL"
        action = "Oversold + cold weather support"
        signal_type = 'natgas_buy'
    elif signal_score <= -2:
        signal = "🔴 GO KOLD"
        action = "Fade the spike - enter KOLD"
        signal_type = 'natgas_short'
    elif signal_score <= -1:
        signal = "🟡 LEAN KOLD"
        action = "Consider reducing BOIL or small KOLD"
        signal_type = 'natgas_warning'
    elif signal_score >= 1:
        signal = "🟢 LEAN BOIL"
        action = "Conditions favor BOIL"
        signal_type = 'natgas_buy'
    else:
        signal = "⚪ NEUTRAL"
        action = "No clear signal - stay flat or hold"
        signal_type = 'natgas_neutral'
    
    boil_status['signal'] = signal
    boil_status['action'] = action
    
    # Create alert if actionable
    if signal_type in ['natgas_exit', 'natgas_short', 'natgas_buy']:
        alert_title = f"🔥 BOIL/KOLD: {signal}"
        alert_msg = f"""BOIL ${price:.2f} | RSI {rsi10:.1f} | 5d {ret_5d:+.1f}%
   Weather: {cold_risk} cold risk | Outlook: {outlook}
   Action: {action}
   Score: {signal_score:+.1f}"""
        alerts.append((alert_title, alert_msg, signal_type))
    elif signal_type == 'natgas_hold' and weather_override:
        alert_title = f"🔥 BOIL/KOLD: {signal}"
        alert_msg = f"""BOIL ${price:.2f} | RSI {rsi10:.1f} | 5d {ret_5d:+.1f}%
   Weather: {cold_risk} cold risk | Outlook: {outlook}
   Action: {action}
   Score: {signal_score:+.1f}"""
        alerts.append((alert_title, alert_msg, 'natgas_warning'))
    
    return alerts, boil_status

# =============================================================================
# ORIGINAL SIGNAL CHECKS (from v2.1)
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
            
            # Get latest values as scalars
            price = safe_float(close.iloc[-1])
            rsi10 = safe_float(calculate_rsi_wilder(close, 10).iloc[-1])
            rsi50 = safe_float(calculate_rsi_wilder(close, 50).iloc[-1])
            sma200 = safe_float(close.rolling(window=200).mean().iloc[-1])
            sma50 = safe_float(close.rolling(window=50).mean().iloc[-1])
            ema21 = safe_float(close.ewm(span=21, adjust=False).mean().iloc[-1])
            
            indicators[ticker] = {
                'price': price,
                'rsi10': rsi10,
                'rsi50': rsi50,
                'sma200': sma200,
                'sma50': sma50,
                'ema21': ema21,
            }
            
            # Calculate % above SMA200
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
        
        # EXIT Signals
        if smh['pct_above_sma200'] >= 40:
            alerts.append(('🔴 SOXL EXIT', f"SMH {smh['pct_above_sma200']:.1f}% above SMA(200) - SELL SOXL", 'exit'))
        elif smh['pct_above_sma200'] >= 35:
            alerts.append(('🟡 SOXL WARNING', f"SMH {smh['pct_above_sma200']:.1f}% above SMA(200) - Approaching sell zone", 'warning'))
        elif smh['pct_above_sma200'] >= 30:
            alerts.append(('🟡 SOXL TRIM', f"SMH {smh['pct_above_sma200']:.1f}% above SMA(200) - Consider trimming 25-50%", 'warning'))
        
        # Death Cross
        if smh['sma50'] < smh['sma200'] and smh['sma200'] > 0:
            alerts.append(('🔴 DEATH CROSS', f"SMH SMA(50) below SMA(200) - Bearish trend", 'exit'))
        
        # BUY Signals - Days below SMA200
        if 'SMH' in data:
            smh_df = data['SMH']
            close = smh_df['Close']
            sma200_series = close.rolling(window=200).mean()
            
            # Count consecutive days below
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
        
        # Double Signal: GLD > 79 AND USDU < 25
        if gld['rsi10'] > 79 and usdu['rsi10'] < 25:
            alerts.append(('🟢🔥 DOUBLE SIGNAL ACTIVE', 
                f"GLD RSI={gld['rsi10']:.1f} > 79 AND USDU RSI={usdu['rsi10']:.1f} < 25\n"
                f"   → Long TQQQ: 88% win, +7% avg (5d)\n"
                f"   → Long UPRO: 85% win, +5.2% avg (5d)\n"
                f"   → AMD/NVDA: 86% win, +5-8% avg (5d)", 'buy'))
            
            # Triple Signal: Add XLP > 65
            if 'XLP' in indicators and indicators['XLP']['rsi10'] > 65:
                xlp = indicators['XLP']
                alerts.append(('🟢🔥🔥 TRIPLE SIGNAL ACTIVE', 
                    f"GLD RSI={gld['rsi10']:.1f} + USDU RSI={usdu['rsi10']:.1f} + XLP RSI={xlp['rsi10']:.1f}\n"
                    f"   → Long TQQQ: 100% win, +11.6% avg (5d) - RARE!", 'buy'))
        
        # Individual GLD overbought
        elif gld['rsi10'] > 79:
            alerts.append(('🟢 GLD OVERBOUGHT', 
                f"GLD RSI={gld['rsi10']:.1f} > 79 → Long TQQQ: 72% win, +3.2% avg (5d)", 'buy'))
    
    # =========================================================================
    # SIGNAL GROUP 3: Defensive Rotation
    # =========================================================================
    defensive_ob = False
    for ticker in ['XLP', 'XLU', 'XLV']:
        if ticker in indicators and indicators[ticker]['rsi10'] > 79:
            defensive_ob = True
            break
    
    if defensive_ob:
        spy_ob = 'SPY' in indicators and indicators['SPY']['rsi10'] > 79
        qqq_ob = 'QQQ' in indicators and indicators['QQQ']['rsi10'] > 79
        
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
                f"QQQ RSI={qqq['rsi10']:.1f} < 20 → Long TQQQ 5d: 69% win, +26% CAGR", 'buy'))
    
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
            uvxy_low = 'UVXY' in indicators and indicators['UVXY']['rsi10'] < 40
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
    # SIGNAL GROUP 8: 3x ETF Signals (CURE, FAS, LABU)
    # =========================================================================
    
    # CURE
    if 'CURE' in indicators:
        cure = indicators['CURE']
        if cure['rsi10'] < 21:
            alerts.append(('🟢 CURE STRONG BUY', 
                f"CURE RSI={cure['rsi10']:.1f} < 21 → Buy CURE: 85% win, +7.3% avg (5d) | n=33", 'buy'))
        elif cure['rsi10'] < 25:
            alerts.append(('🟢 CURE BUY', 
                f"CURE RSI={cure['rsi10']:.1f} < 25 → Buy CURE: 81% win, +5.4% avg (5d) | n=70", 'buy'))
        if cure['rsi10'] > 79:
            alerts.append(('🔴 CURE OVERBOUGHT', 
                f"CURE RSI={cure['rsi10']:.1f} > 79 → Exit CURE: Only 40% win (5d) | n=95", 'exit'))
    
    # FAS
    if 'FAS' in indicators:
        fas = indicators['FAS']
        if fas['rsi10'] < 30:
            alerts.append(('🟢 FAS BUY', 
                f"FAS RSI={fas['rsi10']:.1f} < 30 → Buy FAS: 63% win, +3.3% avg (5d) | n=195", 'buy'))
        if fas['rsi10'] > 85:
            alerts.append(('🔴 FAS SELL', 
                f"FAS RSI={fas['rsi10']:.1f} > 85 → Sell FAS: Only 8% win! (5d) | n=12", 'exit'))
    
    # LABU
    if 'LABU' in indicators:
        labu = indicators['LABU']
        if labu['rsi10'] < 21:
            alerts.append(('🟢 LABU STRONG BUY', 
                f"LABU RSI={labu['rsi10']:.1f} < 21 → Buy LABU: 73% win, +11.2% avg (5d) | n=11", 'buy'))
        elif labu['rsi10'] < 25:
            alerts.append(('🟢 LABU BUY', 
                f"LABU RSI={labu['rsi10']:.1f} < 25 → Buy LABU: 66% win, +5.7% avg (5d) | n=59", 'buy'))
    
    return alerts, status

# =============================================================================
# EMAIL FORMATTING
# =============================================================================
def format_email(alerts, status, boil_status, weather_data, is_preclose=False):
    """Format the email body"""
    now = datetime.now()
    
    timing = "PRE-CLOSE PREVIEW (3:15 PM)" if is_preclose else "MARKET CLOSE CONFIRMATION (4:05 PM)"
    
    body = f"""
{'='*70}
MARKET SIGNAL MONITOR v3.0 - {timing}
{now.strftime('%Y-%m-%d %H:%M')} ET
{'='*70}

"""
    
    # ==========================================================================
    # BOIL/KOLD SECTION (NEW)
    # ==========================================================================
    if boil_status.get('price'):
        cold_risk = weather_data.get('hazards', {}).get('cold_risk', 'Unknown')
        outlook = weather_data.get('outlook', {}).get('6_10_day', {}).get('eastern_us', 'Unknown')
        
        body += f"""
{'='*70}
🔥 NATURAL GAS (BOIL/KOLD) SIGNAL
{'='*70}

┌────────────────────────────────────────────────────────────────────┐
│  SIGNAL: {boil_status.get('signal', 'N/A'):<55} │
│  ACTION: {boil_status.get('action', 'N/A'):<55} │
│  SCORE:  {boil_status.get('signal_score', 0):+.1f}                                                        │
└────────────────────────────────────────────────────────────────────┘

BOIL TECHNICALS:
  Price:         ${boil_status.get('price', 0):.2f}
  RSI(10):       {boil_status.get('rsi10', 0):.1f}  {"⚠️ OVERBOUGHT" if boil_status.get('rsi10', 0) > 79 else "✓ OVERSOLD" if boil_status.get('rsi10', 0) < 21 else ""}
  5-Day Return:  {boil_status.get('ret_5d', 0):+.1f}%
  10-Day Return: {boil_status.get('ret_10d', 0):+.1f}%
  Daily Change:  {boil_status.get('daily_ret', 0):+.1f}%
  vs EMA(9):     {"ABOVE" if boil_status.get('above_ema9') else "BELOW"}
  vs EMA(20):    {"ABOVE" if boil_status.get('above_ema20') else "BELOW"}

WEATHER FORECAST:
  Cold Risk:     {cold_risk}
  6-10 Day:      {outlook}
  Source:        NOAA CPC

INVENTORY:
  Storage:       {weather_data.get('inventory', {}).get('current_level', 'N/A')}
  vs 5-Yr Avg:   {weather_data.get('inventory', {}).get('vs_5yr_avg', 'N/A')}
  Next Report:   {weather_data.get('inventory', {}).get('next_report', 'Thursday 10:30 AM ET')}

SIGNAL REASONING:
"""
        for reason in boil_status.get('reasoning', []):
            body += f"  • {reason}\n"
        
        body += f"""
BACKTEST REFERENCE (Winter Spike Fade):
  ┌─────────────┬─────────────┬─────────────┬─────────────┐
  │ 5d Rally    │ KOLD Win %  │ Avg Return  │ Sample      │
  ├─────────────┼─────────────┼─────────────┼─────────────┤
  │ >30%        │ 69%         │ +10.2%      │ n=65        │
  │ >40%        │ 84%         │ +18.4%      │ n=19        │
  │ >50%        │ 90%         │ +22.7%      │ n=10        │
  └─────────────┴─────────────┴─────────────┴─────────────┘

KEY THRESHOLDS:
  • RSI > 79 = EXIT BOIL (regardless of weather)
  • 6-10 day "Above Normal" = GO KOLD
  • Weather turns warmer = GO KOLD

"""
    
    # ==========================================================================
    # STANDARD ALERTS
    # ==========================================================================
    if alerts:
        buy_alerts = [a for a in alerts if a[2] in ['buy', 'natgas_buy']]
        exit_alerts = [a for a in alerts if a[2] in ['exit', 'short', 'natgas_exit', 'natgas_short']]
        warning_alerts = [a for a in alerts if a[2] in ['warning', 'hedge', 'watch', 'natgas_warning', 'natgas_hold']]
        
        if buy_alerts:
            body += f"{'='*70}\n"
            body += "🟢 BUY SIGNALS:\n" + "-"*50 + "\n"
            for title, msg, _ in buy_alerts:
                body += f"{title}\n{msg}\n\n"
        
        if exit_alerts:
            body += f"{'='*70}\n"
            body += "🔴 EXIT/SHORT SIGNALS:\n" + "-"*50 + "\n"
            for title, msg, _ in exit_alerts:
                body += f"{title}\n{msg}\n\n"
        
        if warning_alerts:
            body += f"{'='*70}\n"
            body += "🟡 WARNINGS/WATCH:\n" + "-"*50 + "\n"
            for title, msg, _ in warning_alerts:
                body += f"{title}\n{msg}\n\n"
    else:
        body += "No signals triggered today.\n\n"
    
    # ==========================================================================
    # INDICATOR STATUS
    # ==========================================================================
    body += f"""
{'='*70}
CURRENT INDICATOR STATUS
{'='*70}

"""
    
    indicators = status.get('indicators', {})
    
    key_tickers = ['SPY', 'QQQ', 'SMH', 'GLD', 'USDU', 'XLP', 'TLT', 'HYG', 'XLF', 'UVXY', 'BTC-USD', 'AMD', 'NVDA']
    body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}\n"
    body += "-"*50 + "\n"
    
    for ticker in key_tickers:
        if ticker in indicators:
            ind = indicators[ticker]
            price = f"${ind['price']:.2f}" if ind['price'] < 1000 else f"${ind['price']:,.0f}"
            rsi = f"{ind['rsi10']:.1f}"
            pct = f"{ind.get('pct_above_sma200', 0):+.1f}%"
            body += f"{ticker:<10} {price:>12} {rsi:>10} {pct:>12}\n"
    
    # 3x Leveraged ETFs
    body += f"""
{'='*70}
3x LEVERAGED ETFs
{'='*70}
"""
    body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}  Signal\n"
    body += "-"*65 + "\n"
    
    leveraged_tickers = ['BOIL', 'KOLD', 'NAIL', 'CURE', 'FAS', 'LABU', 'TQQQ', 'SOXL']
    for ticker in leveraged_tickers:
        if ticker in indicators:
            ind = indicators[ticker]
            price = f"${ind['price']:.2f}"
            rsi = f"{ind['rsi10']:.1f}"
            pct = f"{ind.get('pct_above_sma200', 0):+.1f}%"
            
            # Signal status
            rsi_val = ind['rsi10']
            if rsi_val < 21:
                signal = "🟢 OVERSOLD"
            elif rsi_val < 30:
                signal = "🟢 Watch"
            elif rsi_val > 85:
                signal = "🔴 OVERBOUGHT"
            elif rsi_val > 79:
                signal = "🟡 Extended"
            else:
                signal = ""
            
            body += f"{ticker:<10} {price:>12} {rsi:>10} {pct:>12}  {signal}\n"
    
    # SMH/SOXL Levels
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
    
    if is_preclose:
        body += f"""
{'='*70}
NOTE: This is a PRE-CLOSE preview. Signals may change by market close.
Final confirmation email will be sent at 4:05 PM ET.
{'='*70}
"""
    
    body += f"""
{'='*70}
DATA SOURCES
{'='*70}
  Weather: NOAA Climate Prediction Center (cpc.ncep.noaa.gov)
  Inventory: EIA Weekly Natural Gas Storage
  Prices: Yahoo Finance via yfinance
  
  For intraday HDD trends: celsiusenergy.net/p/weather-data.html
{'='*70}
"""
    
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
    
    # =========================================================================
    # FETCH WEATHER DATA
    # =========================================================================
    weather_data = get_all_weather_data()
    
    # =========================================================================
    # FETCH MARKET DATA
    # =========================================================================
    tickers = [
        # Core Indices
        'SMH', 'SPY', 'QQQ', 'IWM',
        # Defensive Sectors
        'XLP', 'XLU', 'XLV',
        # Safe Havens & Macro
        'GLD', 'TLT', 'HYG', 'LQD', 'TMV',
        'USDU', 'UCO',
        # Natural Gas (NEW)
        'BOIL', 'KOLD',
        # Volatility
        'UVXY',
        # International
        'EDC', 'YINN', 'KORU', 'EURL', 'INDL',
        # Crypto
        'BTC-USD',
        # Individual Stocks
        'AMD', 'NVDA',
        # 3x Leveraged ETFs
        'NAIL', 'CURE', 'FAS', 'LABU',
        'TQQQ', 'SOXL', 'TECL', 'DRN',
        # Style/Factor ETFs
        'VOOV', 'VOOG', 'VTV', 'QQQE',
        # Energy
        'XLE', 'XLF',
    ]
    
    print("Downloading market data...")
    data = download_data(tickers)
    print(f"Downloaded data for {len(data)} tickers")
    
    # =========================================================================
    # CHECK SIGNALS
    # =========================================================================
    
    # Standard signals
    alerts, status = check_signals(data)
    
    # BOIL/KOLD signals with weather integration
    boil_alerts, boil_status = check_boil_kold_signals(data, weather_data)
    
    # Combine alerts
    all_alerts = alerts + boil_alerts
    
    # =========================================================================
    # DETERMINE EMAIL SUBJECT
    # =========================================================================
    if all_alerts:
        buy_count = len([a for a in all_alerts if 'buy' in a[2].lower()])
        exit_count = len([a for a in all_alerts if 'exit' in a[2].lower() or 'short' in a[2].lower()])
        natgas_alert = any('natgas' in a[2] for a in all_alerts)
        
        if exit_count > 0:
            emoji = "🔴"
            urgency = "EXIT SIGNALS"
        elif buy_count > 0:
            emoji = "🟢"
            urgency = "BUY SIGNALS"
        else:
            emoji = "🟡"
            urgency = "WATCH"
        
        if natgas_alert:
            emoji = "🔥" + emoji
        
        timing = "PRE-CLOSE" if IS_PRECLOSE else "CLOSE"
        subject = f"{emoji} [{timing}] Market Signals: {len(all_alerts)} Alert(s) - {urgency}"
    else:
        timing = "PRE-CLOSE" if IS_PRECLOSE else "CLOSE"
        subject = f"📊 [{timing}] Market Signals: No Alerts"
    
    # =========================================================================
    # FORMAT AND SEND EMAIL
    # =========================================================================
    body = format_email(all_alerts, status, boil_status, weather_data, IS_PRECLOSE)
    send_email(subject, body)
    
    print(f"\n{len(all_alerts)} signal(s) detected")
    for title, msg, _ in all_alerts:
        print(f"  {title}")

if __name__ == "__main__":
    main()
