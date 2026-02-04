#!/usr/bin/env python3
"""
Comprehensive Market Signal Monitor v5.2
========================================
Monitors all backtested trading signals and sends alerts.

SCHEDULE: Two emails daily (weekdays)
- 12:00 PM ET: Pre-close preview (noon)
- 4:05 PM ET: Market close confirmation

NEW IN v5.2 (Feb 4, 2026):
- Added SMH/IGV Rotation Framework from backtest analysis
- When SMH leads IGV by RSI spread > 25 AND IGV < 35: Long TECL (75% win, +10.5% avg 10d)
- When IGV leads SMH by RSI spread < -25 AND SMH < 35: Long SOXL (80% win, +12.1% avg 10d)
- RSI spread convergence is reliable but PRICE leadership persists
- Best signal: Spread > 30 + IGV < 35 → TECL: 78% win, +13.2% avg (10d)
- Pre-close moved to noon ET for more actionable timing

FROM v5.1 (Jan 28, 2026):
- Added FNGO (2x FANG+ ETN) signals from backtest analysis
- FNGO responds to GLD/USDU combo: 91% win, +8.9% avg (5d) | n=11
- FNGO RSI < 25: 100% win, +12.1% avg (5d) | n=11
- FNGO EXIT when SPY/QQQ > 79: Only 32-36% win rate
- KEY: FNGO does NOT follow momentum (unlike BTC) - overbought = SELL

FROM v5.0 (Jan 28, 2026):
- Enhanced XLP/XLU/XLV defensive rotation signals based on backtesting
- XLP RSI 75-79 "transition zone" = UVXY hedge (56% win, +1.69% 1-day)
- XLP RSI > 82 = next-day UVXY trade (67% win, +4.81%)
- XLU overbought = SHORT via SDP (76% win, +2.34% 20d) - WORKS!
- XLV overbought = DO NOT SHORT (only 42% win) - use as signal only
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

# =============================================================================
# CONFIGURATION
# =============================================================================
SENDER_EMAIL = os.environ.get('SENDER_EMAIL', '')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD', '')
RECIPIENT_EMAIL = os.environ.get('RECIPIENT_EMAIL', '')
PHONE_EMAIL = os.environ.get('PHONE_EMAIL', '')

IS_PRECLOSE = len(sys.argv) > 1 and sys.argv[1] == 'preclose'

# NYC coordinates for weather (proxy for Eastern US heating demand)
NYC_LAT = 40.74
NYC_LON = -74.04

# =============================================================================
# WEATHER DATA FUNCTIONS (Open-Meteo API - Free, Reliable)
# =============================================================================
def fetch_openmeteo_forecast():
    """
    Fetch 7-day temperature forecast from Open-Meteo API.
    Returns dict with forecast data for BOIL/KOLD signal generation.
    """
    forecast = {
        'fetched': False,
        'source': 'Open-Meteo API',
        'current_temp': None,
        'temps_7d': [],
        'temp_change_7d': None,
        'cold_coming': False,
        'warming_coming': False,
        'error': None
    }
    
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={NYC_LAT}&longitude={NYC_LON}"
            f"&daily=temperature_2m_max,temperature_2m_min"
            f"&temperature_unit=fahrenheit"
            f"&timezone=America/New_York"
            f"&forecast_days=8"
        )
        
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            daily = data.get('daily', {})
            
            temps_max = daily.get('temperature_2m_max', [])
            temps_min = daily.get('temperature_2m_min', [])
            dates = daily.get('time', [])
            
            if temps_max and temps_min:
                temps_mean = [(mx + mn) / 2 for mx, mn in zip(temps_max, temps_min)]
                forecast['temps_7d'] = temps_mean
                forecast['dates'] = dates
                forecast['current_temp'] = temps_mean[0] if temps_mean else None
                
                if len(temps_mean) >= 8:
                    forecast['temp_change_7d'] = temps_mean[0] - temps_mean[7]
                    
                    if forecast['temp_change_7d'] >= 20:
                        forecast['cold_coming'] = True
                        forecast['cold_intensity'] = 'SEVERE'
                    elif forecast['temp_change_7d'] >= 15:
                        forecast['cold_coming'] = True
                        forecast['cold_intensity'] = 'MODERATE'
                    elif forecast['temp_change_7d'] >= 10:
                        forecast['cold_coming'] = True
                        forecast['cold_intensity'] = 'MILD'
                    elif forecast['temp_change_7d'] <= -10:
                        forecast['warming_coming'] = True
                        forecast['warming_intensity'] = 'SIGNIFICANT'
                    elif forecast['temp_change_7d'] <= -5:
                        forecast['warming_coming'] = True
                        forecast['warming_intensity'] = 'MILD'
                
                forecast['fetched'] = True
                
    except Exception as e:
        forecast['error'] = str(e)
        print(f"Open-Meteo fetch error: {e}")
    
    return forecast

def get_weather_data():
    """Fetch weather forecast data."""
    print("Fetching weather forecast data...")
    
    weather_data = {
        'forecast': fetch_openmeteo_forecast(),
        'fetch_time': datetime.now().strftime('%Y-%m-%d %H:%M ET')
    }
    
    print(f"  Open-Meteo fetched: {weather_data['forecast'].get('fetched', False)}")
    if weather_data['forecast'].get('temp_change_7d') is not None:
        print(f"  7-day temp change: {weather_data['forecast']['temp_change_7d']:+.1f}°F")
    
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
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    return data

# =============================================================================
# BOIL/KOLD SIGNAL LOGIC
# =============================================================================
def check_boil_kold_signals(data, weather_data, indicators):
    """
    Check BOIL/KOLD signals with refined logic from Jan 2026 backtest.
    """
    alerts = []
    boil_status = {
        'signal': 'NEUTRAL',
        'action': 'No clear signal',
        'tier': None,
        'reasoning': [],
        'weather_override': False
    }
    
    if 'BOIL' not in data:
        boil_status['signal'] = '⚠️ NO DATA'
        boil_status['action'] = 'BOIL data unavailable'
        return alerts, boil_status
    
    boil_df = data['BOIL']
    close = boil_df['Close']
    
    price = safe_float(close.iloc[-1])
    rsi10 = safe_float(calculate_rsi_wilder(close, 10).iloc[-1])
    
    # Calculate gains
    gain_5d = 0
    gain_7d = 0
    if len(close) >= 6:
        gain_5d = (safe_float(close.iloc[-1]) / safe_float(close.iloc[-6]) - 1) * 100
    if len(close) >= 8:
        gain_7d = (safe_float(close.iloc[-1]) / safe_float(close.iloc[-8]) - 1) * 100
    
    # Get macro indicators
    uco_rsi = indicators.get('UCO', {}).get('rsi10', 50)
    uvxy_rsi = indicators.get('UVXY', {}).get('rsi10', 50)
    usdu_rsi = indicators.get('USDU', {}).get('rsi10', 50)
    
    # Weather data
    forecast = weather_data.get('forecast', {})
    temp_change_7d = forecast.get('temp_change_7d', 0) or 0
    current_temp = forecast.get('current_temp', 'N/A')
    cold_coming = forecast.get('cold_coming', False)
    
    boil_status.update({
        'price': price,
        'rsi10': rsi10,
        'gain_5d': gain_5d,
        'gain_7d': gain_7d,
        'uco_rsi': uco_rsi,
        'uvxy_rsi': uvxy_rsi,
        'usdu_rsi': usdu_rsi,
        'temp_change_7d': temp_change_7d,
        'current_temp': current_temp,
    })
    
    # Determine signals
    kold_tier = 0
    boil_signal = False
    boil_type = None
    
    # KOLD fade signals (5-day gain bands)
    if gain_5d >= 50:
        kold_tier = 1
        boil_status['reasoning'].append(f"KOLD T1: 5d gain {gain_5d:+.1f}% >= 50% | 100% win, +25.4% avg")
    elif gain_5d >= 40:
        kold_tier = 1
        boil_status['reasoning'].append(f"KOLD T1: 5d gain {gain_5d:+.1f}% >= 40% | 89% win, +18.5% avg")
    elif gain_5d >= 30:
        kold_tier = 2
        boil_status['reasoning'].append(f"KOLD T2: 5d gain {gain_5d:+.1f}% >= 30% | 88% win, +14.5% avg")
    elif gain_5d >= 20:
        kold_tier = 3
        boil_status['reasoning'].append(f"KOLD Watch: 5d gain {gain_5d:+.1f}% approaching fade zone")
    
    # UCO filter enhancement
    if kold_tier > 0:
        if uco_rsi > 50:
            boil_status['reasoning'].append(f"UCO RSI {uco_rsi:.1f} > 50 = Enhanced (77% win)")
        else:
            boil_status['reasoning'].append(f"⚠️ UCO RSI {uco_rsi:.1f} < 50 = Weaker signal (57% win)")
    
    # Supply shock signal
    if uvxy_rsi > 70 and uco_rsi > 60:
        boil_signal = True
        boil_type = 'supply_shock'
        boil_status['reasoning'].append(f"Supply shock: UVXY>{uvxy_rsi:.0f} + UCO>{uco_rsi:.0f} | 73% win, +23.5%")
    
    # Weather-based BOIL signal
    if cold_coming and temp_change_7d >= 15 and rsi10 < 50:
        boil_signal = True
        boil_type = 'weather'
        boil_status['reasoning'].append(f"Cold front: {temp_change_7d:+.1f}°F + RSI {rsi10:.1f} < 50")
    
    # BOIL oversold
    if rsi10 < 25 and not kold_tier:
        boil_signal = True
        boil_type = 'oversold'
        boil_status['reasoning'].append(f"BOIL oversold: RSI {rsi10:.1f} < 25")
    
    # Weather override for KOLD
    if kold_tier > 0 and cold_coming and temp_change_7d >= 15:
        boil_status['weather_override'] = True
        boil_status['reasoning'].append(f"⚠️ Cold front ({temp_change_7d:+.1f}°F) blocking KOLD fade")
    
    # Determine final signal
    signal_type = 'natgas_neutral'
    
    if kold_tier == 1 and not boil_status['weather_override']:
        signal = "🔴 KOLD - TIER 1 FADE"
        action = f"Strong fade: 5d gain {gain_5d:+.1f}% | Enter KOLD position"
        signal_type = 'natgas_kold_t1'
    elif kold_tier == 2 and not boil_status['weather_override']:
        signal = "🟡 KOLD - TIER 2 FADE"
        action = f"Moderate fade: 5d gain {gain_5d:+.1f}% | Scale into KOLD"
        signal_type = 'natgas_kold_t2'
    elif boil_status['weather_override']:
        signal = "🟡 HOLD - WEATHER BLOCK"
        action = f"Cold front blocking fade - wait for weather to clear"
        signal_type = 'natgas_hold'
    elif boil_signal:
        if boil_type == 'supply_shock':
            signal = "🟢 BOIL - SUPPLY SHOCK"
            action = "Rare geopolitical/supply signal - consider BOIL"
            signal_type = 'natgas_boil_shock'
        elif boil_type == 'weather':
            signal = "🟢 BOIL - WEATHER"
            action = f"Cold coming ({temp_change_7d:+.1f}°F) + RSI low - buy BOIL"
            signal_type = 'natgas_boil_weather'
        else:
            signal = "🟢 BOIL - OVERSOLD"
            action = "RSI oversold - mean reversion likely"
            signal_type = 'natgas_boil_oversold'
    elif kold_tier == 3:
        signal = "🟡 KOLD WATCH"
        action = f"5d gain {gain_5d:+.1f}% - approaching fade territory"
        signal_type = 'natgas_watch'
    else:
        signal = "⚪ NEUTRAL"
        action = "No clear signal"
        signal_type = 'natgas_neutral'
    
    boil_status['signal'] = signal
    boil_status['action'] = action
    
    # Create alert if actionable
    if signal_type in ['natgas_kold_t1', 'natgas_kold_t2', 'natgas_boil_shock', 
                       'natgas_boil_weather', 'natgas_boil_oversold']:
        alert_title = f"🔥 NATGAS: {signal}"
        alert_msg = f"""BOIL ${price:.2f} | RSI {rsi10:.1f} | 5d {gain_5d:+.1f}%
   UCO RSI: {uco_rsi:.1f} | UVXY RSI: {uvxy_rsi:.1f}
   Weather: {temp_change_7d:+.1f}°F change (7d) | Temp: {current_temp}°F
   Action: {action}"""
        alerts.append((alert_title, alert_msg, signal_type))
    
    elif signal_type == 'natgas_hold':
        alert_title = f"🔥 NATGAS: {signal}"
        alert_msg = f"""BOIL ${price:.2f} | RSI {rsi10:.1f} | 5d {gain_5d:+.1f}%
   Weather: {temp_change_7d:+.1f}°F cold coming - blocking fade
   Action: {action}"""
        alerts.append((alert_title, alert_msg, 'natgas_warning'))
    
    return alerts, boil_status

# =============================================================================
# SMH/IGV ROTATION SIGNALS - NEW in v5.2
# =============================================================================
def check_smh_igv_rotation(indicators):
    """
    Check SMH/IGV rotation signals based on Feb 2026 backtest.
    
    KEY FINDINGS:
    - RSI spread (SMH - IGV) mean reverts over 10-20 days
    - But PRICE leadership can persist even as RSI converges
    - Best trade: Long the LAGGARD with 3x leverage when spread extreme + laggard oversold
    
    LONG TECL (IGV lagging):
    - Spread > 30 + IGV < 35: 78% win, +13.2% avg (10d) | n=13
    - Spread > 25 + IGV < 35: 75% win, +10.5% avg (10d) | n=19
    - Spread > 25 + IGV < 40: 70% win, +6.9% avg (10d) | n=30
    
    LONG SOXL (SMH lagging):
    - Spread < -15 + SMH < 30: 88% win, +14.9% avg (10d) | n=34
    - Spread < -25 + SMH < 35: 80% win, +12.1% avg (10d) | n=21
    - Spread < -15 + SMH < 35: 74% win, +12.8% avg (10d) | n=71
    """
    alerts = []
    rotation_status = {
        'smh_rsi': 0,
        'igv_rsi': 0,
        'rsi_spread': 0,
        'signal': 'NEUTRAL',
        'action': None
    }
    
    if 'SMH' not in indicators or 'IGV' not in indicators:
        return alerts, rotation_status
    
    smh_rsi = indicators['SMH']['rsi10']
    igv_rsi = indicators['IGV']['rsi10']
    rsi_spread = smh_rsi - igv_rsi
    
    gld_rsi = indicators.get('GLD', {}).get('rsi10', 50)
    
    rotation_status.update({
        'smh_rsi': smh_rsi,
        'igv_rsi': igv_rsi,
        'rsi_spread': rsi_spread,
    })
    
    # =========================================================================
    # LONG TECL SIGNALS (IGV lagging, SMH leading)
    # =========================================================================
    
    # Tier 1: Best signal - extreme spread + IGV deeply oversold
    if rsi_spread > 30 and igv_rsi < 35:
        rotation_status['signal'] = 'TECL_T1'
        rotation_status['action'] = 'Long TECL'
        alerts.append(('🟢🔥 SMH/IGV ROTATION - LONG TECL', 
            f"RSI Spread: {rsi_spread:+.1f} (SMH leading)\n"
            f"   SMH RSI: {smh_rsi:.1f} | IGV RSI: {igv_rsi:.1f}\n"
            f"   → Spread > 30 + IGV < 35: 78% win, +13.2% avg (10d) | n=13\n"
            f"   → RSI spread will converge - IGV/TECL bounces\n"
            f"   → Hold 10 trading days", 'buy'))
    
    # Tier 2: Good signal - moderate spread + IGV oversold
    elif rsi_spread > 25 and igv_rsi < 35:
        rotation_status['signal'] = 'TECL_T2'
        rotation_status['action'] = 'Long TECL'
        alerts.append(('🟢 SMH/IGV ROTATION - LONG TECL', 
            f"RSI Spread: {rsi_spread:+.1f} (SMH leading)\n"
            f"   SMH RSI: {smh_rsi:.1f} | IGV RSI: {igv_rsi:.1f}\n"
            f"   → Spread > 25 + IGV < 35: 75% win, +10.5% avg (10d) | n=19\n"
            f"   → Hold 10 trading days", 'buy'))
    
    # Tier 2 with GLD enhancement
    elif rsi_spread > 20 and igv_rsi < 35 and gld_rsi > 70:
        rotation_status['signal'] = 'TECL_T2_GLD'
        rotation_status['action'] = 'Long TECL'
        alerts.append(('🟢 SMH/IGV ROTATION + GLD - LONG TECL', 
            f"RSI Spread: {rsi_spread:+.1f} + GLD RSI: {gld_rsi:.1f} > 70\n"
            f"   SMH RSI: {smh_rsi:.1f} | IGV RSI: {igv_rsi:.1f}\n"
            f"   → Spread > 20 + IGV < 35 + GLD > 70: ~80% win (10d)\n"
            f"   → GLD filter improves signal quality\n"
            f"   → Hold 10 trading days", 'buy'))
    
    # Tier 3: Watch zone
    elif rsi_spread > 25 and igv_rsi < 40:
        rotation_status['signal'] = 'TECL_WATCH'
        rotation_status['action'] = 'Watch TECL'
        alerts.append(('🟡 SMH/IGV SPREAD ELEVATED - WATCH TECL', 
            f"RSI Spread: {rsi_spread:+.1f} (SMH leading)\n"
            f"   SMH RSI: {smh_rsi:.1f} | IGV RSI: {igv_rsi:.1f}\n"
            f"   → Spread > 25 + IGV < 40: 70% win, +6.9% avg (10d) | n=30\n"
            f"   → Wait for IGV < 35 for better entry", 'watch'))
    
    # =========================================================================
    # LONG SOXL SIGNALS (SMH lagging, IGV leading)
    # =========================================================================
    
    # Tier 1: Best signal - extreme spread + SMH deeply oversold
    elif rsi_spread < -15 and smh_rsi < 30:
        rotation_status['signal'] = 'SOXL_T1'
        rotation_status['action'] = 'Long SOXL'
        alerts.append(('🟢🔥 SMH/IGV ROTATION - LONG SOXL', 
            f"RSI Spread: {rsi_spread:+.1f} (IGV leading)\n"
            f"   SMH RSI: {smh_rsi:.1f} | IGV RSI: {igv_rsi:.1f}\n"
            f"   → Spread < -15 + SMH < 30: 88% win, +14.9% avg (10d) | n=34\n"
            f"   → RSI spread will converge - SMH/SOXL bounces\n"
            f"   → Hold 10 trading days", 'buy'))
    
    # Tier 2: Good signal
    elif rsi_spread < -25 and smh_rsi < 35:
        rotation_status['signal'] = 'SOXL_T2'
        rotation_status['action'] = 'Long SOXL'
        alerts.append(('🟢 SMH/IGV ROTATION - LONG SOXL', 
            f"RSI Spread: {rsi_spread:+.1f} (IGV leading)\n"
            f"   SMH RSI: {smh_rsi:.1f} | IGV RSI: {igv_rsi:.1f}\n"
            f"   → Spread < -25 + SMH < 35: 80% win, +12.1% avg (10d) | n=21\n"
            f"   → Hold 10 trading days", 'buy'))
    
    # Tier 3: Watch zone
    elif rsi_spread < -15 and smh_rsi < 35:
        rotation_status['signal'] = 'SOXL_WATCH'
        rotation_status['action'] = 'Watch SOXL'
        alerts.append(('🟡 SMH/IGV SPREAD - WATCH SOXL', 
            f"RSI Spread: {rsi_spread:+.1f} (IGV leading)\n"
            f"   SMH RSI: {smh_rsi:.1f} | IGV RSI: {igv_rsi:.1f}\n"
            f"   → Spread < -15 + SMH < 35: 74% win, +12.8% avg (10d) | n=71\n"
            f"   → Consider entry or wait for SMH < 30", 'watch'))
    
    # =========================================================================
    # EXTREME SPREAD WARNING (informational)
    # =========================================================================
    elif abs(rsi_spread) > 20:
        direction = "SMH leading" if rsi_spread > 0 else "IGV leading"
        rotation_status['signal'] = 'SPREAD_ELEVATED'
        alerts.append(('ℹ️ SMH/IGV SPREAD ELEVATED', 
            f"RSI Spread: {rsi_spread:+.1f} ({direction})\n"
            f"   SMH RSI: {smh_rsi:.1f} | IGV RSI: {igv_rsi:.1f}\n"
            f"   → Spread typically converges over 10-20 days\n"
            f"   → Wait for laggard RSI < 35 for entry signal", 'info'))
    
    return alerts, rotation_status

# =============================================================================
# ORIGINAL SIGNAL CHECKS
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
                'price': price,
                'rsi10': rsi10,
                'rsi50': rsi50,
                'sma200': sma200,
                'sma50': sma50,
                'ema21': ema21,
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
    # SIGNAL GROUP 0: SMH/IGV ROTATION (NEW in v5.2)
    # =========================================================================
    rotation_alerts, rotation_status = check_smh_igv_rotation(indicators)
    alerts.extend(rotation_alerts)
    status['rotation'] = rotation_status
    
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
    # SIGNAL GROUP 3: ENHANCED Defensive Rotation
    # =========================================================================
    xlp_rsi = indicators.get('XLP', {}).get('rsi10', 0)
    xlu_rsi = indicators.get('XLU', {}).get('rsi10', 0)
    xlv_rsi = indicators.get('XLV', {}).get('rsi10', 0)
    spy_rsi = indicators.get('SPY', {}).get('rsi10', 0)
    qqq_rsi = indicators.get('QQQ', {}).get('rsi10', 0)
    smh_rsi = indicators.get('SMH', {}).get('rsi10', 0)
    
    defensive_status = []
    if xlp_rsi > 79:
        defensive_status.append(f"XLP={xlp_rsi:.1f}")
    if xlu_rsi > 79:
        defensive_status.append(f"XLU={xlu_rsi:.1f}")
    if xlv_rsi > 79:
        defensive_status.append(f"XLV={xlv_rsi:.1f}")
    
    if 75 <= xlp_rsi < 79:
        alerts.append(('🟡 XLP TRANSITION ZONE', 
            f"XLP RSI={xlp_rsi:.1f} in 75-79 range\n"
            f"   → Small UVXY hedge (1-2 day hold): 56% win, +1.69% avg", 'hedge'))
    
    if xlp_rsi > 82:
        alerts.append(('🟡 XLP EXTREME - UVXY', 
            f"XLP RSI={xlp_rsi:.1f} > 82\n"
            f"   → Next-day UVXY trade: 67% win, +4.81% avg (1-day only!)", 'hedge'))
    
    if xlu_rsi > 79 and spy_rsi < 79 and qqq_rsi < 79:
        alerts.append(('🟢 XLU OVERBOUGHT - SHORT UTILITIES', 
            f"XLU RSI={xlu_rsi:.1f} > 79 + SPY/QQQ not overbought\n"
            f"   → Short XLU (via SDP): 76% win, +2.34% avg (20d)", 'buy'))
        
        if xlu_rsi > 82:
            alerts.append(('🟢🔥 XLU EXTREME - STRONG SHORT', 
                f"XLU RSI={xlu_rsi:.1f} > 82\n"
                f"   → Short XLU (via SDP): 89% win, +2.98% avg (20d)", 'buy'))
    
    any_defensive_ob = xlp_rsi > 79 or xlu_rsi > 79 or xlv_rsi > 79
    
    if any_defensive_ob and spy_rsi < 79 and qqq_rsi < 79:
        if 50 <= smh_rsi <= 70:
            smh_note = "SMH in goldilocks zone (50-70) - BEST for growth longs"
        elif smh_rsi < 50:
            smh_note = "⚠️ SMH weak (<50) - Favor SDP short over SOXL long"
        else:
            smh_note = "⚠️ SMH extended (>70) - Rotation risk, favor SDP short"
        
        alerts.append(('🟢 DEFENSIVE ROTATION SIGNAL', 
            f"Defensive OB: {', '.join(defensive_status) if defensive_status else 'None'}\n"
            f"SPY RSI={spy_rsi:.1f}, QQQ RSI={qqq_rsi:.1f} (not OB)\n"
            f"   → BEST: 50% SOXL + 50% SDP pairs trade\n"
            f"   {smh_note}", 'buy'))
        
        if xlv_rsi > 79:
            alerts.append(('ℹ️ XLV NOTE - DO NOT SHORT', 
                f"XLV RSI={xlv_rsi:.1f} > 79 - Healthcare does NOT mean-revert!\n"
                f"   → Use XLV as SIGNAL only, short UTILITIES (XLU) instead", 'info'))
    
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
    # SIGNAL GROUP 8: 3x ETF Signals (CURE, FAS, LABU, NAIL)
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
        
        if 'GLD' in indicators and 'USDU' in indicators:
            gld = indicators['GLD']
            usdu = indicators['USDU']
            if gld['rsi10'] > 79 and usdu['rsi10'] < 25:
                alerts.append(('🟢 FAS SIGNAL', 
                    f"GLD>{gld['rsi10']:.0f} + USDU<{usdu['rsi10']:.0f} → Long FAS 10d: 92% win, +5.8% avg | n=13", 'buy'))
        
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
    
    # NAIL
    if 'NAIL' in indicators and 'GLD' in indicators and 'USDU' in indicators and 'XLF' in indicators:
        nail = indicators['NAIL']
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
    
    # =========================================================================
    # SIGNAL GROUP 9: FNGO (2x FANG+) Signals
    # =========================================================================
    if 'FNGO' in indicators:
        fngo = indicators['FNGO']
        fngo_rsi = fngo['rsi10']
        
        if 'GLD' in indicators and 'USDU' in indicators:
            gld = indicators['GLD']
            usdu = indicators['USDU']
            
            if gld['rsi10'] > 79 and usdu['rsi10'] < 25:
                alerts.append(('🟢 FNGO SIGNAL', 
                    f"GLD RSI={gld['rsi10']:.1f} > 79 + USDU RSI={usdu['rsi10']:.1f} < 25\n"
                    f"   → Long FNGO: 91% win, +8.9% avg (5d), +11.8% avg (10d) | n=11", 'buy'))
                
                if 'XLP' in indicators and indicators['XLP']['rsi10'] > 65:
                    xlp = indicators['XLP']
                    alerts.append(('🟢🔥 FNGO TRIPLE SIGNAL', 
                        f"GLD>{gld['rsi10']:.0f} + USDU<{usdu['rsi10']:.0f} + XLP>{xlp['rsi10']:.0f}\n"
                        f"   → Long FNGO: 100% win, +9.2% avg (5d) | n=4 ⚠️ Low sample", 'buy'))
        
        if fngo_rsi < 25:
            alerts.append(('🟢 FNGO OVERSOLD', 
                f"FNGO RSI={fngo_rsi:.1f} < 25 → Buy FNGO: 100% win, +12.1% avg (5d) | n=11", 'buy'))
        elif fngo_rsi < 30:
            alerts.append(('🟢 FNGO WATCH', 
                f"FNGO RSI={fngo_rsi:.1f} < 30 → Consider FNGO: 73% win, +5.5% avg (5d) | n=67", 'buy'))
        
        if 'SPY' in indicators and indicators['SPY']['rsi10'] > 79:
            spy = indicators['SPY']
            alerts.append(('🔴 FNGO EXIT - SPY OB', 
                f"SPY RSI={spy['rsi10']:.1f} > 79 → Exit FNGO: Only 36% win, -3.9% avg (5d)", 'exit'))
        
        elif 'QQQ' in indicators and indicators['QQQ']['rsi10'] > 79:
            qqq = indicators['QQQ']
            alerts.append(('🔴 FNGO EXIT - QQQ OB', 
                f"QQQ RSI={qqq['rsi10']:.1f} > 79 → Exit FNGO: Only 32% win, -2.9% avg (5d)", 'exit'))
        
        if fngo_rsi > 85:
            alerts.append(('🔴 FNGO OVERBOUGHT', 
                f"FNGO RSI={fngo_rsi:.1f} > 85 → TRIM FNGO: Only 46% win, -4.4% avg (5d)", 'exit'))
        elif fngo_rsi > 79:
            alerts.append(('🟡 FNGO EXTENDED', 
                f"FNGO RSI={fngo_rsi:.1f} > 79 → Watch FNGO: Only 48% win, -1.3% avg (5d)", 'warning'))
        
        if 'QQQ' in indicators:
            qqq = indicators['QQQ']
            rsi_diff = fngo_rsi - qqq['rsi10']
            if rsi_diff > 10:
                alerts.append(('🔴 FNGO DIVERGENCE WARNING', 
                    f"FNGO RSI={fngo_rsi:.1f} > QQQ RSI={qqq['rsi10']:.1f} by {rsi_diff:.1f}\n"
                    f"   → EXIT FNGO when divergence > 10 points: Only 30% win", 'exit'))
    
    return alerts, status

# =============================================================================
# EMAIL FORMATTING
# =============================================================================
def format_email(alerts, status, boil_status, weather_data, is_preclose=False):
    """Format the email body"""
    now = datetime.now()
    
    timing = "PRE-CLOSE PREVIEW (12:00 PM)" if is_preclose else "MARKET CLOSE CONFIRMATION (4:05 PM)"
    
    body = f"""
{'='*70}
MARKET SIGNAL MONITOR v5.2 - {timing}
{now.strftime('%Y-%m-%d %H:%M')} ET
{'='*70}

"""
    
    # ==========================================================================
    # SMH/IGV ROTATION STATUS (NEW in v5.2)
    # ==========================================================================
    rotation = status.get('rotation', {})
    if rotation.get('smh_rsi', 0) > 0:
        body += f"""{'='*70}
🔄 SMH/IGV ROTATION STATUS
{'='*70}
SMH RSI(10): {rotation.get('smh_rsi', 0):.1f}
IGV RSI(10): {rotation.get('igv_rsi', 0):.1f}
RSI Spread:  {rotation.get('rsi_spread', 0):+.1f} ({'SMH leading' if rotation.get('rsi_spread', 0) > 0 else 'IGV leading'})

Signal: {rotation.get('signal', 'NEUTRAL')}
{f"Action: {rotation.get('action')}" if rotation.get('action') else ""}

Quick Reference:
  Spread > 25 + IGV < 35 → TECL: 75% win, +10.5% avg (10d)
  Spread > 30 + IGV < 35 → TECL: 78% win, +13.2% avg (10d)
  Spread < -25 + SMH < 35 → SOXL: 80% win, +12.1% avg (10d)
  Spread < -15 + SMH < 30 → SOXL: 88% win, +14.9% avg (10d)

"""
    
    # ==========================================================================
    # BOIL/KOLD STATUS
    # ==========================================================================
    body += f"""{'='*70}
🔥 NATURAL GAS (BOIL/KOLD) STATUS
{'='*70}
Signal: {boil_status.get('signal', 'N/A')}
Action: {boil_status.get('action', 'N/A')}

BOIL: ${boil_status.get('price', 0):.2f} | RSI(10): {boil_status.get('rsi10', 0):.1f}
5-Day Gain: {boil_status.get('gain_5d', 0):+.1f}% | 7-Day Gain: {boil_status.get('gain_7d', 0):+.1f}%

Macro Filters:
  UCO RSI: {boil_status.get('uco_rsi', 0):.1f} {'(>50 ✓ Enhanced)' if boil_status.get('uco_rsi', 0) > 50 else '(<50 ⚠️ Weaker)'}
  UVXY RSI: {boil_status.get('uvxy_rsi', 0):.1f}

Weather (7-day forecast):
  Current Temp: {boil_status.get('current_temp', 'N/A')}°F
  7-Day Change: {boil_status.get('temp_change_7d', 0):+.1f}°F {'(COLD COMING)' if boil_status.get('temp_change_7d', 0) >= 15 else ''}

"""
    
    # ==========================================================================
    # ALERTS SECTION
    # ==========================================================================
    if alerts:
        buy_alerts = [a for a in alerts if a[2] == 'buy']
        exit_alerts = [a for a in alerts if a[2] in ['exit', 'short']]
        warning_alerts = [a for a in alerts if a[2] in ['warning', 'hedge', 'watch']]
        info_alerts = [a for a in alerts if a[2] == 'info']
        natgas_alerts = [a for a in alerts if 'natgas' in a[2]]
        
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
        
        if info_alerts:
            body += "ℹ️ INFO:\n" + "-"*50 + "\n"
            for title, msg, _ in info_alerts:
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
    
    key_tickers = ['SPY', 'QQQ', 'SMH', 'IGV', 'GLD', 'USDU', 'XLP', 'TLT', 'HYG', 'XLF', 'UVXY', 'BTC-USD', 'AMD', 'NVDA']
    body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}\n"
    body += "-"*50 + "\n"
    
    for ticker in key_tickers:
        if ticker in indicators:
            ind = indicators[ticker]
            price = f"${ind['price']:.2f}" if ind['price'] < 1000 else f"${ind['price']:,.0f}"
            rsi = f"{ind['rsi10']:.1f}"
            pct = f"{ind.get('pct_above_sma200', 0):+.1f}%"
            body += f"{ticker:<10} {price:>12} {rsi:>10} {pct:>12}\n"
    
    # ==========================================================================
    # QUICK REFERENCE
    # ==========================================================================
    body += f"""
{'='*70}
SMH/IGV ROTATION QUICK REFERENCE (NEW)
{'='*70}
LONG TECL (IGV lagging):
  Spread > 30 + IGV < 35: 78% win, +13.2% avg (10d) | n=13
  Spread > 25 + IGV < 35: 75% win, +10.5% avg (10d) | n=19
  Spread > 25 + IGV < 40: 70% win, +6.9% avg (10d) | n=30

LONG SOXL (SMH lagging):
  Spread < -15 + SMH < 30: 88% win, +14.9% avg (10d) | n=34
  Spread < -25 + SMH < 35: 80% win, +12.1% avg (10d) | n=21

Key insight: RSI spread converges, but PRICE leadership can persist.
Trade the LAGGARD with leverage when spread extreme + laggard oversold.

{'='*70}
DEFENSIVE ROTATION QUICK REFERENCE
{'='*70}
XLP RSI 75-79: UVXY hedge (56% win, +1.69% 1-day)
XLP RSI > 82:  UVXY next-day trade (67% win, +4.81%)
XLU RSI > 79:  Short XLU via SDP (76% win, +2.34% 20d)
XLU RSI > 82:  Strong short (89% win, +2.98% 20d)
XLV RSI > 79:  DO NOT SHORT - use as signal only

Best trade when defensives OB + SPY/QQQ not OB:
  → 50% SOXL + 50% SDP (pairs trade)
"""
    
    if is_preclose:
        body += f"""
{'='*70}
NOTE: This is a PRE-CLOSE preview (12:00 PM ET).
Signals may change by market close.
Final confirmation email will be sent at 4:05 PM ET.
{'='*70}
"""
    
    body += f"""
{'='*70}
DATA SOURCES
{'='*70}
  Weather: Open-Meteo API (api.open-meteo.com)
  Prices: Yahoo Finance via yfinance
  
  SMH/IGV Rotation: Based on Feb 4, 2026 backtest
  - RSI spread convergence is reliable
  - Trade laggard with 3x leverage when extreme
  
  Defensive Rotation: Based on Jan 28, 2026 backtest
  - XLU shorts WORK (76% win at 20d)
  - XLV shorts DO NOT work (42% win)
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
    print(f"Mode: {'PRE-CLOSE (12:00 PM)' if IS_PRECLOSE else 'MARKET CLOSE (4:05 PM)'}")
    
    # Fetch weather data
    weather_data = get_weather_data()
    
    # Fetch market data
    tickers = [
        # Core Indices
        'SMH', 'IGV', 'SPY', 'QQQ', 'IWM',
        # Defensive Sectors
        'XLP', 'XLU', 'XLV',
        # Safe Havens & Macro
        'GLD', 'TLT', 'HYG', 'LQD', 'TMV',
        'USDU', 'UCO',
        # Natural Gas
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
        'TQQQ', 'SOXL', 'TECL', 'UPRO',
        # 2x Leveraged ETFs
        'FNGO',
        # Style/Factor ETFs
        'VOOV', 'VOOG', 'VTV', 'QQQE',
        # Energy & Financials
        'XLE', 'XLF',
    ]
    
    print("Downloading market data...")
    data = download_data(tickers)
    print(f"Downloaded data for {len(data)} tickers")
    
    # Check signals
    alerts, status = check_signals(data)
    indicators = status.get('indicators', {})
    
    # BOIL/KOLD signals
    boil_alerts, boil_status = check_boil_kold_signals(data, weather_data, indicators)
    
    # Combine alerts
    all_alerts = alerts + boil_alerts
    
    # Determine email subject
    if all_alerts:
        buy_count = len([a for a in all_alerts if 'buy' in a[2].lower()])
        exit_count = len([a for a in all_alerts if 'exit' in a[2].lower() or 'short' in a[2].lower()])
        
        if exit_count > 0:
            emoji = "🔴"
            urgency = "EXIT SIGNALS"
        elif buy_count > 0:
            emoji = "🟢"
            urgency = "BUY SIGNALS"
        else:
            emoji = "🟡"
            urgency = "WATCH"
        
        timing = "PRE-CLOSE" if IS_PRECLOSE else "CLOSE"
        subject = f"{emoji} [{timing}] Market Signals: {len(all_alerts)} Alert(s) - {urgency}"
    else:
        timing = "PRE-CLOSE" if IS_PRECLOSE else "CLOSE"
        subject = f"📊 [{timing}] Market Signals: No Alerts"
    
    # Format and send email
    body = format_email(all_alerts, status, boil_status, weather_data, IS_PRECLOSE)
    send_email(subject, body)
    
    print(f"\n{len(all_alerts)} signal(s) detected")
    for title, msg, _ in all_alerts:
        print(f"  {title}")

if __name__ == "__main__":
    main()
