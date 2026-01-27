#!/usr/bin/env python3
"""
Comprehensive Market Signal Monitor v4.0
========================================
Monitors all backtested trading signals and sends alerts.

SCHEDULE: Two emails daily (weekdays)
- 3:15 PM ET: Pre-close preview
- 4:05 PM ET: Market close confirmation

NEW IN v4.0:
- Refined BOIL/KOLD signals with 5-day gain bands (from Jan 2026 backtest)
- UCO > 50 enhancement filter for KOLD fade (77% vs 68% win rate)
- Supply shock signal: UVXY > 70 + UCO > 60 (73% win, +23.5% avg)
- Open-Meteo API for reliable weather forecasts (replaces NOAA scraping)
- Strong dollar winter signal (USDU > 75 → 73% BOIL win rate)
- Weather override only blocks fades when RSI < 70
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
                # Calculate daily mean temps
                temps_mean = [(mx + mn) / 2 for mx, mn in zip(temps_max, temps_min)]
                forecast['temps_7d'] = temps_mean
                forecast['dates'] = dates
                forecast['current_temp'] = temps_mean[0] if temps_mean else None
                
                # Calculate 7-day temperature change
                if len(temps_mean) >= 8:
                    forecast['temp_change_7d'] = temps_mean[0] - temps_mean[7]  # Positive = colder coming
                    
                    # Signal thresholds from backtest
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
    """
    Fetch weather forecast data.
    Returns combined dict with forecast info.
    """
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
# BOIL/KOLD SIGNAL LOGIC (Updated with Jan 2026 backtest findings)
# =============================================================================
def check_boil_kold_signals(data, weather_data, indicators):
    """
    Check BOIL/KOLD signals with refined logic from Jan 2026 backtest.
    
    KEY FINDINGS:
    - 5-day gain bands are MORE predictive than RSI for KOLD entry
    - KOLD 5d gain >= 30%: 88% win, +14.5% avg (n=24)
    - KOLD 5d gain >= 40%: 89% win, +18.5% avg (n=9)
    - KOLD 5d gain >= 50%: 100% win, +25.4% avg (n=7)
    - UCO > 50 enhancement: 77% win, +10.1% avg (vs 68% baseline)
    - UCO < 50 warning: only 57% win rate
    - Weather forecast improves BOIL entry, NOT KOLD entry
    - Supply shock (UVXY > 70 + UCO > 60): 73% win, +23.5% avg (n=11)
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
        print("BOIL data not available")
        return alerts, boil_status
    
    boil_df = data['BOIL']
    if len(boil_df) < 50:
        print("Insufficient BOIL data")
        return alerts, boil_status
    
    close = boil_df['Close']
    
    # Calculate BOIL indicators
    price = safe_float(close.iloc[-1])
    rsi10 = safe_float(calculate_rsi_wilder(close, 10).iloc[-1])
    
    # Calculate 5-day gain (KEY metric for KOLD entry)
    if len(close) >= 6:
        price_5d_ago = safe_float(close.iloc[-6])
        gain_5d = (price / price_5d_ago - 1) * 100
    else:
        gain_5d = 0
    
    # Calculate 7-day gain
    if len(close) >= 8:
        price_7d_ago = safe_float(close.iloc[-8])
        gain_7d = (price / price_7d_ago - 1) * 100
    else:
        gain_7d = 0
    
    # Get macro indicators
    uco_rsi = indicators.get('UCO', {}).get('rsi10', 50)
    uvxy_rsi = indicators.get('UVXY', {}).get('rsi10', 50)
    usdu_rsi = indicators.get('USDU', {}).get('rsi10', 50)
    
    # Get weather forecast
    forecast = weather_data.get('forecast', {})
    temp_change_7d = forecast.get('temp_change_7d', 0) or 0
    cold_coming = forecast.get('cold_coming', False)
    warming_coming = forecast.get('warming_coming', False)
    current_temp = forecast.get('current_temp', 'N/A')
    
    # Determine if winter (Nov-Feb)
    current_month = datetime.now().month
    is_winter = current_month in [11, 12, 1, 2]
    
    # Store status
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
        'is_winter': is_winter
    })
    
    reasoning = []
    
    # =========================================================================
    # KOLD ENTRY SIGNALS (Fade the spike)
    # =========================================================================
    
    kold_signal = False
    kold_tier = None
    kold_enhanced = False
    kold_warning = False
    
    # Check UCO filter
    if uco_rsi > 50:
        kold_enhanced = True
        reasoning.append(f"UCO RSI {uco_rsi:.1f} > 50 → Enhanced KOLD (77% win vs 68%)")
    elif uco_rsi < 50:
        kold_warning = True
        reasoning.append(f"⚠️ UCO RSI {uco_rsi:.1f} < 50 → Weaker fade (57% win rate)")
    
    # TIER 1: Extreme spike (highest conviction)
    if gain_5d >= 50:
        kold_signal = True
        kold_tier = 1
        reasoning.append(f"🔥 TIER 1: 5d gain {gain_5d:+.1f}% >= 50% → 100% win, +25.4% avg (n=7)")
    
    elif gain_5d >= 40:
        kold_signal = True
        kold_tier = 1
        if rsi10 > 70:
            reasoning.append(f"🔥 TIER 1: 5d gain {gain_5d:+.1f}% >= 40% + RSI {rsi10:.1f} > 70 → 100% win, +24% avg (n=6)")
        else:
            reasoning.append(f"🔥 TIER 1: 5d gain {gain_5d:+.1f}% >= 40% → 89% win, +18.5% avg (n=9)")
    
    # TIER 2: Strong spike
    elif gain_5d >= 30 and rsi10 > 70:
        kold_signal = True
        kold_tier = 2
        reasoning.append(f"🟢 TIER 2: 5d gain {gain_5d:+.1f}% >= 30% + RSI {rsi10:.1f} > 70 → 92% win, +16% avg (n=12)")
    
    elif gain_5d >= 30:
        kold_signal = True
        kold_tier = 2
        reasoning.append(f"🟢 TIER 2: 5d gain {gain_5d:+.1f}% >= 30% → 88% win, +14.5% avg (n=24)")
    
    # TIER 3: Moderate spike (watch)
    elif gain_5d >= 20:
        kold_tier = 3
        reasoning.append(f"🟡 TIER 3: 5d gain {gain_5d:+.1f}% >= 20% → 66% win, +7.6% avg (n=76)")
    
    boil_status['tier'] = kold_tier
    
    # =========================================================================
    # BOIL ENTRY SIGNALS (Buy the dip / weather play)
    # =========================================================================
    
    boil_signal = False
    boil_type = None
    
    # SUPPLY SHOCK SIGNAL (rare but powerful)
    if uvxy_rsi > 70 and uco_rsi > 60:
        boil_signal = True
        boil_type = 'supply_shock'
        reasoning.append(f"🔥 SUPPLY SHOCK: UVXY {uvxy_rsi:.1f} > 70 + UCO {uco_rsi:.1f} > 60 → 73% win, +23.5% avg (n=11)")
    
    # WEATHER-BASED ENTRY (winter only)
    elif is_winter and cold_coming and rsi10 < 50:
        boil_signal = True
        boil_type = 'weather'
        intensity = forecast.get('cold_intensity', 'UNKNOWN')
        if temp_change_7d >= 20:
            reasoning.append(f"🟢 WEATHER BUY: {temp_change_7d:+.1f}°F drop coming + RSI {rsi10:.1f} < 50 → 62% win, +7.1% avg")
        else:
            reasoning.append(f"🟡 WEATHER WATCH: {temp_change_7d:+.1f}°F drop coming ({intensity})")
    
    # OVERSOLD BOUNCE
    elif rsi10 < 21:
        boil_signal = True
        boil_type = 'oversold'
        reasoning.append(f"🟢 OVERSOLD: RSI {rsi10:.1f} < 21 → Mean reversion likely")
    
    # STRONG DOLLAR WINTER SIGNAL
    elif is_winter and usdu_rsi > 75 and rsi10 < 50:
        reasoning.append(f"🟡 WINTER DOLLAR: USDU RSI {usdu_rsi:.1f} > 75 in winter → 73% BOIL win, +7.3% avg (n=22)")
    
    # =========================================================================
    # WEATHER OVERRIDE LOGIC
    # =========================================================================
    # Only block KOLD fade if RSI < 70 and severe cold coming
    
    weather_override = False
    if kold_signal and kold_tier in [2, 3] and cold_coming and rsi10 < 70:
        if temp_change_7d >= 15:
            weather_override = True
            reasoning.append(f"⚠️ WEATHER OVERRIDE: Fade blocked - {temp_change_7d:+.1f}°F cold coming, RSI only {rsi10:.1f}")
    
    boil_status['weather_override'] = weather_override
    boil_status['reasoning'] = reasoning
    
    # =========================================================================
    # DETERMINE FINAL SIGNAL
    # =========================================================================
    
    if kold_signal and not weather_override:
        if kold_tier == 1:
            signal = "🔴 KOLD TIER 1"
            action = "Enter KOLD NOW - highest conviction fade"
            signal_type = 'natgas_kold_t1'
        else:
            signal = "🔴 KOLD TIER 2"
            action = "Strong KOLD entry"
            signal_type = 'natgas_kold_t2'
        
        if kold_enhanced:
            signal += " (ENHANCED)"
            action += f" | UCO confirms (77% win)"
        elif kold_warning:
            signal += " (CAUTION)"
            action += f" | UCO weak - only 57% win"
    
    elif weather_override:
        signal = "🟡 HOLD - WEATHER OVERRIDE"
        action = f"Spike says fade, but {temp_change_7d:+.1f}°F cold coming. WAIT."
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
        
        # FAS responds to GLD/USDU signal
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
MARKET SIGNAL MONITOR v4.0 - {timing}
{now.strftime('%Y-%m-%d %H:%M')} ET
{'='*70}

"""
    
    # ==========================================================================
    # BOIL/KOLD STATUS (Top of email for visibility)
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
  USDU RSI: {boil_status.get('usdu_rsi', 0):.1f}

Weather (7-day forecast):
  Current Temp: {boil_status.get('current_temp', 'N/A')}°F
  7-Day Change: {boil_status.get('temp_change_7d', 0):+.1f}°F {'(COLD COMING)' if boil_status.get('temp_change_7d', 0) >= 15 else '(warming)' if boil_status.get('temp_change_7d', 0) <= -10 else ''}

Signal Reasoning:
"""
    for reason in boil_status.get('reasoning', []):
        body += f"  • {reason}\n"
    
    body += f"""
KOLD Entry Thresholds (5-day gain):
  30% → 88% win, +14.5% avg (n=24) {'← ACTIVE' if boil_status.get('gain_5d', 0) >= 30 else ''}
  40% → 89% win, +18.5% avg (n=9) {'← ACTIVE' if boil_status.get('gain_5d', 0) >= 40 else ''}
  50% → 100% win, +25.4% avg (n=7) {'← ACTIVE' if boil_status.get('gain_5d', 0) >= 50 else ''}

"""
    
    # ==========================================================================
    # ALERTS
    # ==========================================================================
    if alerts:
        buy_alerts = [a for a in alerts if 'buy' in a[2].lower()]
        exit_alerts = [a for a in alerts if 'exit' in a[2].lower() or 'short' in a[2].lower() or 'kold' in a[2].lower()]
        warning_alerts = [a for a in alerts if a[2] in ['warning', 'hedge', 'watch', 'natgas_warning']]
        
        if buy_alerts:
            body += f"{'='*70}\n"
            body += "🟢 BUY SIGNALS:\n" + "-"*50 + "\n"
            for title, msg, _ in buy_alerts:
                body += f"{title}\n{msg}\n\n"
        
        if exit_alerts:
            body += f"{'='*70}\n"
            body += "🔴 EXIT/SHORT/KOLD SIGNALS:\n" + "-"*50 + "\n"
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
  Weather: Open-Meteo API (api.open-meteo.com)
  Prices: Yahoo Finance via yfinance
  
  BOIL/KOLD Strategy: Based on Jan 2026 backtest
  - 5-day gain bands primary signal for KOLD fade
  - UCO > 50 = enhanced fade (77% win)
  - Weather forecast for BOIL entry timing only
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
    weather_data = get_weather_data()
    
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
        # Natural Gas
        'BOIL', 'KOLD',
        # Volatility
        'UVXY',
        # International
        'EDC', 'YINN',
        # Crypto
        'BTC-USD',
        # Individual Stocks
        'AMD', 'NVDA',
        # 3x Leveraged ETFs
        'NAIL', 'CURE', 'FAS', 'LABU',
        'TQQQ', 'SOXL', 'UPRO',
        # Energy
        'XLE', 'XLF',
    ]
    
    print("Downloading market data...")
    data = download_data(tickers)
    print(f"Downloaded data for {len(data)} tickers")
    
    # =========================================================================
    # CHECK SIGNALS
    # =========================================================================
    
    # Standard signals (also builds indicators dict)
    alerts, status = check_signals(data)
    indicators = status.get('indicators', {})
    
    # BOIL/KOLD signals with weather integration and macro filters
    boil_alerts, boil_status = check_boil_kold_signals(data, weather_data, indicators)
    
    # Combine alerts
    all_alerts = alerts + boil_alerts
    
    # =========================================================================
    # DETERMINE EMAIL SUBJECT
    # =========================================================================
    if all_alerts:
        buy_count = len([a for a in all_alerts if 'buy' in a[2].lower()])
        exit_count = len([a for a in all_alerts if 'exit' in a[2].lower() or 'short' in a[2].lower() or 'kold' in a[2].lower()])
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
