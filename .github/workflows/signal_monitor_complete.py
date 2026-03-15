#!/usr/bin/env python3
"""
Comprehensive Market Signal Monitor v2.1
========================================
Monitors all backtested trading signals and sends alerts.

SCHEDULE: Two emails daily (weekdays)
- 3:15 PM ET: Pre-close preview
- 4:05 PM ET: Market close confirmation
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
import json

# =============================================================================
# CONFIGURATION
# =============================================================================
SENDER_EMAIL = os.environ.get('SENDER_EMAIL', '')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD', '')
RECIPIENT_EMAIL = os.environ.get('RECIPIENT_EMAIL', '')
PHONE_EMAIL = os.environ.get('PHONE_EMAIL', '')

IS_PRECLOSE = len(sys.argv) > 1 and sys.argv[1] == 'preclose'

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
# SIGNAL CHECKS
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
            
            # New indicators for Crisis Alpha / Deep Value
            daily_ret = close.pct_change()
            
            # Moving Average of Return (10d) — avg daily return in %
            if len(daily_ret) >= 10:
                indicators[ticker]['maret_10'] = safe_float(daily_ret.rolling(10).mean().iloc[-1]) * 100
            else:
                indicators[ticker]['maret_10'] = 0
            
            # Cumulative Return over various windows (in %)
            for win in [10, 30, 50, 100]:
                if len(close) > win:
                    indicators[ticker][f'cumret_{win}'] = (price / safe_float(close.iloc[-win-1]) - 1) * 100
                else:
                    indicators[ticker][f'cumret_{win}'] = 0
            
            # Standard Deviation of Return (10d and 50d) — daily decimal
            if len(daily_ret) >= 50:
                indicators[ticker]['stdret_10'] = safe_float(daily_ret.rolling(10).std().iloc[-1])
                indicators[ticker]['stdret_50'] = safe_float(daily_ret.rolling(50).std().iloc[-1])
            else:
                indicators[ticker]['stdret_10'] = 0
                indicators[ticker]['stdret_50'] = 0
            
            # EMA20 for recovery detection
            indicators[ticker]['ema20'] = safe_float(close.ewm(span=20, adjust=False).mean().iloc[-1])
                
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
            alerts.append(('ðŸ”´ SOXL EXIT', f"SMH {smh['pct_above_sma200']:.1f}% above SMA(200) - SELL SOXL", 'exit'))
        elif smh['pct_above_sma200'] >= 35:
            alerts.append(('ðŸŸ¡ SOXL WARNING', f"SMH {smh['pct_above_sma200']:.1f}% above SMA(200) - Approaching sell zone", 'warning'))
        elif smh['pct_above_sma200'] >= 30:
            alerts.append(('ðŸŸ¡ SOXL TRIM', f"SMH {smh['pct_above_sma200']:.1f}% above SMA(200) - Consider trimming 25-50%", 'warning'))
        
        # Death Cross
        if smh['sma50'] < smh['sma200'] and smh['sma200'] > 0:
            alerts.append(('ðŸ”´ DEATH CROSS', f"SMH SMA(50) below SMA(200) - Bearish trend", 'exit'))
        
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
                    alerts.append(('ðŸŸ¢ SOXL STRONG BUY', f"SMH {days_below} days below SMA(200) + RSI(50)={smh['rsi50']:.1f} < 45 | 97% win, +81% avg", 'buy'))
                else:
                    alerts.append(('ðŸŸ¢ SOXL ACCUMULATE', f"SMH {days_below} days below SMA(200) | 85% win, +54% avg", 'buy'))
            
            status['smh_days_below_sma200'] = days_below
    
    # =========================================================================
    # SIGNAL GROUP 2: GLD/USDU Combo Signals
    # =========================================================================
    if 'GLD' in indicators and 'USDU' in indicators:
        gld = indicators['GLD']
        usdu = indicators['USDU']
        
        # Double Signal: GLD > 79 AND USDU < 25
        if gld['rsi10'] > 79 and usdu['rsi10'] < 25:
            alerts.append(('ðŸŸ¢ðŸ”¥ DOUBLE SIGNAL ACTIVE', 
                f"GLD RSI={gld['rsi10']:.1f} > 79 AND USDU RSI={usdu['rsi10']:.1f} < 25\n"
                f"   â†’ Long TQQQ: 88% win, +7% avg (5d)\n"
                f"   â†’ Long UPRO: 85% win, +5.2% avg (5d)\n"
                f"   â†’ AMD/NVDA: 86% win, +5-8% avg (5d)", 'buy'))
            
            # Triple Signal: Add XLP > 65
            if 'XLP' in indicators and indicators['XLP']['rsi10'] > 65:
                xlp = indicators['XLP']
                alerts.append(('ðŸŸ¢ðŸ”¥ðŸ”¥ TRIPLE SIGNAL ACTIVE', 
                    f"GLD RSI={gld['rsi10']:.1f} + USDU RSI={usdu['rsi10']:.1f} + XLP RSI={xlp['rsi10']:.1f}\n"
                    f"   â†’ Long TQQQ: 100% win, +11.6% avg (5d) - RARE!", 'buy'))
        
        # Individual GLD overbought
        elif gld['rsi10'] > 79:
            alerts.append(('ðŸŸ¢ GLD OVERBOUGHT', 
                f"GLD RSI={gld['rsi10']:.1f} > 79 â†’ Long TQQQ: 72% win, +3.2% avg (5d)", 'buy'))
    
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
            alerts.append(('ðŸŸ¢ DEFENSIVE ROTATION', 
                f"Defensive sector overbought, SPY/QQQ not â†’ Long TQQQ 20d: 70% win, +5% avg", 'buy'))
    
    # =========================================================================
    # SIGNAL GROUP 4: Volatility Hedge Signals
    # =========================================================================
    if 'QQQ' in indicators:
        qqq = indicators['QQQ']
        
        if qqq['rsi10'] > 79:
            alerts.append(('ðŸŸ¡ VOL HEDGE', 
                f"QQQ RSI={qqq['rsi10']:.1f} > 79 â†’ Long UVXY 5d: 67% win, +33% CAGR", 'hedge'))
        
        if qqq['rsi10'] < 20:
            alerts.append(('ðŸŸ¢ QQQ DIP BUY', 
                f"QQQ RSI={qqq['rsi10']:.1f} < 20 â†’ Long TQQQ 5d: 69% win, +26% CAGR", 'buy'))
    
    # =========================================================================
    # SIGNAL GROUP 5: SOXS Short Signals
    # =========================================================================
    if 'SMH' in indicators and 'USDU' in indicators:
        smh = indicators['SMH']
        usdu = indicators['USDU']
        
        if smh['rsi10'] > 79 and usdu['rsi10'] > 70:
            alerts.append(('ðŸ”´ SOXS SIGNAL', 
                f"SMH RSI={smh['rsi10']:.1f} > 79 AND USDU RSI={usdu['rsi10']:.1f} > 70\n"
                f"   â†’ Long SOXS 5d: 100% win, +9.5% avg", 'short'))
        
        if 'IWM' in indicators and smh['rsi10'] > 79 and indicators['IWM']['rsi10'] < 50:
            alerts.append(('ðŸ”´ SOXS DIVERGENCE', 
                f"SMH RSI={smh['rsi10']:.1f} > 79 AND IWM RSI={indicators['IWM']['rsi10']:.1f} < 50\n"
                f"   â†’ Long SOXS 5d: 86% win, +6.9% avg", 'short'))
    
    # =========================================================================
    # SIGNAL GROUP 6: BTC Signals
    # =========================================================================
    if 'BTC-USD' in indicators:
        btc = indicators['BTC-USD']
        
        if btc['rsi10'] > 79:
            alerts.append(('ðŸŸ¢ BTC MOMENTUM', 
                f"BTC RSI={btc['rsi10']:.1f} > 79 â†’ Hold/Add BTC: 67% win, +5.2% avg (5d)", 'buy'))
        
        if btc['rsi10'] < 30:
            uvxy_low = 'UVXY' in indicators and indicators['UVXY']['rsi10'] < 40
            if uvxy_low:
                alerts.append(('ðŸŸ¢ BTC DIP BUY', 
                    f"BTC RSI={btc['rsi10']:.1f} < 30 AND UVXY < 40 â†’ Buy BTC: 77% win, +4.1% avg (5d)", 'buy'))
            else:
                alerts.append(('ðŸŸ¡ BTC OVERSOLD', 
                    f"BTC RSI={btc['rsi10']:.1f} < 30 (wait for UVXY < 40 for better signal)", 'watch'))
    
    # =========================================================================
    # SIGNAL GROUP 7: UPRO Entry/Exit Signals
    # =========================================================================
    if 'SPY' in indicators:
        spy = indicators['SPY']
        
        if spy['rsi10'] > 85:
            alerts.append(('ðŸ”´ UPRO EXIT', 
                f"SPY RSI={spy['rsi10']:.1f} > 85 â†’ Trim/Exit UPRO: Only 36% win, -3.5% avg (5d)", 'exit'))
        elif spy['rsi10'] > 82:
            alerts.append(('ðŸŸ¡ UPRO CAUTION', 
                f"SPY RSI={spy['rsi10']:.1f} > 82 â†’ Watch UPRO: 49% win at 5d", 'warning'))
        
        if spy['rsi10'] < 21:
            alerts.append(('ðŸŸ¢ UPRO STRONG BUY', 
                f"SPY RSI={spy['rsi10']:.1f} < 21 â†’ Add UPRO: 94% win, +8.9% avg (5d)", 'buy'))
        elif spy['rsi10'] < 25:
            alerts.append(('ðŸŸ¢ UPRO BUY', 
                f"SPY RSI={spy['rsi10']:.1f} < 25 â†’ Add UPRO: 74% win, +3.9% avg (5d)", 'buy'))
        elif spy['rsi10'] < 30:
            alerts.append(('ðŸŸ¢ UPRO CONSIDER', 
                f"SPY RSI={spy['rsi10']:.1f} < 30 â†’ Consider UPRO: 69% win, +4.3% avg (5d)", 'buy'))
    
    # =========================================================================
    # SIGNAL GROUP 8: AMD/NVDA Specific
    # =========================================================================
    if 'AMD' in indicators:
        amd = indicators['AMD']
        if amd['rsi10'] > 85:
            alerts.append(('ðŸŸ¡ AMD EXTENDED', 
                f"AMD RSI={amd['rsi10']:.1f} > 85 â†’ Consider taking profits", 'warning'))
    
    if 'NVDA' in indicators:
        nvda = indicators['NVDA']
        if nvda['rsi10'] > 85:
            alerts.append(('ðŸŸ¡ NVDA EXTENDED', 
                f"NVDA RSI={nvda['rsi10']:.1f} > 85 â†’ Consider taking profits", 'warning'))
    
    # =========================================================================
    # SIGNAL GROUP 9: NAIL (3x Homebuilders) Signals
    # =========================================================================
    if 'NAIL' in indicators:
        nail = indicators['NAIL']
        
        # GLD/USDU combo for NAIL (with XLF filter)
        if 'GLD' in indicators and 'USDU' in indicators and 'XLF' in indicators:
            gld = indicators['GLD']
            usdu = indicators['USDU']
            xlf = indicators['XLF']
            
            if gld['rsi10'] > 79 and usdu['rsi10'] < 25 and xlf['rsi10'] < 70:
                alerts.append(('ðŸŸ¢ NAIL SIGNAL', 
                    f"GLD>{gld['rsi10']:.0f} + USDU<{usdu['rsi10']:.0f} + XLF<{xlf['rsi10']:.0f}\n"
                    f"   â†’ Long NAIL: 90% win, +4.9% avg (5d), +14.4% avg (10d) | n=10", 'buy'))
            
            # Warning: XLF strong + USDU weak = danger for NAIL
            if xlf['rsi10'] > 70 and usdu['rsi10'] < 25:
                alerts.append(('ðŸ”´ NAIL DANGER', 
                    f"XLF RSI={xlf['rsi10']:.1f} > 70 + USDU < 25 = Historically BAD for NAIL\n"
                    f"   â†’ 11% win, -11.5% avg (5d) | Consider exit", 'exit'))
        
        # NAIL overbought/oversold
        if nail['rsi10'] > 79:
            alerts.append(('ðŸ”´ NAIL OVERBOUGHT', 
                f"NAIL RSI={nail['rsi10']:.1f} > 79 â†’ Consider exit", 'warning'))
    
    # =========================================================================
    # SIGNAL GROUP 10: CURE (3x Healthcare) Signals
    # =========================================================================
    if 'CURE' in indicators:
        cure = indicators['CURE']
        
        if cure['rsi10'] < 21:
            alerts.append(('ðŸŸ¢ CURE STRONG BUY', 
                f"CURE RSI={cure['rsi10']:.1f} < 21 â†’ Buy CURE: 85% win, +7.3% avg (5d) | n=33", 'buy'))
        elif cure['rsi10'] < 25:
            alerts.append(('ðŸŸ¢ CURE BUY', 
                f"CURE RSI={cure['rsi10']:.1f} < 25 â†’ Buy CURE: 81% win, +5.4% avg (5d) | n=70", 'buy'))
        
        if cure['rsi10'] > 79:
            alerts.append(('ðŸ”´ CURE OVERBOUGHT', 
                f"CURE RSI={cure['rsi10']:.1f} > 79 â†’ Exit CURE: Only 40% win (5d) | n=95", 'exit'))
        elif cure['rsi10'] > 85:
            alerts.append(('ðŸ”´ CURE SELL', 
                f"CURE RSI={cure['rsi10']:.1f} > 85 â†’ Sell CURE: Only 33% win (5d) | n=15", 'exit'))
    
    # =========================================================================
    # SIGNAL GROUP 11: FAS (3x Financials) Signals
    # =========================================================================
    if 'FAS' in indicators:
        fas = indicators['FAS']
        
        # FAS responds to GLD/USDU signal
        if 'GLD' in indicators and 'USDU' in indicators:
            gld = indicators['GLD']
            usdu = indicators['USDU']
            
            if gld['rsi10'] > 79 and usdu['rsi10'] < 25:
                alerts.append(('ðŸŸ¢ FAS SIGNAL', 
                    f"GLD>{gld['rsi10']:.0f} + USDU<{usdu['rsi10']:.0f}\n"
                    f"   â†’ Long FAS 10d: 92% win, +5.8% avg | n=13", 'buy'))
        
        if fas['rsi10'] < 30:
            alerts.append(('ðŸŸ¢ FAS BUY', 
                f"FAS RSI={fas['rsi10']:.1f} < 30 â†’ Buy FAS: 63% win, +3.3% avg (5d) | n=195", 'buy'))
        
        if fas['rsi10'] > 82:
            alerts.append(('ðŸ”´ FAS OVERBOUGHT', 
                f"FAS RSI={fas['rsi10']:.1f} > 82 â†’ Exit FAS: Only 38% win (5d) | n=40", 'exit'))
        elif fas['rsi10'] > 85:
            alerts.append(('ðŸ”´ FAS SELL', 
                f"FAS RSI={fas['rsi10']:.1f} > 85 â†’ Sell FAS: Only 8% win! (5d) | n=12", 'exit'))
    
    # =========================================================================
    # SIGNAL GROUP 12: LABU (3x Biotech) Signals
    # =========================================================================
    if 'LABU' in indicators:
        labu = indicators['LABU']
        
        if labu['rsi10'] < 21:
            alerts.append(('ðŸŸ¢ LABU STRONG BUY', 
                f"LABU RSI={labu['rsi10']:.1f} < 21 â†’ Buy LABU: 73% win, +11.2% avg (5d) | n=11", 'buy'))
        elif labu['rsi10'] < 25:
            alerts.append(('ðŸŸ¢ LABU BUY', 
                f"LABU RSI={labu['rsi10']:.1f} < 25 â†’ Buy LABU: 66% win, +5.7% avg (5d) | n=59", 'buy'))
        
        if labu['rsi10'] > 70:
            alerts.append(('ðŸŸ¡ LABU EXTENDED', 
                f"LABU RSI={labu['rsi10']:.1f} > 70 â†’ Caution: 42% win (5d) | n=180", 'warning'))
        
        # LABU extreme extension warning
        if labu.get('pct_above_sma200', 0) > 80:
            alerts.append(('ðŸŸ¡ LABU EXTREME', 
                f"LABU {labu['pct_above_sma200']:.0f}% above SMA(200) â†’ Very extended, consider profits", 'warning'))
    
    # =========================================================================
    # SIGNAL GROUP 13: DEEP VALUE (Cumulative Return / MaRet Cascade)
    # =========================================================================
    # Rule 1: QQQ MaRet(10) < -1%/day → TQQQ (crash bounce)
    if 'QQQ' in indicators and indicators['QQQ'].get('maret_10', 0) < -1.0:
        qqq = indicators['QQQ']
        alerts.append(('🟢🔥 CRASH BOUNCE',
            f"QQQ MaRet(10)={qqq['maret_10']:+.2f}%/day < -1.0%\n"
            f"   → Long TQQQ: 75% win, +6.3% avg (1d) | n=28\n"
            f"   QQQ losing >{abs(qqq['maret_10']):.1f}%/day avg for 10 sessions", 'buy'))

    # Rule 2: SMH MaRet(10) < -1.5%/day → SOXL
    if 'SMH' in indicators and indicators['SMH'].get('maret_10', 0) < -1.5:
        smh_i = indicators['SMH']
        alerts.append(('🟢🔥 SMH CRASH BOUNCE',
            f"SMH MaRet(10)={smh_i['maret_10']:+.2f}%/day < -1.5%\n"
            f"   → Long SOXL: extreme semi selloff", 'buy'))

    # Rule 3: QQQ CumRet(30) < -20% → TQQQ
    if 'QQQ' in indicators:
        qqq = indicators['QQQ']
        cr30 = qqq.get('cumret_30', 0)
        if cr30 < -20:
            alerts.append(('🟢 QQQ DEEP DRAWDOWN',
                f"QQQ CumRet(30d)={cr30:+.1f}% < -20%\n"
                f"   → Long TQQQ: deep drawdown buy", 'buy'))
        elif cr30 < -15:
            alerts.append(('🟡 QQQ DRAWDOWN WATCH',
                f"QQQ CumRet(30d)={cr30:+.1f}% — approaching -20% trigger", 'watch'))

    # Rule 4: SMH CumRet(30) < -25% → SOXL
    if 'SMH' in indicators:
        smh_i = indicators['SMH']
        cr30 = smh_i.get('cumret_30', 0)
        if cr30 < -25:
            alerts.append(('🟢 SMH DEEP DRAWDOWN',
                f"SMH CumRet(30d)={cr30:+.1f}% < -25%\n"
                f"   → Long SOXL: deep semi drawdown buy", 'buy'))

    # Rule 5: SPY CumRet(50) < -15% + RSI < 35 → UPRO
    if 'SPY' in indicators:
        spy = indicators['SPY']
        cr50 = spy.get('cumret_50', 0)
        if cr50 < -15 and spy['rsi10'] < 35:
            alerts.append(('🟢🔥 SPY DEEP VALUE',
                f"SPY CumRet(50d)={cr50:+.1f}% + RSI={spy['rsi10']:.1f}\n"
                f"   → Long UPRO: 88% win at 10d, +12% avg | n=56", 'buy'))
        elif cr50 < -10:
            alerts.append(('🟡 SPY DRAWDOWN WATCH',
                f"SPY CumRet(50d)={cr50:+.1f}% — approaching -15% trigger", 'watch'))

    # Rule 6: SPY CumRet(100) < -10% → UPRO
    if 'SPY' in indicators:
        spy = indicators['SPY']
        cr100 = spy.get('cumret_100', 0)
        if cr100 < -10 and spy.get('cumret_50', 0) >= -15:
            alerts.append(('🟢 SPY MODERATE DRAWDOWN',
                f"SPY CumRet(100d)={cr100:+.1f}% < -10%\n"
                f"   → Long UPRO: 85% win at 10d | n=92", 'buy'))

    # DANGER: SPY falling + strong dollar
    if 'SPY' in indicators and 'USDU' in indicators:
        spy = indicators['SPY']
        usdu = indicators['USDU']
        cr10 = spy.get('cumret_10', 0)
        if cr10 < -5 and usdu['rsi10'] > 70:
            alerts.append(('🔴 FALLING KNIFE WARNING',
                f"SPY CumRet(10d)={cr10:+.1f}% + USDU RSI={usdu['rsi10']:.1f}\n"
                f"   → DO NOT buy dip: 20% WR when SPY<-5% + USDU>70\n"
                f"   Strong dollar + falling equities = more pain ahead", 'exit'))

    # =========================================================================
    # SIGNAL GROUP 14: CRISIS ALPHA v2 (Vol Compression Regime)
    # =========================================================================
    if 'QQQ' in indicators and 'SPY' in indicators:
        qqq = indicators['QQQ']
        spy = indicators['SPY']

        vol_10 = qqq.get('stdret_10', 0)
        vol_50 = qqq.get('stdret_50', 0)
        vol_compressing = vol_10 < vol_50 and vol_50 > 0
        vol_ratio = vol_10 / vol_50 if vol_50 > 0 else 1.0
        spy_above_sma200 = spy['price'] > spy['sma200'] and spy['sma200'] > 0
        spy_above_ema20 = spy['price'] > spy.get('ema20', 0) and spy.get('ema20', 0) > 0

        if spy_above_sma200 and vol_compressing:
            alerts.append(('🟢 VOL COMPRESSION (BULL)',
                f"QQQ StdDev 10/50 ratio={vol_ratio:.2f} + SPY above SMA200\n"
                f"   → Crisis Alpha: TQQQ/GLD regime | 69% WR 20d (n=1407)", 'buy'))
        elif spy_above_sma200 and not vol_compressing:
            alerts.append(('🟡 VOL EXPANDING (BULL)',
                f"QQQ StdDev 10/50 ratio={vol_ratio:.2f} + SPY above SMA200\n"
                f"   → Crisis Alpha: UPRO/GLD regime (less aggressive)", 'watch'))
        elif not spy_above_sma200 and vol_compressing and spy_above_ema20:
            alerts.append(('🟢 BEAR RECOVERY',
                f"SPY below SMA200 but above EMA20 + vol compressing\n"
                f"   → Crisis Alpha: recovery regime (UPRO/GLD/SHY)", 'buy'))
        elif not spy_above_sma200:
            alerts.append(('🟡 BEAR DEFENSIVE',
                f"SPY below SMA200 + vol ratio={vol_ratio:.2f}\n"
                f"   → Crisis Alpha: defensive (SHY/GLD)", 'warning'))

        status['crisis_alpha'] = {
            'vol_ratio': vol_ratio,
            'vol_compressing': vol_compressing,
            'spy_above_sma200': spy_above_sma200,
            'spy_above_ema20': spy_above_ema20,
        }

    # =========================================================================
    # SIGNAL GROUP 15: SIGNAL DEGRADATION / CALIBRATION WARNINGS
    # =========================================================================
    # Track trailing win rates for key signals and warn when degrading
    degradation_checks = [
        ('UCO RSI>75 → TMV', 'UCO', lambda i: i.get('rsi10',0) > 75, 0.65, 'TMV'),
        ('GLD>79 + USDU<25 → TQQQ', None, lambda i: i.get('GLD',{}).get('rsi10',0) > 79 and i.get('USDU',{}).get('rsi10',50) < 25, 0.88, 'TQQQ'),
        ('SPY>79 → UVXY', 'SPY', lambda i: i.get('rsi10',0) > 79, 0.686, 'UVXY'),
        ('GLD RSI>79 alone', 'GLD', lambda i: i.get('rsi10',0) > 79, 0.72, 'TQQQ'),
    ]
    
    for sig_name, ticker, cond_fn, hist_wr, target in degradation_checks:
        if ticker and ticker in data and target in data:
            try:
                close_sig = data[ticker]['Close']
                close_tgt = data[target]['Close']
                rsi_series = calculate_rsi_wilder(close_sig, 10)
                fwd_5d = close_tgt.shift(-5) / close_tgt - 1
                
                # Build indicator dict for condition check
                if ticker:
                    recent_rsi = rsi_series.iloc[-200:]
                    episodes = []
                    for dt in recent_rsi.index:
                        rsi_val = safe_float(recent_rsi.loc[dt])
                        test_ind = {'rsi10': rsi_val}
                        try:
                            if cond_fn(test_ind) and dt in fwd_5d.index:
                                fr = safe_float(fwd_5d.loc[dt])
                                if not pd.isna(fr):
                                    episodes.append(1 if fr > 0 else 0)
                        except:
                            continue
                    
                    if len(episodes) >= 5:
                        trail_wr = sum(episodes[-16:]) / len(episodes[-16:]) if len(episodes) >= 16 else sum(episodes) / len(episodes)
                        n_trail = min(len(episodes), 16)
                        
                        if trail_wr < 0.40 and hist_wr > 0.60:
                            alerts.append(('🔴 SIGNAL DEGRADATION',
                                f"{sig_name}: Trailing WR {trail_wr:.0%} vs historical {hist_wr:.0%} — signal may be BROKEN (n={n_trail})", 'exit'))
                        elif trail_wr < hist_wr - 0.15 and len(episodes) >= 8:
                            alerts.append(('🟡 SIGNAL CALIBRATION',
                                f"{sig_name}: Trailing WR {trail_wr:.0%} vs historical {hist_wr:.0%} — signal DEGRADING (n={n_trail})", 'warning'))
            except Exception as e:
                pass  # Skip if data issue
    
    # =========================================================================
    # SIGNAL GROUP 16: BOIL/KOLD NATURAL GAS MONITORING
    # =========================================================================
    if 'BOIL' in indicators:
        boil = indicators['BOIL']
        boil_rsi = boil['rsi10']
        
        # BOIL RSI extremes
        if boil_rsi > 79:
            alerts.append(('🟡 BOIL OVERBOUGHT',
                f"BOIL RSI={boil_rsi:.1f} > 79 → Consider KOLD fade\n"
                f"   Winter spikes tend to fade. 44% WR 5d in high-HDD months", 'warning'))
        elif boil_rsi < 21:
            alerts.append(('🟢 BOIL OVERSOLD',
                f"BOIL RSI={boil_rsi:.1f} < 21 → Watch for weather-driven bounce", 'buy'))
        
        # 5-day gain tracking for KOLD entry
        if 'BOIL' in data and len(data['BOIL']) > 5:
            boil_close = data['BOIL']['Close']
            boil_5d_gain = safe_float((boil_close.iloc[-1] / boil_close.iloc[-6] - 1) * 100)
            
            if boil_5d_gain > 30:
                alerts.append(('🔴 BOIL SPIKE → KOLD',
                    f"BOIL 5d gain: {boil_5d_gain:+.1f}% > 30% → KOLD entry zone\n"
                    f"   Historical: 88% of spikes >30% fade within 10d", 'short'))
            elif boil_5d_gain > 20:
                alerts.append(('🟡 BOIL SPIKE WATCH',
                    f"BOIL 5d gain: {boil_5d_gain:+.1f}% — approaching 30% KOLD trigger", 'watch'))
            
            status['boil_5d_gain'] = boil_5d_gain
    
    # Temperature forecast (Open-Meteo NYC proxy for heating demand)
    try:
        import urllib.request
        url = "https://api.open-meteo.com/v1/forecast?latitude=40.71&longitude=-74.01&daily=temperature_2m_max,temperature_2m_min&temperature_unit=fahrenheit&timezone=America/New_York&forecast_days=7"
        with urllib.request.urlopen(url, timeout=5) as resp:
            weather = json.loads(resp.read())
        
        if 'daily' in weather:
            temps = weather['daily']
            avg_temps = [(h + l) / 2 for h, l in zip(temps['temperature_2m_max'], temps['temperature_2m_min'])]
            avg_7d = sum(avg_temps) / len(avg_temps)
            cold_days = sum(1 for t in avg_temps if t < 32)
            
            status['weather'] = {
                'avg_7d_temp': round(avg_7d, 1),
                'cold_days': cold_days,
                'dates': temps.get('time', []),
                'temps': [round(t, 1) for t in avg_temps],
            }
            
            if cold_days >= 5:
                alerts.append(('🟡 COLD SNAP AHEAD',
                    f"NYC 7d forecast: {cold_days}/7 days below freezing, avg {avg_7d:.0f}°F\n"
                    f"   → NatGas demand elevated. BOIL may hold/rise. Delay KOLD fade", 'watch'))
            elif avg_7d > 55:
                alerts.append(('🟡 WARM FORECAST',
                    f"NYC 7d avg: {avg_7d:.0f}°F → Low heating demand\n"
                    f"   → KOLD favored if BOIL is extended", 'watch'))
    except:
        pass  # Weather API optional — don't break monitor if unavailable

    # =========================================================================
    # SIGNAL GROUP 17: DFEN (3x Defense) with Bollinger Band Enhancement
    # =========================================================================
    if 'DFEN' in indicators and 'DFEN' in data:
        dfen = indicators['DFEN']
        dfen_rsi = dfen['rsi10']
        dfen_above_sma200 = dfen['price'] > dfen['sma200'] and dfen['sma200'] > 0
        
        # Compute Bollinger Bands for DFEN
        dfen_close = data['DFEN']['Close']
        dfen_bb_sma = safe_float(dfen_close.rolling(20).mean().iloc[-1])
        dfen_bb_std = safe_float(dfen_close.rolling(20).std().iloc[-1])
        dfen_bb_upper = dfen_bb_sma + 2 * dfen_bb_std
        dfen_bb_lower = dfen_bb_sma - 2 * dfen_bb_std
        dfen_pct_b = (dfen['price'] - dfen_bb_lower) / (dfen_bb_upper - dfen_bb_lower) if (dfen_bb_upper - dfen_bb_lower) > 0 else 0.5
        dfen_bb_width = (dfen_bb_upper - dfen_bb_lower) / dfen_bb_sma * 100 if dfen_bb_sma > 0 else 0
        dfen_below_bb = dfen['price'] < dfen_bb_lower
        
        # Store BB data for dashboard
        dfen['bb_upper'] = round(dfen_bb_upper, 2)
        dfen['bb_lower'] = round(dfen_bb_lower, 2)
        dfen['bb_sma20'] = round(dfen_bb_sma, 2)
        dfen['pct_b'] = round(dfen_pct_b, 3)
        dfen['bb_width'] = round(dfen_bb_width, 1)
        
        # PRIMARY: Bollinger Band signal — DFEN-specific edge BB beats RSI
        # Below BB + RSI>=30: 73.5% WR, +4.28% avg 5d, +17% edge, n=49
        # Below BB + RSI<30: 63.8% WR, +6.83% avg 5d, n=47
        if dfen_below_bb and dfen_rsi >= 30:
            alerts.append(('🟢🔥 DFEN BOLLINGER BUY',
                f"DFEN ${dfen['price']:.2f} BELOW lower BB (${dfen_bb_lower:.2f}) + RSI={dfen_rsi:.1f}≥30\n"
                f"   → 73.5% WR, +4.3% avg (5d) | +17% edge vs unconditional | n=49\n"
                f"   BB catches DFEN dips RSI misses — RSI<30 alone only 57.5% WR for DFEN", 'buy'))
        elif dfen_below_bb and dfen_rsi < 30:
            alerts.append(('🟢 DFEN BB + RSI OVERSOLD',
                f"DFEN ${dfen['price']:.2f} BELOW lower BB (${dfen_bb_lower:.2f}) + RSI={dfen_rsi:.1f}\n"
                f"   → 63.8% WR, +6.8% avg (5d) | n=47 | Double oversold", 'buy'))
        
        # SECONDARY: RSI + SMA200 signal (existing)
        elif dfen_above_sma200:
            if dfen_rsi < 25:
                alerts.append(('🟢🔥 DFEN STRONG BUY',
                    f"DFEN RSI={dfen_rsi:.1f} < 25 + above SMA200\n"
                    f"   → 90% WR, +11% avg (20d) | n=52 | Strong uptrend dip", 'buy'))
            elif dfen_rsi < 30:
                alerts.append(('🟢 DFEN BUY',
                    f"DFEN RSI={dfen_rsi:.1f} < 30 + above SMA200\n"
                    f"   → 90% WR, +11% avg (20d) | n=52 | Uptrend dip buy", 'buy'))
            elif dfen_rsi < 35:
                alerts.append(('🟢 DFEN WATCH',
                    f"DFEN RSI={dfen_rsi:.1f} < 35 + above SMA200\n"
                    f"   → 90% WR, +11% avg (20d) | n=52 | Pullback in uptrend", 'buy'))
        else:
            if dfen_rsi < 25:
                alerts.append(('🟡 DFEN OVERSOLD (no trend)',
                    f"DFEN RSI={dfen_rsi:.1f} < 25 but BELOW SMA200\n"
                    f"   → 63% WR without trend filter — reduced conviction", 'watch'))
        
        # Exit signals
        if dfen_rsi > 85:
            alerts.append(('🔴 DFEN OVERBOUGHT',
                f"DFEN RSI={dfen_rsi:.1f} > 85 → Exit DFEN: 42% WR (20d)", 'exit'))
        elif dfen_rsi > 79:
            alerts.append(('🟡 DFEN EXTENDED',
                f"DFEN RSI={dfen_rsi:.1f} > 79 → Caution: 48% WR (20d)", 'warning'))
        
        # BB status line for dashboard section
        status['dfen_bb'] = {
            'price': dfen['price'],
            'upper': round(dfen_bb_upper, 2),
            'lower': round(dfen_bb_lower, 2),
            'sma20': round(dfen_bb_sma, 2),
            'pct_b': round(dfen_pct_b, 3),
            'width': round(dfen_bb_width, 1),
            'below_band': dfen_below_bb,
            'rsi': dfen_rsi,
        }

    return alerts, status

# =============================================================================
# EMAIL FUNCTIONS
# =============================================================================
def format_email(alerts, status, is_preclose=False):
    """Format the email body"""
    now = datetime.now()
    
    timing = "PRE-CLOSE PREVIEW (3:15 PM)" if is_preclose else "MARKET CLOSE CONFIRMATION (4:05 PM)"
    
    body = f"""
{'='*70}
MARKET SIGNAL MONITOR - {timing}
{now.strftime('%Y-%m-%d %H:%M')} ET
{'='*70}

"""
    
    if alerts:
        buy_alerts = [a for a in alerts if a[2] == 'buy']
        exit_alerts = [a for a in alerts if a[2] in ['exit', 'short']]
        warning_alerts = [a for a in alerts if a[2] in ['warning', 'hedge', 'watch']]
        
        if buy_alerts:
            body += "ðŸŸ¢ BUY SIGNALS:\n" + "-"*50 + "\n"
            for title, msg, _ in buy_alerts:
                body += f"{title}\n{msg}\n\n"
        
        if exit_alerts:
            body += "ðŸ”´ EXIT/SHORT SIGNALS:\n" + "-"*50 + "\n"
            for title, msg, _ in exit_alerts:
                body += f"{title}\n{msg}\n\n"
        
        if warning_alerts:
            body += "ðŸŸ¡ WARNINGS/WATCH:\n" + "-"*50 + "\n"
            for title, msg, _ in warning_alerts:
                body += f"{title}\n{msg}\n\n"
    else:
        body += "No signals triggered today.\n\n"
    
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
    
    # Add 3x Leveraged ETFs Section
    body += f"""
{'='*70}
3x LEVERAGED ETFs
{'='*70}
"""
    body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}  Signal\n"
    body += "-"*65 + "\n"
    
    leveraged_tickers = ['NAIL', 'CURE', 'FAS', 'LABU', 'TQQQ', 'SOXL', 'TECL', 'DRN', 'DFEN']
    for ticker in leveraged_tickers:
        if ticker in indicators:
            ind = indicators[ticker]
            price = f"${ind['price']:.2f}"
            rsi = f"{ind['rsi10']:.1f}"
            pct = f"{ind.get('pct_above_sma200', 0):+.1f}%"
            
            # Signal status
            rsi_val = ind['rsi10']
            if rsi_val < 21:
                signal = "ðŸŸ¢ OVERSOLD"
            elif rsi_val < 30:
                signal = "ðŸŸ¢ Watch"
            elif rsi_val > 85:
                signal = "ðŸ”´ OVERBOUGHT"
            elif rsi_val > 79:
                signal = "ðŸŸ¡ Extended"
            else:
                signal = ""
            
            body += f"{ticker:<10} {price:>12} {rsi:>10} {pct:>12}  {signal}\n"
    
    # Add International/Other ETFs
    body += f"""
{'='*70}
OTHER ETFs
{'='*70}
"""
    body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}\n"
    body += "-"*50 + "\n"
    
    other_tickers = ['XLV', 'XLU', 'XLE', 'TMV', 'VOOV', 'VOOG', 'VTV', 'QQQE', 'BOIL', 'EURL', 'YINN', 'KORU', 'INDL', 'EDC']
    for ticker in other_tickers:
        if ticker in indicators:
            ind = indicators[ticker]
            price = f"${ind['price']:.2f}" if ind['price'] < 1000 else f"${ind['price']:,.0f}"
            rsi = f"{ind['rsi10']:.1f}"
            pct = f"{ind.get('pct_above_sma200', 0):+.1f}%"
            body += f"{ticker:<10} {price:>12} {rsi:>10} {pct:>12}\n"
    
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
    
    # Crisis Alpha / Deep Value Dashboard
    body += f"""
{'='*70}
CRISIS ALPHA / DEEP VALUE DASHBOARD
{'='*70}
"""
    for ticker in ['SPY', 'QQQ', 'SMH', 'USMV']:
        if ticker in indicators:
            ind = indicators[ticker]
            maret = ind.get('maret_10', 0)
            cr10 = ind.get('cumret_10', 0)
            cr30 = ind.get('cumret_30', 0)
            cr50 = ind.get('cumret_50', 0)
            s10 = ind.get('stdret_10', 0)
            s50 = ind.get('stdret_50', 0)
            vr = s10/s50 if s50 > 0 else 0
            body += f"\n  {ticker}:\n"
            body += f"    MaRet(10d): {maret:+.2f}%/day    CumRet 10d/30d/50d: {cr10:+.1f}% / {cr30:+.1f}% / {cr50:+.1f}%\n"
            body += f"    StdDev 10/50: {s10:.4f}/{s50:.4f}  Vol Ratio: {vr:.2f}\n"

    # Crisis Alpha regime
    ca = status.get('crisis_alpha', {})
    if ca:
        regime = "UNKNOWN"
        if ca.get('spy_above_sma200') and ca.get('vol_compressing'):
            regime = "BULL + VOL COMPRESS (TQQQ/GLD)"
        elif ca.get('spy_above_sma200') and not ca.get('vol_compressing'):
            regime = "BULL + VOL EXPAND (UPRO/GLD)"
        elif not ca.get('spy_above_sma200') and ca.get('vol_compressing') and ca.get('spy_above_ema20'):
            regime = "BEAR RECOVERY (UPRO/GLD/SHY)"
        elif not ca.get('spy_above_sma200'):
            regime = "BEAR DEFENSIVE (SHY/GLD)"
        body += f"\n  CRISIS ALPHA REGIME: {regime}\n"
        body += f"    Vol Ratio: {ca.get('vol_ratio', 0):.2f}  SPY>SMA200: {ca.get('spy_above_sma200')}  SPY>EMA20: {ca.get('spy_above_ema20')}\n"
    
    # Deep Value trigger proximity
    body += f"\n  DEEP VALUE TRIGGER PROXIMITY:\n"
    for ticker, thresholds in [('QQQ', [('MaRet(10d)', 'maret_10', -1.0), ('CumRet(30d)', 'cumret_30', -20)]),
                                ('SMH', [('MaRet(10d)', 'maret_10', -1.5), ('CumRet(30d)', 'cumret_30', -25)]),
                                ('SPY', [('CumRet(50d)', 'cumret_50', -15), ('CumRet(100d)', 'cumret_100', -10)])]:
        if ticker in indicators:
            for label, key, thresh in thresholds:
                val = indicators[ticker].get(key, 0)
                dist = val - thresh
                pct_to_trigger = (dist / abs(thresh)) * 100 if thresh != 0 else 0
                status_str = "ACTIVE" if val < thresh else f"{dist:+.1f}% away"
                body += f"    {ticker} {label}: {val:+.1f}% (trigger: {thresh}%) — {status_str}\n"
    
    # DFEN Bollinger Band Status
    dfen_bb = status.get('dfen_bb')
    if dfen_bb:
        bb_status = "🔥 BELOW LOWER BAND" if dfen_bb['below_band'] else "Within bands"
        body += f"""
  DFEN BOLLINGER BANDS (20, 2):
    Price: ${dfen_bb['price']:.2f}  |  RSI: {dfen_bb['rsi']:.1f}
    Upper:  ${dfen_bb['upper']:.2f}
    SMA20:  ${dfen_bb['sma20']:.2f}
    Lower:  ${dfen_bb['lower']:.2f}
    %B: {dfen_bb['pct_b']:.3f}  |  Width: {dfen_bb['width']:.1f}%  |  {bb_status}
    Signal: Below BB+RSI>=30 = 73.5% WR (5d) | Below BB+RSI<30 = 63.8% WR
"""
    
    if is_preclose:
        body += f"""
{'='*70}
NOTE: This is a PRE-CLOSE preview. Signals may change by market close.
Final confirmation email will be sent at 4:05 PM ET.
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
    
    tickers = [
        # Core Indices
        'SMH', 'SPY', 'QQQ', 'IWM',
        # Defensive Sectors
        'XLP', 'XLU', 'XLV',
        # Safe Havens & Macro
        'GLD', 'TLT', 'HYG', 'LQD', 'TMV',
        'USDU', 'UCO', 'BOIL',
        # Volatility
        'UVXY',
        # International
        'EDC', 'YINN', 'KORU', 'EURL', 'INDL',
        # Crypto
        'BTC-USD',
        # Individual Stocks
        'AMD', 'NVDA',
        # 3x Leveraged ETFs (new)
        'NAIL', 'CURE', 'FAS', 'LABU',
        'TQQQ', 'SOXL', 'TECL', 'DRN',
        # Style/Factor ETFs
        'VOOV', 'VOOG', 'VTV', 'QQQE',
        # Energy
        'XLE', 'XLF',
        # Crisis Alpha / Deep Value
        'USMV',
        # NatGas
        'KOLD',
        # Defense
        'DFEN',
    ]
    
    print("Downloading market data...")
    data = download_data(tickers)
    print(f"Downloaded data for {len(data)} tickers")
    
    alerts, status = check_signals(data)
    
    if alerts:
        buy_count = len([a for a in alerts if a[2] == 'buy'])
        exit_count = len([a for a in alerts if a[2] in ['exit', 'short']])
        
        if exit_count > 0:
            emoji = "ðŸ”´"
            urgency = "EXIT SIGNALS"
        elif buy_count > 0:
            emoji = "ðŸŸ¢"
            urgency = "BUY SIGNALS"
        else:
            emoji = "ðŸŸ¡"
            urgency = "WATCH"
        
        timing = "PRE-CLOSE" if IS_PRECLOSE else "CLOSE"
        subject = f"{emoji} [{timing}] Market Signals: {len(alerts)} Alert(s) - {urgency}"
    else:
        timing = "PRE-CLOSE" if IS_PRECLOSE else "CLOSE"
        subject = f"ðŸ“Š [{timing}] Market Signals: No Alerts"
    
    body = format_email(alerts, status, IS_PRECLOSE)
    send_email(subject, body)
    
    print(f"\n{len(alerts)} signal(s) detected")
    for title, msg, _ in alerts:
        print(f"  {title}")

if __name__ == "__main__":
    main()
