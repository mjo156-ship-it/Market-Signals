#!/usr/bin/env python3
"""
Comprehensive Market Signal Monitor v3.0
========================================
Monitors all backtested trading signals and sends alerts.

CHANGES in v3.0:
- Added Bond Momentum indicator (TLT 10d ret vs 0 as BND/TBX proxy)
- Bond momentum conviction flag on UVXY hedge signals
- Added EMA(9), EMA(20), EMA(50), EMA(200) for all tracked tickers
- Added BND to ticker list for bond momentum tracking

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
            
            # EMAs — 9, 20, 50, 200
            ema9 = safe_float(close.ewm(span=9, adjust=False).mean().iloc[-1])
            ema20 = safe_float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = safe_float(close.ewm(span=50, adjust=False).mean().iloc[-1])
            ema200 = safe_float(close.ewm(span=200, adjust=False).mean().iloc[-1])
            
            indicators[ticker] = {
                'price': price,
                'rsi10': rsi10,
                'rsi50': rsi50,
                'sma200': sma200,
                'sma50': sma50,
                'ema9': ema9,
                'ema20': ema20,
                'ema50': ema50,
                'ema200': ema200,
            }
            
            # Calculate % above SMA200
            if sma200 > 0:
                indicators[ticker]['pct_above_sma200'] = (price / sma200 - 1) * 100
            else:
                indicators[ticker]['pct_above_sma200'] = 0

            # EMA trend flags
            indicators[ticker]['above_ema9'] = price > ema9
            indicators[ticker]['above_ema20'] = price > ema20
            indicators[ticker]['above_ema50'] = price > ema50
            indicators[ticker]['above_ema200'] = price > ema200
                
        except Exception as e:
            print(f"Error calculating indicators for {ticker}: {e}")
            continue
    
    status['indicators'] = indicators
    
    # =========================================================================
    # BOND MOMENTUM INDICATOR
    # =========================================================================
    bond_momentum = None
    bond_mom_detail = {}
    if 'TLT' in data and len(data['TLT']) >= 15:
        try:
            tlt_close = data['TLT']['Close']
            tlt_ret10 = safe_float((tlt_close.iloc[-1] / tlt_close.iloc[-11] - 1))
            bonds_rising = tlt_ret10 > 0
            bond_momentum = bonds_rising
            bond_mom_detail = {
                'tlt_ret10': tlt_ret10 * 100,
                'bonds_rising': bonds_rising,
                'direction': 'RISING' if bonds_rising else 'FALLING',
            }
            # Also get BND if available
            if 'BND' in data and len(data['BND']) >= 15:
                bnd_close = data['BND']['Close']
                bnd_ret10 = safe_float((bnd_close.iloc[-1] / bnd_close.iloc[-11] - 1))
                bond_mom_detail['bnd_ret10'] = bnd_ret10 * 100
        except Exception as e:
            print(f"Error calculating bond momentum: {e}")
    
    status['bond_momentum'] = bond_mom_detail
    
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
    # SIGNAL GROUP 4: Volatility Hedge Signals (with Bond Momentum Conviction)
    # =========================================================================
    if 'QQQ' in indicators:
        qqq = indicators['QQQ']
        
        if qqq['rsi10'] > 79:
            # Add bond momentum conviction
            bm_note = ""
            if bond_momentum is not None:
                if not bond_momentum:  # Bonds falling
                    bm_note = " | ⚡ BONDS FALLING = HIGH conviction"
                else:
                    bm_note = " | ⚠️ Bonds rising = moderate conviction"
            
            alerts.append(('🟡 VOL HEDGE', 
                f"QQQ RSI={qqq['rsi10']:.1f} > 79 → Long UVXY 5d: 67% win, +33% CAGR{bm_note}", 'hedge'))
        
        if qqq['rsi10'] < 20:
            alerts.append(('🟢 QQQ DIP BUY', 
                f"QQQ RSI={qqq['rsi10']:.1f} < 20 → Long TQQQ 5d: 69% win, +26% CAGR", 'buy'))
    
    # =========================================================================
    # SIGNAL GROUP 4b: SPY Overbought UVXY (with Bond Momentum)
    # =========================================================================
    if 'SPY' in indicators:
        spy = indicators['SPY']
        
        if spy['rsi10'] > 79:
            bm_note = ""
            if bond_momentum is not None:
                if not bond_momentum:
                    bm_note = "\n   ⚡ BONDS FALLING: 70% win, +7.2% avg (HIGH conviction)"
                else:
                    bm_note = "\n   ⚠️ Bonds rising: 50% win, +1.9% avg (moderate conviction)"
            
            # Dual overbought
            qqq_ob = 'QQQ' in indicators and indicators['QQQ']['rsi10'] > 79
            if qqq_ob:
                alerts.append(('🟡 DUAL OVERBOUGHT → UVXY', 
                    f"SPY RSI={spy['rsi10']:.1f} + QQQ RSI={indicators['QQQ']['rsi10']:.1f} > 79 → UVXY 5d: 76% win, +9.0%{bm_note}", 'hedge'))
            else:
                alerts.append(('🟡 SPY OVERBOUGHT → UVXY', 
                    f"SPY RSI={spy['rsi10']:.1f} > 79 → UVXY 5d: 64% win, +5.9%{bm_note}", 'hedge'))
    
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
    # SIGNAL GROUP 8: AMD/NVDA Specific
    # =========================================================================
    if 'AMD' in indicators:
        amd = indicators['AMD']
        if amd['rsi10'] > 85:
            alerts.append(('🟡 AMD EXTENDED', 
                f"AMD RSI={amd['rsi10']:.1f} > 85 → Consider taking profits", 'warning'))
    
    if 'NVDA' in indicators:
        nvda = indicators['NVDA']
        if nvda['rsi10'] > 85:
            alerts.append(('🟡 NVDA EXTENDED', 
                f"NVDA RSI={nvda['rsi10']:.1f} > 85 → Consider taking profits", 'warning'))
    
    # =========================================================================
    # SIGNAL GROUP 9: NAIL (3x Homebuilders) Signals
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
    
    # =========================================================================
    # SIGNAL GROUP 10: CURE (3x Healthcare) Signals
    # =========================================================================
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
    
    # =========================================================================
    # SIGNAL GROUP 11: FAS (3x Financials) Signals
    # =========================================================================
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
    
    # =========================================================================
    # SIGNAL GROUP 12: LABU (3x Biotech) Signals
    # =========================================================================
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
                f"LABU {labu['pct_above_sma200']:.0f}% above SMA(200) → Very extended, consider profits", 'warning'))
    
    return alerts, status

# =============================================================================
# EMAIL FUNCTIONS
# =============================================================================
def format_ema_line(ind, price):
    """Format EMA status as compact trend arrows"""
    e9 = ind.get('ema9', 0)
    e20 = ind.get('ema20', 0)
    e50 = ind.get('ema50', 0)
    e200 = ind.get('ema200', 0)
    
    # Trend stack: price vs each EMA
    flags = []
    if price > e9: flags.append('9↑')
    else: flags.append('9↓')
    if price > e20: flags.append('20↑')
    else: flags.append('20↓')
    if price > e50: flags.append('50↑')
    else: flags.append('50↓')
    if price > e200: flags.append('200↑')
    else: flags.append('200↓')
    
    return ' '.join(flags)

def format_email(alerts, status, is_preclose=False):
    """Format the email body"""
    now = datetime.now()
    
    timing = "PRE-CLOSE PREVIEW (3:15 PM)" if is_preclose else "MARKET CLOSE CONFIRMATION (4:05 PM)"
    
    body = f"""
{'='*70}
MARKET SIGNAL MONITOR v3.0 - {timing}
{now.strftime('%Y-%m-%d %H:%M')} ET
{'='*70}

"""
    
    # ─── Bond Momentum Status ───
    bm = status.get('bond_momentum', {})
    if bm:
        direction = bm.get('direction', 'UNKNOWN')
        tlt_ret = bm.get('tlt_ret10', 0)
        icon = '📈' if bm.get('bonds_rising') else '📉'
        
        body += f"""{'─'*70}
{icon} BOND MOMENTUM: {direction} (TLT 10d: {tlt_ret:+.2f}%)
"""
        if bm.get('bonds_rising'):
            body += "   Interpretation: Bonds bid → rate-cut expectations / risk-on macro\n"
            body += "   UVXY hedge conviction: MODERATE (50% win when SPY>79)\n"
        else:
            body += "   Interpretation: Bonds selling → rate-rise pressure / risk-off\n"
            body += "   UVXY hedge conviction: HIGH (70% win when SPY>79)\n"
        body += f"{'─'*70}\n\n"
    
    # ─── Signal Alerts ───
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
    
    # ─── Playbook Status ───
    indicators = status.get('indicators', {})
    
    def _rsi(ticker):
        return indicators.get(ticker, {}).get('rsi10')
    
    def _pct_bar(current, threshold, direction='above'):
        """Create a visual proximity bar. direction='above' means signal fires when current > threshold."""
        if current is None:
            return "          —           "
        if direction == 'above':
            pct = (current / threshold) * 100 if threshold > 0 else 0
            active = current >= threshold
        else:  # 'below' — signal fires when current < threshold
            # Invert: closer to firing as current drops toward threshold
            pct = ((100 - current) / (100 - threshold)) * 100 if threshold < 100 else 0
            active = current <= threshold
        
        pct = min(pct, 100)
        filled = int(pct / 100 * 12)
        bar = '█' * filled + '░' * (12 - filled)
        
        if active:
            return f"[{bar}] ✓ ACTIVE"
        else:
            return f"[{bar}] {pct:.0f}%"
    
    gld_rsi = _rsi('GLD')
    usdu_rsi = _rsi('USDU')
    xlp_rsi = _rsi('XLP')
    xlu_rsi = _rsi('XLU')
    xlv_rsi = _rsi('XLV')
    spy_rsi = _rsi('SPY')
    qqq_rsi = _rsi('QQQ')
    smh_rsi = _rsi('SMH')
    xlf_rsi = _rsi('XLF')
    uvxy_rsi = _rsi('UVXY')
    btc_rsi = _rsi('BTC-USD')
    
    # Count combo signal conditions
    triple_met = sum([
        1 if gld_rsi and gld_rsi > 79 else 0,
        1 if usdu_rsi and usdu_rsi < 25 else 0,
        1 if xlp_rsi and xlp_rsi > 65 else 0,
    ])
    double_met = sum([
        1 if gld_rsi and gld_rsi > 79 else 0,
        1 if usdu_rsi and usdu_rsi < 25 else 0,
    ])
    def_rotation_met = sum([
        1 if any(_rsi(t) and _rsi(t) > 79 for t in ['XLP', 'XLU', 'XLV']) else 0,
        1 if spy_rsi and spy_rsi < 79 else 0,
        1 if qqq_rsi and qqq_rsi < 79 else 0,
    ])
    soxs_squeeze_met = sum([
        1 if smh_rsi and smh_rsi > 79 else 0,
        1 if usdu_rsi and usdu_rsi > 70 else 0,
    ])
    
    body += f"""
{'='*70}
PLAYBOOK STATUS — Signal Proximity
{'='*70}

COMBO SIGNALS
{'─'*50}
  Triple Signal ({triple_met}/3 conditions):
    GLD RSI > 79:    {gld_rsi if gld_rsi else 0:>5.1f}  {_pct_bar(gld_rsi, 79, 'above')}
    USDU RSI < 25:   {usdu_rsi if usdu_rsi else 0:>5.1f}  {_pct_bar(usdu_rsi, 25, 'below')}
    XLP RSI > 65:    {xlp_rsi if xlp_rsi else 0:>5.1f}  {_pct_bar(xlp_rsi, 65, 'above')}

  Double Signal ({double_met}/2 conditions):
    GLD RSI > 79:    {gld_rsi if gld_rsi else 0:>5.1f}  {_pct_bar(gld_rsi, 79, 'above')}
    USDU RSI < 25:   {usdu_rsi if usdu_rsi else 0:>5.1f}  {_pct_bar(usdu_rsi, 25, 'below')}

DEFENSIVE ROTATION ({def_rotation_met}/3 conditions):
{'─'*50}
    XLP RSI > 79:    {xlp_rsi if xlp_rsi else 0:>5.1f}  {_pct_bar(xlp_rsi, 79, 'above') if xlp_rsi else '—'}
    XLU RSI > 79:    {xlu_rsi if xlu_rsi else 0:>5.1f}  {_pct_bar(xlu_rsi, 79, 'above') if xlu_rsi else '—'}
    XLV RSI > 79:    {xlv_rsi if xlv_rsi else 0:>5.1f}  {_pct_bar(xlv_rsi, 79, 'above') if xlv_rsi else '—'}
    SPY RSI < 79:    {spy_rsi if spy_rsi else 0:>5.1f}  {'✓ met' if spy_rsi and spy_rsi < 79 else '✗ SPY overbought'}
    QQQ RSI < 79:    {qqq_rsi if qqq_rsi else 0:>5.1f}  {'✓ met' if qqq_rsi and qqq_rsi < 79 else '✗ QQQ overbought'}

VOL HEDGE
{'─'*50}
    SPY RSI > 79:    {spy_rsi if spy_rsi else 0:>5.1f}  {_pct_bar(spy_rsi, 79, 'above') if spy_rsi else '—'}
    QQQ RSI > 79:    {qqq_rsi if qqq_rsi else 0:>5.1f}  {_pct_bar(qqq_rsi, 79, 'above') if qqq_rsi else '—'}

SOXS DOLLAR SQUEEZE ({soxs_squeeze_met}/2 conditions):
{'─'*50}
    SMH RSI > 79:    {smh_rsi if smh_rsi else 0:>5.1f}  {_pct_bar(smh_rsi, 79, 'above') if smh_rsi else '—'}
    USDU RSI > 70:   {usdu_rsi if usdu_rsi else 0:>5.1f}  {_pct_bar(usdu_rsi, 70, 'above') if usdu_rsi else '—'}

DANGER SIGNALS
{'─'*50}
    XLF > 70 + USDU < 25 (NAIL danger):  XLF={xlf_rsi if xlf_rsi else 0:.1f}  USDU={usdu_rsi if usdu_rsi else 0:.1f}  {'⚠️ ACTIVE' if xlf_rsi and usdu_rsi and xlf_rsi > 70 and usdu_rsi < 25 else '— clear'}
    SPY RSI > 85 (UPRO exit):   {spy_rsi if spy_rsi else 0:>5.1f}  {_pct_bar(spy_rsi, 85, 'above') if spy_rsi else '—'}
    FAS RSI > 85 (FAS exit):    {_rsi('FAS') or 0:>5.1f}  {_pct_bar(_rsi('FAS'), 85, 'above') if _rsi('FAS') else '—'}

DIP BUY PROXIMITY
{'─'*50}
    SPY RSI < 25:    {spy_rsi if spy_rsi else 0:>5.1f}  {_pct_bar(spy_rsi, 25, 'below') if spy_rsi else '—'}
    QQQ RSI < 20:    {qqq_rsi if qqq_rsi else 0:>5.1f}  {_pct_bar(qqq_rsi, 20, 'below') if qqq_rsi else '—'}
    BTC RSI < 30:    {btc_rsi if btc_rsi else 0:>5.1f}  {_pct_bar(btc_rsi, 30, 'below') if btc_rsi else '—'}
    CURE RSI < 25:   {_rsi('CURE') or 0:>5.1f}  {_pct_bar(_rsi('CURE'), 25, 'below') if _rsi('CURE') else '—'}
    LABU RSI < 25:   {_rsi('LABU') or 0:>5.1f}  {_pct_bar(_rsi('LABU'), 25, 'below') if _rsi('LABU') else '—'}
    FAS RSI < 30:    {_rsi('FAS') or 0:>5.1f}  {_pct_bar(_rsi('FAS'), 30, 'below') if _rsi('FAS') else '—'}

"""
    
    # ─── Current Indicator Status ───
    body += f"""
{'='*70}
CURRENT INDICATOR STATUS
{'='*70}

"""
    
    key_tickers = ['SPY', 'QQQ', 'SMH', 'GLD', 'USDU', 'XLP', 'TLT', 'HYG', 'XLF', 'UVXY', 'BTC-USD', 'AMD', 'NVDA']
    body += f"{'Ticker':<10} {'Price':>10} {'RSI(10)':>8} {'vsSMA200':>9}  {'EMA Trend':>20}\n"
    body += "-"*62 + "\n"
    
    for ticker in key_tickers:
        if ticker in indicators:
            ind = indicators[ticker]
            price = ind['price']
            price_str = f"${price:.2f}" if price < 1000 else f"${price:,.0f}"
            rsi = f"{ind['rsi10']:.1f}"
            pct = f"{ind.get('pct_above_sma200', 0):+.1f}%"
            ema_trend = format_ema_line(ind, price)
            body += f"{ticker:<10} {price_str:>10} {rsi:>8} {pct:>9}  {ema_trend:>20}\n"
    
    # ─── 3x Leveraged ETFs ───
    body += f"""
{'='*70}
3x LEVERAGED ETFs
{'='*70}
"""
    body += f"{'Ticker':<8} {'Price':>10} {'RSI(10)':>8} {'vsSMA200':>9}  {'EMA Trend':>20}  Signal\n"
    body += "-"*75 + "\n"
    
    leveraged_tickers = ['NAIL', 'CURE', 'FAS', 'LABU', 'TQQQ', 'SOXL', 'TECL', 'DRN']
    for ticker in leveraged_tickers:
        if ticker in indicators:
            ind = indicators[ticker]
            price = f"${ind['price']:.2f}"
            rsi = f"{ind['rsi10']:.1f}"
            pct = f"{ind.get('pct_above_sma200', 0):+.1f}%"
            ema_trend = format_ema_line(ind, ind['price'])
            
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
            
            body += f"{ticker:<8} {price:>10} {rsi:>8} {pct:>9}  {ema_trend:>20}  {signal}\n"
    
    # ─── Other ETFs ───
    body += f"""
{'='*70}
OTHER ETFs
{'='*70}
"""
    body += f"{'Ticker':<8} {'Price':>10} {'RSI(10)':>8} {'vsSMA200':>9}  {'EMA Trend':>20}\n"
    body += "-"*60 + "\n"
    
    other_tickers = ['XLV', 'XLU', 'XLE', 'TMV', 'VOOV', 'VOOG', 'VTV', 'QQQE', 'BOIL', 'EURL', 'YINN', 'KORU', 'INDL', 'EDC']
    for ticker in other_tickers:
        if ticker in indicators:
            ind = indicators[ticker]
            price = ind['price']
            price_str = f"${price:.2f}" if price < 1000 else f"${price:,.0f}"
            rsi = f"{ind['rsi10']:.1f}"
            pct = f"{ind.get('pct_above_sma200', 0):+.1f}%"
            ema_trend = format_ema_line(ind, price)
            body += f"{ticker:<8} {price_str:>10} {rsi:>8} {pct:>9}  {ema_trend:>20}\n"
    
    # ─── EMA Detail Table (Key Tickers) ───
    body += f"""
{'='*70}
EMA DETAIL — KEY TICKERS
{'='*70}
"""
    body += f"{'Ticker':<8} {'Price':>10} {'EMA(9)':>10} {'EMA(20)':>10} {'EMA(50)':>10} {'EMA(200)':>10}\n"
    body += "-"*62 + "\n"
    
    ema_tickers = ['SPY', 'QQQ', 'SMH', 'GLD', 'TLT', 'USDU', 'XLP', 'XLF', 'UVXY', 'BTC-USD', 
                   'TQQQ', 'SOXL', 'UPRO', 'TECL', 'NAIL', 'CURE', 'FAS', 'LABU']
    for ticker in ema_tickers:
        if ticker in indicators:
            ind = indicators[ticker]
            p = ind['price']
            fmt = lambda v: f"${v:.2f}" if v < 1000 else f"${v:,.0f}"
            body += f"{ticker:<8} {fmt(p):>10} {fmt(ind['ema9']):>10} {fmt(ind['ema20']):>10} {fmt(ind['ema50']):>10} {fmt(ind['ema200']):>10}\n"
    
    # ─── SMH/SOXL Levels ───
    if 'SMH' in indicators:
        smh = indicators['SMH']
        sma200 = smh['sma200']
        body += f"""
{'='*70}
SMH/SOXL LEVELS
{'='*70}
Current Price:    ${smh['price']:.2f}
SMA(200):         ${sma200:.2f}
EMA(9):           ${smh['ema9']:.2f}  {'✓ above' if smh['above_ema9'] else '✗ below'}
EMA(20):          ${smh['ema20']:.2f}  {'✓ above' if smh['above_ema20'] else '✗ below'}
EMA(50):          ${smh['ema50']:.2f}  {'✓ above' if smh['above_ema50'] else '✗ below'}
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
        # Bonds (for bond momentum)
        'BND',
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
        'TQQQ', 'SOXL', 'TECL', 'DRN', 'UPRO',
        # Style/Factor ETFs
        'VOOV', 'VOOG', 'VTV', 'QQQE',
        # Energy/Financials
        'XLE', 'XLF',
    ]
    
    print("Downloading market data...")
    data = download_data(tickers)
    print(f"Downloaded data for {len(data)} tickers")
    
    alerts, status = check_signals(data)
    
    if alerts:
        buy_count = len([a for a in alerts if a[2] == 'buy'])
        exit_count = len([a for a in alerts if a[2] in ['exit', 'short']])
        
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
        subject = f"{emoji} [{timing}] Market Signals: {len(alerts)} Alert(s) - {urgency}"
    else:
        timing = "PRE-CLOSE" if IS_PRECLOSE else "CLOSE"
        subject = f"📊 [{timing}] Market Signals: No Alerts"
    
    body = format_email(alerts, status, IS_PRECLOSE)
    send_email(subject, body)
    
    print(f"\n{len(alerts)} signal(s) detected")
    for title, msg, _ in alerts:
        print(f"  {title}")

if __name__ == "__main__":
    main()
