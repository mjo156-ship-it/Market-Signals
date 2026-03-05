#!/usr/bin/env python3
"""
Comprehensive Market Signal Monitor v4.0
========================================
Monitors all backtested trading signals and sends alerts.

SCHEDULE: Two emails daily (weekdays)
- 3:15 PM ET: Pre-close preview
- 4:05 PM ET: Market close confirmation

v4.0 ADDITIONS (March 2026):
- Signal Group 13: Oil Supply Shock (USO/UCO + USDU combos)
- Signal Group 14: Dispersion Trade Regime (sector RSI spread + VIX routing)
- Signal Group 15: Commodity Playbook (PALL, PDBC manual hold alerts)
- Signal Group 16: KOLD Oil Reversal (XLE>79 + USDU amplifier)
- Signal Group 17: AND Combination Signals (UVXY+XLU, BTAL+QQQ, XLE+XLF)
- Dashboard: Oil Supply Shock status with consecutive day tracking
- Dashboard: Dispersion regime indicator with hedge routing
- Dashboard: Commodity watchlist
- Dashboard: Managed Futures status (CTA/DBMF/KMLM)
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
    # SIGNAL GROUP 13: OIL SUPPLY SHOCK (USO/UCO + USDU)
    # Priority addition - strongest new signal found March 2026
    # Causal: Oil up + Dollar up = supply-driven shock, crushes equities
    # Contrast: Oil up + Dollar down = demand-driven, benign for equities
    # =========================================================================
    
    # Track consecutive days for oil supply shock signals
    oil_consec_uco79 = 0
    oil_consec_uco82 = 0
    oil_consec_uso75 = 0
    
    if 'UCO' in data and 'USDU' in data:
        uco_df = data['UCO']
        usdu_df = data['USDU']
        uco_close = uco_df['Close']
        usdu_close = usdu_df['Close']
        
        uco_rsi_series = calculate_rsi_wilder(uco_close, 10)
        usdu_rsi_series = calculate_rsi_wilder(usdu_close, 10)
        
        # Align on common dates
        common_dates = uco_rsi_series.dropna().index.intersection(usdu_rsi_series.dropna().index)
        if len(common_dates) > 0:
            common_dates = common_dates.sort_values()
            # Count UCO>79 + USDU>55 consecutive
            for d in reversed(common_dates):
                try:
                    uco_r = safe_float(uco_rsi_series.loc[d])
                    usdu_r = safe_float(usdu_rsi_series.loc[d])
                    if uco_r > 79 and usdu_r > 55:
                        oil_consec_uco79 += 1
                    else:
                        break
                except:
                    break
            
            # Count UCO>82 + USDU>55
            for d in reversed(common_dates):
                try:
                    uco_r = safe_float(uco_rsi_series.loc[d])
                    usdu_r = safe_float(usdu_rsi_series.loc[d])
                    if uco_r > 82 and usdu_r > 55:
                        oil_consec_uco82 += 1
                    else:
                        break
                except:
                    break
    
    if 'USO' in data and 'USDU' in data:
        uso_df = data['USO']
        uso_close = uso_df['Close']
        uso_rsi_series = calculate_rsi_wilder(uso_close, 10)
        usdu_close2 = data['USDU']['Close']
        usdu_rsi_series2 = calculate_rsi_wilder(usdu_close2, 10)
        
        common2 = uso_rsi_series.dropna().index.intersection(usdu_rsi_series2.dropna().index)
        if len(common2) > 0:
            common2 = common2.sort_values()
            for d in reversed(common2):
                try:
                    uso_r = safe_float(uso_rsi_series.loc[d])
                    usdu_r = safe_float(usdu_rsi_series2.loc[d])
                    if uso_r > 75 and usdu_r > 55:
                        oil_consec_uso75 += 1
                    else:
                        break
                except:
                    break
    
    status['oil_consec_uco79_usdu55'] = oil_consec_uco79
    status['oil_consec_uco82_usdu55'] = oil_consec_uco82
    status['oil_consec_uso75_usdu55'] = oil_consec_uso75
    
    # Generate oil supply shock alerts
    if 'UCO' in indicators and 'USDU' in indicators:
        uco = indicators['UCO']
        usdu = indicators['USDU']
        
        if uco['rsi10'] >= 85 and usdu['rsi10'] > 60:
            alerts.append(('OIL SUPPLY SHOCK EXTREME',
                f"UCO RSI={uco['rsi10']:.1f} + USDU RSI={usdu['rsi10']:.1f}\n"
                f"   Consecutive days UCO>79+USDU>55: {oil_consec_uco79}\n"
                f"   HISTORICAL (UCO>85+USDU>55, n=5): SPY 5d 20% WR, TQQQ -5.25%\n"
                f"   -> HEDGE: 33% UVXY / 33% BTAL / 33% SQQQ (5d hold)\n"
                f"   -> Episode avg: +7.78% basket return\n"
                f"   -> BTAL 100% WR at >=3 consec days, UVXY +9.95%\n"
                f"   -> EXIT: Cut when UCO RSI <79 or basket negative after D+2", 'exit'))
        elif uco['rsi10'] > 82 and usdu['rsi10'] > 55:
            alerts.append(('OIL SUPPLY SHOCK STRONG',
                f"UCO RSI={uco['rsi10']:.1f} + USDU RSI={usdu['rsi10']:.1f}\n"
                f"   Consecutive days UCO>82+USDU>55: {oil_consec_uco82}\n"
                f"   HISTORICAL (n=13): SPY 5d 23% WR, -1.25% avg\n"
                f"   At >=2 consec days: SPY 12.5% WR, BTAL 100%, UVXY +8.36%\n"
                f"   -> HEDGE: 33% UVXY / 33% BTAL / 33% SQQQ\n"
                f"   -> Avoid adding TQQQ/SOXL/UPRO", 'exit'))
        elif uco['rsi10'] > 79 and usdu['rsi10'] > 55:
            alerts.append(('OIL SUPPLY SHOCK WARNING',
                f"UCO RSI={uco['rsi10']:.1f} + USDU RSI={usdu['rsi10']:.1f}\n"
                f"   Consecutive days UCO>79+USDU>55: {oil_consec_uco79}\n"
                f"   HISTORICAL (n=23): SPY 5d 30% WR, -0.96% avg\n"
                f"   -> Caution on leveraged equity adds\n"
                f"   -> Monitor for escalation to UCO>82", 'warning'))
        elif uco['rsi10'] > 75 and usdu['rsi10'] > 50:
            alerts.append(('OIL SUPPLY WATCH',
                f"UCO RSI={uco['rsi10']:.1f} + USDU RSI={usdu['rsi10']:.1f}\n"
                f"   Approaching supply shock zone. Monitor for escalation.", 'watch'))
        
        # CONTRAST: Oil up + Dollar WEAK = benign
        if uco['rsi10'] > 79 and usdu['rsi10'] < 40:
            alerts.append(('OIL DEMAND SIGNAL (BULLISH)',
                f"UCO RSI={uco['rsi10']:.1f} + USDU RSI={usdu['rsi10']:.1f} (weak dollar)\n"
                f"   Oil up + Dollar weak = demand-driven, BULLISH\n"
                f"   QQQ 5d: 84.6% WR, +1.00% avg | n=13", 'buy'))
    
    # USO-based confirmations
    if 'USO' in indicators and 'USDU' in indicators:
        uso_ind = indicators['USO']
        usdu_ind = indicators['USDU']
        
        if uso_ind['rsi10'] > 79 and usdu_ind['rsi10'] > 60:
            alerts.append(('USO SUPPLY SHOCK CONFIRM',
                f"USO RSI={uso_ind['rsi10']:.1f} + USDU RSI={usdu_ind['rsi10']:.1f}\n"
                f"   HISTORICAL (n=15, 3 episodes): SPY 5d 13.3% WR, -1.69%\n"
                f"   SOXL -9.20%, TQQQ -7.42%, BTAL 100% WR +2.70%\n"
                f"   Consec days USO>75+USDU>55: {oil_consec_uso75}", 'exit'))
        
        # Inflation squeeze
        if 'TLT' in indicators:
            tlt_ind = indicators['TLT']
            if uso_ind['rsi10'] > 75 and tlt_ind['rsi10'] < 30:
                alerts.append(('INFLATION SQUEEZE',
                    f"USO RSI={uso_ind['rsi10']:.1f} + TLT RSI={tlt_ind['rsi10']:.1f}\n"
                    f"   Oil up + Bonds crushed = inflation panic\n"
                    f"   QQQ 20d: 33% WR, -2.07% avg | n=21, 9 episodes\n"
                    f"   XLE keeps running, tech suffers most", 'warning'))
    
    # =========================================================================
    # SIGNAL GROUP 14: DISPERSION TRADE REGIME
    # =========================================================================
    sector_rsis = {}
    for ticker in ['XLP', 'XLU', 'XLV', 'XLF', 'XLE', 'XLY', 'SMH']:
        if ticker in indicators:
            sector_rsis[ticker] = indicators[ticker]['rsi10']
    
    if len(sector_rsis) >= 5:
        rsi_values = list(sector_rsis.values())
        rsi_range = max(rsi_values) - min(rsi_values)
        rsi_max_ticker = max(sector_rsis, key=sector_rsis.get)
        rsi_min_ticker = min(sector_rsis, key=sector_rsis.get)
        
        status['dispersion_range'] = rsi_range
        status['dispersion_leader'] = f"{rsi_max_ticker} ({sector_rsis[rsi_max_ticker]:.1f})"
        status['dispersion_laggard'] = f"{rsi_min_ticker} ({sector_rsis[rsi_min_ticker]:.1f})"
        
        smh_xlp_gap = 0
        if 'SMH' in sector_rsis and 'XLP' in sector_rsis:
            smh_xlp_gap = sector_rsis['SMH'] - sector_rsis['XLP']
            status['smh_xlp_gap'] = smh_xlp_gap
        
        vix_level = 0
        if 'VIX' in indicators:
            vix_level = indicators['VIX']['price']
        
        # Defensive rotation with VIX-based hedge routing
        if 'XLP' in sector_rsis and 'SMH' in sector_rsis:
            if sector_rsis['XLP'] > 65 and sector_rsis['SMH'] < 50:
                if vix_level < 20:
                    hedge_rec = "GLD is primary hedge (+4.8% avg 20d in dispersion-only)"
                elif vix_level < 25:
                    hedge_rec = "CTA/KMLM are primary hedges (+3.9% avg 20d when VIX>20)"
                else:
                    hedge_rec = "VIX>25 = possible bottom, CTA/KMLM may whipsaw"
                
                alerts.append(('DEFENSIVE ROTATION + DISPERSION',
                    f"XLP RSI={sector_rsis['XLP']:.1f} (>65) + SMH RSI={sector_rsis['SMH']:.1f} (<50)\n"
                    f"   Sector RSI range: {rsi_range:.1f} | VIX: {vix_level:.1f}\n"
                    f"   Leader: {rsi_max_ticker} ({sector_rsis[rsi_max_ticker]:.1f})\n"
                    f"   Laggard: {rsi_min_ticker} ({sector_rsis[rsi_min_ticker]:.1f})\n"
                    f"   -> {hedge_rec}", 'warning'))
        
        if rsi_range > 45:
            alerts.append(('EXTREME DISPERSION',
                f"Sector RSI range: {rsi_range:.1f} (>45 threshold)\n"
                f"   Leader: {rsi_max_ticker} ({sector_rsis[rsi_max_ticker]:.1f})\n"
                f"   Laggard: {rsi_min_ticker} ({sector_rsis[rsi_min_ticker]:.1f})\n"
                f"   SMH-XLP gap: {smh_xlp_gap:+.1f}\n"
                f"   Watch for contagion from laggards to leaders", 'warning'))
    
    # IGV mean reversion
    if 'IGV' in indicators:
        igv = indicators['IGV']
        if igv['rsi10'] < 25:
            alerts.append(('IGV MEAN REVERSION',
                f"IGV RSI={igv['rsi10']:.1f} < 25 | Software oversold\n"
                f"   83% win at 10d, +4.96% avg | n=12\n"
                f"   Buy IGV or QLD for leveraged exposure", 'buy'))
    
    # =========================================================================
    # SIGNAL GROUP 15: COMMODITY PLAYBOOK (Manual Hold Alerts)
    # =========================================================================
    if 'PALL' in indicators:
        pall = indicators['PALL']
        pall_above_sma = pall['price'] > pall['sma200'] if pall['sma200'] > 0 else False
        
        if pall['rsi10'] < 25 and pall_above_sma:
            alerts.append(('PALL OVERSOLD (PLAYBOOK)',
                f"PALL RSI={pall['rsi10']:.1f} < 25 AND above SMA(200)\n"
                f"   MANUAL 10-DAY HOLD: 86% win, +10.06% avg | n=14\n"
                f"   SMA(200) filter critical. NOT a Composer signal.", 'buy'))
        elif pall['rsi10'] < 30 and pall_above_sma:
            alerts.append(('PALL WATCH (PLAYBOOK)',
                f"PALL RSI={pall['rsi10']:.1f} < 30 AND above SMA(200)\n"
                f"   MANUAL 10-DAY HOLD: 71% win, +4.95% avg | n=35", 'buy'))
    
    if 'PDBC' in indicators:
        pdbc = indicators['PDBC']
        pdbc_above_sma = pdbc['price'] > pdbc['sma200'] if pdbc['sma200'] > 0 else False
        
        if pdbc['rsi10'] < 30 and pdbc_above_sma:
            alerts.append(('PDBC PULLBACK (PLAYBOOK)',
                f"PDBC RSI={pdbc['rsi10']:.1f} < 30 AND above SMA(200)\n"
                f"   MANUAL 10-DAY HOLD: 82% win, +1.82% avg | n=33\n"
                f"   Pullback-in-uptrend. NOT a Composer signal.", 'buy'))
    
    # =========================================================================
    # SIGNAL GROUP 16: KOLD OIL REVERSAL
    # =========================================================================
    if 'XLE' in indicators:
        xle = indicators['XLE']
        
        if xle['rsi10'] > 79:
            usdu_amp = ""
            if 'USDU' in indicators and indicators['USDU']['rsi10'] > 60:
                usdu_amp = f"\n   USDU amplifier ACTIVE (RSI={indicators['USDU']['rsi10']:.1f}): KOLD 5d +2.63% avg (n=21)"
            
            alerts.append(('KOLD SIGNAL',
                f"XLE RSI={xle['rsi10']:.1f} > 79\n"
                f"   -> KOLD 1d: 59% win, +1.38% avg (post-2020: +2.39%){usdu_amp}", 'hedge'))
        
        if xle['rsi10'] > 79 and 'XLF' in indicators and indicators['XLF']['rsi10'] > 79:
            alerts.append(('CYCLICAL EUPHORIA TOP',
                f"XLE RSI={xle['rsi10']:.1f} + XLF RSI={indicators['XLF']['rsi10']:.1f} > 79\n"
                f"   -> UVXY 1d: 71% WR, +5.91% avg | n=8\n"
                f"   -> SPY 5d: 37.5% WR, -1.01% avg", 'exit'))
    
    # =========================================================================
    # SIGNAL GROUP 17: AND COMBINATION SIGNALS
    # =========================================================================
    if 'UVXY' in indicators and 'XLU' in indicators:
        if indicators['UVXY']['rsi10'] > 79 and indicators['XLU']['rsi10'] < 30:
            alerts.append(('CRISIS RECOVERY SIGNAL',
                f"UVXY RSI={indicators['UVXY']['rsi10']:.1f} >79 + XLU RSI={indicators['XLU']['rsi10']:.1f} <30\n"
                f"   -> SOXL: 91.7% WR, +13.11% avg | 6 crisis episodes\n"
                f"   n=6 episodes - high conviction but small sample", 'buy'))
    
    if 'BTAL' in indicators and 'QQQ' in indicators:
        if indicators['BTAL']['rsi10'] > 75 and indicators['QQQ']['rsi10'] < 30:
            alerts.append(('QUALITY-TO-GROWTH ROTATION',
                f"BTAL RSI={indicators['BTAL']['rsi10']:.1f} >75 + QQQ RSI={indicators['QQQ']['rsi10']:.1f} <30\n"
                f"   -> SOXL: n=40, 18 episodes\n"
                f"   Peak quality positioning + oversold tech = rotation", 'buy'))
    
    if 'VIXM' in indicators and 'SPY' in indicators:
        if indicators['VIXM']['rsi10'] < 30 and indicators['SPY']['rsi10'] > 79:
            alerts.append(('CALM-MARKET HEDGE',
                f"VIXM RSI={indicators['VIXM']['rsi10']:.1f} <30 + SPY RSI={indicators['SPY']['rsi10']:.1f} >79\n"
                f"   Calm vol + overbought equity = snap-back risk", 'hedge'))
    
    # Managed futures status
    mf_status = {}
    for mf in ['CTA', 'DBMF', 'KMLM']:
        if mf in indicators:
            mf_status[mf] = indicators[mf]
    status['managed_futures'] = mf_status
    

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
    
    leveraged_tickers = ['NAIL', 'CURE', 'FAS', 'LABU', 'TQQQ', 'SOXL', 'TECL', 'DRN']
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
    
    # Oil Supply Shock Dashboard
    body += f"""
{'='*70}
OIL SUPPLY SHOCK STATUS
{'='*70}
"""
    if 'UCO' in indicators and 'USO' in indicators:
        uco_i = indicators.get('UCO', {})
        uso_i = indicators.get('USO', {})
        usdu_i = indicators.get('USDU', {})
        
        uco_rsi = uco_i.get('rsi10', 0)
        uso_rsi = uso_i.get('rsi10', 0)
        usdu_rsi = usdu_i.get('rsi10', 0)
        
        # Signal level
        if uco_rsi > 85 and usdu_rsi > 60:
            oil_signal = "EXTREME - HEDGE NOW"
        elif uco_rsi > 82 and usdu_rsi > 55:
            oil_signal = "STRONG - Reduce equity"
        elif uco_rsi > 79 and usdu_rsi > 55:
            oil_signal = "WARNING - Caution"
        elif uco_rsi > 75 and usdu_rsi > 50:
            oil_signal = "Watch"
        elif uco_rsi > 79 and usdu_rsi < 40:
            oil_signal = "DEMAND-DRIVEN (Bullish)"
        else:
            oil_signal = "Inactive"
        
        body += f"""Signal Level:     {oil_signal}
UCO:  ${uco_i.get('price',0):.2f}  RSI={uco_rsi:.1f}
USO:  ${uso_i.get('price',0):.2f}  RSI={uso_rsi:.1f}
USDU: ${usdu_i.get('price',0):.2f}  RSI={usdu_rsi:.1f}
Consecutive Days (UCO>79+USDU>55):  {status.get('oil_consec_uco79_usdu55', 0)}
Consecutive Days (UCO>82+USDU>55):  {status.get('oil_consec_uco82_usdu55', 0)}
Consecutive Days (USO>75+USDU>55):  {status.get('oil_consec_uso75_usdu55', 0)}

Duration Playbook (UCO>79+USDU>55):
  >=1d: SPY 5d 30% WR  | BTAL 78% WR  | UVXY +3.93%
  >=2d: SPY 5d  8% WR  | BTAL 92% WR  | UVXY +9.34%
  >=3d: SPY 5d 11% WR  | BTAL 100% WR | UVXY +9.95%
  Hedge basket (33/33/33 UVXY/BTAL/SQQQ): 3/3 episodes, +7.78% avg
  EXIT: When UCO RSI drops <79 or basket negative after D+2
"""
    
    # Dispersion Dashboard
    body += f"""
{'='*70}
DISPERSION & REGIME DASHBOARD
{'='*70}
"""
    disp_range = status.get('dispersion_range', 0)
    leader = status.get('dispersion_leader', 'N/A')
    laggard = status.get('dispersion_laggard', 'N/A')
    smh_xlp = status.get('smh_xlp_gap', 0)
    
    vix_price = indicators.get('VIX', {}).get('price', 0)
    if vix_price < 20:
        vix_regime = "LOW (<20) -> GLD is primary dispersion hedge"
    elif vix_price < 25:
        vix_regime = "ELEVATED (20-25) -> CTA/KMLM are primary hedges"
    else:
        vix_regime = "HIGH (>25) -> Possible bottom, CTA/KMLM may whipsaw"
    
    body += f"""Sector RSI Range:  {disp_range:.1f} {'(EXTREME >45)' if disp_range > 45 else '(elevated >35)' if disp_range > 35 else ''}
Leader:            {leader}
Laggard:           {laggard}
SMH-XLP Gap:       {smh_xlp:+.1f} {'(DEFENSIVE ROTATION <-20)' if smh_xlp < -20 else ''}
VIX:               {vix_price:.1f} - {vix_regime}
"""
    
    # Managed Futures Status
    mf = status.get('managed_futures', {})
    if mf:
        body += f"""
{'='*70}
MANAGED FUTURES & ALTERNATIVES
{'='*70}
"""
        body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}\n"
        body += "-"*50 + "\n"
        for ticker in ['CTA', 'DBMF', 'KMLM', 'BTAL', 'VIXM', 'IGV']:
            if ticker in indicators:
                ind = indicators[ticker]
                price = f"${ind['price']:.2f}"
                rsi = f"{ind['rsi10']:.1f}"
                pct = f"{ind.get('pct_above_sma200', 0):+.1f}%"
                body += f"{ticker:<10} {price:>12} {rsi:>10} {pct:>12}\n"
    
    # Commodity Watchlist
    body += f"""
{'='*70}
COMMODITY WATCHLIST (Playbook Signals)
{'='*70}
"""
    body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}  Signal\n"
    body += "-"*65 + "\n"
    for ticker in ['PALL', 'PDBC', 'USO', 'UCO', 'BOIL', 'GLD']:
        if ticker in indicators:
            ind = indicators[ticker]
            price = f"${ind['price']:.2f}"
            rsi = f"{ind['rsi10']:.1f}"
            pct = f"{ind.get('pct_above_sma200', 0):+.1f}%"
            above_sma = ind['price'] > ind['sma200'] if ind['sma200'] > 0 else False
            
            signal = ""
            if ticker == 'PALL':
                if ind['rsi10'] < 25 and above_sma:
                    signal = "BUY (10d hold)"
                elif ind['rsi10'] < 30 and above_sma:
                    signal = "Watch (<25)"
            elif ticker == 'PDBC':
                if ind['rsi10'] < 30 and above_sma:
                    signal = "BUY (10d hold)"
            
            body += f"{ticker:<10} {price:>12} {rsi:>10} {pct:>12}  {signal}\n"
    
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
        'XLP', 'XLU', 'XLV', 'XLY',
        # Safe Havens & Macro
        'GLD', 'TLT', 'HYG', 'LQD', 'TMV',
        'USDU', 'UCO', 'USO', 'BOIL',
        # Volatility
        'UVXY', 'VIXM', '^VIX',
        # International
        'EDC', 'YINN', 'KORU', 'EURL', 'INDL',
        # Crypto
        'BTC-USD',
        # Individual Stocks
        'AMD', 'NVDA',
        # 3x Leveraged ETFs
        'NAIL', 'CURE', 'FAS', 'LABU',
        'TQQQ', 'SOXL', 'TECL', 'DRN',
        # Inverse/Hedge ETFs
        'SOXS', 'SQQQ', 'BTAL',
        # Style/Factor ETFs
        'VOOV', 'VOOG', 'VTV', 'QQQE',
        # Energy/Financials
        'XLE', 'XLF',
        # Managed Futures
        'CTA', 'DBMF', 'KMLM',
        # Software/Dispersion
        'IGV',
        # Commodities (Playbook)
        'PALL', 'PDBC',
    ]
    
    print("Downloading market data...")
    data = download_data(tickers)
    # Rename ^VIX to VIX for cleaner references
    if '^VIX' in data:
        data['VIX'] = data.pop('^VIX')
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
