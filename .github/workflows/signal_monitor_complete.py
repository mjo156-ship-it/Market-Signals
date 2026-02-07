#!/usr/bin/env python3
"""
Comprehensive Market Signal Monitor v3.0
========================================
Monitors all backtested trading signals and sends alerts.

SCHEDULE: Two emails daily (weekdays)
- 3:15 PM ET: Pre-close preview
- 4:05 PM ET: Market close confirmation

CHANGELOG v3.0 (2026-02-07):
- Added VIXM confirmation filter for TQQQ/SOXL dip buy signals
  - VIXM 10d ROC > +5% boosts TQQQ RSI<30 from 67% to 76% win rate
  - VIXM RSI > 60 confirms elevated vol regime for dip buying
  - VIXM RSI < 40 flags danger (vol not confirming the dip)
- Added BOIL/KOLD natural gas signals
  - BOIL RSI < 21 → Buy BOIL (52% 5d, 69% 20d, +11.4%)
  - BOIL RSI > 79 OR 5d rally > 30% → Exit BOIL / Enter KOLD (70% win, +7.2%)
  - BOIL 5d rally > 40% → Strong KOLD (88% win)
  - Geopolitical: UVXY > 70 + UCO > 60 → Buy BOIL (73% win, +23.5%)
- Active signals now lead the email body
- BOIL/KOLD reference section at end unless signals are active
- Restored comprehensive ETF status table (58 tickers)
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
        if len(df) < 20:
            continue
        try:
            close = df['Close']
            price = safe_float(close.iloc[-1])
            rsi10 = safe_float(calculate_rsi_wilder(close, 10).iloc[-1])

            indicators[ticker] = {
                'price': price,
                'rsi10': rsi10,
            }

            if len(df) >= 50:
                indicators[ticker]['rsi50'] = safe_float(calculate_rsi_wilder(close, 50).iloc[-1])
                indicators[ticker]['sma50'] = safe_float(close.rolling(window=50).mean().iloc[-1])
                indicators[ticker]['ema21'] = safe_float(close.ewm(span=21, adjust=False).mean().iloc[-1])
            else:
                indicators[ticker]['rsi50'] = 0.0
                indicators[ticker]['sma50'] = 0.0
                indicators[ticker]['ema21'] = 0.0

            if len(df) >= 200:
                indicators[ticker]['sma200'] = safe_float(close.rolling(window=200).mean().iloc[-1])
            else:
                indicators[ticker]['sma200'] = 0.0

            sma200 = indicators[ticker]['sma200']
            if sma200 > 0:
                indicators[ticker]['pct_above_sma200'] = (price / sma200 - 1) * 100
            else:
                indicators[ticker]['pct_above_sma200'] = 0

            # VIXM 10d rate of change
            if ticker == 'VIXM' and len(df) >= 11:
                price_10d_ago = safe_float(close.iloc[-11])
                if price_10d_ago > 0:
                    indicators[ticker]['roc_10d'] = (price / price_10d_ago - 1) * 100
                else:
                    indicators[ticker]['roc_10d'] = 0.0

            # BOIL 5d rally for KOLD fade signal
            if ticker == 'BOIL' and len(df) >= 6:
                price_5d_ago = safe_float(close.iloc[-6])
                if price_5d_ago > 0:
                    indicators[ticker]['rally_5d'] = (price / price_5d_ago - 1) * 100
                else:
                    indicators[ticker]['rally_5d'] = 0.0

        except Exception as e:
            print(f"Error calculating indicators for {ticker}: {e}")
            continue

    status['indicators'] = indicators

    # =========================================================================
    # SIGNAL GROUP 1: SOXL/SMH Long-Term Signals
    # =========================================================================
    if 'SMH' in indicators and indicators['SMH']['sma200'] > 0:
        smh = indicators['SMH']

        if smh['pct_above_sma200'] >= 40:
            alerts.append(('🔴 SOXL EXIT', f"SMH {smh['pct_above_sma200']:.1f}% above SMA(200) - SELL SOXL", 'exit'))
        elif smh['pct_above_sma200'] >= 35:
            alerts.append(('🟡 SOXL WARNING', f"SMH {smh['pct_above_sma200']:.1f}% above SMA(200) - Approaching sell zone", 'warning'))
        elif smh['pct_above_sma200'] >= 30:
            alerts.append(('🟡 SOXL TRIM', f"SMH {smh['pct_above_sma200']:.1f}% above SMA(200) - Consider trimming 25-50%", 'warning'))

        if smh['sma50'] > 0 and smh['sma50'] < smh['sma200']:
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
    # SIGNAL GROUP 8: AMD/NVDA Specific
    # =========================================================================
    if 'AMD' in indicators and indicators['AMD']['rsi10'] > 85:
        alerts.append(('🟡 AMD EXTENDED',
            f"AMD RSI={indicators['AMD']['rsi10']:.1f} > 85 → Consider taking profits", 'warning'))
    if 'NVDA' in indicators and indicators['NVDA']['rsi10'] > 85:
        alerts.append(('🟡 NVDA EXTENDED',
            f"NVDA RSI={indicators['NVDA']['rsi10']:.1f} > 85 → Consider taking profits", 'warning'))

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

    # =========================================================================
    # SIGNAL GROUP 13: VIXM Confirmation for TQQQ/SOXL Dip Buys
    # =========================================================================
    if 'VIXM' in indicators:
        vixm = indicators['VIXM']
        vixm_roc = vixm.get('roc_10d', 0.0)
        vixm_rsi = vixm['rsi10']
        vixm_above_sma200 = vixm.get('pct_above_sma200', 0) > 0

        if 'TQQQ' in indicators:
            tqqq = indicators['TQQQ']
            if tqqq['rsi10'] < 30:
                vixm_flags = []
                if vixm_roc > 5:
                    vixm_flags.append(f"VIXM ROC={vixm_roc:+.1f}% ✓")
                if vixm_rsi > 60:
                    vixm_flags.append(f"VIXM RSI={vixm_rsi:.0f} ✓")
                if vixm_above_sma200:
                    vixm_flags.append("VIXM>SMA200 ✓")

                if vixm_roc > 5 and vixm_rsi > 60:
                    alerts.append(('🟢🔥 TQQQ DIP + VIXM CONFIRMED',
                        f"TQQQ RSI={tqqq['rsi10']:.1f} < 30 with VIXM confirmation\n"
                        f"   {' | '.join(vixm_flags)}\n"
                        f"   → Long TQQQ: 76% win, +4.8% avg (5d) | n=34 episodes\n"
                        f"   → 20d: 74% win, +7.3% avg", 'buy'))
                elif vixm_roc > 5:
                    alerts.append(('🟢 TQQQ DIP + VIXM RISING',
                        f"TQQQ RSI={tqqq['rsi10']:.1f} < 30 + VIXM 10d ROC={vixm_roc:+.1f}%\n"
                        f"   → Long TQQQ: 76% win, +4.8% avg (5d) | n=34 episodes", 'buy'))
                elif vixm_rsi > 60:
                    alerts.append(('🟢 TQQQ DIP + VIXM ELEVATED',
                        f"TQQQ RSI={tqqq['rsi10']:.1f} < 30 + VIXM RSI={vixm_rsi:.0f} > 60\n"
                        f"   → Long TQQQ: 69% win, +3.7% avg (5d) | n=39 episodes\n"
                        f"   → 20d: 74% win, +7.6% avg", 'buy'))
                elif vixm_rsi < 40:
                    alerts.append(('🔴 TQQQ DIP - VIXM DANGER',
                        f"TQQQ RSI={tqqq['rsi10']:.1f} < 30 BUT VIXM RSI={vixm_rsi:.0f} < 40\n"
                        f"   → Vol NOT confirming dip - caution! Low sample but 0% win at 20d\n"
                        f"   → Wait for VIXM RSI > 60 for higher conviction", 'warning'))
                else:
                    alerts.append(('🟢 TQQQ OVERSOLD',
                        f"TQQQ RSI={tqqq['rsi10']:.1f} < 30 (VIXM neutral: RSI={vixm_rsi:.0f}, ROC={vixm_roc:+.1f}%)\n"
                        f"   → Baseline TQQQ: 67% win, +4.1% avg (5d) | n=141", 'buy'))

        if 'SOXL' in indicators:
            soxl = indicators['SOXL']
            if soxl['rsi10'] < 30:
                if vixm_roc > 5:
                    alerts.append(('🟢 SOXL DIP + VIXM CONFIRMED',
                        f"SOXL RSI={soxl['rsi10']:.1f} < 30 + VIXM 10d ROC={vixm_roc:+.1f}%\n"
                        f"   → Long SOXL: 70% win, +7.2% avg (5d) | n=84\n"
                        f"   → 20d: 68% win, +13.8% avg", 'buy'))
                elif vixm_rsi > 60:
                    alerts.append(('🟢 SOXL DIP + VIXM ELEVATED',
                        f"SOXL RSI={soxl['rsi10']:.1f} < 30 + VIXM RSI={vixm_rsi:.0f} > 60\n"
                        f"   → Long SOXL: 69% win, +6.9% avg (5d) | n=98\n"
                        f"   → 20d: 67% win, +14.4% avg", 'buy'))

    # =========================================================================
    # SIGNAL GROUP 14: BOIL/KOLD Natural Gas Signals
    # =========================================================================
    boil_kold_alerts = []  # Separate list - placed at end unless active

    if 'BOIL' in indicators:
        boil = indicators['BOIL']
        boil_rally_5d = boil.get('rally_5d', 0.0)

        # --- KOLD ENTRY (fade BOIL spike) ---
        if boil_rally_5d >= 50:
            boil_kold_alerts.append(('🟢🔥 KOLD STRONG ENTRY',
                f"BOIL 5d rally={boil_rally_5d:+.1f}% (>50%) + RSI={boil['rsi10']:.1f}\n"
                f"   → Long KOLD: ~100% win rate at 50%+ rally | n=10\n"
                f"   → Hold 10d or until BOIL RSI < 50", 'buy'))
        elif boil_rally_5d >= 40:
            boil_kold_alerts.append(('🟢 KOLD ENTRY',
                f"BOIL 5d rally={boil_rally_5d:+.1f}% (>40%) + RSI={boil['rsi10']:.1f}\n"
                f"   → Long KOLD: 89% win, +7.2% avg | n=18\n"
                f"   → Hold 10d or until BOIL RSI < 50", 'buy'))
        elif boil_rally_5d >= 30 or boil['rsi10'] > 79:
            trigger = []
            if boil_rally_5d >= 30:
                trigger.append(f"5d rally={boil_rally_5d:+.1f}%")
            if boil['rsi10'] > 79:
                trigger.append(f"RSI={boil['rsi10']:.1f}>79")
            boil_kold_alerts.append(('🟢 KOLD CONSIDER',
                f"BOIL {' + '.join(trigger)}\n"
                f"   → Long KOLD: 70% win, +7.2% avg | n=63\n"
                f"   → Exit BOIL if holding", 'buy'))

        # --- BOIL BUY (oversold) ---
        if boil['rsi10'] < 21:
            boil_kold_alerts.append(('🟢 BOIL OVERSOLD',
                f"BOIL RSI={boil['rsi10']:.1f} < 21 → Buy BOIL: 52% (5d), 69% (20d), +11.4% avg (20d) | n=62", 'buy'))

        # --- Geopolitical supply shock: UVXY > 70 + UCO > 60 ---
        if 'UVXY' in indicators and 'UCO' in indicators:
            if indicators['UVXY']['rsi10'] > 70 and indicators['UCO']['rsi10'] > 60:
                boil_kold_alerts.append(('🟢🔥 BOIL SUPPLY SHOCK',
                    f"UVXY RSI={indicators['UVXY']['rsi10']:.1f} > 70 + UCO RSI={indicators['UCO']['rsi10']:.1f} > 60\n"
                    f"   → Geopolitical energy signal: 73% win, +23.5% avg | n=11\n"
                    f"   → Fear + Oil strong = potential supply disruption", 'buy'))

        # --- UCO filter for KOLD quality ---
        if boil['rsi10'] > 79 and 'UCO' in indicators:
            uco_rsi = indicators['UCO']['rsi10']
            if uco_rsi > 50:
                boil_kold_alerts.append(('🟢 KOLD + UCO CONFIRM',
                    f"BOIL overbought + UCO RSI={uco_rsi:.1f} > 50\n"
                    f"   → Enhanced KOLD: 77% win | UCO>50 confirms fade", 'buy'))
            elif uco_rsi < 50:
                boil_kold_alerts.append(('🟡 KOLD UCO CAUTION',
                    f"BOIL overbought BUT UCO RSI={uco_rsi:.1f} < 50\n"
                    f"   → KOLD only 57% win when oil weak | Size down", 'warning'))

    status['boil_kold_alerts'] = boil_kold_alerts

    return alerts, status

# =============================================================================
# EMAIL FUNCTIONS
# =============================================================================
def format_email(alerts, status, is_preclose=False):
    """Format the email body"""
    now = datetime.now()
    timing = "PRE-CLOSE PREVIEW (3:15 PM)" if is_preclose else "MARKET CLOSE CONFIRMATION (4:05 PM)"

    boil_kold_alerts = status.get('boil_kold_alerts', [])
    boil_kold_active = len(boil_kold_alerts) > 0

    # Merge BOIL/KOLD into main alerts ONLY if active (they appear at top with others)
    all_alerts = alerts + boil_kold_alerts if boil_kold_active else alerts

    body = f"""
{'='*70}
MARKET SIGNAL MONITOR v3.0 - {timing}
{now.strftime('%Y-%m-%d %H:%M')} ET
{'='*70}

"""

    if all_alerts:
        buy_alerts = [a for a in all_alerts if a[2] == 'buy']
        exit_alerts = [a for a in all_alerts if a[2] in ['exit', 'short']]
        warning_alerts = [a for a in all_alerts if a[2] in ['warning', 'hedge', 'watch']]

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

    # =========================================================================
    # CURRENT INDICATOR STATUS - Key Tickers
    # =========================================================================
    indicators = status.get('indicators', {})

    body += f"""
{'='*70}
CURRENT INDICATOR STATUS
{'='*70}

"""
    key_tickers = ['SPY', 'QQQ', 'SMH', 'GLD', 'USDU', 'XLP', 'TLT', 'HYG', 'XLF', 'UVXY', 'BTC-USD', 'AMD', 'NVDA']
    body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}\n"
    body += "-"*50 + "\n"

    for ticker in key_tickers:
        if ticker in indicators:
            ind = indicators[ticker]
            price = f"${ind['price']:.2f}" if ind['price'] < 1000 else f"${ind['price']:,.0f}"
            rsi = f"{ind['rsi10']:.1f}"
            pct = f"{ind.get('pct_above_sma200', 0):+.1f}%" if ind.get('sma200', 0) > 0 else "N/A"
            body += f"{ticker:<10} {price:>12} {rsi:>10} {pct:>12}\n"

    # =========================================================================
    # VIXM STATUS
    # =========================================================================
    if 'VIXM' in indicators:
        vixm = indicators['VIXM']
        vixm_roc = vixm.get('roc_10d', 0.0)
        vixm_above = vixm.get('pct_above_sma200', 0) > 0
        regime = "ABOVE" if vixm_above else "BELOW"

        vixm_signal = "Neutral"
        if vixm_roc > 5:
            vixm_signal = "🟢 RISING (dip buy confirmation active)"
        elif vixm_roc > 0:
            vixm_signal = "Rising (not yet at +5% threshold)"
        elif vixm_roc < -5:
            vixm_signal = "Falling (vol compressing)"

        body += f"""
{'='*70}
VIXM STATUS (Mid-Term Vol - Dip Buy Confirmation)
{'='*70}
Price:          ${vixm['price']:.2f}
RSI(10):        {vixm['rsi10']:.1f}
10d ROC:        {vixm_roc:+.1f}%  (threshold: +5% for dip confirmation)
vs SMA(200):    {vixm.get('pct_above_sma200', 0):+.1f}% ({regime})
Signal:         {vixm_signal}

VIXM Dip Buy Filter Rules:
  TQQQ/SOXL RSI<30 + VIXM ROC>+5%  → Higher conviction (76%/70% win)
  TQQQ/SOXL RSI<30 + VIXM RSI>60   → Elevated vol confirms (69% win)
  TQQQ/SOXL RSI<30 + VIXM RSI<40   → DANGER: vol not confirming
"""

    # =========================================================================
    # 3x LEVERAGED ETFs
    # =========================================================================
    body += f"""
{'='*70}
3x LEVERAGED ETFs
{'='*70}
"""
    body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}  Signal\n"
    body += "-"*65 + "\n"

    leveraged_tickers = ['TQQQ', 'SOXL', 'UPRO', 'TECL', 'QLD', 'NAIL', 'CURE', 'FAS', 'LABU',
                         'HIBL', 'DUSL', 'TPOR', 'BOIL', 'UCO', 'EDC', 'YINN', 'FNGO']
    for ticker in leveraged_tickers:
        if ticker in indicators:
            ind = indicators[ticker]
            price = f"${ind['price']:.2f}"
            rsi = f"{ind['rsi10']:.1f}"
            pct = f"{ind.get('pct_above_sma200', 0):+.1f}%" if ind.get('sma200', 0) > 0 else "N/A"
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

    # =========================================================================
    # SMH/SOXL LEVELS
    # =========================================================================
    if 'SMH' in indicators and indicators['SMH'].get('sma200', 0) > 0:
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
    # COMPREHENSIVE ETF STATUS - All Monitored Tickers
    # =========================================================================
    body += f"""
{'='*70}
COMPREHENSIVE ETF STATUS - ALL TICKERS
{'='*70}
"""

    etf_groups = [
        ('Core Indices', ['SPY', 'QQQ', 'IWM', 'SMH']),
        ('Sectors', ['XLP', 'XLU', 'XLV', 'XLF', 'XLE', 'XLY', 'VOX']),
        ('Style & Factor', ['VOOV', 'VOOG', 'VTV', 'QQQE', 'IGV', 'PSI']),
        ('Safe Havens & Bonds', ['GLD', 'TLT', 'SHY', 'HYG', 'LQD', 'BOND', 'FBND', 'TMV']),
        ('Macro & Dollar', ['USDU', 'DBC', 'UCO', 'BOIL']),
        ('Volatility', ['UVXY', 'SVXY', 'VIXY', 'VIXM', 'TAIL']),
        ('3x Leveraged (Long)', ['TQQQ', 'SOXL', 'UPRO', 'TECL', 'QLD', 'NAIL', 'CURE', 'FAS', 'LABU',
                                  'HIBL', 'DUSL', 'TPOR', 'FNGO']),
        ('Inverse', ['SOXS']),
        ('International', ['EDC', 'YINN', 'EWJ']),
        ('Alternatives', ['DBMF', 'MNA', 'IAK']),
        ('Individual Stocks', ['AMD', 'NVDA', 'MSFT']),
        ('Crypto', ['BTC-USD']),
    ]

    for group_name, tickers in etf_groups:
        group_tickers = [t for t in tickers if t in indicators]
        if not group_tickers:
            continue

        body += f"\n  {group_name}\n"
        body += f"  {'Ticker':<10} {'Price':>12} {'RSI(10)':>8} {'vs SMA200':>10}  Note\n"
        body += f"  {'-'*58}\n"

        for ticker in group_tickers:
            ind = indicators[ticker]
            price = f"${ind['price']:.2f}" if ind['price'] < 10000 else f"${ind['price']:,.0f}"
            rsi = f"{ind['rsi10']:.1f}"

            if ind.get('sma200', 0) > 0:
                pct = f"{ind['pct_above_sma200']:+.1f}%"
            else:
                pct = "N/A"

            note = ""
            rsi_val = ind['rsi10']
            if rsi_val < 21:
                note = "⚡ OVERSOLD"
            elif rsi_val < 25:
                note = "↓ Oversold"
            elif rsi_val < 30:
                note = "↓ Weak"
            elif rsi_val > 85:
                note = "⚡ OVERBOUGHT"
            elif rsi_val > 79:
                note = "↑ Extended"
            elif rsi_val > 70:
                note = "↑ Strong"

            pct_val = ind.get('pct_above_sma200', 0)
            if ind.get('sma200', 0) > 0:
                if pct_val > 40:
                    note += " 🔴>40%SMA"
                elif pct_val < -20:
                    note += " 🟢<-20%SMA"

            if ticker == 'VIXM':
                roc = ind.get('roc_10d', 0)
                note += f" ROC10d={roc:+.1f}%"

            if ticker == 'BOIL':
                rally = ind.get('rally_5d', 0)
                if abs(rally) > 5:
                    note += f" 5dRally={rally:+.0f}%"

            body += f"  {ticker:<10} {price:>12} {rsi:>8} {pct:>10}  {note}\n"

    # =========================================================================
    # BOIL/KOLD REFERENCE (at end, always shown for context)
    # =========================================================================
    if 'BOIL' in indicators:
        boil = indicators['BOIL']
        boil_rally = boil.get('rally_5d', 0.0)
        active_tag = " *** SIGNALS ACTIVE - SEE TOP ***" if boil_kold_active else ""
        body += f"""
{'='*70}
BOIL/KOLD STATUS (Natural Gas){active_tag}
{'='*70}
BOIL Price:     ${boil['price']:.2f}
BOIL RSI(10):   {boil['rsi10']:.1f}
BOIL 5d Rally:  {boil_rally:+.1f}%
vs SMA(200):    {boil.get('pct_above_sma200', 0):+.1f}%

KOLD Fade Rules:
  5d rally > 50%  → KOLD ~100% win | n=10
  5d rally > 40%  → KOLD  89% win  | n=18
  5d rally > 30% OR RSI > 79 → KOLD 70% win | n=63
  UCO RSI > 50 enhances KOLD to 77% win
  UCO RSI < 50 reduces KOLD to 57% win - size down

BOIL Buy Rules:
  RSI < 21       → Buy BOIL: 52% (5d), 69% (20d) | n=62
  Geopolitical: UVXY>70 + UCO>60 → 73% win, +23.5% | n=11
"""

    # =========================================================================
    # PRE-CLOSE NOTE
    # =========================================================================
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
        # Sectors
        'XLP', 'XLU', 'XLV', 'XLF', 'XLE', 'XLY', 'VOX',
        # Style/Factor
        'VOOV', 'VOOG', 'VTV', 'QQQE', 'IGV', 'PSI',
        # Safe Havens & Bonds
        'GLD', 'TLT', 'SHY', 'HYG', 'LQD', 'BOND', 'FBND', 'TMV',
        # Macro & Dollar
        'USDU', 'DBC', 'UCO', 'BOIL',
        # Volatility
        'UVXY', 'SVXY', 'VIXY', 'VIXM', 'TAIL',
        # 3x Leveraged (Long)
        'TQQQ', 'SOXL', 'UPRO', 'TECL', 'QLD', 'NAIL', 'CURE', 'FAS', 'LABU',
        'HIBL', 'DUSL', 'TPOR', 'FNGO',
        # Inverse
        'SOXS',
        # International
        'EDC', 'YINN', 'EWJ',
        # Alternatives
        'DBMF', 'MNA', 'IAK',
        # Individual Stocks
        'AMD', 'NVDA', 'MSFT',
        # Crypto
        'BTC-USD',
    ]

    print("Downloading market data...")
    data = download_data(tickers)
    print(f"Downloaded data for {len(data)} tickers")

    alerts, status = check_signals(data)

    # Include BOIL/KOLD in alert count for subject line
    boil_kold_alerts = status.get('boil_kold_alerts', [])
    all_alerts = alerts + boil_kold_alerts

    if all_alerts:
        buy_count = len([a for a in all_alerts if a[2] == 'buy'])
        exit_count = len([a for a in all_alerts if a[2] in ['exit', 'short']])

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

    body = format_email(alerts, status, IS_PRECLOSE)
    send_email(subject, body)

    print(f"\n{len(all_alerts)} signal(s) detected")
    for title, msg, _ in all_alerts:
        print(f"  {title}")

if __name__ == "__main__":
    main()
