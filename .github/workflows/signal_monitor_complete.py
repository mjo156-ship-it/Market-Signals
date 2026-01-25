#!/usr/bin/env python3
"""
Comprehensive Market Signal Monitor v2.0
========================================
Monitors all backtested trading signals and sends alerts.

SCHEDULE: Two emails daily (weekdays)
- 3:15 PM ET: Pre-close preview
- 4:05 PM ET: Market close confirmation

Signals monitored:
- SOXL/SMH long-term accumulation & exit signals
- GLD/USDU combo (TQQQ, UPRO, AMD, NVDA)
- Triple signal: GLD + USDU + XLP
- Defensive rotation signals
- Volatility hedge signals
- SOXS short signals
- BTC signals
- UPRO entry/exit signals
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
# CONFIGURATION - Read from environment variables
# =============================================================================
SENDER_EMAIL = os.environ.get('SENDER_EMAIL', '')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD', '')
RECIPIENT_EMAIL = os.environ.get('RECIPIENT_EMAIL', '')
PHONE_EMAIL = os.environ.get('PHONE_EMAIL', '')  # Optional SMS

# Check if this is pre-close (3:15) or post-close (4:05) run
# Pass 'preclose' as command line argument for 3:15 run
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

def calculate_sma(prices, period):
    """Calculate Simple Moving Average"""
    return prices.rolling(window=period).mean()

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=period, adjust=False).mean()

def download_data(tickers, period='2y'):
    """Download data for multiple tickers"""
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False)
            if len(df) > 0:
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
        close = df['Close']
        indicators[ticker] = {
            'price': close.iloc[-1],
            'rsi10': calculate_rsi_wilder(close, 10).iloc[-1],
            'rsi50': calculate_rsi_wilder(close, 50).iloc[-1],
            'sma200': calculate_sma(close, 200).iloc[-1],
            'sma50': calculate_sma(close, 50).iloc[-1],
            'ema21': calculate_ema(close, 21).iloc[-1],
        }
        # Calculate % above SMA200
        if indicators[ticker]['sma200'] > 0:
            indicators[ticker]['pct_above_sma200'] = (close.iloc[-1] / indicators[ticker]['sma200'] - 1) * 100
        else:
            indicators[ticker]['pct_above_sma200'] = 0
    
    # Store indicators for status report
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
        if smh['sma50'] < smh['sma200']:
            alerts.append(('🔴 DEATH CROSS', f"SMH SMA(50) below SMA(200) - Bearish trend", 'exit'))
        
        # BUY Signals - Days below SMA200
        if 'SMH' in data:
            smh_df = data['SMH']
            close = smh_df['Close']
            sma200 = calculate_sma(close, 200)
            below_sma = close < sma200
            
            # Count consecutive days below
            days_below = 0
            for i in range(len(below_sma)-1, -1, -1):
                if below_sma.iloc[i]:
                    days_below += 1
                else:
                    break
            
            if days_below >= 100:
                if smh['rsi50'] < 45:
                    alerts.append(('🟢 SOXL STRONG BUY', f"SMH {days_below} days below SMA(200) + RSI(50)={smh['rsi50']:.1f} < 45 | 97% win, +81% avg", 'buy'))
                else:
                    alerts.append(('🟢 SOXL ACCUMULATE', f"SMH {days_below} days below SMA(200) | 85% win, +54% avg", 'buy'))
            
            status['smh_days_below_sma200'] = days_below
    
    # =========================================================================
    # SIGNAL GROUP 2: GLD/USDU Combo Signals (MAJOR DISCOVERY)
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
        
        # Individual GLD overbought (weaker but still valid)
        elif gld['rsi10'] > 79:
            alerts.append(('🟢 GLD OVERBOUGHT', 
                f"GLD RSI={gld['rsi10']:.1f} > 79 → Long TQQQ: 72% win, +3.2% avg (5d)", 'buy'))
    
    # =========================================================================
    # SIGNAL GROUP 3: Defensive Rotation (XLP/XLU/XLV)
    # =========================================================================
    defensive_ob = False
    for ticker in ['XLP', 'XLU', 'XLV']:
        if ticker in indicators and indicators[ticker]['rsi10'] > 79:
            defensive_ob = True
            break
    
    if defensive_ob:
        spy_ob = 'SPY' in indicators and indicators['SPY']['rsi10'] > 79
        qqq_ob = 'QQQ' in indicators and indicators['QQQ']['rsi10'] > 79
        vix_ok = '^VIX' in indicators and indicators['^VIX']['price'] < 30
        
        if not spy_ob and not qqq_ob and (vix_ok or '^VIX' not in indicators):
            alerts.append(('🟢 DEFENSIVE ROTATION', 
                f"Defensive sector overbought, SPY/QQQ not → Long TQQQ 20d: 70% win, +5% avg", 'buy'))
    
    # =========================================================================
    # SIGNAL GROUP 4: Volatility Hedge Signals
    # =========================================================================
    if 'QQQ' in indicators:
        qqq = indicators['QQQ']
        spy_ob = 'SPY' in indicators and indicators['SPY']['rsi10'] > 79
        
        # QQQ overbought → UVXY
        if qqq['rsi10'] > 79:
            alerts.append(('🟡 VOL HEDGE', 
                f"QQQ RSI={qqq['rsi10']:.1f} > 79 → Long UVXY 5d: 67% win, +33% CAGR", 'hedge'))
        
        # QQQ oversold → TQQQ dip buy
        if qqq['rsi10'] < 20:
            alerts.append(('🟢 QQQ DIP BUY', 
                f"QQQ RSI={qqq['rsi10']:.1f} < 20 → Long TQQQ 5d: 69% win, +26% CAGR", 'buy'))
    
    # =========================================================================
    # SIGNAL GROUP 5: SOXS Short Signals (Short Semis)
    # =========================================================================
    if 'SMH' in indicators and 'USDU' in indicators:
        smh = indicators['SMH']
        usdu = indicators['USDU']
        
        # SMH overbought + Dollar strong → SOXS
        if smh['rsi10'] > 79 and usdu['rsi10'] > 70:
            alerts.append(('🔴 SOXS SIGNAL', 
                f"SMH RSI={smh['rsi10']:.1f} > 79 AND USDU RSI={usdu['rsi10']:.1f} > 70\n"
                f"   → Long SOXS 5d: 100% win, +9.5% avg", 'short'))
        
        # SMH overbought + Small caps weak
        if 'IWM' in indicators and smh['rsi10'] > 79 and indicators['IWM']['rsi10'] < 50:
            alerts.append(('🔴 SOXS DIVERGENCE', 
                f"SMH RSI={smh['rsi10']:.1f} > 79 AND IWM RSI={indicators['IWM']['rsi10']:.1f} < 50\n"
                f"   → Long SOXS 5d: 86% win, +6.9% avg (narrow leadership)", 'short'))
    
    # =========================================================================
    # SIGNAL GROUP 6: Credit Signals
    # =========================================================================
    if 'LQD' in indicators:
        lqd = indicators['LQD']
        if lqd['rsi10'] > 79:
            alerts.append(('🟢 CREDIT SIGNAL', 
                f"LQD RSI={lqd['rsi10']:.1f} > 79 → Long TQQQ 10d: 84% win", 'buy'))
    
    # =========================================================================
    # SIGNAL GROUP 7: EM/China Volatility Signal
    # =========================================================================
    edc_ob = 'EDC' in indicators and indicators['EDC']['rsi10'] > 79
    yinn_ob = 'YINN' in indicators and indicators['YINN']['rsi10'] > 79
    
    if edc_ob or yinn_ob:
        alerts.append(('🟡 EM OVERBOUGHT → VOL', 
            f"EDC/YINN overbought → Long UVXY 5d: 77% win, +6% avg", 'hedge'))
    
    # =========================================================================
    # SIGNAL GROUP 8: Oil Short Signal
    # =========================================================================
    if 'UCO' in indicators:
        uco = indicators['UCO']
        pct_above_ema21 = (uco['price'] / uco['ema21'] - 1) * 100
        if pct_above_ema21 > 15:
            alerts.append(('🔴 OIL SHORT', 
                f"UCO {pct_above_ema21:.1f}% above EMA(21) → Long SCO 1d: 76% win", 'short'))
    
    # =========================================================================
    # SIGNAL GROUP 9: BTC Signals
    # =========================================================================
    if 'BTC-USD' in indicators:
        btc = indicators['BTC-USD']
        
        # BTC momentum (overbought = bullish for crypto!)
        if btc['rsi10'] > 79:
            alerts.append(('🟢 BTC MOMENTUM', 
                f"BTC RSI={btc['rsi10']:.1f} > 79 → Hold/Add BTC: 67% win, +5.2% avg (5d)", 'buy'))
        
        # BTC dip buy with low vol
        if btc['rsi10'] < 30:
            uvxy_low = 'UVXY' in indicators and indicators['UVXY']['rsi10'] < 40
            if uvxy_low:
                alerts.append(('🟢 BTC DIP BUY', 
                    f"BTC RSI={btc['rsi10']:.1f} < 30 AND UVXY < 40 → Buy BTC: 77% win, +4.1% avg (5d)", 'buy'))
            else:
                alerts.append(('🟡 BTC OVERSOLD', 
                    f"BTC RSI={btc['rsi10']:.1f} < 30 (wait for UVXY < 40 for better signal)", 'watch'))
    
    # =========================================================================
    # SIGNAL GROUP 10: UPRO Entry/Exit Signals
    # =========================================================================
    if 'SPY' in indicators:
        spy = indicators['SPY']
        
        # UPRO Exit
        if spy['rsi10'] > 85:
            alerts.append(('🔴 UPRO EXIT', 
                f"SPY RSI={spy['rsi10']:.1f} > 85 → Trim/Exit UPRO: Only 36% win, -3.5% avg (5d)", 'exit'))
        elif spy['rsi10'] > 82:
            alerts.append(('🟡 UPRO CAUTION', 
                f"SPY RSI={spy['rsi10']:.1f} > 82 → Watch UPRO: 49% win at 5d", 'warning'))
        
        # UPRO Add
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
    # SIGNAL GROUP 11: AMD/NVDA Specific
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
        # Group alerts by type
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
    
    # Status summary
    body += f"""
{'='*70}
CURRENT INDICATOR STATUS
{'='*70}

"""
    
    indicators = status.get('indicators', {})
    
    # Key indicators table
    key_tickers = ['SPY', 'QQQ', 'SMH', 'GLD', 'USDU', 'XLP', 'UVXY', 'BTC-USD', 'AMD', 'NVDA']
    body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}\n"
    body += "-"*50 + "\n"
    
    for ticker in key_tickers:
        if ticker in indicators:
            ind = indicators[ticker]
            price = f"${ind['price']:.2f}" if ind['price'] < 1000 else f"${ind['price']:,.0f}"
            rsi = f"{ind['rsi10']:.1f}"
            pct = f"{ind.get('pct_above_sma200', 0):+.1f}%" if 'pct_above_sma200' in ind else "N/A"
            body += f"{ticker:<10} {price:>12} {rsi:>10} {pct:>12}\n"
    
    # SMH specific status
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
⚠️ NOTE: This is a PRE-CLOSE preview. Signals may change by market close.
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
        
        # Also send SMS if configured
        if PHONE_EMAIL:
            sms_msg = MIMEText(subject[:160])
            sms_msg['From'] = SENDER_EMAIL
            sms_msg['To'] = PHONE_EMAIL
            sms_msg['Subject'] = ''
            
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(sms_msg)
            server.quit()
        
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
    
    # Tickers to monitor
    tickers = [
        'SMH', 'SPY', 'QQQ', 'IWM',           # Major indices
        'XLP', 'XLU', 'XLV',                   # Defensive sectors
        'GLD', 'TLT', 'HYG', 'LQD',           # Safe havens & credit
        'USDU', 'UCO',                         # Dollar & oil
        'UVXY',                                # Volatility
        'EDC', 'YINN',                         # EM/China
        'BTC-USD',                             # Bitcoin
        'AMD', 'NVDA',                         # Individual stocks
        '^VIX',                                # VIX
    ]
    
    # Download data
    print("Downloading market data...")
    data = download_data(tickers)
    print(f"Downloaded data for {len(data)} tickers")
    
    # Check signals
    alerts, status = check_signals(data)
    
    # Determine email subject
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
    
    # Format and send email
    body = format_email(alerts, status, IS_PRECLOSE)
    send_email(subject, body)
    
    # Print summary
    print(f"\n{len(alerts)} signal(s) detected")
    for title, msg, _ in alerts:
        print(f"  {title}")

if __name__ == "__main__":
    main()
