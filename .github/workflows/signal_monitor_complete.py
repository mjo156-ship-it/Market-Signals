#!/usr/bin/env python3
"""
Quantitative Signal Monitor
Complete monitoring of all backtested trading signals

Signals included:
- SOXL long-term accumulation (days below SMA200, RSI50)
- SOXL/SMH top signals (% above SMA200, death cross)
- Defensive rotation (XLP/XLU/XLV overbought → TQQQ)
- Volatility hedge (QQQ/SPY overbought → UVXY/VIXY)
- Dip-buy signals (QQQ/SMH oversold → TQQQ/SOXL)
- Credit signals (LQD/HYG overbought → TQQQ)
- Oil shorts (UCO overbought → SCO)
- EM/China signals (EDC/YINN overbought → UVXY)
- Gold oversold (GLD → TQQQ)

Setup instructions at bottom of file
"""

import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

# =============================================================================
# CONFIGURATION - EDIT THESE
# =============================================================================

SENDER_EMAIL = os.environ.get('SENDER_EMAIL', '')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD', '')
RECIPIENT_EMAIL = os.environ.get('RECIPIENT_EMAIL', '')

# Optional SMS (leave empty to skip)
PHONE_EMAIL = ""  # e.g., "5551234567@vtext.com"

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
    return prices.rolling(window=period).mean()

def calculate_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()

def get_data():
    """Download all required data"""
    tickers = [
        'SMH', 'QQQ', 'SPY', 'IWM',           # Equity indexes
        'XLP', 'XLU', 'XLV',                   # Defensive sectors
        'HYG', 'LQD', 'TLT',                   # Credit/Bonds
        'UCO', 'GLD',                          # Commodities
        'EDC', 'YINN',                         # EM/China
        '^VIX'                                 # Volatility
    ]
    
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="2y", progress=False)
            if not df.empty:
                df.columns = df.columns.get_level_values(0)
                data[ticker.replace('^', '')] = df['Close']
        except:
            pass
    
    return pd.DataFrame(data)

def analyze_signals(df):
    """Analyze all signals and return alerts"""
    
    alerts = []
    status_lines = []
    
    # Calculate indicators for each ticker
    indicators = {}
    
    for ticker in df.columns:
        prices = df[ticker].dropna()
        if len(prices) < 200:
            continue
            
        ind = {}
        ind['price'] = prices.iloc[-1]
        ind['prev_price'] = prices.iloc[-2]
        
        # RSI
        ind['RSI5'] = calculate_rsi_wilder(prices, 5).iloc[-1]
        ind['RSI10'] = calculate_rsi_wilder(prices, 10).iloc[-1]
        ind['RSI14'] = calculate_rsi_wilder(prices, 14).iloc[-1]
        ind['RSI50'] = calculate_rsi_wilder(prices, 50).iloc[-1]
        
        # Moving averages
        ind['SMA50'] = calculate_sma(prices, 50).iloc[-1]
        ind['SMA200'] = calculate_sma(prices, 200).iloc[-1]
        ind['EMA21'] = calculate_ema(prices, 21).iloc[-1]
        
        # Percent from SMA200
        ind['pct_SMA200'] = (ind['price'] / ind['SMA200'] - 1) * 100
        ind['pct_EMA21'] = (ind['price'] / ind['EMA21'] - 1) * 100
        
        # Days below/above SMA200
        below_sma = (prices < calculate_sma(prices, 200)).astype(int)
        days_below = 0
        for val in below_sma.iloc[::-1]:
            if val == 1:
                days_below += 1
            else:
                break
        ind['days_below_SMA200'] = days_below
        
        above_sma = (prices > calculate_sma(prices, 200)).astype(int)
        days_above = 0
        for val in above_sma.iloc[::-1]:
            if val == 1:
                days_above += 1
            else:
                break
        ind['days_above_SMA200'] = days_above
        
        # Death cross check
        sma50_series = calculate_sma(prices, 50)
        sma200_series = calculate_sma(prices, 200)
        ind['death_cross'] = (sma50_series.iloc[-1] < sma200_series.iloc[-1]) and \
                            (sma50_series.iloc[-2] >= sma200_series.iloc[-2])
        ind['golden_cross'] = (sma50_series.iloc[-1] > sma200_series.iloc[-1]) and \
                             (sma50_series.iloc[-2] <= sma200_series.iloc[-2])
        ind['below_SMA200'] = ind['price'] < ind['SMA200']
        
        indicators[ticker] = ind
    
    # =========================================================================
    # TIER 1: SOXL LONG-TERM SIGNALS (Your accumulation strategy)
    # =========================================================================
    
    if 'SMH' in indicators:
        smh = indicators['SMH']
        
        # SELL SIGNALS
        if smh['pct_SMA200'] >= 40:
            alerts.append("🔴 SOXL SELL: SMH 40%+ above SMA(200)!")
            alerts.append(f"   Currently: {smh['pct_SMA200']:.1f}% | Historical 3m win: 26%")
            alerts.append("   → EXIT most/all SOXL position")
        elif smh['pct_SMA200'] >= 35:
            alerts.append("🟡 SOXL WARNING: SMH 35%+ above SMA(200)")
            alerts.append(f"   Currently: {smh['pct_SMA200']:.1f}% | Approaching 40% sell zone")
        elif smh['pct_SMA200'] >= 30:
            alerts.append("🟡 SOXL TRIM: SMH 30%+ above SMA(200)")
            alerts.append(f"   Currently: {smh['pct_SMA200']:.1f}% | Consider reducing 25-50%")
        
        if smh['death_cross']:
            alerts.append("🔴 SOXL SELL: Death Cross (SMA50 < SMA200)!")
            alerts.append("   Historical 3m win: 43% | → EXIT position")
        
        if smh['below_SMA200'] and not indicators['SMH'].get('prev_below', True):
            alerts.append("🔴 SOXL WARNING: Price crossed below SMA(200)")
        
        # BUY SIGNALS
        if smh['days_below_SMA200'] >= 100:
            alerts.append(f"🟢 SOXL ACCUMULATE: {smh['days_below_SMA200']} days below SMA(200)")
            alerts.append("   Historical 6m win: 85% | → Accumulate SOXL")
            if smh['RSI50'] < 45:
                alerts.append("   + RSI(50) < 45 = STRONG BUY (97% win rate)")
        elif smh['days_below_SMA200'] >= 50:
            alerts.append(f"🟡 SOXL WATCH: {smh['days_below_SMA200']} days below SMA(200)")
            alerts.append("   Approaching 100-day accumulation signal")
        
        if smh['RSI50'] < 40 and smh['days_below_SMA200'] < 50:
            alerts.append(f"🟡 SOXL ALERT: RSI(50) = {smh['RSI50']:.1f} (oversold)")
            alerts.append("   Better when combined with days below SMA200")
    
    # =========================================================================
    # TIER 2: DEFENSIVE ROTATION (XLP/XLU/XLV → TQQQ)
    # =========================================================================
    
    xlp_ob = indicators.get('XLP', {}).get('RSI10', 0) > 79
    xlu_ob = indicators.get('XLU', {}).get('RSI10', 0) > 79
    xlv_ob = indicators.get('XLV', {}).get('RSI10', 0) > 79
    spy_ob = indicators.get('SPY', {}).get('RSI10', 0) > 79
    qqq_ob = indicators.get('QQQ', {}).get('RSI10', 0) > 79
    
    vix = indicators.get('VIX', {}).get('price', 20)
    
    defensives_ob = sum([xlp_ob, xlu_ob, xlv_ob])
    
    if defensives_ob >= 1 and not spy_ob and not qqq_ob and vix < 30:
        ob_names = []
        if xlp_ob: ob_names.append(f"XLP={indicators['XLP']['RSI10']:.0f}")
        if xlu_ob: ob_names.append(f"XLU={indicators['XLU']['RSI10']:.0f}")
        if xlv_ob: ob_names.append(f"XLV={indicators['XLV']['RSI10']:.0f}")
        
        alerts.append(f"🟢 DEFENSIVE ROTATION: {', '.join(ob_names)} RSI(10) > 79")
        alerts.append(f"   SPY/QQQ not overbought, VIX={vix:.1f}")
        alerts.append("   → Long TQQQ, hold 20 days | Win: 70%, Avg: +5%")
    
    # =========================================================================
    # TIER 3: VOLATILITY HEDGE (QQQ/SPY overbought → UVXY/VIXY)
    # =========================================================================
    
    if 'QQQ' in indicators:
        qqq = indicators['QQQ']
        spy = indicators.get('SPY', {})
        
        if qqq.get('RSI10', 0) > 79:
            alerts.append(f"🟢 VOL SIGNAL: QQQ RSI(10) = {qqq['RSI10']:.1f} > 79")
            alerts.append("   → Long UVXY 5 days | CAGR: +33%, Win: 67%")
        elif spy.get('RSI10', 0) > 79 and qqq.get('RSI10', 0) <= 79:
            alerts.append(f"🟢 VOL SIGNAL: SPY RSI(10) = {spy['RSI10']:.1f} > 79 (QQQ not)")
            alerts.append("   → Long VIXY 5 days")
    
    # =========================================================================
    # TIER 4: DIP-BUY SIGNALS (Oversold → TQQQ/SOXL)
    # =========================================================================
    
    if 'QQQ' in indicators:
        qqq = indicators['QQQ']
        
        if qqq.get('RSI5', 50) < 20:
            alerts.append(f"🟢 DIP-BUY: QQQ RSI(5) = {qqq['RSI5']:.1f} < 20")
            alerts.append("   → Long TQQQ 5 days | CAGR: +26%, Win: 69%")
        elif qqq.get('RSI10', 50) < 25:
            alerts.append(f"🟢 DIP-BUY: QQQ RSI(10) = {qqq['RSI10']:.1f} < 25")
            alerts.append("   → Long TQQQ 5 days | CAGR: +20%, Win: 76%")
    
    if 'SMH' in indicators:
        smh = indicators['SMH']
        
        if smh.get('RSI5', 50) < 20:
            alerts.append(f"🟢 SEMI DIP-BUY: SMH RSI(5) = {smh['RSI5']:.1f} < 20")
            alerts.append("   → Long SOXL 10 days | CAGR: +21%, Win: 59%")
    
    # =========================================================================
    # TIER 5: CREDIT SIGNALS (LQD/HYG overbought → TQQQ)
    # =========================================================================
    
    if 'LQD' in indicators:
        lqd = indicators['LQD']
        if lqd.get('RSI10', 0) > 79:
            alerts.append(f"🟢 CREDIT RISK-ON: LQD RSI(10) = {lqd['RSI10']:.1f} > 79")
            alerts.append("   → Long TQQQ 10 days | CAGR: +19%, Win: 84%")
    
    if 'HYG' in indicators:
        hyg = indicators['HYG']
        if hyg.get('RSI10', 0) > 79:
            alerts.append(f"🟢 CREDIT RISK-ON: HYG RSI(10) = {hyg['RSI10']:.1f} > 79")
            alerts.append("   → Long TQQQ 10 days | Win: 79%")
    
    # =========================================================================
    # TIER 6: OIL SHORT SIGNALS (UCO overbought → SCO)
    # =========================================================================
    
    if 'UCO' in indicators:
        uco = indicators['UCO']
        
        if uco.get('pct_EMA21', 0) > 15:
            alerts.append(f"🟢 OIL SHORT: UCO {uco['pct_EMA21']:.1f}% above EMA(21)")
            alerts.append("   → Long SCO 1 day | Win: 76%")
        elif uco.get('RSI10', 0) > 85:
            alerts.append(f"🟢 OIL SHORT: UCO RSI(10) = {uco['RSI10']:.1f} > 85")
            alerts.append("   → Long SCO 1 day | Win: 70%")
    
    if 'TLT' in indicators:
        tlt = indicators['TLT']
        if tlt.get('RSI10', 0) > 79:
            alerts.append(f"🟢 OIL SHORT: TLT RSI(10) = {tlt['RSI10']:.1f} > 79 (flight to safety)")
            alerts.append("   → Long SCO 10 days | CAGR: +19%")
    
    # =========================================================================
    # TIER 7: EM/CHINA SIGNALS (Overbought → UVXY)
    # =========================================================================
    
    edc_ob = indicators.get('EDC', {}).get('RSI10', 0) > 79
    yinn_ob = indicators.get('YINN', {}).get('RSI10', 0) > 79
    
    if edc_ob and yinn_ob:
        alerts.append("🟢 EM/CHINA FROTHY: EDC AND YINN RSI(10) > 79")
        alerts.append("   → Long UVXY 1-2 days | Win: 77%")
    elif edc_ob or yinn_ob:
        which = "EDC" if edc_ob else "YINN"
        val = indicators.get(which, {}).get('RSI10', 0)
        alerts.append(f"🟡 EM/CHINA WATCH: {which} RSI(10) = {val:.1f} > 79")
        alerts.append("   → Long UVXY 3 days | Win: 57%")
    
    # =========================================================================
    # TIER 8: GOLD OVERSOLD (GLD → TQQQ)
    # =========================================================================
    
    if 'GLD' in indicators:
        gld = indicators['GLD']
        if gld.get('RSI10', 50) < 21:
            alerts.append(f"🟢 GOLD OVERSOLD: GLD RSI(10) = {gld['RSI10']:.1f} < 21")
            alerts.append("   → Long TQQQ 10 days | CAGR: +19%, Win: 70%")
    
    # =========================================================================
    # BUILD STATUS SUMMARY
    # =========================================================================
    
    status_lines.append("=" * 70)
    status_lines.append(f"DAILY SIGNAL STATUS - {datetime.now().strftime('%Y-%m-%d')}")
    status_lines.append("=" * 70)
    status_lines.append("")
    
    # SOXL Status
    if 'SMH' in indicators:
        smh = indicators['SMH']
        status_lines.append("SOXL/SMH STATUS:")
        status_lines.append(f"  Price: ${smh['price']:.2f}")
        status_lines.append(f"  % Above SMA(200): {smh['pct_SMA200']:+.1f}%")
        status_lines.append(f"  RSI(50): {smh['RSI50']:.1f}")
        status_lines.append(f"  Days Above SMA200: {smh['days_above_SMA200']}")
        status_lines.append(f"  Days Below SMA200: {smh['days_below_SMA200']}")
        status_lines.append(f"  30% Trim Level: ${smh['SMA200'] * 1.30:.2f}")
        status_lines.append(f"  40% Sell Level: ${smh['SMA200'] * 1.40:.2f}")
        status_lines.append("")
    
    # Key RSI Levels
    status_lines.append("KEY RSI(10) LEVELS:")
    for ticker in ['QQQ', 'SPY', 'SMH', 'XLP', 'XLU', 'XLV', 'LQD', 'HYG', 'UCO', 'TLT', 'GLD']:
        if ticker in indicators:
            rsi = indicators[ticker].get('RSI10', 0)
            flag = "🔴" if rsi > 79 else ("🟢" if rsi < 25 else "  ")
            status_lines.append(f"  {flag} {ticker}: {rsi:.1f}")
    
    status_lines.append("")
    status_lines.append("RSI > 79 = Overbought | RSI < 25 = Oversold")
    status_lines.append("=" * 70)
    
    return alerts, status_lines

def send_email(subject, body):
    """Send email alert"""
    if not SENDER_EMAIL or SENDER_EMAIL == "your.email@gmail.com":
        print("Email not configured - printing to console:")
        print(f"Subject: {subject}")
        print(body)
        return
    
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        
        if PHONE_EMAIL:
            short_msg = MIMEText(body[:160])
            short_msg['From'] = SENDER_EMAIL
            short_msg['To'] = PHONE_EMAIL
            short_msg['Subject'] = subject[:50]
            server.sendmail(SENDER_EMAIL, PHONE_EMAIL, short_msg.as_string())
        
        server.quit()
        print(f"Alert sent to {RECIPIENT_EMAIL}")
    except Exception as e:
        print(f"Email failed: {e}")

def main():
    """Main function"""
    print(f"Running signal check at {datetime.now()}")
    
    try:
        df = get_data()
        alerts, status = analyze_signals(df)
        
        body_parts = []
        if alerts:
            body_parts.append("⚠️ SIGNALS DETECTED ⚠️")
            body_parts.append("")
            body_parts.extend(alerts)
            body_parts.append("")
        
        body_parts.extend(status)
        body = "\n".join(body_parts)
        
        # Determine priority
        if any("🔴" in a for a in alerts):
            subject = "🔴 SIGNAL ALERT: Action Required!"
        elif any("🟢" in a for a in alerts):
            subject = "🟢 SIGNAL ALERT: Trading Opportunity"
        elif any("🟡" in a for a in alerts):
            subject = "🟡 Signal Alert: Watch List"
        else:
            subject = "Daily Signal Status"
            # Uncomment to only send when signals fire:
            # return
        
        send_email(subject, body)
        
    except Exception as e:
        send_email("❌ Signal Monitor Error", f"Error: {e}")

if __name__ == "__main__":
    main()


# =============================================================================
# SIGNAL SUMMARY (What this monitors)
# =============================================================================
"""
SELL SIGNALS:
  🔴 SMH 40%+ above SMA(200) → Exit SOXL (26% win rate)
  🔴 SMH Death Cross → Exit SOXL (43% win rate)
  🔴 SMH below SMA(200) → Warning

BUY/ACCUMULATE SIGNALS:
  🟢 SMH 100+ days below SMA(200) → Accumulate SOXL (85% win, +54% avg)
  🟢 SMH 100+ days below + RSI(50) < 45 → STRONG BUY (97% win)
  
TRADING SIGNALS:
  🟢 XLP/XLU/XLV RSI(10) > 79 (SPY/QQQ not) → TQQQ 20d (70% win, +5%)
  🟢 QQQ RSI(10) > 79 → UVXY 5d (+33% CAGR, 67% win)
  🟢 QQQ RSI(5) < 20 → TQQQ 5d (+26% CAGR, 69% win)
  🟢 QQQ RSI(10) < 25 → TQQQ 5d (+20% CAGR, 76% win)
  🟢 SMH RSI(5) < 20 → SOXL 10d (+21% CAGR, 59% win)
  🟢 LQD RSI(10) > 79 → TQQQ 10d (+19% CAGR, 84% win)
  🟢 UCO RSI(10) > 85 → SCO 1d (70% win)
  🟢 UCO 15%+ above EMA(21) → SCO 1d (76% win)
  🟢 TLT RSI(10) > 79 → SCO 10d (+19% CAGR)
  🟢 EDC AND YINN RSI(10) > 79 → UVXY 1-2d (77% win)
  🟢 GLD RSI(10) < 21 → TQQQ 10d (+19% CAGR, 70% win)
"""

# =============================================================================
# SETUP INSTRUCTIONS
# =============================================================================
"""
STEP 1: CREATE GMAIL APP PASSWORD
1. Go to https://myaccount.google.com/apppasswords
2. Generate password for "Mail" / "Other"
3. Copy the 16-character password

STEP 2: SETUP PYTHONANYWHERE (FREE)
1. Go to https://www.pythonanywhere.com - create free account
2. Files tab → Upload this script
3. Consoles tab → Bash → run: pip install yfinance --user
4. Test: python signal_monitor.py
5. Tasks tab → Schedule at 21:30 UTC (4:30 PM ET)

STEP 3: EDIT CONFIGURATION
Change these at the top of the script:
  SENDER_EMAIL = "your.email@gmail.com"
  SENDER_PASSWORD = "xxxx xxxx xxxx xxxx"  
  RECIPIENT_EMAIL = "your.email@gmail.com"
"""
