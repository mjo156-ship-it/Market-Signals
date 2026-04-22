#!/usr/bin/env python3
"""
Comprehensive Market Signal Monitor v4.8
========================================
Monitors all backtested trading signals and sends alerts.

SCHEDULE: Two emails daily (weekdays)
- 11:00 AM ET: Mid-day preview
- 4:05 PM ET: Market close confirmation
Ca
v4.8: Validated signal additions (Groups 31-37):
      - Group 31: Key OB alert system (always fires on key ticker RSI>79 / XLP>76)
      - Group 32: QQQ>79 streak-aware (Day 1 / Day 2 EXCLUDE / Day 3+ differentiation)
      - Group 33: Deep RSI>=82 / >=85 (T1 ROBUST, strongest single-factor edges)
      - Group 34: Breadth-conditioned OB (RSP/SPY and QQQE/QQQ divergence)
      - Group 35: Volume-conditioned OB (low vol confirmation)
      - Group 36: Core 4x stretched persistence override
      - Group 37: QQQ-QQQE RSI divergence (mega-cap driven)
      All passed pre/post-2020 + Bonferroni + bootstrap + MC + synthetic null tests
v4.7: Composer dry-run trades in daily emails (close email only)
      Portfolio & symphony win rate tracking (daily/weekly/monthly)
      Composer API integration for live portfolio data
v4.6: SPY/TLT Mid-Month Contrarian Rotation (Group 25)
      Robot James signal: buy MTD loser on trading day 15, hold through month-end
      63.7% WR, Sharpe 1.03, MaxDD -8.6%, SPY R=-0.03, n=281
      Manual execution only — daily rebalance kills edge
v4.5: FXY carry trade (Group 20), CPER copper regime (Group 21),
      LABU/SOXL 3x dip buy, UVXY SMA200 cross (Group 29),
      Bayesian Kelly on alerts, confidence tier labels,
      Oil Supply Shock EITHER/OR trigger (UCO OR USO + USDU>55)
v4.4: UVXY vol regime (Group 13), DRIF velocity filter (Group 30),
      MOVE signals (19A/B/C), multi-oversold breadth,
      ZBT/McClellan/breadth, Fibonacci context
v4.3: Audit signals (Groups 19-29), breadth regime
v4.2: Rolling beta, GLD & miners (Group 18)
"""

import os
import json
import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
import requests as req_lib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import sys

SENDER_EMAIL = os.environ.get('SENDER_EMAIL', '')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD', '')
RECIPIENT_EMAIL = os.environ.get('RECIPIENT_EMAIL', '')
PHONE_EMAIL = os.environ.get('PHONE_EMAIL', '')
IS_PRECLOSE = len(sys.argv) > 1 and sys.argv[1] == 'preclose'

# Composer API (optional — set COMPOSER_KEY_ID and COMPOSER_KEY_SECRET)
COMPOSER_KEY_ID = os.environ.get('COMPOSER_KEY_ID', '')
COMPOSER_KEY_SECRET = os.environ.get('COMPOSER_KEY_SECRET', '')
COMPOSER_BASE = "https://api.composer.trade/api/v0.1"

def calculate_rsi_wilder(prices, period):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def sf(value):
    if isinstance(value, pd.Series):
        return float(value.iloc[-1]) if len(value) > 0 else 0.0
    elif isinstance(value, np.ndarray):
        return float(value[-1]) if len(value) > 0 else 0.0
    elif pd.isna(value):
        return 0.0
    return float(value)

def bayesian_kelly(wins, losses, avg_win, avg_loss):
    if wins + losses == 0: return 0.0
    b = abs(avg_win / avg_loss) if avg_loss != 0 else 10
    p_samples = np.random.beta(wins + 1, losses + 1, 5000)
    kelly_samples = np.array([max(0, (b*p - (1-p)) / b) for p in p_samples])
    return round(float(np.mean(kelly_samples)) * 100, 0)

# ═══════════════════════════════════════════════════════════════════
# v4.8 HELPER FUNCTIONS (validated signal framework)
# ═══════════════════════════════════════════════════════════════════

def compute_qqq_streak_from_history(qqq_df, threshold=79):
    """Compute current QQQ>threshold streak day from historical RSI.
    Returns 0 if not currently firing, or N = consecutive days including today.
    Self-contained — no state file required, works across GitHub Actions runs."""
    try:
        close = qqq_df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        rsi = calculate_rsi_wilder(close, 10)
        streak = 0
        for i in range(len(rsi) - 1, max(len(rsi) - 30, 0) - 1, -1):
            v = sf(rsi.iloc[i])
            if v > threshold:
                streak += 1
            else:
                break
        return streak
    except Exception as e:
        print(f"  streak compute error: {e}")
        return 0


def compute_breadth_divergence(data):
    """Compute RSP/SPY and QQQE/QQQ 10-day ratio changes for narrow-breadth signals."""
    result = {'rsp_spy_10d': None, 'qqqe_qqq_10d': None}
    try:
        for num_tkr, den_tkr, key in [('RSP', 'SPY', 'rsp_spy_10d'),
                                       ('QQQE', 'QQQ', 'qqqe_qqq_10d')]:
            if num_tkr not in data or den_tkr not in data:
                continue
            n_df = data[num_tkr]; d_df = data[den_tkr]
            if len(n_df) <= 10 or len(d_df) <= 10:
                continue
            n_c = n_df['Close']; d_c = d_df['Close']
            if isinstance(n_c, pd.DataFrame): n_c = n_c.iloc[:, 0]
            if isinstance(d_c, pd.DataFrame): d_c = d_c.iloc[:, 0]
            n_t = sf(n_c.iloc[-1]); d_t = sf(d_c.iloc[-1])
            n_10 = sf(n_c.iloc[-11]); d_10 = sf(d_c.iloc[-11])
            if d_t > 0 and d_10 > 0 and n_10 > 0:
                result[key] = round(((n_t / d_t) / (n_10 / d_10) - 1) * 100, 2)
    except Exception as e:
        print(f"  breadth divergence error: {e}")
    return result


def compute_qqq_volume_ratio(data):
    """Compute QQQ volume vs 20d avg and consecutive low-vol day count."""
    try:
        if 'QQQ' not in data:
            return None, 0
        qqq_df = data['QQQ']
        if 'Volume' not in qqq_df.columns or len(qqq_df) < 22:
            return None, 0
        vol = qqq_df['Volume']
        if isinstance(vol, pd.DataFrame): vol = vol.iloc[:, 0]
        vol_today = sf(vol.iloc[-1])
        vol_20d_avg = sf(vol.iloc[-21:-1].mean())
        if vol_20d_avg <= 0:
            return None, 0
        ratio = round(vol_today / vol_20d_avg, 2)
        low_vol_days = 0
        for i in range(10):
            if len(vol) < 22 + i:
                break
            v_i = sf(vol.iloc[-(1 + i)])
            v_avg_i = sf(vol.iloc[-(21 + i):-(1 + i)].mean())
            if v_avg_i > 0 and v_i < v_avg_i:
                low_vol_days += 1
            else:
                break
        return ratio, low_vol_days
    except Exception as e:
        print(f"  volume compute error: {e}")
        return None, 0


def compute_calendar_position():
    """Intramonth momentum cycle (Nathan, Suominen & Tasa 2026).
    T-9 to T-4 = institutional selling pressure window."""
    today = datetime.now()
    if today.month == 12:
        month_end = datetime(today.year + 1, 1, 1) - timedelta(days=1)
    else:
        month_end = datetime(today.year, today.month + 1, 1) - timedelta(days=1)
    trading_days = 0
    d = today
    while d <= month_end:
        if d.weekday() < 5:
            trading_days += 1
        d += timedelta(days=1)
    in_window = 4 <= trading_days <= 9
    if trading_days >= 10:
        zone = f"EARLY MONTH (T-{trading_days}) — Normal returns"
        emoji = "🟢"
    elif in_window:
        zone = f"SELLING PRESSURE WINDOW (T-{trading_days}) — Lev equity suppressed"
        emoji = "🟡"
    else:
        zone = f"LATE MONTH (T-{trading_days}) — Pressure clearing"
        emoji = "🟢"
    return {'days': trading_days, 'zone': zone, 'emoji': emoji, 'in_window': in_window}

def download_data(tickers, period='2y'):
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

def check_signals(data):
    alerts = []
    status = {}
    indicators = {}

    for ticker, df in data.items():
        if len(df) < 50: continue
        try:
            close = df['Close']
            if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
            price = sf(close.iloc[-1])
            rsi10 = sf(calculate_rsi_wilder(close, 10).iloc[-1])
            rsi7 = sf(calculate_rsi_wilder(close, 7).iloc[-1]) if len(close) >= 50 else rsi10
            rsi14 = sf(calculate_rsi_wilder(close, 14).iloc[-1]) if len(close) >= 50 else rsi10
            rsi50 = sf(calculate_rsi_wilder(close, 50).iloc[-1]) if len(close) >= 200 else 50
            sma200 = sf(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else 0
            sma50 = sf(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else 0
            sma20 = sf(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else 0
            ema9 = sf(close.ewm(span=9, adjust=False).mean().iloc[-1])
            ema20 = sf(close.ewm(span=20, adjust=False).mean().iloc[-1])
            cr5 = sf(close.pct_change(5).iloc[-1]) * 100 if len(close) > 5 else 0
            cr7 = sf(close.pct_change(7).iloc[-1]) * 100 if len(close) > 7 else 0
            cr10 = sf(close.pct_change(10).iloc[-1]) * 100 if len(close) > 10 else 0
            cr30 = sf(close.pct_change(30).iloc[-1]) * 100 if len(close) > 30 else 0
            cr50 = sf(close.pct_change(50).iloc[-1]) * 100 if len(close) > 50 else 0
            cr100 = sf(close.pct_change(100).iloc[-1]) * 100 if len(close) > 100 else 0
            rets = close.pct_change()
            std10 = sf(rets.rolling(10).std().iloc[-1]) if len(rets) > 10 else 0
            std50 = sf(rets.rolling(50).std().iloc[-1]) if len(rets) > 50 else 0
            rsi_full = calculate_rsi_wilder(close, 10)
            rsi_5ago = sf(rsi_full.iloc[-6]) if len(rsi_full) > 6 else 50
            indicators[ticker] = {
                'price': price, 'rsi10': rsi10, 'rsi7': rsi7, 'rsi14': rsi14, 'rsi50': rsi50,
                'sma200': sma200, 'sma50': sma50, 'sma20': sma20,
                'ema9': ema9, 'ema20': ema20,
                'pct_above_sma200': (price / sma200 - 1) * 100 if sma200 > 0 else 0,
                'cumRet5d': cr5, 'cumRet7d': cr7, 'cumRet10d': cr10,
                'cumRet30d': cr30, 'cumRet50d': cr50, 'cumRet100d': cr100,
                'maret10d': cr10 / 10 if cr10 != 0 else 0,
                'std10': std10, 'std50': std50, 'vol_ratio': std10 / std50 if std50 > 0 else 1.0,
                'rsi_velocity': round(rsi10 - rsi_5ago, 1),
                'above_ema9': price > ema9, 'above_ema20': price > ema20,
                'above_sma50': price > sma50 if sma50 > 0 else None,
                'above_sma200': price > sma200 if sma200 > 0 else None,
            }
        except Exception as e:
            print(f"Error {ticker}: {e}")

    status['indicators'] = indicators
    def r(t): return indicators.get(t, {}).get('rsi10', 50)
    def p(t): return indicators.get(t, {}).get('price', 0)

    # === GROUP 1: SOXL/SMH ===
    if 'SMH' in indicators:
        smh = indicators['SMH']
        if smh['pct_above_sma200'] >= 40:
            alerts.append(('[EXIT] SOXL EXIT', f"SMH {smh['pct_above_sma200']:.1f}% above SMA(200) - SELL ALL", 'exit'))
        elif smh['pct_above_sma200'] >= 35:
            alerts.append(('[WARN] SOXL WARNING', f"SMH {smh['pct_above_sma200']:.1f}% above SMA(200)", 'warning'))
        elif smh['pct_above_sma200'] >= 30:
            alerts.append(('[WARN] SOXL TRIM', f"SMH {smh['pct_above_sma200']:.1f}% above SMA(200) - Trim 25-50%", 'warning'))
        if smh['sma50'] < smh['sma200'] and smh['sma200'] > 0:
            alerts.append(('[EXIT] DEATH CROSS', f"SMH SMA(50) below SMA(200)", 'exit'))
        if 'SMH' in data:
            close = data['SMH']['Close']
            if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
            s200s = close.rolling(200).mean()
            days_below = 0
            for i in range(len(close)-1, max(len(close)-500, 199), -1):
                try:
                    if sf(s200s.iloc[i]) > 0 and sf(close.iloc[i]) < sf(s200s.iloc[i]):
                        days_below += 1
                    else: break
                except: break
            if days_below >= 100:
                if smh['rsi50'] < 45:
                    alerts.append(('[BUY] SOXL STRONG BUY [T1]', f"SMH {days_below}d below SMA(200) + RSI(50)={smh['rsi50']:.1f}<45 | 97% WR +81% n=33 | BK=96%", 'buy'))
                else:
                    alerts.append(('[BUY] SOXL ACCUMULATE', f"SMH {days_below}d below SMA(200) | 85% WR n=52", 'buy'))
            status['smh_days_below_sma200'] = days_below

    # === GROUP 2: GLD/USDU ===
    gld_r, usdu_r = r('GLD'), r('USDU')
    if gld_r > 79 and usdu_r < 25:
        bk = int(bayesian_kelly(12, 1, 7.0, 3.0))
        alerts.append(('[BUY] DOUBLE SIGNAL [T2]', f"GLD RSI={gld_r:.1f}>79 + USDU={usdu_r:.1f}<25 | TQQQ 88% WR | BK={bk}% | n=13", 'buy'))
        if r('XLP') > 65:
            alerts.append(('[BUY] TRIPLE SIGNAL [LOW-N]', f"+ XLP={r('XLP'):.1f}>65 | TQQQ 100% WR +11.6% | BK=80% | n=5", 'buy'))
    elif gld_r > 79:
        alerts.append(('[BUY] GLD OB', f"GLD RSI={gld_r:.1f}>79 | TQQQ 72% WR +3.2% 5d | n=97", 'buy'))

    # === GROUP 3: Defensive Rotation ===
    if any(r(t) > 79 for t in ['XLP','XLU','XLV']) and r('SPY') < 79 and r('QQQ') < 79:
        alerts.append(('[BUY] DEFENSIVE ROTATION [T2]', f"Defensive OB, SPY/QQQ not | TQQQ 20d: 70% WR +5%", 'buy'))

    # === GROUP 4: QQQ Dip Buy (OB vol hedge now routed through v4.8 Group 32 streak-aware) ===
    if r('QQQ') < 20:
        bk = int(bayesian_kelly(12, 0, 5.4, 3.7))
        alerts.append(('[BUY] QQQ DIP BUY [T1]', f"QQQ RSI={r('QQQ'):.1f}<20 | TQQQ 100% WR n=12 | BK={bk}%", 'buy'))

    # === GROUP 5: SOXS ===
    if r('SMH') > 79 and usdu_r > 70:
        alerts.append(('[EXIT] SOXS DOLLAR SQUEEZE [T1]', f"SMH={r('SMH'):.1f}>79 + USDU={usdu_r:.1f}>70 | SOXS 100% WR +9.5% n=8 | BK=85%", 'short'))
    if r('SMH') > 79 and r('IWM') < 50:
        alerts.append(('[EXIT] SOXS DIVERGENCE', f"SMH={r('SMH'):.1f}>79 + IWM={r('IWM'):.1f}<50 | SOXS 86% WR n=7", 'short'))

    # === GROUP 6: BTC ===
    if r('BTC-USD') > 79:
        alerts.append(('[BUY] BTC MOMENTUM', f"BTC RSI={r('BTC-USD'):.1f}>79 | Hold/Add 67% WR +5.2% n=215", 'buy'))
    if r('BTC-USD') < 30:
        if r('UVXY') < 40:
            alerts.append(('[BUY] BTC DIP BUY', f"BTC RSI={r('BTC-USD'):.1f}<30 + UVXY<40 | 77% WR n=30", 'buy'))
        else:
            alerts.append(('[WATCH] BTC OVERSOLD', f"BTC RSI={r('BTC-USD'):.1f}<30 (wait UVXY<40)", 'watch'))

    # === GROUP 7: UPRO ===
    spy_r = r('SPY')
    if spy_r > 85:
        alerts.append(('[EXIT] UPRO EXIT', f"SPY RSI={spy_r:.1f}>85 | Only 36% WR n=11", 'exit'))
    elif spy_r > 82:
        alerts.append(('[WARN] UPRO CAUTION', f"SPY RSI={spy_r:.1f}>82 | 49% WR n=35", 'warning'))
    if spy_r < 21:
        bk = int(bayesian_kelly(20, 3, 8.9, 3.5))
        alerts.append(('[BUY] UPRO STRONG BUY [T1]', f"SPY RSI={spy_r:.1f}<21 | 87% WR +8.9% | BK={bk}% | n=23", 'buy'))
    elif spy_r < 25:
        alerts.append(('[BUY] UPRO BUY', f"SPY RSI={spy_r:.1f}<25 | 74% WR n=42", 'buy'))
    elif spy_r < 30:
        alerts.append(('[BUY] UPRO CONSIDER', f"SPY RSI={spy_r:.1f}<30 | 69% WR n=108", 'buy'))

    # === GROUP 8: AMD/NVDA ===
    for t in ['AMD','NVDA']:
        if r(t) > 85:
            alerts.append((f'[WARN] {t} EXTENDED', f"{t} RSI={r(t):.1f}>85 | Take profits", 'warning'))

    # === GROUP 9: NAIL ===
    if gld_r > 79 and usdu_r < 25 and r('XLF') < 70:
        alerts.append(('[BUY] NAIL SIGNAL [LOW-N]', f"GLD>{gld_r:.0f}+USDU<{usdu_r:.0f}+XLF<{r('XLF'):.0f} | 90% WR +4.9% n=10 | BK=88%", 'buy'))
    if r('XLF') > 70 and usdu_r < 25:
        alerts.append(('[EXIT] NAIL DANGER', f"XLF RSI={r('XLF'):.1f}>70+USDU<25 | 11% WR -11.5% | EXIT", 'exit'))
    if r('NAIL') < 21:
        alerts.append(('[BUY] NAIL RSI<21', f"NAIL RSI={r('NAIL'):.1f} | Oversold", 'buy'))

    # === GROUP 10: CURE ===
    cure_r = r('CURE')
    if cure_r < 21:
        bk = int(bayesian_kelly(28, 5, 7.3, 3.0))
        alerts.append(('[BUY] CURE STRONG BUY [T1]', f"CURE RSI={cure_r:.1f}<21 | 85% WR +7.3% | BK={bk}% | n=33", 'buy'))
    elif cure_r < 25:
        alerts.append(('[BUY] CURE BUY [T1]', f"CURE RSI={cure_r:.1f}<25 | 81% WR +5.4% | n=70", 'buy'))
    if cure_r > 85:
        alerts.append(('[EXIT] CURE SELL', f"CURE RSI={cure_r:.1f}>85 | 33% WR n=15", 'exit'))
    elif cure_r > 79:
        alerts.append(('[EXIT] CURE OB', f"CURE RSI={cure_r:.1f}>79 | 40% WR n=95", 'exit'))

    # === GROUP 11: FAS ===
    fas_r = r('FAS')
    if gld_r > 79 and usdu_r < 25:
        alerts.append(('[BUY] FAS SIGNAL', f"GLD>{gld_r:.0f}+USDU<{usdu_r:.0f} | FAS 10d: 92% WR n=13", 'buy'))
    if fas_r < 30:
        alerts.append(('[BUY] FAS BUY [T2]', f"FAS RSI={fas_r:.1f}<30 | 63% WR n=195", 'buy'))
    if fas_r > 85:
        alerts.append(('[EXIT] FAS SELL', f"FAS RSI={fas_r:.1f}>85 | 8% WR n=12", 'exit'))
    elif fas_r > 82:
        alerts.append(('[EXIT] FAS OB', f"FAS RSI={fas_r:.1f}>82 | 38% WR n=40", 'exit'))

    # === GROUP 12: LABU + LABU/SOXL Dip Buy (Wishlist Item 1) ===
    labu_r = r('LABU')
    if labu_r < 25:
        pick = 'LABU'
        if 'SOXL' in indicators:
            pick = 'SOXL' if indicators['SOXL'].get('cumRet30d',0) < indicators['LABU'].get('cumRet30d',0) else 'LABU'
        if labu_r < 22:
            bk = int(bayesian_kelly(11, 0, 12.3, 5.0))
            alerts.append(('[BUY] LABU/SOXL CORE DIP [T1]', f"LABU RSI={labu_r:.1f}<22 -> {pick} | 100% WR +12.3% | BK={bk}% | n=11", 'buy'))
        else:
            bk = int(bayesian_kelly(23, 5, 7.0, 4.0))
            alerts.append(('[BUY] LABU/SOXL DIP [T1]', f"LABU RSI={labu_r:.1f}<25 -> {pick} | 82% WR +7.0% | BK={bk}% | n=28 | SPY R=0.18", 'buy'))
    if labu_r > 70:
        alerts.append(('[WARN] LABU EXTENDED', f"LABU RSI={labu_r:.1f}>70 | 42% WR n=180", 'warning'))

    # === GROUP 13: UVXY Vol Regime Shift ===
    if 'UVXY' in indicators and indicators['UVXY']['sma200'] > 0:
        pct_ab = indicators['UVXY']['pct_above_sma200']
        if pct_ab >= 30:
            alerts.append(('[BUY] VOL REGIME EXTREME', f"UVXY {pct_ab:+.1f}% above SMA200 | SPY 20d: 94% WR | 40d+: 100% n=18", 'buy'))
        elif pct_ab >= 20:
            alerts.append(('[BUY] VOL REGIME HIGH', f"UVXY {pct_ab:+.1f}% above SMA200 | SPY 20d: 92% WR n=24", 'buy'))
        elif pct_ab >= 0:
            alerts.append(('[BUY] VOL REGIME SHIFT', f"UVXY {pct_ab:+.1f}% above SMA200 | SPY 20d: 83% WR n=52", 'buy'))
        elif pct_ab >= -10:
            alerts.append(('[WATCH] VOL REGIME APPROACHING', f"UVXY {pct_ab:+.1f}% vs SMA200 | Threshold: ${indicators['UVXY']['sma200']:.2f}", 'watch'))
        status['uvxy_vol'] = {'pct': pct_ab, 'sma200': indicators['UVXY']['sma200']}

    # === GROUP 17: DFEN BB ===
    if 'DFEN' in data and 'DFEN' in indicators:
        dc = data['DFEN']['Close']
        if isinstance(dc, pd.DataFrame): dc = dc.iloc[:, 0]
        if len(dc) >= 20:
            bb_sma = sf(dc.rolling(20).mean().iloc[-1])
            bb_std = sf(dc.rolling(20).std().iloc[-1])
            bb_u = bb_sma + 2*bb_std; bb_l = bb_sma - 2*bb_std
            dp = indicators['DFEN']['price']
            pctB = (dp - bb_l) / (bb_u - bb_l) if (bb_u - bb_l) > 0 else 0.5
            status['dfen_bb'] = {'price':dp,'rsi':indicators['DFEN']['rsi10'],'upper':round(bb_u,2),'sma20':round(bb_sma,2),'lower':round(bb_l,2),'pctB':round(pctB,3),'width':round((bb_u-bb_l)/bb_sma*100,1) if bb_sma>0 else 0}
            if pctB < 0:
                wr = '73.5%' if indicators['DFEN']['rsi10'] >= 30 else '63.8%'
                alerts.append(('[BUY] DFEN BELOW BB [T2]', f"DFEN %B={pctB:.3f} | RSI={indicators['DFEN']['rsi10']:.1f} | {wr} WR 5d", 'buy'))

    # === GROUP 18: GLD & Miners ===
    gdxj_r, gdx_r = r('GDXJ'), r('GDX')
    if gdxj_r < 21:
        alerts.append(('[BUY] GDXJ<21 -> JNUG', f"GDXJ RSI={gdxj_r:.1f} | 59% WR +8.43% 1d n=17", 'buy'))
    elif gdxj_r < 25:
        alerts.append(('[BUY] GDXJ<25 -> JNUG [T2]', f"GDXJ RSI={gdxj_r:.1f} | 63% WR +3.55% 1d n=59", 'buy'))
    if gdx_r < 21 and gdxj_r >= 25:
        alerts.append(('[BUY] GDX<21 -> NUGT', f"GDX RSI={gdx_r:.1f} | 56% WR n=25", 'buy'))
    if gdx_r > 85:
        alerts.append(('[WARN] GDX EXTENDED - DO NOT SHORT', f"GDX RSI={gdx_r:.1f} | DUST loses", 'warning'))
    if gld_r > 75 and usdu_r > 60:
        alerts.append(('[HEDGE] MINER SHORT WINDOW', f"GLD={gld_r:.1f}>75+USDU={usdu_r:.1f}>60 | JDST 59% WR n=34", 'hedge'))

    # === GROUP 19: HYG Credit Euphoria + Oil Supply Shock ===
    hyg_r = r('HYG')
    if hyg_r > 80:
        hyg_p, hyg_s200 = p('HYG'), indicators.get('HYG',{}).get('sma200',0)
        if hyg_p > hyg_s200 > 0:
            alerts.append(('[BUY] HYG CREDIT EUPHORIA [T1]', f"HYG RSI~{hyg_r:.1f}+above SMA200 | TQQQ 1d: 80.6% WR n=36", 'buy'))
    uco_r = r('UCO'); uso_r = r('USO') if 'USO' in indicators else 50
    if (uco_r > 79 or uso_r > 79) and usdu_r > 55:
        sig = f"UCO={uco_r:.1f}" if uco_r > 79 else f"USO={uso_r:.1f}"
        conv = " HIGH CONVICTION" if usdu_r > 60 else ""
        alerts.append(('[EXIT] OIL SUPPLY SHOCK -> TMV [T1]', f"{sig}+USDU={usdu_r:.1f}>55{conv} | TMV 80% WR n=25", 'short'))
    elif uco_r > 75:
        alerts.append(('[WATCH] UCO>75 -> TMV', f"UCO RSI={uco_r:.1f}", 'watch'))

    # === GROUP 20: FXY Carry Trade (NEW - Wishlist Item 7) ===
    if 'FXY' in indicators:
        fxy_r = r('FXY')
        if fxy_r > 75:
            alerts.append(('[WARN] FXY CARRY STRESS', f"FXY RSI={fxy_r:.1f}>75 | Yen strengthening", 'warning'))
        if fxy_r > 70 and 'TLT' in indicators and not indicators['TLT'].get('above_sma200', True):
            alerts.append(('[HEDGE] FXY+TLT CARRY UNWIND [T2]', f"FXY={fxy_r:.1f}>70+TLT<SMA200 | BTAL 86.7% WR 1d n=15", 'hedge'))
        if fxy_r > 70 and usdu_r > 60:
            alerts.append(('[HEDGE] DUAL SAFE HAVEN [LOW-N]', f"FXY={fxy_r:.1f}>70+USDU={usdu_r:.1f}>60 | BTAL 87% WR 1d", 'hedge'))

    # === GROUP 21: CPER Copper Regime (NEW - Wishlist Item 9) ===
    if 'CPER' in indicators:
        cper = indicators['CPER']
        spy_ae9 = indicators.get('SPY',{}).get('above_ema9', True)
        copx_ae9 = indicators.get('COPX',{}).get('above_ema9', True) if 'COPX' in indicators else True
        if cper['above_ema9'] and not spy_ae9 and copx_ae9:
            alerts.append(('[BUY] COPPER REGIME -> TQQQ [T2]', f"CPER>EMA9+SPY<EMA9+COPX>EMA9 | 40.2% CAGR, 0.23 SPY R | Composer active", 'buy'))
        elif cper['above_ema9'] and not copx_ae9:
            alerts.append(('[WARN] COPPER SUPPLY DISRUPTION', f"CPER>EMA9 but COPX<EMA9 | Possible false positive", 'warning'))

    # === GROUP 24: ILS Cat Bond Monitoring (Wishlist Item 8D) ===
    if 'ILS' in indicators:
        ils = indicators['ILS']
        if ils['cumRet5d'] < -3:
            alerts.append(('[WARN] ILS CAT BOND DROP [24A]', f"ILS 5d: {ils['cumRet5d']:+.1f}% | DO NOT SELL — drawdowns recover 1-3 months", 'warning'))
        now_month = datetime.now().month
        if 6 <= now_month <= 11:
            alerts.append(('[WATCH] HURRICANE SEASON [24B]', f"Month {now_month} — binary risk window open for ILS", 'watch'))

    # === GROUP 27: SPHB/SPLV Ratio RSI ===
    if 'SPHB' in data and 'SPLV' in data:
        try:
            sc = data['SPHB']['Close']; lc = data['SPLV']['Close']
            if isinstance(sc, pd.DataFrame): sc = sc.iloc[:,0]
            if isinstance(lc, pd.DataFrame): lc = lc.iloc[:,0]
            ratio_rsi = sf(calculate_rsi_wilder(sc/lc, 10).iloc[-1])
            if ratio_rsi < 25:
                alerts.append(('[BUY] RISK ROTATION EXHAUST [T2]', f"SPHB/SPLV RSI={ratio_rsi:.1f}<25 | TQQQ 10d: 75.5% WR n=53 | MANUAL", 'buy'))
            status['sphb_splv_rsi'] = round(ratio_rsi, 1)
        except: pass

    # === GROUP 28: Vol Recovery Alpha ===
    uvxy_p, uvxy_s200 = p('UVXY'), indicators.get('UVXY',{}).get('sma200',0)
    vixm_p, vixm_s50 = p('VIXM'), indicators.get('VIXM',{}).get('sma50',0)
    if uvxy_s200 > 0 and vixm_s50 > 0:
        if uvxy_p > uvxy_s200 and vixm_p < vixm_s50:
            alerts.append(('[BUY] VOL RECOVERY ALPHA [T2]', f"UVXY>{uvxy_s200:.0f}SMA200+VIXM<{vixm_s50:.0f}SMA50 | SOXL 10d: 90.3% WR n=31", 'buy'))
        elif uvxy_p > uvxy_s200:
            alerts.append(('[WATCH] VOL STILL ELEVATED', f"UVXY ${uvxy_p:.2f}>SMA200 — waiting VIXM normalization", 'watch'))

    # === GROUP 29: UVXY SMA200 Cross -> SOXL 5d (NEW - Wishlist 8C) ===
    if 'UVXY' in data and 'UVXY' in indicators:
        uc = data['UVXY']['Close']
        if isinstance(uc, pd.DataFrame): uc = uc.iloc[:,0]
        if len(uc) >= 201:
            us = uc.rolling(200).mean()
            today_ab = sf(uc.iloc[-1]) > sf(us.iloc[-1])
            yest_ab = sf(uc.iloc[-2]) > sf(us.iloc[-2])
            if today_ab and not yest_ab:
                alerts.append(('[BUY] UVXY SMA200 CROSS -> SOXL [T2]', f"UVXY crossed ABOVE SMA200 | Buy SOXL hold 5d | 80% WR +8.3% PF=6.75 n=10 | MANUAL", 'buy'))

    # === GROUP 30: DRIF Velocity Filter ===
    drif_data = {}
    for t, lever, thresholds in [
        ('SPY','UPRO',[(25,'cumRet5d',-5,'100%',10,'55.6%',9,'5d'),(30,'cumRet7d',-5,'76.1%',46,'56.2%',32,'20d')]),
        ('QQQ','TQQQ',[(25,'cumRet7d',-8,'87.5%',16,'57.1%',7,'5d')]),
        ('SMH','SOXL',[(25,'cumRet5d',-5,'68.8%',16,'16.7%',6,'5d')]),
    ]:
        if t not in indicators: continue
        ind = indicators[t]
        entry = {'ticker':t,'lever':lever,'rsi':ind['rsi10'],'velocity':ind['rsi_velocity'],'cumRet5d':ind['cumRet5d'],'cumRet7d':ind['cumRet7d'],'gate':'---','label':'NOT OVERSOLD'}
        for rsi_th, rf, rg, pwr, pn, fwr, fn, hold in thresholds:
            if ind['rsi10'] < rsi_th:
                rv = ind.get(rf, 0)
                if rv > rg:
                    entry['gate']='PASS'; entry['label']='STABILIZED DIP'
                    alerts.append((f'[BUY] DRIF: {t} CONFIRMED', f"{t} RSI={ind['rsi10']:.1f}+{rf}={rv:+.1f}%>{rg}% | {pwr} WR n={pn}", 'buy'))
                else:
                    entry['gate']='FAIL'; entry['label']='FALLING KNIFE'
                    alerts.append((f'[WARN] DRIF: {t} FALLING KNIFE', f"{t} RSI={ind['rsi10']:.1f} BUT {rf}={rv:+.1f}%<{rg}% | Only {fwr} WR n={fn}", 'warning'))
                break
        drif_data[t] = entry
    status['drif'] = drif_data

    # === USMV Overbought ===
    if r('USMV') > 82:
        alerts.append(('[HEDGE] USMV COMPLACENCY [T1]', f"USMV RSI={r('USMV'):.1f}>82 | UVXY 1d: 75% WR n=24", 'hedge'))

    # === VIXM<25 -> HIBL ===
    if r('VIXM') < 25:
        alerts.append(('[BUY] VIXM<25 -> HIBL (B2)', f"VIXM RSI={r('VIXM'):.1f}<25 | POST-2020 edge only", 'buy'))

    # === GLD Oversold ===
    if gld_r < 20:
        alerts.append(('[BUY] GLD DEEP OVERSOLD [T2]', f"GLD RSI={gld_r:.1f}<20 | TQQQ 10d: 70.6% WR PF=5.99", 'buy'))
    elif gld_r < 22:
        alerts.append(('[BUY] GLD OVERSOLD', f"GLD RSI={gld_r:.1f}<22 | TQQQ 10d: 73.3% WR", 'buy'))

    # === Multi-Oversold Breadth ===
    if all(r(t) < 30 for t in ['SPY','USMV','VTV','VOOV','UPRO'] if t in indicators):
        msg = f'SPY+USMV+VTV+VOOV+UPRO all RSI<30 | UPRO 5d: 77.8% WR n=45'
        if hyg_r < 30: msg += ' | HYG confirmed'
        alerts.append(('[BUY] MULTI-OVERSOLD BREADTH', msg, 'buy'))

    # === DANGER: Falling knife + strong dollar ===
    if 'SPY' in indicators and indicators['SPY']['cumRet10d'] < -5 and usdu_r > 70:
        alerts.append(('[EXIT] DANGER: KNIFE+DOLLAR', f"SPY CumRet10d={indicators['SPY']['cumRet10d']:.1f}%+USDU>{usdu_r:.1f} | 20% WR", 'exit'))

    # Group 25 (SPY/TLT 2-way mid-month) superseded by the Robot James
    # 4-way ensemble (SPY/QQQ/SMH/TLT). See rj_compute_state() in main() —
    # it writes state + generates TD1/TD15/EOM action alerts directly.

    # === Regime ===
    spy_d = indicators.get('SPY', {})

    # Load Polygon breadth for ZBT confirmation (Wishlist Item 2)
    zbt_confirm = False
    zbt_thrust = False
    breadth_json = os.path.join(os.environ.get('BREADTH_DATA_DIR', './data/breadth'), 'latest_indicators.json')
    if os.path.exists(breadth_json):
        try:
            import json
            with open(breadth_json) as f:
                bi = json.load(f)
            if bi.get('zbt_zone') == 'OVERSOLD':
                zbt_confirm = True
            if bi.get('zbt_thrust'):
                zbt_thrust = True
                alerts.append(('[BUY] ZBT THRUST SIGNAL',
                    f"Zweig Breadth Thrust fired | ZBT surged from <0.40 to >0.615\n"
                    f"   Historically near-100% forward returns at 6 and 12 months", 'buy'))
        except: pass

    # Annotate existing dip-buy alerts with ZBT confirmation
    if zbt_confirm:
        for i, (title, msg, typ) in enumerate(alerts):
            if typ == 'buy' and ('UPRO' in title or 'TQQQ' in title or 'DIP' in title):
                alerts[i] = (title, msg + ' | ZBT OVERSOLD CONFIRMED', typ)

    a200 = spy_d.get('above_sma200', False)
    a_e20 = spy_d.get('above_ema20', False)
    vol_exp = indicators.get('QQQ', {}).get('vol_ratio', 1.0) > 1.0
    if a200 and not vol_exp: regime = 'BULL + VOL COMPRESS'
    elif a200 and vol_exp: regime = 'BULL + VOL EXPAND'
    elif not a200 and a_e20: regime = 'BEAR RECOVERY'
    elif not a200: regime = 'BEAR DEFENSIVE'
    else: regime = 'UNKNOWN'
    status['regime'] = regime
    if not a200 and not a_e20:
        alerts.append(('[WARN] BEAR DEFENSIVE', f"SPY below SMA200+EMA20 -> SHY/GLD defensive", 'warning'))
    elif not a200 and a_e20:
        alerts.append(('[WATCH] BEAR RECOVERY', f"SPY below SMA200 but above EMA20", 'watch'))

    # ═══════════════════════════════════════════════════════════════════
    # v4.8 VALIDATED SIGNAL ADDITIONS (Groups 31-37)
    # All passed: pre/post-2020 stability, Bonferroni/Holm/BH, bootstrap CI,
    # Monte Carlo null, block bootstrap, synthetic null (3 methods)
    # ═══════════════════════════════════════════════════════════════════

    # === GROUP 31: Key OB Alert System (mandatory floor — always emit) ===
    KEY_OB_79 = ['QQQ','SPY','VTV','VOOV','VOOG','QQQE','XLY','FAS','TQQQ','TECL']
    KEY_OB_76 = ['XLP']
    key_ob_fires = []
    for tkr in KEY_OB_79:
        v = r(tkr)
        if v > 79:
            key_ob_fires.append((tkr, v, 79))
    for tkr in KEY_OB_76:
        v = r(tkr)
        if v > 76:
            key_ob_fires.append((tkr, v, 76))
    if key_ob_fires:
        msg_parts = [f"{t} RSI={v:.1f}>{th}" for t, v, th in key_ob_fires]
        alerts.append(('[OB ALERT] Key Tickers Overbought',
                       ' | '.join(msg_parts),
                       'warning'))
    status['key_ob_fires'] = [{'ticker': t, 'rsi': round(v, 1), 'threshold': th}
                               for t, v, th in key_ob_fires]

    # === GROUP 32: QQQ>79 Streak-Aware Signal [T1 ROBUST] ===
    qqq_streak = compute_qqq_streak_from_history(data.get('QQQ'), threshold=79) if 'QQQ' in data else 0
    status['qqq_79_streak_day'] = qqq_streak

    if qqq_streak == 1:
        bk = int(bayesian_kelly(36, 13, 3.1, 2.5))
        alerts.append(('[HEDGE] QQQ>79 DAY 1 -> UVXY [T1 ROBUST]',
                       f"Fresh Day 1 fire | QQQ={r('QQQ'):.1f}>79 | UVXY 1d: 74% WR +3.1% worst -9.66% | n=49 | BK={bk}%",
                       'hedge'))
    elif qqq_streak == 2:
        alerts.append(('[SKIP] QQQ>79 DAY 2 — EXCLUDE standalone',
                       f"Day 2 of streak | FAILED validation: 50% WR, 0% avg, worst -10.5% | n=30 | Use only with Core 4x confirm (Group 36)",
                       'watch'))
    elif qqq_streak >= 3:
        bk = int(bayesian_kelly(35, 17, 3.4, 2.0))
        alerts.append(('[HEDGE] QQQ>79 DAY 3+ -> UVXY [T1]',
                       f"Day {qqq_streak} of streak | UVXY 1d: 67% WR +3.4% worst -4.95% | n=52 | BK={bk}%",
                       'hedge'))

    # === GROUP 33: Deep OB Levels RSI>=82 / >=85 [T1 ROBUST] ===
    if r('QQQ') >= 85:
        bk = int(bayesian_kelly(18, 4, 5.5, 2.0))
        alerts.append(('[HEDGE] QQQ RSI>=85 EXTREME -> UVXY [T1 ROBUST]',
                       f"QQQ={r('QQQ'):.1f}>=85 | UVXY 1d: 82% WR +5.5% worst -2.63% | n=22 | BK={bk}% | DEEPEST EDGE",
                       'hedge'))
    elif r('QQQ') >= 82:
        bk = int(bayesian_kelly(39, 13, 4.3, 2.5))
        alerts.append(('[HEDGE] QQQ RSI>=82 DEEP -> UVXY [T1 ROBUST]',
                       f"QQQ={r('QQQ'):.1f}>=82 | UVXY 1d: 75% WR +4.3% worst -8.67% | n=52 | BK={bk}%",
                       'hedge'))

    # === GROUP 34: Breadth-Conditioned OB [T1 ROBUST] ===
    breadth_div = compute_breadth_divergence(data)
    status['breadth_divergence'] = breadth_div

    if r('QQQ') > 79:
        qqqe_qqq = breadth_div.get('qqqe_qqq_10d')
        rsp_spy = breadth_div.get('rsp_spy_10d')

        if qqqe_qqq is not None and qqqe_qqq < -2.0:
            bk = int(bayesian_kelly(14, 4, 3.0, 1.5))
            alerts.append(('[HEDGE] QQQ>79 + VERY NARROW TECH [T1 ROBUST]',
                           f"QQQ>79 + QQQE/QQQ 10d={qqqe_qqq:+.1f}%<-2 | UVXY 1d: 78% WR +3.0% worst -0.89% | n=18 | BK={bk}% | TIGHTEST TAIL",
                           'hedge'))
        elif qqqe_qqq is not None and qqqe_qqq < -1.0:
            alerts.append(('[HEDGE] QQQ>79 + NARROW TECH [T1 ROBUST]',
                           f"QQQ>79 + QQQE/QQQ 10d={qqqe_qqq:+.1f}%<-1 | UVXY 1d: 64% WR +2.0% | n=44",
                           'hedge'))

        if rsp_spy is not None and rsp_spy < -1.0:
            bk = int(bayesian_kelly(18, 8, 2.5, 1.5))
            alerts.append(('[HEDGE] QQQ>79 + VERY NARROW BREADTH [T1 ROBUST]',
                           f"QQQ>79 + RSP/SPY 10d={rsp_spy:+.1f}%<-1 | UVXY 1d: 69% WR +2.5% worst -2.1% | n=26 | BK={bk}%",
                           'hedge'))
        elif rsp_spy is not None and rsp_spy < -0.5:
            alerts.append(('[HEDGE] QQQ>79 + NARROW BREADTH [T1 ROBUST]',
                           f"QQQ>79 + RSP/SPY 10d={rsp_spy:+.1f}%<-0.5 | UVXY 1d: 62% WR +2.7% | n=50",
                           'hedge'))

    # === GROUP 35: Volume-Conditioned OB [T1 ROBUST] ===
    vol_ratio, low_vol_consecutive = compute_qqq_volume_ratio(data)
    status['qqq_vol_ratio_20d'] = vol_ratio
    status['qqq_low_vol_consecutive'] = low_vol_consecutive

    if r('QQQ') > 79 and vol_ratio is not None:
        if vol_ratio < 0.85:
            alerts.append(('[HEDGE] QQQ>79 + VERY LOW VOL [T1 ROBUST]',
                           f"QQQ>79 + vol={vol_ratio:.2f}x<85% of 20d | UVXY 1d: 69% WR +2.5% worst -6.30% | n=42",
                           'hedge'))
        elif vol_ratio < 1.0:
            alerts.append(('[HEDGE] QQQ>79 + LOW VOL [T1 ROBUST]',
                           f"QQQ>79 + vol={vol_ratio:.2f}x<20d avg | UVXY 1d: 68% WR +2.0% worst -9.83% | n=72",
                           'hedge'))

    if low_vol_consecutive >= 5 and r('QQQ') > 79:
        alerts.append(('[CONFIRM] 5-day low vol streak + QQQ>79',
                       f"{low_vol_consecutive} consecutive sessions below 20d vol avg | LPL-style mechanical rally pattern",
                       'watch'))

    # === GROUP 36: Core 4x Stretched Persistence [T2 VALIDATED] ===
    core_4x_fires = all(r(t) > 79 for t in ['TQQQ','QQQ','SOXL','TECL'])
    full_complex = ['TQQQ','QQQ','SOXL','TECL','UPRO','SPHB','VOOG','SPY','IWM']
    complex_count = sum(1 for t in full_complex if r(t) > 79)
    status['core_4x_active'] = core_4x_fires
    status['ob_complex_count'] = complex_count

    if core_4x_fires and qqq_streak >= 2:
        alerts.append(('[HEDGE] CORE 4x DAY 2+ PERSISTENT [T2]',
                       f"Core 4x all>79, Day {qqq_streak} of streak | UVXY 1d: 75% WR +4.1% worst -0.32% | n=7 (small-n caveat)",
                       'hedge'))
    if complex_count >= 8 and qqq_streak >= 2:
        alerts.append(('[HEDGE] 8+ TICKER OB PERSISTENCE [T2]',
                       f"{complex_count}/9 tickers >79, Day {qqq_streak} | UVXY 1d: 75% WR +3.5% worst -1.55% | n=8",
                       'hedge'))

    # === GROUP 37: QQQ-QQQE RSI Divergence (Mega-Cap Driven) [T1 ROBUST] ===
    qqq_rsi_v = r('QQQ')
    qqqe_rsi_v = r('QQQE')
    rsi_div_val = qqq_rsi_v - qqqe_rsi_v
    status['qqq_qqqe_rsi_div'] = round(rsi_div_val, 2)

    if qqq_rsi_v > 79 and rsi_div_val > 5:
        alerts.append(('[HEDGE] QQQ>79 + RSI DIV>5 MEGA-CAP DRIVEN [T1 ROBUST]',
                       f"QQQ-QQQE RSI diff={rsi_div_val:+.1f}>5 | Mega-cap driven rally | UVXY 1d: 65% WR +2.8% | n=66",
                       'hedge'))

    return alerts, status


def compute_move_section(data, indicators):
    mk = '^MOVE'
    if mk not in data or mk not in indicators: return {}
    c = data[mk]['Close']
    if isinstance(c, pd.DataFrame): c = c.iloc[:,0]
    price = indicators[mk]['price']; rsi = indicators[mk]['rsi10']
    ch20 = 0
    if len(c) >= 21:
        prev = float(c.iloc[-21])
        if prev > 0: ch20 = ((price/prev)-1)*100
    rs = calculate_rsi_wilder(c, 10)
    was_ob = float(rs.iloc[-10:].max()) > 79 if len(rs) >= 10 else False
    return {'price':round(price,2),'rsi':round(rsi,1),'change_20d':round(ch20,1),'19A':price>115,'19B':ch20>50,'19C':was_ob and rsi<60}


def compute_rolling_betas(data, regime='UNKNOWN'):
    if 'SPY' not in data: return []
    sc = data['SPY']['Close']
    if isinstance(sc, pd.DataFrame): sc = sc.iloc[:,0]
    sr = sc.pct_change()
    def _b(t, w):
        if t not in data: return None
        ac = data[t]['Close']
        if isinstance(ac, pd.DataFrame): ac = ac.iloc[:,0]
        ar = ac.pct_change()
        ci = sr.dropna().index.intersection(ar.dropna().index)
        if len(ci) < w+10: return None
        cov = ar.loc[ci].rolling(w).cov(sr.loc[ci])
        var = sr.loc[ci].rolling(w).var()
        v = (cov/var).iloc[-1]
        return round(float(v),3) if not pd.isna(v) else None
    wts = {'Equity Sleeve':0.49,'Lev Equity':0.13,'MF Rotation':0.15,'Gold':0.07,'Vol/Hedge':0.10,'Bonds':0.07}
    groups = [('Equity Sleeve',[('SPY',1.0)]),('Lev Equity',[('UPRO',1.0)]),('MF Rotation',[('CTA',.25),('DBMF',.25),('BTAL',.3),('KMLM',.2)]),('Gold',[('GLD',1.0)]),('Vol/Hedge',[('UVXY',1.0)]),('Bonds',[('TLT',.5),('SHY',.5)])]
    results = []; blend = {'name':'Est. Blend','b63':0,'b126':0,'b252':0}
    for gn, tw in groups:
        row = {'name':gn}
        for wn, w in [('b63',63),('b126',126),('b252',252)]:
            gb, tot = 0, 0
            for t, wt in tw:
                b = _b(t, w)
                if b is not None: gb += b*wt; tot += wt
            if tot > 0: gb = gb/tot; row[wn] = round(gb,3); blend[wn] += gb*wts.get(gn,0)
            else: row[wn] = None
        results.append(row)
    for k in ['b63','b126','b252']: blend[k] = round(blend[k],3)
    results.append(blend)
    return results


# =============================================================================
# HORMUZ INTELLIGENCE (Windward)
# =============================================================================
# Pulls structured Hormuz intel from insights.windward.ai. Writes to a shared
# history file so the dashboard and monitor stay in sync. Provides a compact
# email block + alerts for attack jumps, dark-activity spikes, blockade flips.
HORMUZ_HISTORY_PATH = os.environ.get('HORMUZ_HISTORY_PATH', '/tmp/hormuz_history.json')


def fetch_hormuz_windward():
    """Scrape insights.windward.ai. Returns dict or None on any failure."""
    try:
        import requests as _req
        import re as _re

        headers = {
            "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                           "Version/17.0 Safari/605.1.15"),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "DNT": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Upgrade-Insecure-Requests": "1",
        }
        resp = _req.get("https://insights.windward.ai/", timeout=25, headers=headers)
        if resp.status_code != 200:
            print(f"[Hormuz] HTTP {resp.status_code}")
            return None
        raw = resp.text

        def _strip(s):
            return _re.sub(r'<[^>]+>', '', s or '').strip()

        def _kpi(label):
            pat = _re.compile(
                r'class="kpi-label"[^>]*>\s*' + _re.escape(label) + r'\s*</div>'
                r'\s*<div class="kpi-v"[^>]*>\s*([\d,\-]+)\s*</div>'
                r'(?:\s*<div class="kpi-sub"[^>]*>(.*?)</div>)?'
                r'(?:\s*<span class="kpi-delta[^"]*"[^>]*>(.*?)</span>)?',
                _re.DOTALL
            )
            m = pat.search(raw)
            if not m:
                return None
            try:
                return {"value": int(m.group(1).replace(",", "")),
                        "sub": _strip(m.group(2)),
                        "delta": _strip(m.group(3))}
            except ValueError:
                return None

        kpis = {
            "vessels_in_gulf": _kpi("Vessels in Gulf"),
            "inbound": _kpi("Hormuz Inbound Crossings"),
            "outbound": _kpi("Hormuz Outbound Crossings"),
            "dark_activity": _kpi("Dark Activity Events"),
            "attacks": _kpi("Vessels Attacked"),
        }
        if sum(1 for v in kpis.values() if v) < 3:
            print("[Hormuz] schema drift — <3 KPIs parsed")
            return None

        m = _re.search(r'Data as of\s+([0-9]{1,2}\s+[A-Z][a-z]+\s+[0-9]{4})', raw)
        as_of = m.group(1) if m else None

        m = _re.search(r'Daily Intelligence Summary.*?<p[^>]*>(.*?)</p>',
                       raw, _re.DOTALL | _re.IGNORECASE)
        intel = _strip(m.group(1)) if m else None

        risk = {"high": None, "moderate": None, "low": None}
        m = _re.search(r'id="risk-summary"(.*?)</div>\s*</div>\s*</div>', raw, _re.DOTALL)
        if m:
            for num_m in _re.finditer(
                r'<span[^>]*>\s*([\d,]+)\s*</span>\s*<span[^>]*>\s*([^<]+?)\s*</span>',
                m.group(1)
            ):
                try:
                    n = int(num_m.group(1).replace(",", ""))
                except ValueError:
                    continue
                lbl = num_m.group(2).lower()
                if "high" in lbl:
                    risk["high"] = n
                elif "moderate" in lbl:
                    risk["moderate"] = n
                elif "low" in lbl:
                    risk["low"] = n

        iran_count = None
        for m in _re.finditer(
            r'<span class="fl-name"[^>]*>\s*([^<]+?)\s*</span>.*?<span class="fl-cnt"[^>]*>\s*([\d,]+)\s*</span>',
            raw, _re.DOTALL
        ):
            if m.group(1).strip().lower() == "iran":
                try:
                    iran_count = int(m.group(2).replace(",", ""))
                except ValueError:
                    pass
                break

        blockade = bool(_re.search(r'BLOCKADE\s+ACTIVE', raw, _re.IGNORECASE))

        in_v = (kpis["inbound"] or {}).get("value")
        out_v = (kpis["outbound"] or {}).get("value")
        total = (in_v or 0) + (out_v or 0) if (in_v is not None or out_v is not None) else None

        return {
            "as_of": as_of, "total_transits": total,
            "inbound": kpis["inbound"], "outbound": kpis["outbound"],
            "vessels_in_gulf": kpis["vessels_in_gulf"],
            "dark_activity": kpis["dark_activity"], "attacks": kpis["attacks"],
            "risk": risk, "iran_flagged": iran_count,
            "blockade_active": blockade, "intel_summary": intel,
        }
    except Exception as e:
        print(f"[Hormuz] fetch error: {e}")
        return None


def hormuz_upsert_history(hz):
    """Append/update today's row; return (prev_row, today_row) for delta logic."""
    try:
        history = []
        if os.path.exists(HORMUZ_HISTORY_PATH):
            with open(HORMUZ_HISTORY_PATH, 'r') as fh:
                history = json.load(fh)
    except Exception:
        history = []

    row = {
        "ts": datetime.now().isoformat(timespec='seconds'),
        "as_of": hz.get("as_of"),
        "vessels_in_gulf": (hz.get("vessels_in_gulf") or {}).get("value"),
        "total_transits": hz.get("total_transits"),
        "dark_activity": (hz.get("dark_activity") or {}).get("value"),
        "attacks": (hz.get("attacks") or {}).get("value"),
        "iran_flagged": hz.get("iran_flagged"),
        "blockade_active": hz.get("blockade_active"),
    }
    today_key = row.get("as_of") or row["ts"][:10]
    prev_row = None
    for r in reversed(history):
        r_key = r.get("as_of") or (r.get("ts") or "")[:10]
        if r_key and r_key != today_key:
            prev_row = r
            break

    seen = {}
    for r in history:
        k = r.get("as_of") or r.get("ts")
        seen[k] = r
    seen[row.get("as_of") or row["ts"]] = row
    try:
        with open(HORMUZ_HISTORY_PATH, 'w') as fh:
            json.dump(list(seen.values())[-60:], fh, indent=2)
    except Exception as e:
        print(f"[Hormuz] save history failed: {e}")
    return prev_row, row


def hormuz_build_alerts(hz, prev_row, today_row):
    """Return list of (title, msg, kind) matching the new email template style
    using [WARN]/[EXIT]/[WATCH] bracket prefixes."""
    alerts = []

    # Attack-count jump — each new attack has historically been an equity vol event
    today_attacks = today_row.get("attacks")
    prev_attacks = (prev_row or {}).get("attacks")
    if today_attacks is not None and prev_attacks is not None:
        delta = today_attacks - prev_attacks
        if delta >= 1:
            alerts.append((
                '[EXIT] HORMUZ ATTACK(S)',
                f"Windward reports {delta} new vessel attack(s) today "
                f"(cumulative: {today_attacks}, was {prev_attacks}). "
                f"Historically each jump is an equity vol event — watch UVXY/oil names.",
                'exit'
            ))

    # Dark activity spike — >30% day-over-day increase on a non-trivial base
    today_dark = today_row.get("dark_activity")
    prev_dark = (prev_row or {}).get("dark_activity")
    if today_dark is not None and prev_dark and prev_dark >= 20:
        if today_dark > prev_dark * 1.3:
            pct = (today_dark / prev_dark - 1) * 100
            alerts.append((
                '[WARN] HORMUZ DARK ACTIVITY SPIKE',
                f"Dark-activity events {prev_dark} -> {today_dark} (+{pct:.0f}%). "
                f"Leading indicator for sanctions-evasion and interdictions.",
                'warning'
            ))

    # Blockade status change
    today_blockade = today_row.get("blockade_active")
    prev_blockade = (prev_row or {}).get("blockade_active")
    if prev_blockade is not None and today_blockade != prev_blockade:
        if today_blockade:
            alerts.append((
                '[EXIT] HORMUZ BLOCKADE ACTIVATED',
                f"Windward banner shows US blockade now ACTIVE (was inactive). "
                f"UNG/SLV/VLO/STNG/CF/MOS structurally supported.",
                'exit'
            ))
        else:
            alerts.append((
                '[BUY] HORMUZ BLOCKADE LIFTED',
                f"Windward banner shows US blockade INACTIVE (was active). "
                f"Watch for oil/energy mean reversion.",
                'buy'
            ))

    # Transit collapse — sustained low transit count
    today_transits = today_row.get("total_transits")
    if today_transits is not None and today_transits < 15:
        alerts.append((
            '[WATCH] HORMUZ TRANSIT LOW',
            f"Only {today_transits} transits today (baseline ~138). "
            f"Dead-hand framework still in force — insurance withdrawn.",
            'watch'
        ))

    return alerts


def format_hormuz_block(hz, prev_row, today_row):
    """Compact Hormuz section for the email body. Matches the new template style
    (no emoji, ASCII decorations, consistent with the other SIGNAL sections)."""
    if not hz:
        return ""

    lines = [
        "=" * 70,
        "STRAIT OF HORMUZ - WINDWARD INTELLIGENCE",
        "=" * 70,
    ]

    as_of = hz.get("as_of") or "unknown"
    blockade_txt = "[BLOCKADE ACTIVE]" if hz.get("blockade_active") else "[blockade inactive]"
    lines.append(f"Data as of: {as_of}   {blockade_txt}")
    lines.append("")

    def _fmt(kpi):
        if not kpi or kpi.get("value") is None:
            return "--"
        val = kpi["value"]
        delta = kpi.get("delta", "")
        return f"{val:,}" + (f" ({delta})" if delta else "")

    transits = hz.get("total_transits")
    transit_str = f"{transits}" if transits is not None else "--"
    in_v = (hz.get("inbound") or {}).get("value")
    out_v = (hz.get("outbound") or {}).get("value")
    in_sub = (hz.get("inbound") or {}).get("sub", "")
    out_sub = (hz.get("outbound") or {}).get("sub", "")
    lines.append(f"  Total transits:   {transit_str}  ({in_v or 0} in [{in_sub}] / {out_v or 0} out [{out_sub}])")
    lines.append(f"  Vessels in Gulf:  {_fmt(hz.get('vessels_in_gulf'))}")
    lines.append(f"  Dark activity:    {_fmt(hz.get('dark_activity'))}")
    lines.append(f"  Attacks (total):  {_fmt(hz.get('attacks'))}")
    if hz.get("iran_flagged") is not None:
        lines.append(f"  Iran-flagged:     {hz['iran_flagged']}")

    risk = hz.get("risk") or {}
    if risk.get("high") is not None:
        lines.append(f"  Risk tiers:       {risk.get('high','--')} High / {risk.get('moderate','--')} Mod / {risk.get('low','--')} Low")

    if prev_row:
        prev_as_of = prev_row.get("as_of", "prev")
        dow_deltas = []
        for key, label in [("total_transits", "transits"),
                           ("vessels_in_gulf", "gulf"),
                           ("dark_activity", "dark"),
                           ("attacks", "attacks")]:
            t = today_row.get(key)
            p = prev_row.get(key)
            if t is not None and p is not None:
                d = t - p
                sign = "+" if d > 0 else ""
                dow_deltas.append(f"{label} {sign}{d}")
        if dow_deltas:
            lines.append(f"  Delta vs {prev_as_of}: {', '.join(dow_deltas)}")

    intel = hz.get("intel_summary")
    if intel:
        lines.append("")
        lines.append("  Daily Intelligence Summary:")
        words = intel.split()
        line = "    "
        for w in words:
            if len(line) + len(w) + 1 > 92:
                lines.append(line.rstrip())
                line = "    " + w + " "
            else:
                line += w + " "
        if line.strip():
            lines.append(line.rstrip())

    lines.append("")
    lines.append("  Source: insights.windward.ai")
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# ROBOT JAMES MODULE
# =============================================================================
# Ensemble 50/50 mom+below-SMA base + 4-way EO strict overlay.
#
# Base (TD1 -> TD15 close):
#   - Momentum-60d pick: highest trailing 60-day return from SPY/QQQ/SMH
#   - Below-SMA50 pick:  most-below-SMA50 from SPY/QQQ/SMH
#   - If rules AGREE -> 60% that equity + 40% GLD
#   - If rules DISAGREE -> 30% mom_pick + 30% sma_pick + 40% GLD
#
# Overlay (TD15 -> EOM close):
#   - 4-way EO strict: biggest MTD lagger among SPY/QQQ/SMH/TLT
#   - TLT lagger -> 100% TMF (no gate)
#   - Equity lagger -> 100% leveraged ETF only if below SMA50; else STRICT SKIP
#     (hold base through EOM - the skip is the point, not a bug)
#
# Execution: MOC by 3:50 PM ET. After-hours fallback if missed.
#
# State schema matches chf_dashboard_server.py's _rj_compute_state_live() so
# the dashboard's /api/robot_james endpoint can read the file this writes.
# =============================================================================
RJ_STATE_FILE = os.environ.get('ROBOT_JAMES_STATE', 'robot_james_state.json')
RJ_LEV_MAP = {'SPY': 'UPRO', 'QQQ': 'TQQQ', 'SMH': 'SOXL', 'TLT': 'TMF'}


def _rj_safe_close(data, ticker):
    """Return Close series or None."""
    if ticker not in data:
        return None
    df = data[ticker]
    if len(df) == 0:
        return None
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close


def rj_compute_state(data):
    """Compute full Robot James state. Returns None if SPY data missing.
    Writes to RJ_STATE_FILE as a side effect so the dashboard can consume it."""
    try:
        spy = _rj_safe_close(data, 'SPY')
        if spy is None or len(spy) < 61:
            print("[RJ] insufficient SPY data")
            return None

        ref = pd.Timestamp.today().normalize()
        trading_days = sorted(spy.index.normalize().unique().tolist())
        month_start = ref.replace(day=1)
        tds_past = [d for d in trading_days
                    if pd.Timestamp(d).normalize() >= month_start
                    and pd.Timestamp(d).normalize() <= ref]
        if not tds_past:
            print("[RJ] no trading days this month yet")
            return None

        current_td = len(tds_past)
        td1_date = pd.Timestamp(tds_past[0]).normalize()

        # Project forward: how many trading days this whole month?
        next_month = (month_start + pd.DateOffset(months=1)).replace(day=1)
        all_tds = [d for d in trading_days
                   if pd.Timestamp(d).normalize() >= month_start
                   and pd.Timestamp(d).normalize() < next_month]
        last_known = pd.Timestamp(trading_days[-1]).normalize()
        if last_known < next_month - pd.Timedelta(days=1):
            scan = last_known + pd.Timedelta(days=1)
            while scan < next_month:
                if scan.weekday() < 5:
                    all_tds.append(scan)
                scan += pd.Timedelta(days=1)
        all_tds = sorted(set(all_tds))
        total_tds = len(all_tds)
        td14 = pd.Timestamp(all_tds[13]).normalize() if total_tds >= 14 else None
        td15 = pd.Timestamp(all_tds[14]).normalize() if total_tds >= 15 else None
        eom = pd.Timestamp(all_tds[-1]).normalize() if total_tds >= 1 else None

        latest = trading_days[-1]

        # --- Base picks (SPY/QQQ/SMH) ---
        mom60, below_sma = {}, {}
        for t in ['SPY', 'QQQ', 'SMH']:
            c = _rj_safe_close(data, t)
            if c is None or len(c) < 61:
                continue
            c_sub = c[c.index <= latest]
            if len(c_sub) < 61:
                continue
            mom60[t] = float(c_sub.iloc[-1] / c_sub.iloc[-61] - 1)
            sma50 = c_sub.rolling(50).mean()
            if pd.notna(sma50.iloc[-1]) and sma50.iloc[-1] > 0:
                below_sma[t] = float(c_sub.iloc[-1] / sma50.iloc[-1] - 1)

        mom_pick = max(mom60, key=mom60.get) if mom60 else 'SPY'
        sma_pick = min(below_sma, key=below_sma.get) if below_sma else 'SPY'
        agree = (mom_pick == sma_pick)
        weights = {mom_pick: 0.60} if agree else {mom_pick: 0.30, sma_pick: 0.30}

        # --- Overlay projection (SPY/QQQ/SMH/TLT MTD lagger) ---
        # "if TD15 were today" - use MTD through yesterday so projection is stable
        yesterday = trading_days[-2] if len(trading_days) >= 2 else trading_days[-1]
        mtd = {}
        for t in ['SPY', 'QQQ', 'SMH', 'TLT']:
            c = _rj_safe_close(data, t)
            if c is None:
                continue
            idx = c.index.normalize()
            td1_sub = c[idx <= td1_date]
            yday_sub = c[idx <= pd.Timestamp(yesterday).normalize()]
            if len(td1_sub) == 0 or len(yday_sub) == 0:
                continue
            mtd[t] = float(yday_sub.iloc[-1] / td1_sub.iloc[-1] - 1)

        lagger = min(mtd, key=mtd.get) if mtd else 'TLT'

        # Gate check - only matters for equity laggers
        gate_status = {}
        strict_skip = False
        skip_reason = None
        signal_asset = None

        if lagger == 'TLT':
            signal_asset = 'TMF'  # no gate
        elif lagger in ('SPY', 'QQQ', 'SMH'):
            c = _rj_safe_close(data, lagger)
            if c is not None:
                c_sub = c[c.index.normalize() <= pd.Timestamp(latest).normalize()]
                sma50 = c_sub.rolling(50).mean()
                if pd.notna(sma50.iloc[-1]) and sma50.iloc[-1] > 0:
                    pct = float(c_sub.iloc[-1] / sma50.iloc[-1] - 1)
                    gate_status[lagger] = pct
                    if c_sub.iloc[-1] < sma50.iloc[-1]:
                        signal_asset = RJ_LEV_MAP[lagger]
                    else:
                        strict_skip = True
                        skip_reason = f"{lagger} is {pct*100:+.1f}% vs SMA50 (above trend) - gate FAIL"

        # Fill gate status for all equity candidates (dashboard display)
        for t in ['SPY', 'QQQ', 'SMH']:
            if t not in gate_status:
                c = _rj_safe_close(data, t)
                if c is not None:
                    c_sub = c[c.index.normalize() <= pd.Timestamp(latest).normalize()]
                    sma50 = c_sub.rolling(50).mean()
                    if pd.notna(sma50.iloc[-1]) and sma50.iloc[-1] > 0:
                        gate_status[t] = float(c_sub.iloc[-1] / sma50.iloc[-1] - 1)

        # --- Action type ---
        if current_td == 1:
            atype, severity = 'TD1_SET_BASE', 'action_required'
        elif current_td == 15:
            atype, severity = 'TD15_OVERLAY', 'action_required'
        elif eom is not None and ref == eom:
            atype, severity = 'EOM_ROTATE_BACK', 'action_required'
        else:
            atype, severity = 'HOLD', 'none'

        next_date, next_type = None, None
        if current_td < 15 and td15 is not None:
            next_date, next_type = td15, 'TD15_OVERLAY'
        elif current_td < total_tds and eom is not None:
            next_date, next_type = eom, 'EOM_ROTATE_BACK'
        days_until = (next_date - ref).days if next_date is not None else None

        state = {
            'ref_date': str(ref.date()),
            'computed_at': datetime.now().isoformat(timespec='seconds'),
            'current_td': current_td,
            'total_tds_this_month': total_tds,
            'td1_date': str(td1_date.date()) if td1_date is not None else None,
            'td14_date': str(td14.date()) if td14 is not None else None,
            'td15_date': str(td15.date()) if td15 is not None else None,
            'eom_date': str(eom.date()) if eom is not None else None,
            'is_td1': current_td == 1,
            'is_td15': current_td == 15,
            'is_eom': eom is not None and ref == eom,
            'projected_base': {
                'mom_pick': mom_pick,
                'below_sma_pick': sma_pick,
                'weights': weights,
                'gld_weight': 0.40,
                'rules_agree': agree,
                'agree_ticker': mom_pick if agree else None,
                'mom60_values': mom60,
                'below_sma_values': below_sma,
            },
            'projected_overlay': {
                'lagger': lagger,
                'mtd_values': mtd,
                'signal_asset': signal_asset,
                'gate_status': gate_status,
                'strict_skip': strict_skip,
                'skip_reason': skip_reason,
            },
            'action': {
                'type': atype,
                'severity': severity,
                'next_decision_date': str(next_date.date()) if next_date is not None else None,
                'next_decision_type': next_type,
                'days_until_next_decision': days_until,
            },
        }

        # Write state file so the dashboard can consume it
        try:
            with open(RJ_STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[RJ] state file write failed: {e}")

        return state
    except Exception as e:
        print(f"[RJ] compute error: {e}")
        return None


def _rj_pct(v):
    """Format a fraction as +X.XX% with sign, or '--' if None."""
    if v is None:
        return '--'
    return f"{v*100:+.2f}%"


def format_rj_block(state):
    """Render the Robot James section for the email body.
    Matches the v4.8 template style (ASCII, bracketed prefixes)."""
    if state is None:
        return ""

    td = state.get('current_td', 0)
    total = state.get('total_tds_this_month', 21)
    pb = state.get('projected_base', {})
    ov = state.get('projected_overlay', {})
    action = state.get('action', {})

    lines = [
        "=" * 70,
        f"ROBOT JAMES - TD {td} of {total}",
        "=" * 70,
    ]

    # Current position description
    if td > 15 and ov.get('signal_asset') and not ov.get('strict_skip'):
        position = f"100% {ov['signal_asset']} (overlay active - lagger was {ov.get('lagger')})"
    elif td > 15 and ov.get('strict_skip'):
        position = (f"Strict skip - holding base "
                    f"(30% {pb.get('mom_pick')} + 30% {pb.get('below_sma_pick')} + 40% GLD)")
    elif pb.get('rules_agree'):
        position = f"60% {pb.get('agree_ticker')} + 40% GLD (rules agree)"
    elif pb.get('mom_pick') and pb.get('below_sma_pick'):
        position = (f"30% {pb.get('mom_pick')} + 30% {pb.get('below_sma_pick')} + 40% GLD "
                    f"(rules disagree)")
    else:
        position = "Unknown"

    lines.append(f"Position: {position}")
    lines.append("")

    # Action banner
    atype = action.get('type')
    if atype == 'TD1_SET_BASE':
        lines.append("[ACTION] TD1 - Set new base at MOC today")
    elif atype == 'TD15_OVERLAY':
        if ov.get('strict_skip'):
            lines.append(f"[ACTION] TD15 - STRICT SKIP (hold base). {ov.get('skip_reason')}")
        elif ov.get('signal_asset'):
            lines.append(f"[ACTION] TD15 - Rotate to 100% {ov['signal_asset']} at MOC today")
    elif atype == 'EOM_ROTATE_BACK':
        lines.append("[ACTION] EOM - Rotate back to cash/base at MOC today")
    elif atype == 'HOLD':
        lines.append("[HOLD] No action today - hold current position")

    # Next decision
    nd_date = action.get('next_decision_date')
    nd_type = action.get('next_decision_type')
    nd_days = action.get('days_until_next_decision')
    if nd_date:
        lines.append(f"Next: {nd_type} on {nd_date} ({nd_days}d)")
    lines.append("")

    # Base picks table
    lines.append("Base picks (SPY/QQQ/SMH):")
    mom_vals = pb.get('mom60_values', {})
    below_vals = pb.get('below_sma_values', {})
    mom_pick = pb.get('mom_pick')
    sma_pick = pb.get('below_sma_pick')
    for t in ['SPY', 'QQQ', 'SMH']:
        mom_marker = "  <- MOM" if t == mom_pick else ""
        sma_marker = "  <- BELOW-SMA" if t == sma_pick else ""
        lines.append(f"  {t}: 60d mom {_rj_pct(mom_vals.get(t))}{mom_marker}  |  "
                     f"vs SMA50 {_rj_pct(below_vals.get(t))}{sma_marker}")
    if pb.get('rules_agree'):
        lines.append(f"  -> RULES AGREE on {pb.get('agree_ticker')}: 60% + 40% GLD")
    else:
        lines.append(f"  -> RULES DISAGREE: 30% {mom_pick} + 30% {sma_pick} + 40% GLD")
    lines.append("")

    # Overlay projection
    lines.append("Overlay projection (if TD15 were today):")
    mtd = ov.get('mtd_values', {})
    gate = ov.get('gate_status', {})
    lagger = ov.get('lagger')
    for t in ['SPY', 'QQQ', 'SMH', 'TLT']:
        lagger_marker = "  <- LAGGER" if t == lagger else ""
        gate_val = _rj_pct(gate.get(t)) if t != 'TLT' else 'n/a'
        lines.append(f"  {t}: MTD {_rj_pct(mtd.get(t))}{lagger_marker}  |  "
                     f"vs SMA50 {gate_val}")

    if ov.get('strict_skip'):
        lines.append(f"  -> STRICT SKIP: {ov.get('skip_reason')}")
    elif ov.get('signal_asset'):
        lines.append(f"  -> Would rotate to: {ov['signal_asset']}")
    lines.append("")

    lines.append("MANUAL execution only - daily rebalance in Composer destroys the edge.")
    lines.append("")
    return "\n".join(lines)


def format_email(alerts, status, data, is_preclose=False, composer_trades=None, composer_perf=None, hormuz_block="", rj_block=""):
    now = datetime.now()
    timing = "MID-DAY PREVIEW (11:00 AM)" if is_preclose else "MARKET CLOSE CONFIRMATION (4:05 PM)"
    indicators = status.get('indicators', {})
    regime = status.get('regime', 'UNKNOWN')

    body = f"""
{'='*70}
MARKET SIGNAL MONITOR - {timing}
{now.strftime('%Y-%m-%d %H:%M')} ET
{'='*70}

"""
    # Intramonth momentum cycle
    cal = compute_calendar_position()
    body += f"{cal['emoji']} INTRAMONTH CYCLE: T-{cal['days']} | {cal['zone']}\n"
    if cal['in_window']:
        body += f"   Dip-buy conviction: BOOSTED (buying forced institutional selling)\n"
        body += f"   TQQQ avg +0.08%/day in window vs +0.26%/day outside\n"
    else:
        body += f"   Dip-buy conviction: Normal\n"
    body += "\n"

    # Robot James — injected after intramonth context, before Hormuz.
    # Leads the email on action days because subject line already flags them.
    if rj_block:
        body += rj_block + "\n"

    # Hormuz intelligence block — injected right after RJ,
    # before the main signal alerts panel
    if hormuz_block:
        body += hormuz_block + "\n"

    if alerts:
        for cat, types in [("BUY SIGNALS:", ['buy']), ("EXIT/SHORT SIGNALS:", ['exit','short']), ("WARNINGS/WATCH:", ['warning','hedge','watch'])]:
            filtered = [a for a in alerts if a[2] in types]
            if filtered:
                body += f"{cat}\n{'-'*50}\n"
                for title, msg, _ in filtered:
                    body += f"{title}\n{msg}\n\n"
    else:
        body += "No signals triggered today.\n\n"

    # Indicators table
    body += f"\n{'='*70}\nCURRENT INDICATOR STATUS\n{'='*70}\n\n"
    body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}\n{'-'*50}\n"
    for t in ['SPY','QQQ','SMH','GLD','USDU','XLP','TLT','HYG','XLF','UVXY','BTC-USD','AMD','NVDA']:
        if t in indicators:
            i = indicators[t]
            pr = f"${i['price']:.2f}" if i['price'] < 10000 else f"${i['price']:,.0f}"
            body += f"{t:<10} {pr:>12} {i['rsi10']:>10.1f} {i['pct_above_sma200']:>+11.1f}%\n"

    # 3x ETFs
    body += f"\n{'='*70}\n3x LEVERAGED ETFs\n{'='*70}\n"
    body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}  Signal\n{'-'*65}\n"
    for t in ['NAIL','CURE','FAS','LABU','TQQQ','SOXL','TECL','DRN','DFEN']:
        if t in indicators:
            i = indicators[t]
            pr = f"${i['price']:.2f}" if i['price']<10000 else f"${i['price']:,.0f}"
            sig = ""
            if i['rsi10']<21: sig="[BUY] OVERSOLD"
            elif i['rsi10']<30: sig="[BUY] Watch"
            elif i['rsi10']>85: sig="[EXIT] OVERBOUGHT"
            elif i['rsi10']>79: sig="[WARN] Extended"
            body += f"{t:<10} {pr:>12} {i['rsi10']:>10.1f} {i['pct_above_sma200']:>+11.1f}%  {sig}\n"

    # Other ETFs
    body += f"\n{'='*70}\nOTHER ETFs\n{'='*70}\n"
    body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}\n{'-'*50}\n"
    for t in ['XLV','XLU','XLE','TMV','VOOV','VOOG','VTV','QQQE','BOIL','EURL','YINN','KORU','INDL','EDC']:
        if t in indicators:
            i = indicators[t]
            pr = f"${i['price']:.2f}" if i['price']<10000 else f"${i['price']:,.0f}"
            body += f"{t:<10} {pr:>12} {i['rsi10']:>10.1f} {i['pct_above_sma200']:>+11.1f}%\n"

    # SMH levels
    if 'SMH' in indicators:
        s = indicators['SMH']; s200 = s['sma200']
        body += f"\n{'='*70}\nSMH/SOXL LEVELS\n{'='*70}\nPrice: ${s['price']:.2f} | SMA200: ${s200:.2f} | {s['pct_above_sma200']:+.1f}% | Days below: {status.get('smh_days_below_sma200',0)}\n"
        body += f"  30% Trim: ${s200*1.30:.2f} | 35% Warn: ${s200*1.35:.2f} | 40% Sell: ${s200*1.40:.2f}\n"

    # Crisis Alpha
    body += f"\n{'='*70}\nCRISIS ALPHA REGIME: {regime}\n{'='*70}\n"
    for n,d in [('SPY',indicators.get('SPY',{})),('QQQ',indicators.get('QQQ',{})),('SMH',indicators.get('SMH',{}))]:
        if d: body += f"{n}: MaRet(10d)={d.get('maret10d',0):+.2f}%/day | CumRet 10/30/50d: {d.get('cumRet10d',0):.1f}%/{d.get('cumRet30d',0):.1f}%/{d.get('cumRet50d',0):.1f}% | VolR: {d.get('vol_ratio',1):.2f}\n"

    # DFEN BB
    dbb = status.get('dfen_bb',{})
    if dbb:
        body += f"\nDFEN BB: Price=${dbb['price']:.2f} RSI={dbb['rsi']:.1f} | Upper=${dbb['upper']:.2f} SMA20=${dbb['sma20']:.2f} Lower=${dbb['lower']:.2f} | %B={dbb['pctB']:.3f} Width={dbb['width']:.1f}%\n"

    # Rolling Beta
    betas = compute_rolling_betas(data, regime)
    body += f"\n{'='*70}\nROLLING BETA vs SPY\n{'='*70}\n"
    body += f"{'Group':<25} {'63d':>8} {'126d':>8} {'252d':>8}\n{'-'*50}\n"
    for row in betas:
        b63 = f"{row['b63']:+.2f}" if row.get('b63') is not None else 'N/A'
        b126 = f"{row['b126']:+.2f}" if row.get('b126') is not None else 'N/A'
        b252 = f"{row['b252']:+.2f}" if row.get('b252') is not None else 'N/A'
        if row['name']=='Est. Blend': body += '-'*50+'\n'
        body += f"{row['name']:<25} {b63:>8} {b126:>8} {b252:>8}\n"
    bl = next((r for r in betas if r['name']=='Est. Blend'), None)
    if bl and bl.get('b63') and bl['b63'] > 2.0:
        body += "\nWARNING: HIGH leverage. Consider increasing KMLM/CTA/GLD.\n"

    # GLD & Miners
    body += f"\n{'='*70}\nGLD & MINERS\n{'='*70}\n"
    for t in ['GLD','GDX','GDXJ','JNUG','NUGT']:
        if t in indicators:
            i = indicators[t]
            body += f" {t:>5} ${i['price']:>10.2f} RSI={i['rsi10']:.1f} {i['pct_above_sma200']:+.1f}%\n"

    # UVXY Vol Regime
    uv = status.get('uvxy_vol', {})
    if uv:
        body += f"\n{'='*70}\nUVXY VOL REGIME SHIFT\n{'='*70}\nUVXY: ${indicators.get('UVXY',{}).get('price',0):.2f} | SMA200: ${uv.get('sma200',0):.2f} | {uv.get('pct',0):+.1f}%\n"

    # FXY (NEW)
    if 'FXY' in indicators:
        f = indicators['FXY']
        body += f"\n{'='*70}\nFXY CARRY TRADE (Group 20)\n{'='*70}\nFXY: ${f['price']:.2f} RSI={f['rsi10']:.1f} | 20A(>75):{'ON' if f['rsi10']>75 else 'off'} | 20B(>70+TLT broken):{'ON' if f['rsi10']>70 and not indicators.get('TLT',{}).get('above_sma200',True) else 'off'}\n"

    # CPER (NEW)
    if 'CPER' in indicators:
        c = indicators['CPER']
        body += f"\n{'='*70}\nCPER COPPER REGIME (Group 21)\n{'='*70}\nCPER: ${c['price']:.2f} {'> EMA9' if c['above_ema9'] else '< EMA9'} | SPY {'> EMA9' if indicators.get('SPY',{}).get('above_ema9') else '< EMA9'}\n"

    # Mid-Month Rotation section superseded by Robot James block at the top
    # of the email (see rj_block injection). status['midmonth'] is no longer
    # populated; the 4-way ensemble renders in its own section instead.

    # DRIF
    drif = status.get('drif', {})
    if drif:
        body += f"\n{'='*70}\nDRIF VELOCITY FILTER\n{'='*70}\n"
        body += f"{'Ticker':>6} {'RSI':>6} {'5d Ret':>8} {'7d Ret':>8} {'Vel':>6} {'Gate':>6}  Status\n{'-'*70}\n"
        for t in ['SPY','QQQ','SMH']:
            d = drif.get(t,{})
            if d: body += f"{t:>6} {d['rsi']:>6.1f} {d['cumRet5d']:>+7.1f}% {d['cumRet7d']:>+7.1f}% {d['velocity']:>+5.0f} {d['gate']:>6}  {d['label']}\n"

    # MOVE
    move = compute_move_section(data, indicators)
    if move:
        body += f"\n{'='*70}\nMOVE INDEX\n{'='*70}\nPrice: {move['price']} | RSI: {move['rsi']} | 20d: {move['change_20d']:+.1f}%\n19A(>115):{'Active' if move['19A'] else '-'} | 19B(20d>50%):{'Active' if move['19B'] else '-'} | 19C(crush):{'Active' if move['19C'] else '-'}\n"

    # MARKET INTERNALS (Polygon breadth if available)
    breadth_json = os.path.join(os.environ.get('BREADTH_DATA_DIR', './data/breadth'), 'latest_indicators.json')
    if os.path.exists(breadth_json):
        try:
            import json
            with open(breadth_json) as f:
                bi = json.load(f)
            body += f"""
{'='*70}
MARKET INTERNALS (Polygon Breadth)
{'='*70}

 ZBT (10d EMA):         {bi.get('zbt_ema','?'):>8}   {bi.get('zbt_zone','')}
 McClellan Oscillator:  {bi.get('mcclellan','?'):>8}   {bi.get('mcl_zone','')} ({bi.get('mcl_direction','')})
   19d EMA: {bi.get('mcl_ema19','?')}  |  39d EMA: {bi.get('mcl_ema39','?')}  |  Summation: {bi.get('mcl_summation','?')}

 Advancing: {bi.get('advancing','?')}  |  Declining: {bi.get('declining','?')}  |  Ratio: {bi.get('ratio','?')}
"""
            if bi.get('zbt_thrust'):
                body += " *** ZBT THRUST SIGNAL — historically near-100% forward returns ***\n"
        except Exception as e:
            print(f"  Breadth JSON read error: {e}")

    # Composer Dry-Run Trades (both emails — see pending trades before close)
    if composer_trades:
        body += f"\n{'='*70}\nCOMPOSER PENDING TRADES (Dry Run)\n{'='*70}\n"

        # First: consolidated net portfolio view (what you'll hold AFTER trades)
        net_positions = {}  # ticker → {current_notional, trade_notional, next_notional, next_weight_sum, sym_count}
        total_portfolio = 0

        for acct in composer_trades:
            for sym in acct['symphonies']:
                sym_val = sym.get('value', 0) or 0
                total_portfolio += sym_val
                for t in sym.get('trades', []):
                    ticker = t['ticker']
                    notional = t.get('notional', 0)
                    next_wt = t.get('next_weight', 0) / 100  # convert from pct
                    if ticker not in net_positions:
                        net_positions[ticker] = {'trade_notional': 0, 'next_notional': 0, 'sym_count': 0}
                    net_positions[ticker]['trade_notional'] += notional
                    net_positions[ticker]['next_notional'] += sym_val * next_wt
                    net_positions[ticker]['sym_count'] += 1

        if net_positions:
            # Sort by absolute post-trade value
            sorted_net = sorted(net_positions.items(), key=lambda x: -abs(x[1]['next_notional']))

            body += f"\n  NET PORTFOLIO AFTER TRADES (all symphonies combined):\n"
            body += f"  {'Ticker':<8} {'Post-Trade $':>12} {'% Portfolio':>12} {'Net Change':>12} {'# Syms':>7}\n"
            body += f"  {'-'*55}\n"

            # Flag volatile instruments
            volatile = {'UVXY', 'SOXL', 'TQQQ', 'SOXS', 'UPRO', 'TECL', 'LABU', 'NAIL', 'FAS', 'CURE', 'TMV', 'BOIL'}

            for ticker, pos in sorted_net:
                post_val = pos['next_notional']
                pct = (post_val / total_portfolio * 100) if total_portfolio > 0 else 0
                net_chg = pos['trade_notional']
                flag = ' ⚠️' if ticker in volatile and abs(pct) > 10 else ''
                if abs(post_val) < 1:
                    continue
                body += f"  {ticker:<8} ${post_val:>11,.0f} {pct:>+11.1f}% ${net_chg:>+11,.0f} {pos['sym_count']:>6}{flag}\n"

            # Concentration warnings
            warnings = [(t, p['next_notional']/total_portfolio*100) for t, p in net_positions.items()
                        if t in volatile and total_portfolio > 0 and abs(p['next_notional']/total_portfolio*100) > 8]
            if warnings:
                body += f"\n  ⚠️ CONCENTRATION WARNINGS:\n"
                for t, pct in sorted(warnings, key=lambda x: -abs(x[1])):
                    body += f"    {t}: {pct:+.1f}% of portfolio — high-vol instrument, monitor closely\n"

        body += f"\n  {'─'*55}\n"
        body += f"  Per-symphony detail:\n"

        for acct in composer_trades:
            body += f"\n  {acct['account']}:\n"
            for sym in acct['symphonies']:
                trades = sym['trades']
                if not trades:
                    continue
                body += f"  {sym['symphony']} (${sym['value']:,.0f})\n"
                for t in trades:
                    arrow = '→' if t['prev_weight'] != t['next_weight'] else '='
                    body += f"    {t['side']:>4} {t['ticker']:<6} ${abs(t['notional']):>10,.2f}  {t['prev_weight']:>5.1f}% {arrow} {t['next_weight']:>5.1f}%\n"
        body += f"\n  NOTE: Trades execute at next rebalance. Dry-run is a preview, not a commitment.\n"

    # Composer Portfolio Performance & Win Rates
    if composer_perf:
        body += f"\n{'='*70}\nPORTFOLIO PERFORMANCE & WIN RATES\n{'='*70}\n"

        # Per-account summary
        for acct in composer_perf.get('accounts', []):
            wr = acct.get('win_rates', {})
            body += f"\n  {acct['account']}: ${acct['value']:,.0f} | Today: {acct['today_pct']:+.2f}%\n"

            if wr:
                streak = wr.get('streak', 0)
                streak_str = f"+{streak}d" if streak > 0 else f"{streak}d"
                streak_label = "winning" if streak > 0 else "losing"
                body += f"  Win Rates: "
                parts = []
                if 'daily_20d' in wr: parts.append(f"20d:{wr['daily_20d']:.0f}%")
                if 'daily_60d' in wr: parts.append(f"60d:{wr['daily_60d']:.0f}%")
                if 'daily_all' in wr: parts.append(f"All:{wr['daily_all']:.0f}%")
                body += ' | '.join(parts)
                if 'weekly_12w' in wr: body += f" | Wk(12w):{wr['weekly_12w']:.0f}%"
                if 'monthly_all' in wr: body += f" | Mo:{wr['monthly_all']:.0f}%"
                body += f" | Streak: {streak_str} ({streak_label})\n"

            # Symphony table
            syms = acct.get('symphonies', [])
            if syms:
                body += f"  {'Symphony':<28} {'Value':>10} {'Today':>8} {'Ann.Ret':>8} {'Sharpe':>7} {'MaxDD':>7}\n"
                body += f"  {'-'*68}\n"
                for s in sorted(syms, key=lambda x: -(x.get('value') or 0)):
                    name = s['name'][:27]
                    body += f"  {name:<28} ${s['value']:>9,.0f} {s['today_pct']:>+7.2f}% {s['ann_return']:>+7.1f}% {s['sharpe']:>6.2f} {s['max_dd']:>+6.1f}%\n"

        # Consolidated win rates
        cwr = composer_perf.get('consolidated_wr', {})
        if cwr:
            body += f"\n  CONSOLIDATED WIN RATES:\n"
            parts = []
            if 'daily_20d' in cwr: parts.append(f"20d: {cwr['daily_20d']:.0f}%")
            if 'daily_60d' in cwr: parts.append(f"60d: {cwr['daily_60d']:.0f}%")
            if 'daily_all' in cwr: parts.append(f"All-time: {cwr['daily_all']:.0f}%")
            body += f"  Daily: {' | '.join(parts)}\n"

    # Fibonacci
    body += f"\n{'='*70}\nFIBONACCI CONTEXT\n{'='*70}\n"
    for sym in ['SPY','QQQ','SMH']:
        if sym not in data: continue
        try:
            df = data[sym]; c = df['Close']
            if isinstance(c, pd.DataFrame): c = c.iloc[:,0]
            cl = float(c.iloc[-1])
            h = df['High'] if not isinstance(df['High'], pd.DataFrame) else df['High'].iloc[:,0]
            l = df['Low'] if not isinstance(df['Low'], pd.DataFrame) else df['Low'].iloc[:,0]
            h30=float(h.tail(30).max()); l30=float(l.tail(30).min()); d=h30-l30; up=cl>(h30+l30)/2
            body += f"\n {sym} (30d): H={h30:.2f} L={l30:.2f} C={cl:.2f} [{'UP' if up else 'DOWN'}]\n"
            for pct in [0.236,0.382,0.500,0.618]:
                lvl = (h30-d*pct) if up else (l30+d*pct)
                dist = (cl-lvl)/cl*100
                near = " <-- NEAR" if abs(dist)<1.5 else ""
                body += f"    {pct*100:.1f}%: ${lvl:.2f} ({dist:+.1f}%){near}\n"
        except: pass

    if is_preclose:
        body += f"\n{'='*70}\nNOTE: MID-DAY preview. Final at 4:05 PM ET.\n{'='*70}\n"
    return body


def send_email(subject, body):
    if not SENDER_EMAIL or not SENDER_PASSWORD or not RECIPIENT_EMAIL:
        print("Email not configured - printing to console:")
        print(f"Subject: {subject}")
        print(body)
        return False
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL; msg['To'] = RECIPIENT_EMAIL; msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        server = smtplib.SMTP('smtp.gmail.com', 587); server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD); server.send_message(msg); server.quit()
        print(f"Email sent to {RECIPIENT_EMAIL}"); return True
    except Exception as e:
        print(f"Email failed: {e}"); return False


# =============================================================================
# COMPOSER API — DRY RUN + PERFORMANCE
# =============================================================================
def _composer_get(path, timeout=15):
    if not COMPOSER_KEY_ID or not COMPOSER_KEY_SECRET:
        return None
    try:
        resp = req_lib.get(f"{COMPOSER_BASE}{path}",
            headers={"x-api-key-id": COMPOSER_KEY_ID, "authorization": f"Bearer {COMPOSER_KEY_SECRET}"},
            timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  Composer API error ({path}): {e}")
        return None

def _composer_post(path, payload=None, timeout=15):
    if not COMPOSER_KEY_ID or not COMPOSER_KEY_SECRET:
        return None
    try:
        resp = req_lib.post(f"{COMPOSER_BASE}{path}",
            headers={"x-api-key-id": COMPOSER_KEY_ID, "authorization": f"Bearer {COMPOSER_KEY_SECRET}",
                      "content-type": "application/json"},
            json=payload or {}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  Composer API error ({path}): {e}")
        return None


def fetch_composer_dry_run():
    """Fetch pending rebalance trades from Composer dry-run API."""
    if not COMPOSER_KEY_ID:
        return None
    import time as _t
    print("  Fetching Composer dry-run trades...")
    result = _composer_post("/dry-run", {"send_segment_event": False})
    if not result:
        return None

    trades_by_account = []
    for acct in result:
        acct_name = acct.get('account_name', acct.get('account_type', 'Unknown'))
        dry_run = acct.get('dry_run_result', {})
        acct_trades = []
        for sym_id, details in dry_run.items():
            sym_name = details.get('symphony_name', sym_id[:12])
            rebalanced = details.get('rebalanced', False)
            trades = details.get('recommended_trades', [])
            sym_val = details.get('symphony_value', 0)
            if trades:
                acct_trades.append({
                    'symphony': sym_name,
                    'value': sym_val,
                    'rebalanced': rebalanced,
                    'trades': [{
                        'ticker': t.get('ticker', ''),
                        'notional': t.get('notional', 0),
                        'quantity': t.get('quantity', 0),
                        'side': 'BUY' if t.get('notional', 0) > 0 else 'SELL',
                        'prev_weight': round((t.get('prev_weight', 0) or 0) * 100, 1),
                        'next_weight': round((t.get('next_weight', 0) or 0) * 100, 1),
                    } for t in trades if abs(t.get('notional', 0)) > 1],
                })
        if acct_trades:
            trades_by_account.append({'account': acct_name, 'symphonies': acct_trades})

    print(f"  Composer dry-run: {sum(len(a['symphonies']) for a in trades_by_account)} symphonies with pending trades")
    return trades_by_account


def fetch_composer_performance():
    """Fetch portfolio history and compute win rates."""
    if not COMPOSER_KEY_ID:
        return None
    import time as _t
    print("  Fetching Composer performance data...")

    acct_resp = _composer_get("/accounts/list")
    if not acct_resp or 'accounts' not in acct_resp:
        return None

    all_performance = []
    consolidated_daily_rets = []

    for acct in acct_resp['accounts']:
        aid = acct['account_uuid']
        atype = acct.get('account_type', 'Unknown')
        _t.sleep(1.1)

        # Get symphony stats (has today's change, annualized return, sharpe, etc.)
        sym_stats = _composer_get(f"/portfolio/accounts/{aid}/symphony-stats-meta")
        _t.sleep(1.1)

        # Get portfolio history for win rate calculation
        port_hist = _composer_get(f"/portfolio/accounts/{aid}/portfolio-history")
        _t.sleep(1.1)

        # Get total stats
        total_stats = _composer_get(f"/portfolio/accounts/{aid}/total-stats")

        # Compute portfolio-level win rates from history
        portfolio_wr = {}
        if port_hist and 'series' in port_hist and len(port_hist['series']) > 5:
            vals = np.array(port_hist['series'], dtype=float)
            daily_rets = np.diff(vals) / vals[:-1]
            consolidated_daily_rets.extend(daily_rets.tolist())

            n = len(daily_rets)
            # Daily win rate (trailing windows)
            if n >= 20:
                portfolio_wr['daily_20d'] = round(np.mean(daily_rets[-20:] > 0) * 100, 1)
            if n >= 60:
                portfolio_wr['daily_60d'] = round(np.mean(daily_rets[-60:] > 0) * 100, 1)
            portfolio_wr['daily_all'] = round(np.mean(daily_rets > 0) * 100, 1)

            # Weekly win rate (group into 5-day blocks)
            if n >= 10:
                weekly_rets = []
                for i in range(0, n - 4, 5):
                    chunk = vals[i:i+6]
                    if len(chunk) >= 2:
                        weekly_rets.append(chunk[-1] / chunk[0] - 1)
                if weekly_rets:
                    portfolio_wr['weekly_all'] = round(np.mean(np.array(weekly_rets) > 0) * 100, 1)
                    if len(weekly_rets) >= 12:
                        portfolio_wr['weekly_12w'] = round(np.mean(np.array(weekly_rets[-12:]) > 0) * 100, 1)

            # Monthly win rate (group into ~21-day blocks)
            if n >= 42:
                monthly_rets = []
                for i in range(0, n - 20, 21):
                    chunk = vals[i:i+22]
                    if len(chunk) >= 2:
                        monthly_rets.append(chunk[-1] / chunk[0] - 1)
                if monthly_rets:
                    portfolio_wr['monthly_all'] = round(np.mean(np.array(monthly_rets) > 0) * 100, 1)

            # Current streak
            streak = 0
            for r in reversed(daily_rets):
                if r > 0:
                    streak += 1
                else:
                    break
            if streak == 0:
                for r in reversed(daily_rets):
                    if r <= 0:
                        streak -= 1
                    else:
                        break
            portfolio_wr['streak'] = streak

        # Parse symphonies
        symphonies = []
        if sym_stats and 'symphonies' in sym_stats:
            for s in sym_stats['symphonies']:
                symphonies.append({
                    'name': s.get('name', 'Unknown'),
                    'value': s.get('value', 0),
                    'today_pct': round((s.get('last_percent_change', 0) or 0) * 100, 2),
                    'ann_return': round((s.get('annualized_rate_of_return', 0) or 0) * 100, 1),
                    'sharpe': round(s.get('sharpe_ratio', 0) or 0, 2),
                    'max_dd': round((s.get('max_drawdown', 0) or 0) * 100, 1),
                })

        acct_value = total_stats.get('portfolio_value', 0) if total_stats else 0
        acct_today_pct = total_stats.get('todays_percent_change', 0) if total_stats else 0
        label = 'Roth IRA' if 'roth' in atype.lower() else 'Traditional IRA' if 'trad' in atype.lower() else atype

        all_performance.append({
            'account': label,
            'value': acct_value,
            'today_pct': round(acct_today_pct * 100, 2) if acct_today_pct and abs(acct_today_pct) < 1 else round(acct_today_pct, 2) if acct_today_pct else 0,
            'win_rates': portfolio_wr,
            'symphonies': symphonies,
        })

    # Consolidated win rates
    consolidated_wr = {}
    if consolidated_daily_rets:
        dr = np.array(consolidated_daily_rets)
        n = len(dr)
        if n >= 20:
            consolidated_wr['daily_20d'] = round(np.mean(dr[-20:] > 0) * 100, 1)
        if n >= 60:
            consolidated_wr['daily_60d'] = round(np.mean(dr[-60:] > 0) * 100, 1)
        consolidated_wr['daily_all'] = round(np.mean(dr > 0) * 100, 1)

    print(f"  Composer performance: {len(all_performance)} accounts, {sum(len(a['symphonies']) for a in all_performance)} symphonies")
    return {'accounts': all_performance, 'consolidated_wr': consolidated_wr}


def main():
    print(f"Signal Monitor v4.8 at {datetime.now()}")
    print(f"Mode: {'MID-DAY (11:00 AM)' if IS_PRECLOSE else 'MARKET CLOSE (4:05 PM)'}")
    print(f"Composer API: {'configured' if COMPOSER_KEY_ID else 'not set'}")
    tickers = [
        'SMH','SPY','QQQ','IWM','XLP','XLU','XLV','XLY',
        'GLD','TLT','HYG','LQD','TMV','USDU','UCO','USO','BOIL',
        'UVXY','VIXM','SVXY','EDC','YINN','KORU','EURL','INDL',
        'BTC-USD','AMD','NVDA',
        'NAIL','CURE','FAS','LABU','TQQQ','SOXL','TECL','DRN','UPRO','DFEN',
        'VOOV','VOOG','VTV','QQQE','USMV','XLE','XLF',
        'GDX','GDXJ','JNUG','NUGT',
        'BTAL','DBMF','KMLM','CTA','FNGO','SLV','UUP','DBC',
        'RSP','SPHB','SPLV','^MOVE','SHY',
        'FXY','CPER','COPX','ILS',
        'TMF',  # Robot James overlay (leveraged bonds)
    ]
    print("Downloading market data...")
    data = download_data(tickers)
    print(f"Downloaded {len(data)} tickers")
    alerts, status = check_signals(data)

    # Hormuz intelligence (Windward). Never raises — degrades silently if offline.
    hormuz_block = ""
    try:
        hz = fetch_hormuz_windward()
        if hz:
            prev_row, today_row = hormuz_upsert_history(hz)
            hz_alerts = hormuz_build_alerts(hz, prev_row, today_row)
            hormuz_block = format_hormuz_block(hz, prev_row, today_row)
            print(f"[Hormuz] as_of={hz.get('as_of')} "
                  f"transits={hz.get('total_transits')} "
                  f"attacks={(hz.get('attacks') or {}).get('value')} "
                  f"alerts={len(hz_alerts)}")
            # Prepend Hormuz alerts so they surface in the subject line + alerts panel
            alerts = hz_alerts + alerts
    except Exception as e:
        print(f"[Hormuz] unexpected error: {e}")

    # Robot James — compute state + render email block. Writes state file so
    # the dashboard's /api/robot_james endpoint can consume it. Action-day
    # alerts are injected into the main alerts list so subject-line summary
    # flags TD1 / TD15 / EOM appropriately.
    rj_block = ""
    try:
        rj_state = rj_compute_state(data)
        if rj_state:
            rj_block = format_rj_block(rj_state)
            atype = (rj_state.get('action') or {}).get('type')
            pb = rj_state.get('projected_base', {})
            ov = rj_state.get('projected_overlay', {})
            if atype == 'TD1_SET_BASE':
                if pb.get('rules_agree'):
                    msg = f"Set base at MOC: 60% {pb.get('agree_ticker')} + 40% GLD (rules agree)"
                else:
                    msg = f"Set base at MOC: 30% {pb.get('mom_pick')} + 30% {pb.get('below_sma_pick')} + 40% GLD"
                alerts = [('[ACTION] ROBOT JAMES TD1', msg, 'buy')] + alerts
            elif atype == 'TD15_OVERLAY':
                if ov.get('strict_skip'):
                    msg = f"STRICT SKIP: {ov.get('skip_reason')}. Hold base through EOM."
                    alerts = [('[WATCH] ROBOT JAMES TD15 SKIP', msg, 'watch')] + alerts
                elif ov.get('signal_asset'):
                    msg = f"Rotate to 100% {ov['signal_asset']} at MOC (lagger: {ov.get('lagger')})"
                    alerts = [('[ACTION] ROBOT JAMES TD15', msg, 'buy')] + alerts
            elif atype == 'EOM_ROTATE_BACK':
                alerts = [('[ACTION] ROBOT JAMES EOM',
                          'Rotate back to cash/base at MOC — prepare for new TD1 tomorrow',
                          'exit')] + alerts
            print(f"[RJ] TD {rj_state.get('current_td')}/{rj_state.get('total_tds_this_month')} "
                  f"action={atype} state written to {RJ_STATE_FILE}")
    except Exception as e:
        print(f"[RJ] unexpected error: {e}")

    # Composer data (close email only for dry-run, both for performance)
    composer_trades = None
    composer_perf = None
    if COMPOSER_KEY_ID:
        try:
            composer_perf = fetch_composer_performance()
            composer_trades = fetch_composer_dry_run()
        except Exception as e:
            print(f"  Composer fetch error: {e}")

    bc = len([a for a in alerts if a[2]=='buy'])
    ec = len([a for a in alerts if a[2] in ['exit','short']])
    urgency = "EXIT SIGNALS" if ec>0 else "BUY SIGNALS" if bc>0 else "WATCH" if alerts else "No Alerts"
    timing = "MID-DAY" if IS_PRECLOSE else "CLOSE"
    subject = f"[{timing}] Market Signals: {len(alerts)} Alert(s) - {urgency}" if alerts else f"[{timing}] Market Signals: No Alerts"
    body = format_email(alerts, status, data, IS_PRECLOSE, composer_trades, composer_perf, hormuz_block, rj_block)
    send_email(subject, body)
    print(f"\n{len(alerts)} signal(s) detected")
    for t, m, _ in alerts: print(f"  {t}")

if __name__ == "__main__":
    main()
