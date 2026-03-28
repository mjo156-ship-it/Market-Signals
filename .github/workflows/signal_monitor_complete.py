#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Market Signal Monitor v4.4
========================================
Groups 1-12: Core RSI signals
Group 13: UVXY Vol Regime Shift (v4.4)
Group 13b: Deep Value (MaRet/CumRet crash+drawdown)
Group 14: Crisis Alpha v2 (vol compression regime)
Group 15: Signal Degradation (trailing WR monitoring)
Group 17: DFEN (RSI+SMA200+Bollinger Band)
Group 18: GLD & Miners
Group 19A/B/C: MOVE Index (v4.4)
Group 21b: Multi-Oversold Breadth (v4.4)
Group 30: DRIF Velocity Filter (v4.4)
+ Rolling Beta vs SPY
+ TLT bond momentum banner
+ FRED Credit Spread monitor
+ Hormuz transit (IMF PortWatch)
+ Market Breadth Intelligence (ZBT/A/D lines)
+ Rolling Brier Score persistence

SCHEDULE: Two emails daily (weekdays)
- 3:15 PM ET: Pre-close preview
- 4:05 PM ET: Market close confirmation
"""
import os, sys, smtplib, json
import yfinance as yf
import pandas as pd
import numpy as np
import requests as req_lib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

SENDER_EMAIL = os.environ.get('SENDER_EMAIL', '')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD', '')
RECIPIENT_EMAIL = os.environ.get('RECIPIENT_EMAIL', '')
PHONE_EMAIL = os.environ.get('PHONE_EMAIL', '')
FRED_API_KEY = os.environ.get('FRED_API_KEY', '')
BRIER_JSON_PATH = os.environ.get('BRIER_JSON_PATH', './brier_scores.json')
BREADTH_CACHE_DIR = os.environ.get('BREADTH_CACHE_DIR', './breadth_cache')
IS_PRECLOSE = len(sys.argv) > 1 and sys.argv[1] == 'preclose'

# Broad market tickers for breadth computation (ZBT, McClellan, %Above50SMA)
BREADTH_TICKERS = [
    'AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','BRK-B','UNH','JNJ',
    'JPM','V','PG','HD','MA','ABBV','MRK','PEP','KO','COST',
    'AVGO','LLY','WMT','MCD','CSCO','TMO','ABT','CRM','ACN','DHR',
    'TXN','NEE','LIN','PM','UNP','RTX','HON','AMGN','LOW','SBUX',
    'FTNT','ODFL','DECK','POOL','WST','GNRC','MPWR','PAYC','ENPH','DXCM',
    'ALGN','MKTX','TER','ZBRA','PODD','TECH','LULU','FICO','CPRT','IDXX',
    'RMD','CSGP','TRMB','NDSN','WSO','EPAM','KEYS','CDW','PCAR',
    'F','GM','FCX','NUE','CLF','AA','RIG','HAL','SLB',
    'DVN','FANG','OXY','MPC','VLO','PSX','CF','MOS','IFF','EMN',
    'IP','PKG','FMC','CE','HUN','OLN','CC','AXTA','RPM',
]

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

def calculate_rolling_betas(data):
    if 'SPY' not in data: return None
    spy_ret = data['SPY']['Close'].pct_change()
    def _beta(ticker, window):
        if ticker not in data: return None
        ar = data[ticker]['Close'].pct_change()
        common = spy_ret.dropna().index.intersection(ar.dropna().index)
        if len(common) < window + 10: return None
        sr, a = spy_ret.loc[common], ar.loc[common]
        b = a.rolling(window).cov(sr) / sr.rolling(window).var()
        v = b.iloc[-1]
        return round(float(v), 3) if not pd.isna(v) else None
    groups = [
        ('Equity Sleeve', [('UPRO',0.4),('TQQQ',0.3),('SOXL',0.3)], 0.55),
        ('MF Rotation', [('CTA',0.33),('DBMF',0.33),('BTAL',0.34)], 0.15),
        ('GLD', [('GLD',1.0)], 0.15),
        ('KMLM', [('KMLM',1.0)], 0.10),
        ('BTAL', [('BTAL',1.0)], 0.05),
    ]
    results = {}
    for wname, w in [('63d',63),('126d',126),('252d',252)]:
        results[wname] = {}
        blend = 0
        for gname, tw, bw in groups:
            gb, valid = 0, True
            for t, twt in tw:
                b = _beta(t, w)
                if b is not None: gb += b * twt
                else: valid = False
            results[wname][gname] = round(gb,3) if valid else None
            if valid: blend += gb * bw
        results[wname]['Est. Blend'] = round(blend, 3)
    return results


def compute_market_breadth(breadth_data):
    """Compute ZBT, McClellan Oscillator, %Above50SMA from broad market data."""
    if not breadth_data:
        return {}
    # Build daily advance/decline from all breadth tickers
    # Get a reference date index from any ticker
    ref_ticker = next(iter(breadth_data))
    dates = breadth_data[ref_ticker].index
    
    daily = []
    for i in range(1, len(dates)):
        adv, dec = 0, 0
        for ticker, df in breadth_data.items():
            try:
                c = df['Close']
                if isinstance(c, pd.DataFrame): c = c.iloc[:,0]
                today = float(c.iloc[i]) if i < len(c) and not pd.isna(c.iloc[i]) else None
                yesterday = float(c.iloc[i-1]) if (i-1) < len(c) and not pd.isna(c.iloc[i-1]) else None
                if today is None or yesterday is None: continue
                if today > yesterday: adv += 1
                elif today < yesterday: dec += 1
            except: pass
        total = adv + dec
        ratio = adv / total if total > 0 else 0.5
        daily.append({'date': dates[i], 'adv': adv, 'dec': dec, 'ratio': ratio, 'net': adv - dec})
    
    if not daily:
        return {}
    bdf = pd.DataFrame(daily)
    
    # ZBT: 10-day EMA of advance ratio
    bdf['zbt_ema'] = bdf['ratio'].ewm(alpha=1.0/10, adjust=False).mean()
    zbt_val = float(bdf['zbt_ema'].iloc[-1])
    zbt_ratio = float(bdf['ratio'].iloc[-1])
    zbt_oversold_recent = bool((bdf['zbt_ema'].tail(10) < 0.40).any())
    if zbt_val < 0.40: zbt_zone = 'OVERSOLD'
    elif zbt_val >= 0.615: zbt_zone = 'THRUST'
    else: zbt_zone = 'NEUTRAL'
    
    # Check for thrust: was oversold (<0.40) and surged to >0.615 within 10 days
    zbt_thrust = False
    if zbt_val >= 0.615 and zbt_oversold_recent:
        zbt_thrust = True
    
    # McClellan Oscillator: 19d EMA - 39d EMA of net advances
    bdf['ema19'] = bdf['net'].ewm(span=19, adjust=False).mean()
    bdf['ema39'] = bdf['net'].ewm(span=39, adjust=False).mean()
    bdf['mcclellan'] = bdf['ema19'] - bdf['ema39']
    mcl_val = float(bdf['mcclellan'].iloc[-1])
    mcl_prev = float(bdf['mcclellan'].iloc[-2]) if len(bdf) >= 2 else mcl_val
    mcl_direction = 'RISING' if mcl_val > mcl_prev else 'FALLING'
    if mcl_val < -100: mcl_zone = 'OVERSOLD'
    elif mcl_val > 100: mcl_zone = 'OVERBOUGHT'
    elif mcl_val > 0: mcl_zone = 'POSITIVE'
    else: mcl_zone = 'NEGATIVE'
    
    # McClellan Summation Index (cumulative)
    mcl_summation = float(bdf['mcclellan'].cumsum().iloc[-1])
    
    # %Above 50-day SMA
    above50, total50 = 0, 0
    for ticker, df in breadth_data.items():
        try:
            c = df['Close']
            if isinstance(c, pd.DataFrame): c = c.iloc[:,0]
            if len(c) >= 50:
                sma50 = float(c.rolling(50).mean().iloc[-1])
                if not pd.isna(sma50):
                    total50 += 1
                    if float(c.iloc[-1]) > sma50:
                        above50 += 1
        except: pass
    pct_above_50 = (above50 / total50 * 100) if total50 > 0 else 0
    
    return {
        'zbt_ema': round(zbt_val, 4),
        'zbt_ratio': round(zbt_ratio, 4),
        'zbt_zone': zbt_zone,
        'zbt_thrust': zbt_thrust,
        'zbt_oversold_recent': zbt_oversold_recent,
        'mcclellan': round(mcl_val, 1),
        'mcl_ema19': round(float(bdf['ema19'].iloc[-1]), 1),
        'mcl_ema39': round(float(bdf['ema39'].iloc[-1]), 1),
        'mcl_direction': mcl_direction,
        'mcl_zone': mcl_zone,
        'mcl_summation': round(mcl_summation, 0),
        'pct_above_50sma': round(pct_above_50, 1),
        'above50_count': above50,
        'above50_total': total50,
        'adv': int(bdf['adv'].iloc[-1]),
        'dec': int(bdf['dec'].iloc[-1]),
    }


def compute_fibonacci_levels(data):
    """Compute Fibonacci retracement levels for SPY, QQQ, SMH."""
    fib_results = {}
    for sym in ['SPY', 'QQQ', 'SMH']:
        if sym not in data:
            continue
        df = data[sym]
        c = df['Close']
        if isinstance(c, pd.DataFrame): c = c.iloc[:,0]
        close = float(c.iloc[-1])
        
        sym_fibs = {}
        for label, days in [('30d', 30), ('60d', 60)]:
            if len(df) < days:
                continue
            recent = df.tail(days)
            high_col = recent['High']
            low_col = recent['Low']
            if isinstance(high_col, pd.DataFrame): high_col = high_col.iloc[:,0]
            if isinstance(low_col, pd.DataFrame): low_col = low_col.iloc[:,0]
            high = float(high_col.max())
            low = float(low_col.min())
            diff = high - low
            is_up = close > (high + low) / 2
            
            levels = {}
            for pct in [0.236, 0.382, 0.500, 0.618, 0.786]:
                if is_up:
                    level = high - diff * pct
                else:
                    level = low + diff * pct
                dist = (close - level) / close * 100
                near = abs(dist) < 1.5
                levels[f'{pct*100:.1f}'] = {'level': round(level, 2), 'dist_pct': round(dist, 1), 'near': near}
            
            sym_fibs[label] = {
                'high': round(high, 2), 'low': round(low, 2),
                'close': round(close, 2),
                'trend': 'UP' if is_up else 'DOWN',
                'levels': levels,
            }
        fib_results[sym] = sym_fibs
    return fib_results


def check_signals(data):
    alerts = []
    status = {}
    indicators = {}
    for ticker, df in data.items():
        if len(df) < 200: continue
        try:
            close = df['Close']
            price = sf(close.iloc[-1])
            rsi10 = sf(calculate_rsi_wilder(close, 10).iloc[-1])
            rsi50 = sf(calculate_rsi_wilder(close, 50).iloc[-1])
            sma200 = sf(close.rolling(200).mean().iloc[-1])
            sma50 = sf(close.rolling(50).mean().iloc[-1])
            ema9 = sf(close.ewm(span=9,adjust=False).mean().iloc[-1])
            ema20 = sf(close.ewm(span=20,adjust=False).mean().iloc[-1])
            ema21 = sf(close.ewm(span=21,adjust=False).mean().iloc[-1])
            rets = close.pct_change()
            maret_10 = sf(rets.rolling(10).mean().iloc[-1])*100
            cumret_10 = sf((close.iloc[-1]/close.iloc[-10]-1))*100 if len(close)>10 else 0
            cumret_30 = sf((close.iloc[-1]/close.iloc[-30]-1))*100 if len(close)>30 else 0
            cumret_50 = sf((close.iloc[-1]/close.iloc[-50]-1))*100 if len(close)>50 else 0
            cumret_100 = sf((close.iloc[-1]/close.iloc[-100]-1))*100 if len(close)>100 else 0
            std_10 = sf(rets.rolling(10).std().iloc[-1])
            std_50 = sf(rets.rolling(50).std().iloc[-1])
            vol_ratio = std_10/std_50 if std_50>0 else 1.0
            bb_sma = sf(close.rolling(20).mean().iloc[-1])
            bb_std = sf(close.rolling(20).std().iloc[-1])
            bb_upper = bb_sma + 2*bb_std
            bb_lower = bb_sma - 2*bb_std
            bb_pctb = (price-bb_lower)/(bb_upper-bb_lower) if (bb_upper-bb_lower)>0 else 0.5
            bb_width = (bb_upper-bb_lower)/bb_sma*100 if bb_sma>0 else 0
            chg5d = sf((close.iloc[-1]/close.iloc[-5]-1))*100 if len(close)>5 else 0
            # DRIF velocity indicators
            cum_ret_5d = sf((close.iloc[-1]/close.iloc[-6]-1))*100 if len(close)>6 else 0
            cum_ret_7d = sf((close.iloc[-1]/close.iloc[-8]-1))*100 if len(close)>8 else 0
            rsi_series_full = calculate_rsi_wilder(close, 10)
            rsi_5d_ago = sf(rsi_series_full.iloc[-6]) if len(rsi_series_full)>6 else 50
            rsi_velocity = rsi10 - rsi_5d_ago
            indicators[ticker] = {
                'price':price,'rsi10':rsi10,'rsi50':rsi50,
                'sma200':sma200,'sma50':sma50,'ema9':ema9,'ema20':ema20,'ema21':ema21,
                'maret_10':maret_10,'cumret_10':cumret_10,'cumret_30':cumret_30,
                'cumret_50':cumret_50,'cumret_100':cumret_100,
                'std_10':std_10,'std_50':std_50,'vol_ratio':vol_ratio,
                'bb_sma':bb_sma,'bb_upper':bb_upper,'bb_lower':bb_lower,
                'bb_pctb':bb_pctb,'bb_width':bb_width,'chg5d':chg5d,
                'cum_ret_5d':cum_ret_5d,'cum_ret_7d':cum_ret_7d,'rsi_velocity':rsi_velocity,
            }
            if sma200>0: indicators[ticker]['pct_above_sma200']=(price/sma200-1)*100
            else: indicators[ticker]['pct_above_sma200']=0
        except Exception as e:
            print(f"Error calculating {ticker}: {e}"); continue
    status['indicators'] = indicators

    # === GROUP 1: SMH/SOXL ===
    if 'SMH' in indicators:
        smh = indicators['SMH']
        if smh['pct_above_sma200']>=40: alerts.append(('[EXIT] SOXL EXIT',f"SMH {smh['pct_above_sma200']:.1f}% above SMA200 - SELL SOXL",'exit'))
        elif smh['pct_above_sma200']>=35: alerts.append(('[WARN] SOXL WARNING',f"SMH {smh['pct_above_sma200']:.1f}% above SMA200 - Approaching sell",'warning'))
        elif smh['pct_above_sma200']>=30: alerts.append(('[WARN] SOXL TRIM',f"SMH {smh['pct_above_sma200']:.1f}% above SMA200 - Trim 25-50%",'warning'))
        if smh['sma50']<smh['sma200'] and smh['sma200']>0: alerts.append(('[EXIT] DEATH CROSS','SMH SMA50 below SMA200','exit'))
        if 'SMH' in data:
            close=data['SMH']['Close']; sma200s=close.rolling(200).mean()
            days_below=0
            for i in range(len(close)-1,max(len(close)-500,199),-1):
                try:
                    if sf(sma200s.iloc[i])>0 and sf(close.iloc[i])<sf(sma200s.iloc[i]): days_below+=1
                    else: break
                except: break
            if days_below>=100:
                if smh['rsi50']<45: alerts.append(('[BUY] SOXL STRONG BUY',f"SMH {days_below}d below SMA200 + RSI50={smh['rsi50']:.1f}<45 | 97% win +81%",'buy'))
                else: alerts.append(('[BUY] SOXL ACCUMULATE',f"SMH {days_below}d below SMA200 | 85% win +54%",'buy'))
            status['smh_days_below_sma200']=days_below

    # === GROUP 2: GLD/USDU Combo ===
    if 'GLD' in indicators and 'USDU' in indicators:
        gld,usdu=indicators['GLD'],indicators['USDU']
        if gld['rsi10']>79 and usdu['rsi10']<25:
            alerts.append(('[BUY] DOUBLE SIGNAL',f"GLD RSI={gld['rsi10']:.1f}>79 + USDU RSI={usdu['rsi10']:.1f}<25\n   -> TQQQ 88% WR +7% (5d) | UPRO 85% | AMD/NVDA 86%",'buy'))
            if 'XLP' in indicators and indicators['XLP']['rsi10']>65:
                alerts.append(('[BUY] TRIPLE SIGNAL',f"+ XLP RSI={indicators['XLP']['rsi10']:.1f}>65 | TQQQ 100% WR +11.6% (5d) RARE!",'buy'))
        elif gld['rsi10']>79: alerts.append(('[BUY] GLD OVERBOUGHT',f"GLD RSI={gld['rsi10']:.1f}>79 -> TQQQ 72% WR +3.2% (5d)",'buy'))

    # === GROUP 3: Defensive Rotation ===
    def_ob=any(indicators.get(t,{}).get('rsi10',0)>79 for t in ['XLP','XLU','XLV'])
    if def_ob and indicators.get('SPY',{}).get('rsi10',50)<79 and indicators.get('QQQ',{}).get('rsi10',50)<79:
        alerts.append(('[BUY] DEFENSIVE ROTATION','Def sector OB, SPY/QQQ not -> TQQQ 70% WR 20d','buy'))

    # === GROUP 4: Vol Hedge ===
    qqq_r=indicators.get('QQQ',{}).get('rsi10',50)
    if qqq_r>79: alerts.append(('[HEDGE] QQQ>79 -> UVXY',f"QQQ RSI={qqq_r:.1f} | 67% WR 5d",'hedge'))
    if qqq_r<20: alerts.append(('[BUY] QQQ DIP BUY',f"QQQ RSI={qqq_r:.1f}<20 -> TQQQ 100% WR [TIER 1]",'buy'))

    # === GROUP 5: SOXS ===
    smh_r=indicators.get('SMH',{}).get('rsi10',50); usdu_r=indicators.get('USDU',{}).get('rsi10',50)
    if smh_r>79 and usdu_r>70: alerts.append(('[SHORT] SOXS SQUEEZE',f"SMH={smh_r:.1f}>79 + USDU={usdu_r:.1f}>70 | 100% WR +9.5%",'short'))
    if smh_r>79 and indicators.get('IWM',{}).get('rsi10',50)<50:
        alerts.append(('[SHORT] SOXS DIVERGENCE',f"SMH={smh_r:.1f}>79 + IWM<50 | 86% WR +6.9%",'short'))

    # === GROUP 6: BTC ===
    btc_r=indicators.get('BTC-USD',{}).get('rsi10',50)
    if btc_r>79: alerts.append(('[BUY] BTC MOMENTUM',f"BTC RSI={btc_r:.1f}>79 | Hold/Add 67% WR +5.2%",'buy'))
    if btc_r<30:
        if indicators.get('UVXY',{}).get('rsi10',50)<40: alerts.append(('[BUY] BTC DIP BUY',f"BTC RSI={btc_r:.1f}<30 + UVXY<40 | 77% WR +4.1%",'buy'))
        else: alerts.append(('[WATCH] BTC OVERSOLD',f"BTC RSI={btc_r:.1f}<30 (wait UVXY<40)",'watch'))

    # === GROUP 7: UPRO ===
    spy_r=indicators.get('SPY',{}).get('rsi10',50)
    if spy_r>85: alerts.append(('[EXIT] UPRO EXIT',f"SPY RSI={spy_r:.1f}>85 | Only 36% WR -3.5%",'exit'))
    elif spy_r>82: alerts.append(('[WARN] UPRO CAUTION',f"SPY RSI={spy_r:.1f}>82 | 49% WR",'warning'))
    if spy_r<21: alerts.append(('[BUY] UPRO STRONG BUY',f"SPY RSI={spy_r:.1f}<21 | 87% WR +8.9% [TIER 1]",'buy'))
    elif spy_r<25: alerts.append(('[BUY] UPRO BUY',f"SPY RSI={spy_r:.1f}<25 | 74% WR +3.9%",'buy'))
    elif spy_r<30: alerts.append(('[BUY] UPRO CONSIDER',f"SPY RSI={spy_r:.1f}<30 | 69% WR +4.3%",'buy'))

    # === GROUP 8: AMD/NVDA ===
    if indicators.get('AMD',{}).get('rsi10',50)>85: alerts.append(('[WARN] AMD EXTENDED',f"AMD RSI={indicators['AMD']['rsi10']:.1f}>85",'warning'))
    if indicators.get('NVDA',{}).get('rsi10',50)>85: alerts.append(('[WARN] NVDA EXTENDED',f"NVDA RSI={indicators['NVDA']['rsi10']:.1f}>85",'warning'))

    # === GROUP 9: NAIL ===
    if 'NAIL' in indicators:
        nail=indicators['NAIL']
        if 'GLD' in indicators and 'USDU' in indicators and 'XLF' in indicators:
            g,u,x=indicators['GLD'],indicators['USDU'],indicators['XLF']
            if g['rsi10']>79 and u['rsi10']<25 and x['rsi10']<70: alerts.append(('[BUY] NAIL SIGNAL',f"GLD>{g['rsi10']:.0f}+USDU<{u['rsi10']:.0f}+XLF<{x['rsi10']:.0f} | 90% WR n=10",'buy'))
            if x['rsi10']>70 and u['rsi10']<25: alerts.append(('[EXIT] NAIL DANGER',f"XLF>{x['rsi10']:.0f}+USDU<25 = 11% WR -11.5%",'exit'))
        if nail['rsi10']<21: alerts.append(('[BUY] NAIL RSI<21',f"NAIL RSI={nail['rsi10']:.1f} | Oversold",'buy'))
        elif nail['rsi10']>79: alerts.append(('[EXIT] NAIL OB',f"NAIL RSI={nail['rsi10']:.1f}>79",'warning'))

    # === GROUP 10: CURE ===
    cure_r=indicators.get('CURE',{}).get('rsi10',50)
    if cure_r<21: alerts.append(('[BUY] CURE STRONG BUY',f"CURE RSI={cure_r:.1f}<21 | 85% WR +7.3% n=33",'buy'))
    elif cure_r<25: alerts.append(('[BUY] CURE BUY',f"CURE RSI={cure_r:.1f}<25 | 81% WR +5.4% n=70",'buy'))
    if cure_r>85: alerts.append(('[EXIT] CURE SELL',f"CURE RSI={cure_r:.1f}>85 | 33% WR",'exit'))
    elif cure_r>79: alerts.append(('[EXIT] CURE OB',f"CURE RSI={cure_r:.1f}>79 | 40% WR",'exit'))

    # === GROUP 11: FAS ===
    fas_r=indicators.get('FAS',{}).get('rsi10',50)
    if 'GLD' in indicators and 'USDU' in indicators:
        if indicators['GLD']['rsi10']>79 and indicators['USDU']['rsi10']<25: alerts.append(('[BUY] FAS SIGNAL',f"GLD>79+USDU<25 -> FAS 10d: 92% WR +5.8%",'buy'))
    if fas_r<30: alerts.append(('[BUY] FAS BUY',f"FAS RSI={fas_r:.1f}<30 | 63% WR n=195",'buy'))
    if fas_r>85: alerts.append(('[EXIT] FAS SELL',f"FAS RSI={fas_r:.1f}>85 | 8% WR!",'exit'))
    elif fas_r>82: alerts.append(('[EXIT] FAS OB',f"FAS RSI={fas_r:.1f}>82 | 38% WR",'exit'))

    # === GROUP 12: LABU ===
    labu_r=indicators.get('LABU',{}).get('rsi10',50)
    if labu_r<21: alerts.append(('[BUY] LABU STRONG BUY',f"LABU RSI={labu_r:.1f}<21 | 73% WR +11.2% n=11",'buy'))
    elif labu_r<25: alerts.append(('[BUY] LABU BUY',f"LABU RSI={labu_r:.1f}<25 | 66% WR +5.7% n=59",'buy'))
    if labu_r>70: alerts.append(('[WARN] LABU EXTENDED',f"LABU RSI={labu_r:.1f}>70 | 42% WR",'warning'))

    # === GROUP 13: Deep Value ===
    if 'QQQ' in indicators:
        q=indicators['QQQ']
        if q['maret_10']<-1.0: alerts.append(('[BUY] QQQ CRASH BOUNCE',f"QQQ MaRet(10d)={q['maret_10']:.2f}%/day -> TQQQ bounce",'buy'))
        if q['cumret_30']<-20: alerts.append(('[BUY] QQQ DEEP DRAWDOWN',f"QQQ CumRet(30d)={q['cumret_30']:.1f}% -> TQQQ deep value",'buy'))
    if 'SMH' in indicators:
        s=indicators['SMH']
        if s['maret_10']<-1.5: alerts.append(('[BUY] SMH CRASH BOUNCE',f"SMH MaRet(10d)={s['maret_10']:.2f}%/day -> SOXL",'buy'))
        if s['cumret_30']<-25: alerts.append(('[BUY] SMH DEEP DRAWDOWN',f"SMH CumRet(30d)={s['cumret_30']:.1f}% -> SOXL",'buy'))
    if 'SPY' in indicators:
        sp=indicators['SPY']
        if sp['cumret_50']<-15 and sp['rsi10']<35: alerts.append(('[BUY] SPY DEEP VALUE',f"SPY CumRet(50d)={sp['cumret_50']:.1f}%+RSI={sp['rsi10']:.1f}<35 -> UPRO",'buy'))
        elif sp['cumret_100']<-10: alerts.append(('[BUY] SPY MOD DRAWDOWN',f"SPY CumRet(100d)={sp['cumret_100']:.1f}% -> UPRO accumulate",'buy'))
        if sp['cumret_10']<-5 and indicators.get('USDU',{}).get('rsi10',50)>70:
            alerts.append(('[WARN] FALLING KNIFE',f"SPY 10d={sp['cumret_10']:.1f}%+USDU>70 = DON\'T catch knife",'warning'))

    # === GROUP 14: Crisis Alpha Regime ===
    crisis_regime='UNKNOWN'
    if 'SPY' in indicators and 'QQQ' in indicators:
        sp,q=indicators['SPY'],indicators['QQQ']
        above200=sp['price']>sp['sma200'] if sp['sma200']>0 else False
        above_ema20=sp['price']>sp['ema20'] if sp['ema20']>0 else False
        vol_exp=q['vol_ratio']>1.0
        if above200 and not vol_exp: crisis_regime='BULL + VOL COMPRESS'
        elif above200 and vol_exp:
            crisis_regime='BULL + VOL EXPAND'
            alerts.append(('[WATCH] VOL EXPANDING (BULL)',f"QQQ StdDev 10/50 ratio={q['vol_ratio']:.2f} + SPY above SMA200\n  -> Crisis Alpha: UPRO/GLD regime (less aggressive)",'watch'))
        elif not above200 and above_ema20:
            crisis_regime='BEAR RECOVERY'
            alerts.append(('[WATCH] BEAR RECOVERY',f"SPY below SMA200 but above EMA20 -> Recovery mode",'watch'))
        elif not above200:
            crisis_regime='BEAR DEFENSIVE'
            alerts.append(('[WARN] BEAR DEFENSIVE',f"SPY below SMA200+EMA20 -> SHY/GLD defensive",'warning'))
    status['crisis_regime']=crisis_regime

    # === GROUP 15: Signal Degradation ===
    if 'GLD' in indicators:
        gld=indicators['GLD']
        if gld['rsi10']>79 and indicators.get('USDU',{}).get('rsi10',50)>=25:
            alerts.append(('[WARN] SIGNAL CALIBRATION',f"GLD RSI>79 alone: Trailing WR 50% vs historical 72% -- signal DEGRADING (n=16)",'warning'))

    # === GROUP 17: DFEN Bollinger ===
    if 'DFEN' in indicators:
        d=indicators['DFEN']
        above200=d['price']>d['sma200'] if d['sma200']>0 else False
        below_bb=d['price']<d['bb_lower']
        if below_bb and d['rsi10']>=30 and above200:
            alerts.append(('[BUY] DFEN BB+RSI OVERSOLD',f"DFEN below BB+RSI={d['rsi10']:.1f}>=30 | 73.5% WR +6.8% n=49\n   BB dominates RSI for DFEN. Pullback in uptrend.",'buy'))
        elif below_bb and d['rsi10']<30 and above200:
            alerts.append(('[BUY] DFEN BB+RSI<30',f"DFEN below BB+RSI={d['rsi10']:.1f}<30 | 63.8% WR",'buy'))
        elif d['rsi10']<35 and above200:
            alerts.append(('[BUY] DFEN WATCH',f"DFEN RSI={d['rsi10']:.1f}<35 + above SMA200\n  -> 90% WR, +11% avg (20d) | n=52 | Pullback in uptrend",'buy'))

    # === GROUP 18: GLD & Miners ===
    gdxj_r=indicators.get('GDXJ',{}).get('rsi10',50)
    gdx_r=indicators.get('GDX',{}).get('rsi10',50)
    if gdxj_r<21: alerts.append(('[BUY] JNUG STRONG BUY [TIER 2]',f"GDXJ RSI={gdxj_r:.1f}<21 -> JNUG\n   1d: 59% WR +8.43% n=17 | 5d: +14.9% | 20d: +18.9%",'buy'))
    elif gdxj_r<25: alerts.append(('[BUY] JNUG DIP BUY [TIER 2]',f"GDXJ RSI={gdxj_r:.1f}<25 -> JNUG\n   1d: 63% WR +3.55% n=59",'buy'))
    elif gdxj_r<30: alerts.append(('[WATCH] GDXJ APPROACHING',f"GDXJ RSI={gdxj_r:.1f} -> JNUG buy zone <25",'watch'))
    if gdx_r<21 and gdxj_r>=25: alerts.append(('[BUY] NUGT BUY',f"GDX RSI={gdx_r:.1f}<21 -> NUGT 56% WR n=25",'buy'))
    if gdx_r>85: alerts.append(('[WARN] GDX EXTENDED - DO NOT SHORT',f"GDX RSI={gdx_r:.1f}>85 | Miners continue when OB. DUST loses.",'warning'))
    gld_r2=indicators.get('GLD',{}).get('rsi10',50); usdu_r2=indicators.get('USDU',{}).get('rsi10',50)
    if gld_r2>75 and usdu_r2>60: alerts.append(('[HEDGE] MINER SHORT WINDOW',f"GLD={gld_r2:.1f}>75+USDU={usdu_r2:.1f}>60 | JDST 5d 59% WR n=34",'hedge'))

    # UCO>75 -> TMV
    uco_r=indicators.get('UCO',{}).get('rsi10',50)
    if uco_r>75: alerts.append(('[BUY] UCO>75 -> TMV',f"UCO RSI={uco_r:.1f} | Oil->Bond weakness",'buy'))

    # === GROUP 13: UVXY Vol Regime Shift (v4.4) ===
    if 'UVXY' in indicators:
        uvxy=indicators['UVXY']
        uvxy_pct=uvxy.get('pct_above_sma200', -999)
        status['uvxy_vol_regime']={'price':uvxy['price'],'sma200':uvxy['sma200'],'pct_above':uvxy_pct,'active':uvxy_pct>0}
        if uvxy_pct>=30: alerts.append(('[BUY] VOL REGIME EXTREME',f"UVXY {uvxy_pct:+.1f}% above SMA200 | SPY 20d: 94% WR +7.3% | 40d/60d: 100% | n=18\n   TQQQ 60d: 100% +65% | SOXL 60d: 96% +81%",'buy'))
        elif uvxy_pct>=20: alerts.append(('[BUY] VOL REGIME HIGH',f"UVXY {uvxy_pct:+.1f}% above SMA200 | SPY 20d: 92% WR +6.2% | 60d: 100%\n   Wilson CI 20d: (74%, 98%) | n=24",'buy'))
        elif uvxy_pct>=0: alerts.append(('[BUY] VOL REGIME SHIFT',f"UVXY {uvxy_pct:+.1f}% above SMA200 | SPY 20d: 83% WR +4.3% | 60d: 92%\n   Edge vs UC: +3.3% (20d), +6.8% (60d) | n=52",'buy'))
        elif uvxy_pct>=-10: alerts.append(('[WATCH] VOL REGIME APPROACHING',f"UVXY {uvxy_pct:+.1f}% vs SMA200 | Threshold: ${uvxy['sma200']:.2f}",'watch'))

    # === GROUP 19: MOVE Index (v4.4) ===
    if '^MOVE' in indicators and '^MOVE' in data:
        move=indicators['^MOVE']
        mc=data['^MOVE']['Close']
        move_price=move['price']; move_rsi=move['rsi10']
        move_chg_20d=0
        if len(mc)>=21:
            prev=sf(mc.iloc[-21])
            if prev>0: move_chg_20d=((move_price/prev)-1)*100
        status['move_index']={'price':move_price,'rsi':move_rsi,'change_20d_pct':round(move_chg_20d,1)}
        if move_chg_20d>50: alerts.append(('[BUY] MOVE EXTREME SPIKE [19B]',f"MOVE 20d Δ={move_chg_20d:+.1f}% | SPY 20d: 86% WR +5.29% (n=69)\n   QQQ 20d: 83% WR +5.90%  |  Wilson CI: [76%-93%]",'buy'))
        elif move_price>115: alerts.append(('[BUY] MOVE ELEVATED [19A]',f"MOVE={move_price:.0f}>115 | SPY 20d: 72% WR +2.07% (n=389d/29 ep)",'buy'))
        # 19C: RSI was >79 recently and now <60
        move_rsi_series=calculate_rsi_wilder(mc,10)
        if len(move_rsi_series)>=10:
            was_ob=float(move_rsi_series.iloc[-10:].max())>79
            if was_ob and move_rsi<60: alerts.append(('[BUY] MOVE VOL CRUSH [19C]',f"MOVE RSI dropped from >79 to {move_rsi:.1f} | SPY 10d: 92% WR +2.28%\n   UVXY 10d: -16.8% avg | n=24",'buy'))
        # GLD combo
        if move_price>100 and spy_r<25: alerts.append(('[BUY] MOVE+SPY GLD COMBO',f"MOVE={move_price:.0f}>100 + SPY RSI={spy_r:.1f}<25 | GLD 20d: 100% WR +6.43% (n=15)",'buy'))

    # === GROUP 21b: Multi-Oversold Breadth (v4.4) ===
    usmv_r2=indicators.get('USMV',{}).get('rsi10',50)
    vtv_r=indicators.get('VTV',{}).get('rsi10',50)
    voov_r=indicators.get('VOOV',{}).get('rsi10',50)
    upro_r=indicators.get('UPRO',{}).get('rsi10',50)
    if all(r<30 for r in [spy_r,usmv_r2,vtv_r,voov_r,upro_r]):
        hyg_r2=indicators.get('HYG',{}).get('rsi10',50)
        msg=f"SPY+USMV+VTV+VOOV+UPRO all RSI<30 | UPRO 5d: 77.8% WR (n=45, 23 ep)"
        if hyg_r2<30: msg+=' | HYG RSI<30 confirmed — higher conviction'
        alerts.append(('[BUY] MULTI-OS BREADTH [21b]',msg,'buy'))

    # === GROUP 30: DRIF Velocity Filter (v4.4) ===
    drif_status={}
    for ticker_d,lever,thresholds in [
        ('SPY','UPRO',[(25,'cum_ret_5d',-5,'100%',10,'55.6%',9),(30,'cum_ret_7d',-5,'76.1%',46,'56.2%',32)]),
        ('QQQ','TQQQ',[(25,'cum_ret_7d',-8,'87.5%',16,'57.1%',7)]),
        ('SMH','SOXL',[(25,'cum_ret_5d',-5,'68.8%',16,'16.7%',6)]),
    ]:
        if ticker_d not in indicators: continue
        ind=indicators[ticker_d]; rsi=ind['rsi10']; vel=ind.get('rsi_velocity',0)
        drif_status[ticker_d]={'rsi':rsi,'cum_ret_5d':ind.get('cum_ret_5d',0),'cum_ret_7d':ind.get('cum_ret_7d',0),'velocity':vel,'gate':'N/A','label':'NOT OVERSOLD'}
        for rsi_t,ret_f,ret_g,p_wr,p_n,f_wr,f_n in thresholds:
            if rsi<rsi_t:
                ret_val=ind.get(ret_f,0)
                if ret_val>ret_g:
                    drif_status[ticker_d].update({'gate':'PASS','label':'STABILIZED DIP'})
                    alerts.append((f'[BUY] DRIF: {ticker_d} CONFIRMED',f"{ticker_d} RSI={rsi:.1f} + {ret_f.replace('cum_ret_','')}d ret={ret_val:+.1f}% > {ret_g}%\n   VELOCITY GATE PASSED: {p_wr} WR 5d (n={p_n})\n   Long {lever} — stabilized dip, not falling knife",'buy'))
                else:
                    drif_status[ticker_d].update({'gate':'FAIL','label':'FALLING KNIFE'})
                    alerts.append((f'[WARN] DRIF: {ticker_d} FALLING KNIFE',f"{ticker_d} RSI={rsi:.1f} BUT {ret_f.replace('cum_ret_','')}d ret={ret_val:+.1f}% < {ret_g}%\n   VELOCITY GATE FAILED: Only {f_wr} WR (n={f_n})\n   Wait for stabilization before {lever} entry",'warning'))
                break
    status['drif_velocity']=drif_status

    return alerts, status

def format_email(alerts, status, is_preclose=False):
    now = datetime.now()
    timing = "PRE-CLOSE PREVIEW (3:15 PM)" if is_preclose else "MARKET CLOSE CONFIRMATION (4:05 PM)"
    indicators = status.get('indicators', {})

    body = f"""{'='*70}
MARKET SIGNAL MONITOR - {timing}
{now.strftime('%Y-%m-%d %H:%M')} ET
{'='*70}

"""
    # TLT bond momentum banner
    if 'TLT' in indicators:
        tlt_10d = indicators['TLT'].get('cumret_10', 0)
        if tlt_10d < -2: body += f">>> TLT 10d: {tlt_10d:+.1f}% -- Bonds FALLING -- Rates rising -- UVXY hedge conviction HIGH\n\n"
        elif tlt_10d > 2: body += f">>> TLT 10d: {tlt_10d:+.1f}% -- Bonds RISING -- Rates falling -- UVXY hedge conviction LOW\n\n"

    if alerts:
        buy_alerts = [a for a in alerts if a[2] == 'buy']
        exit_alerts = [a for a in alerts if a[2] in ['exit', 'short']]
        warn_alerts = [a for a in alerts if a[2] in ['warning', 'hedge', 'watch']]
        if buy_alerts:
            body += "BUY SIGNALS:\n" + "-"*50 + "\n"
            for t,m,_ in buy_alerts: body += f"{t}\n{m}\n\n"
        if exit_alerts:
            body += "EXIT/SHORT SIGNALS:\n" + "-"*50 + "\n"
            for t,m,_ in exit_alerts: body += f"{t}\n{m}\n\n"
        if warn_alerts:
            body += "WARNINGS/WATCH:\n" + "-"*50 + "\n"
            for t,m,_ in warn_alerts: body += f"{t}\n{m}\n\n"
    else:
        body += "No signals triggered today.\n\n"

    # --- INDICATOR STATUS ---
    body += f"\n{'='*70}\nCURRENT INDICATOR STATUS\n{'='*70}\n\n"
    key_tickers = ['SPY','QQQ','SMH','GLD','USDU','XLP','TLT','HYG','XLF','UVXY','BTC-USD','AMD','NVDA']
    body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}\n" + "-"*50 + "\n"
    for t in key_tickers:
        if t in indicators:
            i=indicators[t]
            p=f"${i['price']:.2f}" if i['price']<1000 else f"${i['price']:,.0f}"
            body += f"{t:<10} {p:>12} {i['rsi10']:>10.1f} {i.get('pct_above_sma200',0):>+11.1f}%\n"

    # --- 3x LEVERAGED ---
    body += f"\n{'='*70}\n3x LEVERAGED ETFs\n{'='*70}\n"
    body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}  Signal\n" + "-"*65 + "\n"
    for t in ['NAIL','CURE','FAS','LABU','TQQQ','SOXL','TECL','DRN','DFEN']:
        if t in indicators:
            i=indicators[t]; r=i['rsi10']
            sig="<< OVERSOLD" if r<21 else "<< Watch" if r<30 else ">> OVERBOUGHT" if r>85 else ">> Extended" if r>79 else ""
            body += f"{t:<10} ${i['price']:>11.2f} {r:>10.1f} {i.get('pct_above_sma200',0):>+11.1f}%  {sig}\n"

    # --- OTHER ETFs ---
    body += f"\n{'='*70}\nOTHER ETFs\n{'='*70}\n"
    body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}\n" + "-"*50 + "\n"
    for t in ['XLV','XLU','XLE','TMV','VOOV','VOOG','VTV','QQQE','BOIL','EURL','YINN','KORU','INDL','EDC']:
        if t in indicators:
            i=indicators[t]
            p=f"${i['price']:.2f}" if i['price']<1000 else f"${i['price']:,.0f}"
            body += f"{t:<10} {p:>12} {i['rsi10']:>10.1f} {i.get('pct_above_sma200',0):>+11.1f}%\n"

    # --- SMH/SOXL LEVELS ---
    if 'SMH' in indicators:
        smh=indicators['SMH']; s200=smh['sma200']
        body += f"\n{'='*70}\nSMH/SOXL LEVELS\n{'='*70}\n"
        body += f"Current Price:    ${smh['price']:.2f}\nSMA(200):         ${s200:.2f}\n"
        body += f"% Above SMA200:   {smh['pct_above_sma200']:+.1f}%\nDays Below SMA:   {status.get('smh_days_below_sma200',0)}\n\n"
        body += f"Key Levels:\n  30% (Trim):     ${s200*1.30:.2f}\n  35% (Warning):  ${s200*1.35:.2f}\n  40% (Sell):     ${s200*1.40:.2f}\n"

    # --- CRISIS ALPHA / DEEP VALUE DASHBOARD ---
    body += f"\n{'='*70}\nCRISIS ALPHA / DEEP VALUE DASHBOARD\n{'='*70}\n"
    body += """
  KEY METRICS EXPLAINED:
  MaRet(10d)  = Avg daily return over 10 days. Measures sell pressure.
                Below -1%/day = crash-level selling -> deep value triggers fire.
  CumRet(Nd)  = Total price change over N days. Measures drawdown depth.
                QQQ -20% over 30d or SPY -15% over 50d = historically rare -> buy signals.
  StdDev 10/50 = Short-term vs long-term daily volatility.
  Vol Ratio   = StdDev(10d)/StdDev(50d). >1.0 means recent vol is ABOVE normal.
                Determines which Opus regime is active (compress vs expand).
  %B (DFEN)   = Position within Bollinger Bands. <0 = below lower band (oversold).

"""
    for t in ['SPY','QQQ','SMH','USMV']:
        if t in indicators:
            i=indicators[t]
            body += f" {t}:\n"
            body += f"   MaRet(10d): {i['maret_10']:+.2f}%/day    CumRet 10d/30d/50d: {i['cumret_10']:.1f}% / {i['cumret_30']:.1f}% / {i['cumret_50']:.1f}%\n"
            body += f"   StdDev 10/50: {i['std_10']:.4f}/{i['std_50']:.4f}  Vol Ratio: {i['vol_ratio']:.2f}\n\n"

    regime=status.get('crisis_regime','UNKNOWN')
    sp=indicators.get('SPY',{}); qq=indicators.get('QQQ',{})
    body += f" CRISIS ALPHA REGIME: {regime}\n"
    body += f"   Vol Ratio: {qq.get('vol_ratio',0):.2f}  SPY>SMA200: {sp.get('price',0)>sp.get('sma200',0)}  SPY>EMA20: {sp.get('price',0)>sp.get('ema20',0)}\n"
    regime_meaning = {
        'BULL + VOL COMPRESS':'Calm bull. Opus B5 (inv-vol MF/GLD/UPRO). Most aggressive equity tilt.',
        'BULL + VOL EXPAND':'Bull but stress rising. Opus B4/B5 (UPRO/GLD). Slightly less aggressive.',
        'BEAR RECOVERY':'Below SMA200 but bouncing above EMA20. Recovery: 50% UPRO / 35% GLD / 15% MF.',
        'BEAR DEFENSIVE':'Below SMA200+EMA20. Full defensive: SHY/GLD/MF. Wait for EMA20 crossover.',
    }
    body += f"   -> {regime_meaning.get(regime,'Check conditions manually')}\n"

    # Crisis Alpha regime commentary
    body += f"\n  WHAT THIS MEANS RIGHT NOW:\n"
    if regime=='BULL + VOL COMPRESS':
        body += f"  Market structure is ideal: price above long-term trend, volatility contracting.\n"
        body += f"  This is the regime where leveraged equity (UPRO/TQQQ) has the highest risk-adjusted\n"
        body += f"  returns. RSI dip-buys have full historical edge. Your Non-Equity Sleeve should be\n"
        body += f"  in T3-T5 tiers (commodities/dollar/SHY) — vol instruments have no edge here.\n"
        body += f"  Portfolio beta target: 0.70-1.00.\n"
    elif regime=='BULL + VOL EXPAND':
        body += f"  Price still above SMA200 (trend intact) but short-term volatility is elevated.\n"
        body += f"  This is the transition zone — not yet defensive, but stress is building.\n"
        body += f"  RSI dip-buys still work but with wider drawdowns. Consider reducing position\n"
        body += f"  sizes on leveraged entries. VIXM/BTAL become relevant hedges.\n"
        body += f"  Non-Equity Sleeve may be nearing T1-T2 (vol/hedge tiers).\n"
        body += f"  Portfolio beta target: 0.50-0.80.\n"
    elif regime=='BEAR RECOVERY':
        body += f"  SPY is below the 200-day SMA (bear territory) but holding above EMA20,\n"
        body += f"  which suggests a bounce is underway. This is the most dangerous regime to\n"
        body += f"  trade — could be a genuine recovery or a bear market rally.\n"
        body += f"  Dip-buys here have lower conviction. Wait for SMA200 reclaim before adding\n"
        body += f"  aggressively to leveraged positions. GLD and BTAL provide protection.\n"
        body += f"  Portfolio beta target: 0.25-0.50.\n"
    elif regime=='BEAR DEFENSIVE':
        body += f"  SPY is below both SMA200 and EMA20 — full bear regime. Capital preservation\n"
        body += f"  is priority. Your Non-Equity Sleeve waterfall should be fully engaged\n"
        body += f"  (VIXM if VIX>25, SH if RSI>79, else commodities/SHY).\n"
        body += f"  DO NOT buy dips in leveraged equity unless DRIF gate PASSES.\n"
        body += f"  Favor GLD, BTAL, KMLM, SHY. Wait for EMA20 crossover as first recovery signal.\n"
        body += f"  Portfolio beta target: 0.00-0.25.\n"

    body += f"\n DEEP VALUE TRIGGER PROXIMITY:\n"
    body += f"   (Distance from each trigger. Negative = trigger has FIRED)\n"
    dv = []
    if 'QQQ' in indicators:
        q=indicators['QQQ']
        dv.append(('QQQ MaRet(10d)',q['maret_10'],-1.0,'Avg daily loss speed. <-1% = crash selling -> TQQQ'))
        dv.append(('QQQ CumRet(30d)',q['cumret_30'],-20.0,'Total 30d drawdown. <-20% = deep value -> TQQQ'))
    if 'SMH' in indicators:
        s=indicators['SMH']
        dv.append(('SMH MaRet(10d)',s['maret_10'],-1.5,'Semi sell pace. <-1.5% = extreme -> SOXL'))
        dv.append(('SMH CumRet(30d)',s['cumret_30'],-25.0,'Semi drawdown. <-25% = generational buy -> SOXL'))
    if 'SPY' in indicators:
        sp=indicators['SPY']
        dv.append(('SPY CumRet(50d)',sp['cumret_50'],-15.0,'Broad mkt ~2.5mo drawdown. <-15% + RSI<35 -> UPRO'))
        dv.append(('SPY CumRet(100d)',sp['cumret_100'],-10.0,'Broad mkt ~5mo drawdown. <-10% -> UPRO accumulate'))
    for label,current,trigger,context in dv:
        away=current-trigger
        body += f"   {label}: {current:.1f} (trigger: {trigger}) -- {away:+.1f} away\n"
        body += f"     [{context}]\n"

    # --- DFEN BOLLINGER ---
    if 'DFEN' in indicators:
        d=indicators['DFEN']
        pos = 'Below lower BB' if d['price']<d['bb_lower'] else 'Above upper BB' if d['price']>d['bb_upper'] else 'Within bands'
        body += f"\n DFEN BOLLINGER BANDS (20, 2):\n"
        body += f"   Price: ${d['price']:.2f}  |  RSI: {d['rsi10']:.1f}\n"
        body += f"   Upper:  ${d['bb_upper']:.2f}\n   SMA20:  ${d['bb_sma']:.2f}\n   Lower:  ${d['bb_lower']:.2f}\n"
        body += f"   %B: {d['bb_pctb']:.3f}  |  Width: {d['bb_width']:.1f}%  |  {pos}\n"
        body += f"   Signal: Below BB+RSI>=30 = 73.5% WR (5d) | Below BB+RSI<30 = 63.8% WR\n"
        body += f"   [BB dominates RSI alone for DFEN. Pullbacks to lower BB in uptrend = high-prob dip buys]\n"

    # --- ROLLING BETA ---
    rb=status.get('rolling_betas')
    if rb:
        body += f"\n{'='*70}\nROLLING BETA vs SPY\n{'='*70}\n"
        body += "  Beta = how much each group moves per 1% SPY move.\n"
        body += "  Blend >2.0 = portfolio is leveraged SPY (diversification minimal).\n"
        body += "  MF Rotation / BTAL should be NEGATIVE (hedge value).\n\n"
        body += f"{'Group':<20} {'63d':>8} {'126d':>8} {'252d':>8}\n" + "-"*50 + "\n"
        for g in ['Equity Sleeve','MF Rotation','GLD','KMLM','BTAL','Est. Blend']:
            row=f"{g:<20}"
            for w in ['63d','126d','252d']:
                v=rb.get(w,{}).get(g)
                row += f" {v:>+7.2f}" if v is not None else f" {'N/A':>7}"
            if g=='Est. Blend':
                b63=rb.get('63d',{}).get('Est. Blend')
                if b63 and b63>2.0: row+="  << HIGH LEVERAGE"
                elif b63 and b63>1.5: row+="  << ELEVATED"
            body += row+"\n"
            if g=='BTAL': body += "-"*50+"\n"
        b63=rb.get('63d',{}).get('Est. Blend'); b252=rb.get('252d',{}).get('Est. Blend')
        if b63 and b252:
            body += f"\nTrend: {b252:+.2f} (252d) -> {b63:+.2f} (63d)\n"
            if b63>2.0: body += "WARNING: HIGH leverage. Holy Grail diversification minimal.\n  -> Consider increasing KMLM/CTA/GLD, reducing equity sleeve.\n"

    # --- GLD & MINERS ---
    body += f"\n{'='*70}\nGLD & MINERS STATUS\n{'='*70}\n"
    body += "  GDXJ/GDX RSI<25 triggers JNUG/NUGT dip-buys (63%/56% WR at 1d).\n"
    body += "  Miners have 5x GLD beta -- their OWN RSI matters, not gold's.\n"
    body += "  DO NOT short overbought miners (GDX>79 -> DUST loses money).\n\n"
    body += f"{'Ticker':<8} {'Price':>10} {'RSI(10)':>8} {'vs SMA200':>10}  Signal\n" + "-"*55 + "\n"
    for t in ['GLD','GDX','GDXJ','JNUG','NUGT']:
        if t in indicators:
            i=indicators[t]; r=i['rsi10']
            flag=""
            if t=='GDXJ' and r<21: flag="JNUG BUY 59% +8.4%"
            elif t=='GDXJ' and r<25: flag="JNUG 63% +3.6%"
            elif t=='GDX' and r<21: flag="NUGT BUY 56%"
            elif t in('GDX','GDXJ') and r>85: flag="DO NOT SHORT"
            elif r<25: flag="Oversold"
            elif r>79: flag="High - momentum"
            body += f"  {t:<6} ${i['price']:>9.2f} {r:>8.1f} {i.get('pct_above_sma200',0):>+9.1f}%  {flag}\n"

    # === v4.4 NEW EMAIL SECTIONS ===

    # --- UVXY VOL REGIME SHIFT ---
    uvxy_regime=status.get('uvxy_vol_regime',{})
    if uvxy_regime:
        up=uvxy_regime.get('price',0); us=uvxy_regime.get('sma200',0); ua=uvxy_regime.get('pct_above',0)
        if ua>=30: ulbl="EXTREME"
        elif ua>=20: ulbl="HIGH"
        elif ua>=0: ulbl="ACTIVE"
        elif ua>=-10: ulbl="APPROACHING"
        else: ulbl="INACTIVE"
        body += f"\n{'='*70}\nUVXY VOL REGIME SHIFT\n{'='*70}\n"
        body += f"Status:         {ulbl}\n"
        body += f"UVXY Price:     ${up:.2f}\n"
        body += f"SMA(200):       ${us:.2f}\n"
        body += f"% Above SMA200: {ua:+.1f}%\n\n"
        body += f"Key Thresholds:\n"
        body += f"  Signal ON:      ${us:.2f}  (UVXY > SMA200)\n"
        body += f"  High Conviction:${us*1.20:.2f}  (20% above -> 92% WR 20d)\n"
        body += f"  Extreme:        ${us*1.30:.2f}  (30% above -> 94% WR, 100% at 40d+)\n"
        body += f"\n  When active: SPY 83-94% WR 20d, 92-100% WR 60d\n"
        body += f"  Action: Favor UPRO/TQQQ/SOXL over defensive positioning\n"

    # --- DRIF VELOCITY FILTER ---
    drif=status.get('drif_velocity',{})
    if drif:
        body += f"\n{'='*70}\nDRIF VELOCITY FILTER (Crash Speed Gate)\n{'='*70}\n"
        body += f"{'Ticker':<8} {'RSI':>6} {'5d Ret':>8} {'7d Ret':>8} {'Vel':>6}  {'Gate':<6} {'Status':<20}\n"
        body += "-"*70 + "\n"
        for td in ['SPY','QQQ','SMH']:
            if td in drif:
                dd=drif[td]
                gi={'PASS':'PASS','FAIL':'FAIL'}.get(dd['gate'],'---')
                body += f"{td:<8} {dd['rsi']:>6.1f} {dd['cum_ret_5d']:>+7.1f}% {dd['cum_ret_7d']:>+7.1f}% {dd['velocity']:>+5.0f}  {gi:<6} {dd['label']:<20}\n"
        body += "\n  Composer gate: RSI(10) < 25 AND cumulative-return(5d) > -5%\n"
        # DRIF commentary
        any_pass=any(drif[t].get('gate')=='PASS' for t in ['SPY','QQQ','SMH'] if t in drif)
        any_fail=any(drif[t].get('gate')=='FAIL' for t in ['SPY','QQQ','SMH'] if t in drif)
        any_os=any(drif[t].get('gate') in ('PASS','FAIL') for t in ['SPY','QQQ','SMH'] if t in drif)
        if any_pass and not any_fail:
            body += "\n  INTERPRETATION: Dip is stabilizing — price found a floor even though RSI\n"
            body += "  is deeply oversold. Historically, stabilized dips recover reliably.\n"
            body += "  This is the textbook leveraged entry point. DRIF confirms your RSI signal.\n"
        elif any_fail:
            body += "\n  INTERPRETATION: Price is in free-fall — RSI is oversold but the decline\n"
            body += "  is accelerating, not stabilizing. Falling knives have dramatically worse\n"
            body += "  win rates (SMH: only 16.7% when gate fails). WAIT for velocity to slow.\n"
            body += "  The dip-buy fires when cumulative return improves above the gate threshold,\n"
            body += "  indicating selling exhaustion. Patience here protects capital.\n"
        elif not any_os:
            body += "\n  INTERPRETATION: SPY/QQQ/SMH not oversold — velocity gate inactive.\n"
            body += "  No dip-buy conditions present. Normal operations.\n"

    # --- MOVE INDEX ---
    move_st=status.get('move_index',{})
    if move_st:
        body += f"\n{'='*70}\nMOVE INDEX (Rates Volatility)\n{'='*70}\n"
        body += f"Price:    {move_st.get('price',0):.2f}  |  RSI: {move_st.get('rsi',0):.1f}  |  20d Change: {move_st.get('change_20d_pct',0):+.1f}%\n"
        body += f"19A (>115): {'ACTIVE' if move_st.get('price',0)>115 else 'Inactive'} | 19B (20d>50%): {'ACTIVE' if move_st.get('change_20d_pct',0)>50 else 'Inactive'}\n"
        # MOVE commentary
        mp=move_st.get('price',0); mc=move_st.get('change_20d_pct',0)
        if mc>50:
            body += f"\n  INTERPRETATION: MOVE spiked {mc:+.0f}% in 20 days — extreme rates volatility.\n"
            body += f"  This is the strongest contrarian equity signal (86% WR 20d). Bond market\n"
            body += f"  panic tends to be transitory, and equities recover strongly once rates vol\n"
            body += f"  normalizes. The edge is at 20d hold, NOT next-day (only 58% at 5d).\n"
            body += f"  Key combo: If SPY RSI drops below 25, the GLD trade becomes near-certain\n"
            body += f"  (100% WR 20d, n=15). Monitor RSI(10) on MOVE for 19C setup.\n"
        elif mp>115:
            body += f"\n  INTERPRETATION: MOVE elevated at {mp:.0f} — rates market is stressed.\n"
            body += f"  This is a moderate contrarian buy signal (72% WR 20d). Not as powerful\n"
            body += f"  as the extreme spike signal, but still positive expected value.\n"
            body += f"  Rates vol spikes tend to resolve. Equities recover as Treasury market\n"
            body += f"  finds equilibrium. Hold period matters: edge builds over 10-20 days.\n"
        elif mp<90:
            body += f"\n  INTERPRETATION: MOVE subdued at {mp:.0f} — rates market calm.\n"
            body += f"  No MOVE-based signal active. Low MOVE supports risk-on positioning.\n"
        else:
            body += f"\n  INTERPRETATION: MOVE at {mp:.0f} — normal range, no signal active.\n"

    # --- FRED CREDIT SPREAD ---
    if FRED_API_KEY:
        try:
            fred_params={"series_id":"BAMLH0A1HYBB","api_key":FRED_API_KEY,"file_type":"json",
                "observation_start":(datetime.now()-timedelta(days=365)).strftime("%Y-%m-%d"),"sort_order":"desc"}
            fred_resp=req_lib.get("https://api.stlouisfed.org/fred/series/observations",params=fred_params,timeout=10)
            fred_data=fred_resp.json().get("observations",[])
            fred_df=pd.DataFrame(fred_data)
            fred_df=fred_df[fred_df["value"]!="."]
            fred_df["value"]=fred_df["value"].astype(float)
            fred_df=fred_df.sort_values("date")
            if len(fred_df)>=20:
                cur_oas=float(fred_df.iloc[-1]["value"])
                chg20=float(cur_oas-fred_df.iloc[-20]["value"])
                lvl="COMPLACENT" if cur_oas<2.0 else "NORMAL" if cur_oas<3.5 else "ELEVATED" if cur_oas<5.0 else "CRISIS"
                trnd="SPIKE" if chg20>0.80 else "DRIFT_UP" if chg20>0.30 else "COMPRESSING" if chg20<-0.30 else "STABLE"
                body += f"\n{'='*70}\nCREDIT SPREAD MONITOR (BB OAS)\n{'='*70}\n"
                body += f"BB OAS:     {cur_oas:.2f}%  [{lvl}]\n"
                body += f"Trend:      {trnd}  (20d: {chg20:+.2f}%)\n"
                body += f"As of:      {fred_df.iloc[-1]['date']}\n"
        except Exception as e:
            print(f"FRED credit spread error: {e}")

    # --- HORMUZ TRANSIT (HormuzTracker.com API — daily updates, CC BY 4.0) ---
    try:
        hz_resp=req_lib.get("https://www.hormuztracker.com/api/data",timeout=15)
        hz_data=hz_resp.json()
        hz_crisis=hz_data.get('crisis',{})
        hz_ships=hz_crisis.get('shipCount',{})
        hz_meta=hz_data.get('meta',{})
        hz_day=hz_meta.get('day',0)
        hz_current=hz_ships.get('current',0)
        hz_baseline=hz_ships.get('baseline',138)
        hz_drop=hz_ships.get('dropPercent',0)
        hz_status=hz_crisis.get('hormuzStatus','unknown')
        hz_severity=hz_crisis.get('severityScore',0)
        hz_verified=hz_ships.get('lastVerified','')[:10]
        hz_trapped=hz_crisis.get('shipsTrapped',{})
        hz_tl=hz_crisis.get('timeline',[])

        body += f"\n{'='*70}\nSTRAIT OF HORMUZ — DAY {hz_day}\n{'='*70}\n"
        body += f"Status:        {'🔴 CLOSED' if hz_status=='closed' else '🟡 RESTRICTED' if hz_status=='restricted' else '🟢 OPEN'}  (Severity: {hz_severity}/10)\n"
        body += f"Ship Count:    ~{hz_current}/day  (baseline: {hz_baseline}/day, ↓{hz_drop}%)\n"
        body += f"Verified:      {hz_verified}\n"
        body += f"Trapped:       ~{hz_trapped.get('insideGulf',0):,} inside Gulf | {hz_trapped.get('seafarersStranded',0):,} seafarers stranded\n"

        # Kill-switch assessment
        if hz_current >= 70:
            body += f"\n🟢 KILL-SWITCH TRIGGERED — transit >70/day, blockade may be breaking\n"
            body += f"   → Watch for insurance reinstatement and carrier resumption.\n"
            body += f"   → If sustained 5 days: UNG/SLV/VLO short, oil longs at risk.\n"
        elif hz_current >= 20:
            body += f"\n🟡 PARTIAL REOPENING — transit rising but below commercial viability\n"
            body += f"   → Permission-based transit only. Insurance still withdrawn.\n"
            body += f"   → Brent likely stays elevated until insurance reinstated.\n"
        else:
            body += f"\n❌ Kill-switch NOT triggered (need >70/day sustained 5 days)\n"
            body += f"   → Blockade intact. Dead-hand framework applies: insurance/reinsurance\n"
            body += f"     breakdown keeps strait commercially closed even if military situation\n"
            body += f"     improves. Reinstating P&I cover takes weeks after physical resolution.\n"
            body += f"   → Your Hormuz positions (UNG/SLV/VLO/STNG/CF/MOS) remain supported.\n"

        # Latest timeline events
        if hz_tl:
            body += f"\nLatest developments:\n"
            for evt in hz_tl[-3:]:
                body += f"  Day {evt.get('day','')} ({evt.get('date','')}):\n"
                body += f"    {evt.get('event','')}\n"
                body += f"    Impact: {evt.get('impact','')}\n"

        body += f"\nSource: HormuzTracker.com (updated daily, CC BY 4.0)\n"
    except Exception as e:
        print(f"Hormuz data error: {e}")

    # --- MARKET INTERNALS (ZBT, McClellan, %Above50SMA, Fibonacci) ---
    breadth = status.get('breadth', {})
    if breadth:
        body += f"\n{'='*70}\nMARKET INTERNALS\n{'='*70}\n"
        # ZBT
        zbt_icon = {'OVERSOLD':'🔴','THRUST':'🟢','NEUTRAL':'⚪'}.get(breadth.get('zbt_zone',''),'⚪')
        body += f"\nBREADTH INDICATORS\n"
        body += f"  ZBT (10d EMA):         {breadth.get('zbt_ema',0):.4f}   {zbt_icon} {breadth.get('zbt_zone','?')}"
        if breadth.get('zbt_oversold_recent') and breadth.get('zbt_zone') != 'OVERSOLD':
            body += " (was oversold in last 10d — watch for thrust)"
        if breadth.get('zbt_thrust'):
            body += "\n  *** ZBT THRUST SIGNAL *** — extremely rare, historically near-100% forward returns"
        body += f"\n  McClellan Oscillator:  {breadth.get('mcclellan',0):>+7.1f}   {breadth.get('mcl_zone','?')} ({breadth.get('mcl_direction','?')})"
        body += f"\n    19d EMA: {breadth.get('mcl_ema19',0):>+7.1f}  |  39d EMA: {breadth.get('mcl_ema39',0):>+7.1f}  |  Summation: {breadth.get('mcl_summation',0):>+.0f}"
        body += f"\n  %Above 50d SMA:        {breadth.get('pct_above_50sma',0):>5.1f}%   ({breadth.get('above50_count',0)}/{breadth.get('above50_total',0)})"
        pct50 = breadth.get('pct_above_50sma', 50)
        if pct50 < 25:
            body += "  🔴 WASHED OUT"
        elif pct50 < 40:
            body += "  🟡 WEAK"
        elif pct50 > 75:
            body += "  🟡 OVEREXTENDED"
        body += f"\n\n  Advancing: {breadth.get('adv',0)}  |  Declining: {breadth.get('dec',0)}  |  Ratio: {breadth.get('zbt_ratio',0):.3f}\n"

        # Market internals interpretation
        body += f"\n  INTERPRETATION:\n"
        if breadth.get('zbt_zone') == 'OVERSOLD' and breadth.get('mcl_zone') == 'OVERSOLD':
            body += f"  Breadth deeply oversold across multiple measures. This is the setup that\n"
            body += f"  precedes ZBT thrust signals. If ZBT surges from <0.40 to >0.615 within\n"
            body += f"  10 days, it's one of the most reliable buy signals in market history.\n"
            body += f"  RSI dip-buy signals have maximum conviction when breadth confirms.\n"
        elif pct50 < 30:
            body += f"  Only {pct50:.0f}% of stocks above 50d SMA — breadth washed out.\n"
            body += f"  Consistent with broad selling pressure, not just mega-cap rotation.\n"
            body += f"  Pairs well with dip-buy signals if McClellan starts improving.\n"
        elif breadth.get('mcl_direction') == 'FALLING' and pct50 > 60:
            body += f"  Breadth momentum deteriorating despite healthy %Above50. Early warning\n"
            body += f"  that rally may be narrowing. Watch for %Above50 to follow McClellan down.\n"
        elif breadth.get('zbt_zone') == 'THRUST':
            body += f"  ZBT thrust signal fired — historically extremely bullish. Near-100% forward\n"
            body += f"  returns at 6 and 12 months. Maximum conviction for equity exposure.\n"
        else:
            body += f"  Breadth in normal range. No extreme readings. Standard signal operations.\n"

    # --- FIBONACCI CONTEXT ---
    fib = status.get('fibonacci', {})
    if fib:
        body += f"\n{'─'*50}\nFIBONACCI CONTEXT\n"
        for sym in ['SPY', 'QQQ', 'SMH']:
            if sym not in fib:
                continue
            for period, pdata in fib[sym].items():
                close = pdata['close']
                trend = pdata['trend']
                body += f"\n  {sym} ({period}): High={pdata['high']:.2f} Low={pdata['low']:.2f} Close={close:.2f} [{trend}]\n"
                for pct_label, lvl in pdata['levels'].items():
                    near_marker = " ◄ NEAR" if lvl['near'] else ""
                    body += f"    {pct_label:>5}% ret: {lvl['level']:>8.2f}  ({lvl['dist_pct']:+.1f}%){near_marker}\n"
                # One-line interpretation for the 30d window
                if period == '30d':
                    l236 = pdata['levels'].get('23.6', {})
                    l382 = pdata['levels'].get('38.2', {})
                    l618 = pdata['levels'].get('61.8', {})
                    if trend == 'DOWN':
                        if l236.get('dist_pct', 0) < -1:
                            body += f"    → Below 23.6% retracement. Downtrend intact, no meaningful bounce yet.\n"
                        elif l236.get('near'):
                            body += f"    → Testing 23.6% retracement. Shallow bounce — needs {l382.get('level',0):.0f} (38.2%) for healthy correction.\n"
                        elif l382.get('near'):
                            body += f"    → At 38.2% retracement — correction zone. Watch for reversal or continuation.\n"
                        elif l618.get('near'):
                            body += f"    → Deep retracement at 61.8%. Prior trend likely resuming if this holds.\n"
                break  # Only show 30d in main section, 60d available in fib dict

    # --- ROLLING BRIER SCORES ---
    if os.path.exists(BRIER_JSON_PATH):
        try:
            with open(BRIER_JSON_PATH,'r') as bf: brier_hist=json.load(bf)
            if brier_hist:
                latest_bd=sorted(brier_hist.keys())[-1]
                latest_bs=brier_hist[latest_bd]
                degraded=[]
                body += f"\n{'='*70}\nSIGNAL CALIBRATION (Rolling Brier)\n{'='*70}\n"
                body += f"As of: {latest_bd}\n"
                body += f"{'Signal':<28} {'WR':>5} {'Trail':>5} {'BSS':>6} {'Health':>8}\n"+"-"*58+"\n"
                for sid,s in sorted(latest_bs.items(),key=lambda x:{'critical':0,'warning':1,'healthy':2}.get(x[1].get('health',''),9)):
                    awr=s.get('actual_wr',0); twr=s.get('trail_wr',0); bss=s.get('bss',0)
                    hlth=s.get('health','?'); act='*' if s.get('active') else ' '
                    body += f"{act}{sid:<27} {awr:>4.0%} {twr:>4.0%} {bss:>+5.3f} {hlth:>8}\n"
                    if awr>0 and twr<awr-0.15 and s.get('n',0)>=15: degraded.append((sid,awr,twr))
                if degraded:
                    body += "\nDEGRADATION ALERTS:\n"
                    for sid,awr,twr in degraded:
                        body += f"  {sid}: Trail WR ({twr:.0%}) is {(awr-twr)*100:.0f}pp below overall ({awr:.0%})\n"
        except Exception as e:
            print(f"Brier scores error: {e}")

    if is_preclose:
        body += f"\n{'='*70}\nNOTE: PRE-CLOSE preview. Signals may change by close.\nFinal confirmation at 4:05 PM ET.\n{'='*70}\n"
    return body

def send_email(subject, body):
    if not SENDER_EMAIL or not SENDER_PASSWORD or not RECIPIENT_EMAIL:
        print("Email not configured - printing to console:")
        print(f"Subject: {subject}")
        print(body)
        return False
    try:
        msg = MIMEMultipart()
        msg['From']=SENDER_EMAIL; msg['To']=RECIPIENT_EMAIL; msg['Subject']=subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls(); server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg); server.quit()
        print(f"Email sent to {RECIPIENT_EMAIL}"); return True
    except Exception as e:
        print(f"Email failed: {e}"); return False

def main():
    print(f"Signal Monitor v4.4 at {datetime.now()}")
    print(f"Mode: {'PRE-CLOSE' if IS_PRECLOSE else 'MARKET CLOSE'}")
    tickers = [
        'SMH','SPY','QQQ','IWM','XLP','XLU','XLV',
        'GLD','TLT','HYG','LQD','TMV','SHY','USDU','UCO','BOIL',
        'UVXY','VIXM','SVXY',
        'EDC','YINN','KORU','EURL','INDL','BTC-USD',
        'AMD','NVDA','NAIL','CURE','FAS','LABU',
        'TQQQ','SOXL','SOXS','TECL','DRN','DFEN',
        'VOOV','VOOG','VTV','QQQE','USMV',
        'XLE','XLF',
        'GDX','GDXJ','JNUG','NUGT',
        'UPRO','CTA','DBMF','BTAL','KMLM',
        'RSP','SPHB','SPLV',
        '^MOVE',
    ]
    print("Downloading market data...")
    data = download_data(tickers)
    print(f"Downloaded {len(data)} tickers")

    # Download breadth tickers for Market Internals (ZBT, McClellan, %Above50SMA)
    print("Downloading breadth tickers...")
    breadth_raw = download_data(BREADTH_TICKERS, period='60d')
    print(f"Downloaded {len(breadth_raw)} breadth tickers")

    rolling_betas = calculate_rolling_betas(data)
    alerts, status = check_signals(data)
    status['rolling_betas'] = rolling_betas

    # Compute Market Internals
    breadth = compute_market_breadth(breadth_raw)
    status['breadth'] = breadth
    if breadth:
        print(f"  ZBT: {breadth['zbt_ema']:.4f} ({breadth['zbt_zone']}) | McClellan: {breadth['mcclellan']:+.1f} | %Above50: {breadth['pct_above_50sma']:.1f}%")
        # Add breadth-based alerts
        if breadth['zbt_thrust']:
            alerts.append(('[BUY] ZBT THRUST SIGNAL','ZBT surged from oversold to >0.615 — historically near-100% forward returns\n   Extremely rare. Maximum conviction for equity exposure.','buy'))
        if breadth['zbt_zone'] == 'OVERSOLD':
            alerts.append(('[WATCH] ZBT OVERSOLD','ZBT EMA < 0.40 — breadth precondition for thrust signal\n   Monitor for rapid recovery to >0.615 within 10 days','watch'))
        if breadth['mcclellan'] < -100:
            alerts.append(('[WATCH] McCLELLAN OVERSOLD',f"McClellan Oscillator at {breadth['mcclellan']:+.1f} — breadth capitulation\n   Pairs with RSI dip-buys for high conviction. Watch for zero crossover.",'watch'))
        elif breadth['mcclellan'] > 100:
            alerts.append(('[WATCH] McCLELLAN OVERBOUGHT',f"McClellan Oscillator at {breadth['mcclellan']:+.1f} — breadth stretched\n   Pairs with SPY RSI>79 for hedge signals.",'watch'))
        if breadth['pct_above_50sma'] < 25:
            alerts.append(('[WATCH] BREADTH WASHED OUT',f"Only {breadth['pct_above_50sma']:.0f}% above 50d SMA — broad selling\n   Supports dip-buy signals if McClellan starts improving.",'watch'))

    # Compute Fibonacci levels
    fib = compute_fibonacci_levels(data)
    status['fibonacci'] = fib
    if alerts:
        buy_n=len([a for a in alerts if a[2]=='buy'])
        exit_n=len([a for a in alerts if a[2] in ['exit','short']])
        urg="EXIT SIGNALS" if exit_n>0 else "BUY SIGNALS" if buy_n>0 else "WATCH"
        tm="PRE-CLOSE" if IS_PRECLOSE else "CLOSE"
        subject=f"[{tm}] Market Signals: {len(alerts)} Alert(s) - {urg}"
    else:
        tm="PRE-CLOSE" if IS_PRECLOSE else "CLOSE"
        subject=f"[{tm}] Market Signals: No Alerts"
    body = format_email(alerts, status, IS_PRECLOSE)
    send_email(subject, body)
    print(f"\n{len(alerts)} signal(s) detected")
    for t,m,_ in alerts: print(f"  {t}")

if __name__ == "__main__":
    main()
