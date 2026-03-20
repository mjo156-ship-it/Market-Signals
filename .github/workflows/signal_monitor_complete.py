#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Market Signal Monitor v4.2
========================================
Groups 1-12: Core RSI signals
Group 13: Deep Value (MaRet/CumRet crash+drawdown)
Group 14: Crisis Alpha v2 (vol compression regime)
Group 15: Signal Degradation (trailing WR monitoring)
Group 17: DFEN (RSI+SMA200+Bollinger Band)
Group 18: GLD & Miners
+ Rolling Beta vs SPY
+ TLT bond momentum banner

SCHEDULE: Two emails daily (weekdays)
- 3:15 PM ET: Pre-close preview
- 4:05 PM ET: Market close confirmation
"""
import os, sys, smtplib
import yfinance as yf
import pandas as pd
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

SENDER_EMAIL = os.environ.get('SENDER_EMAIL', '')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD', '')
RECIPIENT_EMAIL = os.environ.get('RECIPIENT_EMAIL', '')
PHONE_EMAIL = os.environ.get('PHONE_EMAIL', '')
IS_PRECLOSE = len(sys.argv) > 1 and sys.argv[1] == 'preclose'

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
            indicators[ticker] = {
                'price':price,'rsi10':rsi10,'rsi50':rsi50,
                'sma200':sma200,'sma50':sma50,'ema9':ema9,'ema20':ema20,'ema21':ema21,
                'maret_10':maret_10,'cumret_10':cumret_10,'cumret_30':cumret_30,
                'cumret_50':cumret_50,'cumret_100':cumret_100,
                'std_10':std_10,'std_50':std_50,'vol_ratio':vol_ratio,
                'bb_sma':bb_sma,'bb_upper':bb_upper,'bb_lower':bb_lower,
                'bb_pctb':bb_pctb,'bb_width':bb_width,'chg5d':chg5d,
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
    print(f"Signal Monitor v4.2 at {datetime.now()}")
    print(f"Mode: {'PRE-CLOSE' if IS_PRECLOSE else 'MARKET CLOSE'}")
    tickers = [
        'SMH','SPY','QQQ','IWM','XLP','XLU','XLV',
        'GLD','TLT','HYG','LQD','TMV','USDU','UCO','BOIL',
        'UVXY','EDC','YINN','KORU','EURL','INDL','BTC-USD',
        'AMD','NVDA','NAIL','CURE','FAS','LABU',
        'TQQQ','SOXL','TECL','DRN','DFEN',
        'VOOV','VOOG','VTV','QQQE','USMV',
        'XLE','XLF',
        'GDX','GDXJ','JNUG','NUGT',
        'UPRO','CTA','DBMF','BTAL','KMLM',
    ]
    print("Downloading market data...")
    data = download_data(tickers)
    print(f"Downloaded {len(data)} tickers")
    rolling_betas = calculate_rolling_betas(data)
    alerts, status = check_signals(data)
    status['rolling_betas'] = rolling_betas
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
