#!/usr/bin/env python3
"""
CHF Signal Monitor — Dashboard Server v4.6
============================================
Self-hosted real-time market signal dashboard with Brier Score calibration,
rolling beta vs SPY, gold miners signal group, market breadth regime,
SPHB/SPLV risk appetite, FXY carry trade (Group 20), CPER copper regime
(Group 21), LABU/SOXL dip buy, UVXY SMA200 cross, and audit-validated
signals (Groups 19-30+).

Usage:
    pip install flask yfinance pandas numpy requests
    python chf_dashboard_server.py

Then open http://localhost:5050 in your browser.

v4.5: FXY carry trade (Group 20), CPER copper regime (Group 21),
      LABU/SOXL dip buy, UVXY SMA200 cross (Group 29),
      Oil Supply Shock EITHER/OR (UCO OR USO + USDU>55)
v4.4: UVXY Vol Regime Shift (Group 13), DRIF Velocity Filter (Group 30),
      MOVE Index signals (Group 19A/B/C), Multi-Oversold Breadth (Group 21b),
      FRED Credit Spread monitor, ZBT/breadth dashboard enhancements,
      Brier persistence to JSON, rolling Brier in email
v4.3: Groups 19-29 (audit signals), breadth regime card, risk appetite card,
      ILS/RSP/SPLV/SPHB/DFEN tickers added
v4.2: Rolling beta vs SPY, GLD & miners (Group 18), CSS display fix
v4.0: Signal Calibration tab with rolling Brier scores
"""

from flask import Flask, jsonify, Response
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import time
import threading
from datetime import datetime, timedelta
import requests as req_lib

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
TICKERS = [
    'SMH','SPY','QQQ','IWM',
    'XLP','XLU','XLV','XLY','XLE','XLF',
    'GLD','TLT','HYG','LQD','TMV','SHY',
    'USDU','UCO','BOIL','DBC',
    'UVXY','SVXY','VIXM',
    'EDC','YINN','KORU','EURL','INDL',
    'BTC-USD',
    'AMD','NVDA',
    'NAIL','CURE','FAS','LABU',
    'TQQQ','SOXL','SOXS','TECL','DRN','UPRO',
    'VOOV','VOOG','VTV','QQQE','VOX','USMV',
    'BTAL','DBMF','KMLM','CTA',
    'FNGO',
    'UUP','SLV','CPER',
    # Gold Miners (Group 18)
    'GDX','GDXJ','JNUG','NUGT',
    # v4.3 additions
    'RSP','ILS','SPLV','SPHB','DFEN',
    # v4.4 additions
    '^MOVE',
    # v4.5 additions (Wishlist Items 7, 9)
    'FXY','COPX','USO',
    # v4.6 additions
    'IGV',  # SMH/IGV divergence
    '^VIX9D','^VIX','^VIX3M','^VIX6M','^VIX1Y',  # VIX term structure
]

CACHE_SECONDS = 60
HISTORY_PERIOD = '2y'
BREADTH_PERIOD = '60d'

# Broad market tickers for inline breadth (ZBT, McClellan, %Above50SMA)
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

cache = {'data': None, 'ts': 0, 'brier': None, 'brier_ts': 0, 'rolling_betas': None, 'composer': None, 'composer_ts': 0}
lock = threading.Lock()

# FRED Credit Spread config (optional — set FRED_API_KEY env var)
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
BRIER_JSON_PATH = os.environ.get("BRIER_JSON_PATH", "./brier_scores.json")

# Composer API config (optional — set COMPOSER_KEY_ID and COMPOSER_KEY_SECRET env vars)
COMPOSER_KEY_ID = os.environ.get("COMPOSER_KEY_ID", "9006c9ea-ea0c-4827-919a-e91da9e22146")
COMPOSER_KEY_SECRET = os.environ.get("COMPOSER_KEY_SECRET", "27c291d6-3986-40e2-a509-a61787090f92")
COMPOSER_BASE = "https://api.composer.trade/api/v0.1"
COMPOSER_CACHE_SECONDS = 300  # 5-min cache (rate limit: 1 req/sec)
COMPOSER_HISTORY_PATH = os.environ.get("COMPOSER_HISTORY_PATH", "./composer_history.json")

# Fidelity CSV ingest (optional — drop CSV exports in this folder)
FIDELITY_CSV_DIR = os.environ.get("FIDELITY_CSV_DIR", "./fidelity_csv")

# ═══════════════════════════════════════════════════════════════════
# CALCULATIONS
# ═══════════════════════════════════════════════════════════════════
def rsi_wilder(prices, period):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def sf(val):
    if isinstance(val, pd.Series):
        return float(val.iloc[-1]) if len(val) > 0 else 0.0
    elif isinstance(val, np.ndarray):
        return float(val[-1]) if len(val) > 0 else 0.0
    elif pd.isna(val):
        return 0.0
    return float(val)

# ═══════════════════════════════════════════════════════════════════
# ROLLING BETA vs SPY
# ═══════════════════════════════════════════════════════════════════
def compute_rolling_betas(raw_data, regime='UNKNOWN'):
    """Compute rolling beta vs SPY using actual Q1 2026 Roth portfolio weights."""
    if 'SPY' not in raw_data:
        return []
    spy_c = raw_data['SPY']['Close']
    if isinstance(spy_c, pd.DataFrame): spy_c = spy_c.iloc[:,0]
    spy_ret = spy_c.pct_change()

    def _beta(ticker, window):
        if ticker not in raw_data: return None
        ac = raw_data[ticker]['Close']
        if isinstance(ac, pd.DataFrame): ac = ac.iloc[:,0]
        ar = ac.pct_change()
        common = spy_ret.dropna().index.intersection(ar.dropna().index)
        if len(common) < window + 10: return None
        sr = spy_ret.loc[common]
        a = ar.loc[common]
        cov = a.rolling(window).cov(sr)
        var = sr.rolling(window).var()
        b = cov / var
        v = b.iloc[-1]
        return round(float(v), 3) if not pd.isna(v) else None

    regime_weights = {
        'BULL + VOL COMPRESS': {
            'Equity 1x':0.52,'Lev Equity':0.15,'MF/Alts':0.08,
            'Gold/Commod':0.07,'Vol/Hedge':0.04,'Bonds (net)':0.10,'Currency':0.04,
        },
        'BULL + VOL EXPAND': {
            'Equity 1x':0.48,'Lev Equity':0.12,'MF/Alts':0.12,
            'Gold/Commod':0.08,'Vol/Hedge':0.05,'Bonds (net)':0.10,'Currency':0.05,
        },
        'BEAR RECOVERY': {
            'Equity 1x':0.42,'Lev Equity':0.10,'MF/Alts':0.18,
            'Gold/Commod':0.10,'Vol/Hedge':0.06,'Bonds (net)':0.08,'Currency':0.06,
        },
        'BEAR DEFENSIVE': {
            'Equity 1x':0.35,'Lev Equity':0.08,'MF/Alts':0.20,
            'Gold/Commod':0.10,'Vol/Hedge':0.08,'Bonds (net)':0.06,'Currency':0.13,
        },
    }
    default_weights = {
        'Equity 1x':0.45,'Lev Equity':0.12,'MF/Alts':0.13,
        'Gold/Commod':0.08,'Vol/Hedge':0.05,'Bonds (net)':0.10,'Currency':0.07,
    }
    blend_w = regime_weights.get(regime, default_weights)

    groups = [
        ('Equity 1x',   [('SPY',1.0)]),
        ('Lev Equity',  [('UPRO',1.0)]),
        ('MF/Alts',     [('CTA',0.25),('DBMF',0.25),('BTAL',0.30),('KMLM',0.20)]),
        ('Gold/Commod', [('GLD',0.85),('DBC',0.15)]),
        ('Vol/Hedge',   [('UVXY',1.0)]),
        ('Bonds (net)', [('TLT',0.35),('SHY',0.30),('TMV',0.35)]),
        ('Currency',    [('UUP',1.0)]),
    ]
    results = []
    blend = {'name':'Est. Blend','b63':0,'b126':0,'b252':0,'is_blend':True,'regime':regime}
    for gname, tickers_w in groups:
        row = {'name': gname, 'blend_wt': blend_w.get(gname, 0)}
        for wname, w in [('b63',63),('b126',126),('b252',252)]:
            gb, total_w = 0, 0
            for t, tw in tickers_w:
                b = _beta(t, w)
                if b is not None:
                    gb += b * tw
                    total_w += tw
            if total_w > 0:
                gb = gb / total_w
                row[wname] = round(gb, 3)
                blend[wname] += gb * blend_w.get(gname, 0)
            else:
                row[wname] = None
        results.append(row)
    blend['b63']=round(blend['b63'],3); blend['b126']=round(blend['b126'],3); blend['b252']=round(blend['b252'],3)
    results.append(blend)
    return results


# ═══════════════════════════════════════════════════════════════════
# BRIER SIGNAL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════
BRIER_SIGNALS = [
    {'id':'spy_lt21','name':'SPY RSI<21 → UPRO','cond':lambda i: i.get('SPY',{}).get('rsi10',50)<21,
     'target':'UPRO','days':5,'dir':'long','wr':0.87,'tier':1,'min_n':10},
    {'id':'spy_lt30','name':'SPY RSI<30 → UPRO','cond':lambda i: 21<=i.get('SPY',{}).get('rsi10',50)<30,
     'target':'UPRO','days':5,'dir':'long','wr':0.69,'tier':1,'min_n':15},
    {'id':'qqq_lt20','name':'QQQ RSI<20 → TQQQ','cond':lambda i: i.get('QQQ',{}).get('rsi10',50)<20,
     'target':'TQQQ','days':5,'dir':'long','wr':1.00,'tier':1,'min_n':5},
    {'id':'cure_lt21','name':'CURE RSI<21','cond':lambda i: i.get('CURE',{}).get('rsi10',50)<21,
     'target':'CURE','days':5,'dir':'long','wr':0.85,'tier':1,'min_n':10},
    {'id':'cure_lt25','name':'CURE RSI<25','cond':lambda i: 21<=i.get('CURE',{}).get('rsi10',50)<25,
     'target':'CURE','days':5,'dir':'long','wr':0.81,'tier':1,'min_n':10},
    {'id':'spy_gt79_uvxy','name':'SPY RSI>79 → UVXY (1d)','cond':lambda i: i.get('SPY',{}).get('rsi10',50)>79,
     'target':'UVXY','days':1,'dir':'long','wr':0.686,'tier':1,'min_n':15},
    {'id':'qqq_gt79','name':'QQQ RSI>79 → UVXY','cond':lambda i: i.get('QQQ',{}).get('rsi10',50)>79,
     'target':'UVXY','days':5,'dir':'long','wr':0.67,'tier':1,'min_n':15},
    {'id':'spy_gt85','name':'SPY RSI>85 → Exit UPRO','cond':lambda i: i.get('SPY',{}).get('rsi10',50)>85,
     'target':'UPRO','days':5,'dir':'short','wr':0.64,'tier':1,'min_n':5},
    {'id':'uco_gt75','name':'UCO RSI>75 → TMV','cond':lambda i: i.get('UCO',{}).get('rsi10',50)>75,
     'target':'TMV','days':5,'dir':'long','wr':0.65,'tier':1,'min_n':15},
    {'id':'double_sig','name':'GLD>79 + USDU<25 → TQQQ','cond':lambda i: i.get('GLD',{}).get('rsi10',50)>79 and i.get('USDU',{}).get('rsi10',50)<25,
     'target':'TQQQ','days':5,'dir':'long','wr':0.88,'tier':2,'min_n':5},
    {'id':'soxs_squeeze','name':'SMH>79 + USDU>70 → SOXS','cond':lambda i: i.get('SMH',{}).get('rsi10',50)>79 and i.get('USDU',{}).get('rsi10',50)>70,
     'target':'SOXS','days':5,'dir':'long','wr':1.00,'tier':2,'min_n':5},
    {'id':'gld_ob','name':'GLD RSI>79 → TQQQ','cond':lambda i: i.get('GLD',{}).get('rsi10',50)>79 and i.get('USDU',{}).get('rsi10',50)>=25,
     'target':'TQQQ','days':5,'dir':'long','wr':0.72,'tier':2,'min_n':10},
    {'id':'fas_lt30','name':'FAS RSI<30','cond':lambda i: i.get('FAS',{}).get('rsi10',50)<30,
     'target':'FAS','days':5,'dir':'long','wr':0.63,'tier':2,'min_n':15},
    {'id':'labu_lt25','name':'LABU RSI<25','cond':lambda i: i.get('LABU',{}).get('rsi10',50)<25,
     'target':'LABU','days':5,'dir':'long','wr':0.66,'tier':2,'min_n':10},
    {'id':'uvxy_gt82','name':'UVXY RSI>82 → SOXL (B1)','cond':lambda i: i.get('UVXY',{}).get('rsi10',50)>82,
     'target':'SOXL','days':1,'dir':'long','wr':0.81,'tier':2,'min_n':8},
    {'id':'def_rot','name':'Defensive OB → TQQQ','cond':lambda i: any(i.get(t,{}).get('rsi10',0)>79 for t in ['XLP','XLU','XLV']) and i.get('SPY',{}).get('rsi10',0)<79 and i.get('QQQ',{}).get('rsi10',0)<79,
     'target':'TQQQ','days':20,'dir':'long','wr':0.70,'tier':2,'min_n':10},
    {'id':'fas_gt85','name':'FAS RSI>85 → Exit','cond':lambda i: i.get('FAS',{}).get('rsi10',50)>85,
     'target':'FAS','days':5,'dir':'short','wr':0.92,'tier':2,'min_n':5},
    {'id':'cure_gt79','name':'CURE RSI>79 → Exit','cond':lambda i: i.get('CURE',{}).get('rsi10',50)>79,
     'target':'CURE','days':5,'dir':'short','wr':0.60,'tier':2,'min_n':10},
    # v4.3 — Audit signals (Mar 24-25, 2026)
    {'id':'usmv_gt82','name':'USMV RSI>82 → UVXY [T1]','cond':lambda i: i.get('USMV',{}).get('rsi10',50)>82,
     'target':'UVXY','days':1,'dir':'long','wr':0.75,'tier':1,'min_n':10},
    {'id':'gld_lt22','name':'GLD RSI<22 → TQQQ [T2]','cond':lambda i: i.get('GLD',{}).get('rsi10',50)<22,
     'target':'TQQQ','days':10,'dir':'long','wr':0.733,'tier':2,'min_n':10},
    {'id':'gdxj_lt25','name':'GDXJ RSI<25 → GDXJ [T2]','cond':lambda i: i.get('GDXJ',{}).get('rsi10',50)<25,
     'target':'GDXJ','days':10,'dir':'long','wr':0.732,'tier':2,'min_n':10},
    # v4.4 — DRIF velocity signals
    {'id':'spy_drif25','name':'SPY<25+DRIF → UPRO','cond':lambda i: i.get('SPY',{}).get('rsi10',50)<25 and i.get('SPY',{}).get('cumRet5d',-99)>-5,
     'target':'UPRO','days':5,'dir':'long','wr':1.00,'tier':2,'min_n':5},
    {'id':'qqq_drif25','name':'QQQ<25+DRIF → TQQQ','cond':lambda i: i.get('QQQ',{}).get('rsi10',50)<25 and i.get('QQQ',{}).get('cumRet7d',-99)>-8,
     'target':'TQQQ','days':5,'dir':'long','wr':0.875,'tier':2,'min_n':5},
    {'id':'spy_drif30','name':'SPY<30+DRIF → UPRO 20d','cond':lambda i: i.get('SPY',{}).get('rsi10',50)<30 and i.get('SPY',{}).get('cumRet7d',-99)>-5,
     'target':'UPRO','days':20,'dir':'long','wr':0.761,'tier':2,'min_n':10},
    # v4.4 — Multi-Oversold Breadth
    {'id':'multi_os','name':'Multi-OS Breadth → UPRO','cond':lambda i: all(i.get(t,{}).get('rsi10',50)<30 for t in ['SPY','USMV','VTV','VOOV','UPRO']),
     'target':'UPRO','days':5,'dir':'long','wr':0.778,'tier':2,'min_n':5},
    # v4.5 — FXY carry trade, CPER copper, LABU/SOXL dip
    {'id':'labu_dip25','name':'LABU RSI<25 → LABU/SOXL','cond':lambda i: i.get('LABU',{}).get('rsi10',50)<25,
     'target':'LABU','days':5,'dir':'long','wr':0.82,'tier':1,'min_n':10},
    {'id':'hyg_euphoria','name':'HYG RSI>80+>SMA200 → TQQQ','cond':lambda i: i.get('HYG',{}).get('rsi10',50)>80,
     'target':'TQQQ','days':1,'dir':'long','wr':0.806,'tier':1,'min_n':10},
]

def compute_brier(raw_data):
    """Compute Brier scores for all registered signals using downloaded data"""
    # Pre-compute RSI series for all tickers
    rsi_cache = {}
    close_cache = {}
    for ticker, df in raw_data.items():
        if len(df) < 200:
            continue
        c = df['Close']
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        close_cache[ticker] = c
        rsi_cache[ticker] = rsi_wilder(c, 10)

    results = []
    # Pre-compute cumulative return series for DRIF Brier signals
    cumret5d_cache = {}
    cumret7d_cache = {}
    for ticker, c in close_cache.items():
        cumret5d_cache[ticker] = c.pct_change(5) * 100
        cumret7d_cache[ticker] = c.pct_change(7) * 100

    for sig in BRIER_SIGNALS:
        target = sig['target']
        if target not in close_cache:
            continue
        tc = close_cache[target]
        fwd = tc.pct_change(sig['days']).shift(-sig['days'])

        episodes = []
        dates = tc.index[200:]
        for dt in dates:
            snap = {}
            for tk, rsi_s in rsi_cache.items():
                if dt in rsi_s.index:
                    v = rsi_s.loc[dt]
                    if isinstance(v, pd.Series):
                        v = v.iloc[0]
                    if not pd.isna(v):
                        entry = {'rsi10': float(v)}
                        # Add cumRet fields for DRIF
                        if tk in cumret5d_cache and dt in cumret5d_cache[tk].index:
                            cr5 = cumret5d_cache[tk].loc[dt]
                            if isinstance(cr5, pd.Series): cr5 = cr5.iloc[0]
                            if not pd.isna(cr5): entry['cumRet5d'] = float(cr5)
                        if tk in cumret7d_cache and dt in cumret7d_cache[tk].index:
                            cr7 = cumret7d_cache[tk].loc[dt]
                            if isinstance(cr7, pd.Series): cr7 = cr7.iloc[0]
                            if not pd.isna(cr7): entry['cumRet7d'] = float(cr7)
                        snap[tk] = entry
            try:
                if not sig['cond'](snap):
                    continue
            except:
                continue
            fr = fwd.get(dt, np.nan)
            if isinstance(fr, pd.Series):
                fr = fr.iloc[0]
            if pd.isna(fr):
                continue
            outcome = 1 if (float(fr) > 0 if sig['dir'] == 'long' else float(fr) < 0) else 0
            episodes.append({'date': dt.strftime('%Y-%m-%d'), 'ret': round(float(fr)*100, 2), 'win': outcome})

        if not episodes:
            continue

        outcomes = [e['win'] for e in episodes]
        p = sig['wr']
        brier = np.mean([(p - o)**2 for o in outcomes])
        uncond = np.mean(outcomes)
        brier_base = np.mean([(uncond - o)**2 for o in outcomes])
        bss = 1 - brier / brier_base if brier_base > 0 else 0
        recent = outcomes[-20:] if len(outcomes) >= 20 else outcomes
        trail_wr = np.mean(recent)
        trail_brier = np.mean([(p - o)**2 for o in recent])

        # Health status
        n = len(episodes)
        if n < sig['min_n']:
            health = 'insufficient'
        elif trail_wr < 0.40 and sig['wr'] > 0.60:
            health = 'critical'
        elif trail_wr < 0.50 and sig['wr'] > 0.65:
            health = 'warning'
        elif bss < -0.05 and n >= 20:
            health = 'warning'
        elif np.mean(outcomes) < sig['wr'] - 0.15 and n >= 15:
            health = 'warning'
        else:
            health = 'healthy'

        # Bayesian Kelly
        wins_ct = sum(outcomes)
        losses_ct = n - wins_ct
        win_rets = [e['ret'] for e in episodes if e['win'] == 1]
        loss_rets = [e['ret'] for e in episodes if e['win'] == 0]
        avg_win = np.mean(win_rets) if win_rets else 1.0
        avg_loss = abs(np.mean(loss_rets)) if loss_rets else 1.0
        b_ratio = avg_win / avg_loss if avg_loss > 0 else 10
        p_samples = np.random.beta(wins_ct + 1, losses_ct + 1, 5000)
        bk_samples = np.array([max(0, (b_ratio*pp - (1-pp)) / b_ratio) for pp in p_samples])
        bk_frac = round(float(np.mean(bk_samples)) * 100, 0)
        full_kelly = round(max(0, (b_ratio * sig['wr'] - (1 - sig['wr'])) / b_ratio) * 100, 0)
        bk_ratio = round(bk_frac / full_kelly, 2) if full_kelly > 0 else 0

        results.append({
            'id': sig['id'], 'name': sig['name'], 'tier': sig['tier'],
            'hist_wr': sig['wr'], 'actual_wr': round(np.mean(outcomes), 3),
            'n': n, 'brier': round(brier, 4), 'bss': round(bss, 4),
            'trail_n': len(recent), 'trail_wr': round(trail_wr, 3),
            'trail_brier': round(trail_brier, 4), 'health': health,
            'bk': bk_frac, 'full_kelly': full_kelly, 'bk_ratio': bk_ratio,
            'recent': episodes[-8:],
            'active': False,  # will be set below
        })

    # Check which signals are active RIGHT NOW
    snap_now = {}
    for tk, rsi_s in rsi_cache.items():
        if len(rsi_s) > 0:
            v = rsi_s.iloc[-1]
            if isinstance(v, pd.Series):
                v = v.iloc[0]
            if not pd.isna(v):
                entry = {'rsi10': float(v)}
                if tk in cumret5d_cache and len(cumret5d_cache[tk]) > 0:
                    cr5 = cumret5d_cache[tk].iloc[-1]
                    if isinstance(cr5, pd.Series): cr5 = cr5.iloc[0]
                    if not pd.isna(cr5): entry['cumRet5d'] = float(cr5)
                if tk in cumret7d_cache and len(cumret7d_cache[tk]) > 0:
                    cr7 = cumret7d_cache[tk].iloc[-1]
                    if isinstance(cr7, pd.Series): cr7 = cr7.iloc[0]
                    if not pd.isna(cr7): entry['cumRet7d'] = float(cr7)
                snap_now[tk] = entry
    for r in results:
        sig = next((s for s in BRIER_SIGNALS if s['id'] == r['id']), None)
        if sig:
            try:
                r['active'] = sig['cond'](snap_now)
            except:
                pass

    return results

# ═══════════════════════════════════════════════════════════════════
# BRIER PERSISTENCE
# ═══════════════════════════════════════════════════════════════════
def persist_brier(results):
    """Save Brier scores to JSON for rolling history tracking."""
    try:
        history = {}
        if os.path.exists(BRIER_JSON_PATH):
            with open(BRIER_JSON_PATH, 'r') as f:
                history = json.load(f)
        today = datetime.now().strftime('%Y-%m-%d')
        snapshot = {}
        for r in results:
            snapshot[r['id']] = {
                'actual_wr': r['actual_wr'], 'brier': r['brier'],
                'bss': r['bss'], 'trail_wr': r['trail_wr'],
                'n': r['n'], 'health': r['health'], 'active': r['active'],
            }
        history[today] = snapshot
        keys = sorted(history.keys())
        if len(keys) > 90:
            for k in keys[:-90]:
                del history[k]
        with open(BRIER_JSON_PATH, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"  Brier persistence error: {e}")


# ═══════════════════════════════════════════════════════════════════
# DRIF VELOCITY FILTER
# ═══════════════════════════════════════════════════════════════════
def compute_drif_signals(indicators):
    """Compute DRIF velocity gate status for SPY/QQQ/SMH."""
    configs = [
        {'ticker': 'SPY', 'lever': 'UPRO', 'thresholds': [
            {'rsi_thresh': 25, 'ret_field': 'cumRet5d', 'ret_gate': -5,
             'pass_wr': '100%', 'pass_n': 10, 'fail_wr': '55.6%', 'fail_n': 9, 'hold': '5d'},
            {'rsi_thresh': 30, 'ret_field': 'cumRet7d', 'ret_gate': -5,
             'pass_wr': '76.1%', 'pass_n': 46, 'fail_wr': '56.2%', 'fail_n': 32, 'hold': '20d'},
        ]},
        {'ticker': 'QQQ', 'lever': 'TQQQ', 'thresholds': [
            {'rsi_thresh': 25, 'ret_field': 'cumRet7d', 'ret_gate': -8,
             'pass_wr': '87.5%', 'pass_n': 16, 'fail_wr': '57.1%', 'fail_n': 7, 'hold': '5d'},
        ]},
        {'ticker': 'SMH', 'lever': 'SOXL', 'thresholds': [
            {'rsi_thresh': 25, 'ret_field': 'cumRet5d', 'ret_gate': -5,
             'pass_wr': '68.8%', 'pass_n': 16, 'fail_wr': '16.7%', 'fail_n': 6, 'hold': '5d'},
        ]},
    ]
    drif = {}
    for cfg in configs:
        t = cfg['ticker']
        ind = indicators.get(t)
        if not ind:
            continue
        rsi = ind.get('rsi', 50)
        vel = ind.get('rsiVelocity', 0)
        entry = {
            'ticker': t, 'lever': cfg['lever'], 'rsi': rsi, 'velocity': vel,
            'cumRet5d': ind.get('cumRet5d', 0), 'cumRet7d': ind.get('cumRet7d', 0),
            'gate': 'N/A', 'label': 'NOT OVERSOLD', 'level': None,
            'passWr': None, 'passN': None, 'failWr': None, 'failN': None,
            'retField': None, 'retGate': None, 'retVal': None, 'hold': None,
        }
        for thresh in cfg['thresholds']:
            if rsi < thresh['rsi_thresh']:
                ret_val = ind.get(thresh['ret_field'], 0)
                entry['level'] = f"RSI<{thresh['rsi_thresh']}"
                entry['retField'] = thresh['ret_field']
                entry['retGate'] = thresh['ret_gate']
                entry['retVal'] = ret_val
                entry['hold'] = thresh['hold']
                if ret_val > thresh['ret_gate']:
                    entry['gate'] = 'PASS'
                    entry['label'] = 'STABILIZED DIP'
                    entry['passWr'] = thresh['pass_wr']
                    entry['passN'] = thresh['pass_n']
                else:
                    entry['gate'] = 'FAIL'
                    entry['label'] = 'FALLING KNIFE'
                    entry['failWr'] = thresh['fail_wr']
                    entry['failN'] = thresh['fail_n']
                break
        drif[t] = entry
    return drif


# ═══════════════════════════════════════════════════════════════════
# FRED CREDIT SPREAD MONITOR
# ═══════════════════════════════════════════════════════════════════
def compute_fred_credit():
    """Fetch BB OAS from FRED and compute signal status."""
    if not FRED_API_KEY:
        return {}
    try:
        params = {
            "series_id": "BAMLH0A1HYBB",
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            "sort_order": "desc",
        }
        resp = req_lib.get(FRED_BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("observations", [])
        df = pd.DataFrame(data)
        df = df[df["value"] != "."]
        df["value"] = df["value"].astype(float)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        if df.empty:
            return {}
        current = float(df["value"].iloc[-1])
        change_20d = float(current - df["value"].iloc[-20]) if len(df) >= 20 else None
        sma_50 = float(df["value"].rolling(50).mean().iloc[-1]) if len(df) >= 50 else None
        pctile = float((df["value"] <= current).mean() * 100)
        if current < 2.0: level = "COMPLACENT"
        elif current < 3.5: level = "NORMAL"
        elif current < 5.0: level = "ELEVATED"
        else: level = "CRISIS"
        trend = "STABLE"
        if change_20d is not None:
            if change_20d > 0.80: trend = "SPIKE"
            elif change_20d > 0.30: trend = "DRIFT_UP"
            elif change_20d < -0.30: trend = "COMPRESSING"
        alert = False
        alert_reason = ""
        if level == "COMPLACENT" and trend in ("SPIKE", "DRIFT_UP"):
            alert = True
            alert_reason = "Spread widening from tight levels — early stress"
        elif level == "ELEVATED" and trend == "SPIKE":
            alert = True
            alert_reason = "Rapid widening into stress zone"
        elif level == "CRISIS":
            alert = True
            alert_reason = "Crisis territory — full risk-off"
        return {
            'current_oas': round(current, 2),
            'change_20d': round(change_20d, 2) if change_20d else None,
            'sma_50': round(sma_50, 2) if sma_50 else None,
            'above_sma50': current > sma_50 if sma_50 else None,
            'percentile_1y': round(pctile, 1),
            'level': level, 'trend': trend,
            'alert': alert, 'alert_reason': alert_reason,
            'as_of': df.index[-1].strftime("%Y-%m-%d"),
        }
    except Exception as e:
        print(f"  FRED credit spread error: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════
# MOVE INDEX SIGNALS (Group 19A/B/C)
# ═══════════════════════════════════════════════════════════════════
def compute_move_signals(raw_data, indicators):
    """Compute MOVE Index signal group."""
    move_key = '^MOVE'
    if move_key not in raw_data or move_key not in indicators:
        return {}
    ind = indicators[move_key]
    df = raw_data[move_key]
    c = df['Close']
    if isinstance(c, pd.DataFrame): c = c.iloc[:,0]
    price = ind.get('price', 0)
    rsi = ind.get('rsi', 50)
    sma200 = ind.get('sma200', 0)
    pct_above_sma200 = ((price / sma200 - 1) * 100) if sma200 > 0 else 0
    change_20d_pct = 0
    if len(c) >= 21:
        prev = float(c.iloc[-21])
        if prev > 0:
            change_20d_pct = ((price / prev) - 1) * 100
    # 19C: RSI was >79 in last 10 days and now <60
    rsi_series = rsi_wilder(c, 10)
    was_ob_recently = False
    if len(rsi_series) >= 10:
        recent_rsi = rsi_series.iloc[-10:]
        was_ob_recently = float(recent_rsi.max()) > 79
    return {
        'price': round(price, 2), 'rsi': round(rsi, 1),
        'sma200': round(sma200, 2),
        'pct_above_sma200': round(pct_above_sma200, 1),
        'change_20d_pct': round(change_20d_pct, 1),
        '19A_active': price > 115,
        '19B_active': change_20d_pct > 50,
        '19C_active': was_ob_recently and rsi < 60,
        '19C_ready': was_ob_recently,
    }


# ═══════════════════════════════════════════════════════════════════
# UVXY VOL REGIME SHIFT (Group 13)
# ═══════════════════════════════════════════════════════════════════
def compute_uvxy_vol_regime(indicators):
    """Compute UVXY Vol Regime Shift signal."""
    uvxy = indicators.get('UVXY', {})
    price = uvxy.get('price', 0)
    sma200 = uvxy.get('sma200', 0)
    if sma200 <= 0:
        return {}
    pct_above = (price / sma200 - 1) * 100
    if pct_above >= 30: tier, tier_color = 'EXTREME', 'green'
    elif pct_above >= 20: tier, tier_color = 'HIGH', 'green'
    elif pct_above >= 0: tier, tier_color = 'ACTIVE', 'cyan'
    elif pct_above >= -10: tier, tier_color = 'APPROACHING', 'amber'
    else: tier, tier_color = 'INACTIVE', 'gray'
    return {
        'price': round(price, 2), 'sma200': round(sma200, 2),
        'pct_above': round(pct_above, 1),
        'tier': tier, 'tier_color': tier_color,
        'threshold_signal': round(sma200, 2),
        'threshold_high': round(sma200 * 1.20, 2),
        'threshold_extreme': round(sma200 * 1.30, 2),
    }


# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# COMPOSER PORTFOLIO API
# ═══════════════════════════════════════════════════════════════════
def _composer_get(path, timeout=15):
    """Make authenticated GET request to Composer API."""
    if not COMPOSER_KEY_ID or not COMPOSER_KEY_SECRET:
        return None
    try:
        resp = req_lib.get(
            f"{COMPOSER_BASE}{path}",
            headers={"x-api-key-id": COMPOSER_KEY_ID, "authorization": f"Bearer {COMPOSER_KEY_SECRET}"},
            timeout=timeout
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  Composer API error ({path}): {e}")
        return None


def fetch_composer_data():
    """Fetch full portfolio state from Composer API. ~8 calls across 2 accounts."""
    if not COMPOSER_KEY_ID or not COMPOSER_KEY_SECRET:
        return None

    import time as _t
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching Composer portfolio data...")

    # 1. List accounts
    acct_resp = _composer_get("/accounts/list")
    if not acct_resp or 'accounts' not in acct_resp:
        print("  Composer: no accounts returned")
        return None

    accounts = []
    consolidated_holdings = {}
    total_value = 0
    total_today_change = 0

    for acct in acct_resp['accounts']:
        aid = acct['account_uuid']
        atype = acct.get('account_type', 'Unknown')
        _t.sleep(1.1)  # rate limit

        # 2. Total stats
        total_stats = _composer_get(f"/portfolio/accounts/{aid}/total-stats")
        _t.sleep(1.1)

        # 3. Symphony stats
        sym_stats = _composer_get(f"/portfolio/accounts/{aid}/symphony-stats-meta")
        _t.sleep(1.1)

        # 4. Holding stats
        hold_stats = _composer_get(f"/portfolio/accounts/{aid}/holding-stats")
        _t.sleep(1.1)

        # 5. Portfolio history
        port_hist = _composer_get(f"/portfolio/accounts/{aid}/portfolio-history")

        acct_value = total_stats.get('portfolio_value', 0) if total_stats else 0
        acct_today = total_stats.get('todays_dollar_change', 0) if total_stats else 0
        total_value += acct_value
        total_today_change += acct_today

        # Parse symphonies
        symphonies = []
        if sym_stats and 'symphonies' in sym_stats:
            for s in sym_stats['symphonies']:
                sym_val = s.get('value', 0) or 0
                symphonies.append({
                    'id': s.get('id', ''),
                    'name': s.get('name', 'Unknown'),
                    'value': sym_val,
                    'pct_of_account': round(sym_val / acct_value * 100, 1) if acct_value > 0 else 0,
                    'simple_return': s.get('simple_return'),
                    'twr': s.get('time_weighted_return'),
                    'annualized_return': s.get('annualized_rate_of_return'),
                    'sharpe': s.get('sharpe_ratio'),
                    'max_dd': s.get('max_drawdown'),
                    'last_pct_change': s.get('last_percent_change'),
                    'last_dollar_change': s.get('last_dollar_change'),
                    'cash': s.get('cash', 0),
                    'invested_since': s.get('invested_since'),
                    'last_rebalance': s.get('last_rebalance_on'),
                    'next_rebalance': s.get('next_rebalance_date'),
                    'may_rebalance_today': s.get('may_rebalance_today', False),
                    'skip_rebalance_today': s.get('skip_rebalance_today', False),
                    'holdings': [{
                        'ticker': h.get('ticker', ''),
                        'price': h.get('price'),
                        'allocation': h.get('allocation'),
                        'value': h.get('value'),
                        'last_pct_change': h.get('last_percent_change'),
                    } for h in s.get('holdings', [])],
                    'color': s.get('color', '#888'),
                })

        # Parse holdings for consolidated view
        if hold_stats and 'holdings' in hold_stats:
            for h in hold_stats['holdings']:
                sym = h.get('symbol', '')
                if not sym:
                    continue
                val = h.get('notional_value', 0) or 0
                if sym not in consolidated_holdings:
                    consolidated_holdings[sym] = {
                        'symbol': sym,
                        'name': h.get('name', sym),
                        'total_value': 0,
                        'total_amount': 0,
                        'price': h.get('price', 0),
                        'today_pct': h.get('todays_change_percent', 0),
                        'today_dollar': h.get('todays_change', 0),
                        'total_change_pct': h.get('total_change_percent', 0),
                        'cost_basis': 0,
                        'accounts': [],
                    }
                consolidated_holdings[sym]['total_value'] += val
                consolidated_holdings[sym]['total_amount'] += (h.get('direct', {}).get('amount', 0) or 0) + (h.get('symphony', {}).get('amount', 0) or 0)
                consolidated_holdings[sym]['cost_basis'] += h.get('cost_basis', 0) or 0
                consolidated_holdings[sym]['accounts'].append(atype)

        # Parse history
        history = None
        if port_hist and 'epoch_ms' in port_hist:
            history = {
                'dates': [datetime.fromtimestamp(e/1000).strftime('%Y-%m-%d') for e in port_hist['epoch_ms']],
                'values': port_hist.get('series', []),
            }

        accounts.append({
            'id': aid,
            'type': atype,
            'status': acct.get('status', ''),
            'value': acct_value,
            'today_dollar': acct_today,
            'today_pct': total_stats.get('todays_percent_change', 0) if total_stats else 0,
            'twr': total_stats.get('time_weighted_return') if total_stats else None,
            'simple_return': total_stats.get('simple_return') if total_stats else None,
            'net_deposits': total_stats.get('net_deposits', 0) if total_stats else 0,
            'cash': total_stats.get('total_cash', 0) if total_stats else 0,
            'unallocated_cash': total_stats.get('total_unallocated_cash', 0) if total_stats else 0,
            'pending_deploys': total_stats.get('pending_deploys_cash', 0) if total_stats else 0,
            'symphonies': symphonies,
            'history': history,
        })

    # Sort consolidated holdings by value
    holdings_list = sorted(consolidated_holdings.values(), key=lambda x: x['total_value'], reverse=True)
    for h in holdings_list:
        h['pct_of_total'] = round(h['total_value'] / total_value * 100, 2) if total_value > 0 else 0

    # Holy Grail stream mapping
    stream_map = {
        'Equity (1x)': ['SPY', 'QQQ', 'SMH', 'IWM', 'VOO', 'RSP', 'VOOV', 'VOOG', 'VTV', 'QQQE'],
        'Lev Equity': ['UPRO', 'TQQQ', 'SOXL', 'TECL', 'QLD', 'SSO', 'FNGO', 'HIBL', 'SPHB'],
        'MF/Alts': ['BTAL', 'DBMF', 'KMLM', 'CTA', 'RSST'],
        'Gold/Commod': ['GLD', 'GDX', 'GDXJ', 'JNUG', 'NUGT', 'SLV', 'DBC', 'CPER', 'UCO', 'USO', 'BOIL'],
        'Vol/Hedge': ['UVXY', 'SVXY', 'VIXM', 'TAIL', 'SH', 'SQQQ', 'SOXS'],
        'Bonds': ['TLT', 'SHY', 'HYG', 'LQD', 'BOND', 'AGG', 'FBND', 'TMF', 'TMV', 'BIL'],
        'Currency': ['USDU', 'UUP', 'FXE', 'FXY'],
        'Sector': ['XLP', 'XLU', 'XLV', 'XLE', 'XLF', 'XLY', 'NAIL', 'CURE', 'FAS', 'LABU', 'DFEN', 'DRN', 'TPOR', 'DUSL'],
        'Cash': ['$USD'],
    }
    stream_totals = {}
    for h in holdings_list:
        placed = False
        for stream, tickers in stream_map.items():
            if h['symbol'] in tickers:
                stream_totals[stream] = stream_totals.get(stream, 0) + h['total_value']
                placed = True
                break
        if not placed:
            stream_totals['Other'] = stream_totals.get('Other', 0) + h['total_value']

    streams = []
    for s, v in sorted(stream_totals.items(), key=lambda x: -x[1]):
        streams.append({'name': s, 'value': round(v, 2), 'pct': round(v / total_value * 100, 1) if total_value > 0 else 0})

    result = {
        'accounts': accounts,
        'consolidated': {
            'total_value': round(total_value, 2),
            'today_dollar': round(total_today_change, 2),
            'today_pct': round(total_today_change / (total_value - total_today_change) * 100, 2) if total_value > total_today_change else 0,
            'holdings': holdings_list[:50],  # top 50
            'streams': streams,
            'goal_8m_pct': round(total_value / 8_000_000 * 100, 1),
        },
        'ts': datetime.now().isoformat(),
    }

    # Persist daily snapshot for historical tracking
    _persist_composer_snapshot(result)

    print(f"  Composer: {len(accounts)} accounts, ${total_value:,.0f} total, {sum(len(a['symphonies']) for a in accounts)} symphonies")
    return result


def _persist_composer_snapshot(data):
    """Append daily portfolio value snapshot to local JSON for time series tracking."""
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        history = {}
        if os.path.exists(COMPOSER_HISTORY_PATH):
            with open(COMPOSER_HISTORY_PATH) as f:
                history = json.load(f)

        # One entry per day
        history[today] = {
            'total_value': data['consolidated']['total_value'],
            'today_pct': data['consolidated']['today_pct'],
            'accounts': [{
                'type': a['type'],
                'value': a['value'],
                'today_pct': a['today_pct'],
            } for a in data['accounts']],
            'streams': data['consolidated']['streams'],
        }

        with open(COMPOSER_HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"  Composer history persist error: {e}")


# ═══════════════════════════════════════════════════════════════════
# FIDELITY CSV INGEST
# ═══════════════════════════════════════════════════════════════════
def parse_fidelity_csv():
    """Parse Fidelity position CSV exports from the drop folder.
    Reads the most recent CSV file. Fidelity format:
    Account Number,Account Name,Symbol,Description,Quantity,Last Price,
    Last Price Change,Current Value,Today's Gain/Loss Dollar,...
    Values have $, +/- prefixes. Footer has disclaimer text to skip.
    """
    if not os.path.isdir(FIDELITY_CSV_DIR):
        return None

    # Find most recent CSV
    csvs = sorted(
        [f for f in os.listdir(FIDELITY_CSV_DIR) if f.lower().endswith('.csv')],
        key=lambda f: os.path.getmtime(os.path.join(FIDELITY_CSV_DIR, f)),
        reverse=True
    )
    if not csvs:
        return None

    fpath = os.path.join(FIDELITY_CSV_DIR, csvs[0])
    file_date = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime('%Y-%m-%d %H:%M')
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Reading Fidelity CSV: {csvs[0]} (modified {file_date})")

    try:
        # Read raw lines, skip Fidelity disclaimer footer
        with open(fpath, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        # Find where data ends (blank line or disclaimer text)
        data_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('"The data and info') or stripped.startswith('"Brokerage services'):
                break
            data_lines.append(line)

        if len(data_lines) < 2:
            return None

        import io
        df = pd.read_csv(io.StringIO(''.join(data_lines)))

        def clean_money(val):
            if pd.isna(val) or val == '--' or val == 'n/a':
                return 0.0
            s = str(val).replace('$', '').replace(',', '').replace('+', '').replace('%', '').strip()
            try:
                return float(s)
            except:
                return 0.0

        accounts = {}
        for _, row in df.iterrows():
            acct_num = str(row.get('Account Number', '')).strip()
            acct_name = str(row.get('Account Name', '')).strip()
            symbol = str(row.get('Symbol', '')).strip()
            if not symbol or symbol == 'nan':
                continue

            acct_key = f"{acct_num} ({acct_name})" if acct_name else acct_num
            if acct_key not in accounts:
                accounts[acct_key] = {
                    'id': acct_num,
                    'type': acct_name,
                    'source': 'Fidelity',
                    'holdings': [],
                    'total_value': 0,
                    'today_dollar': 0,
                    'cost_basis_total': 0,
                }

            cur_val = clean_money(row.get('Current Value', 0))
            today_gl = clean_money(row.get("Today's Gain/Loss Dollar", 0))
            total_gl = clean_money(row.get('Total Gain/Loss Dollar', 0))
            total_gl_pct = clean_money(row.get('Total Gain/Loss Percent', 0))
            cost_basis = clean_money(row.get('Cost Basis', 0))
            quantity = clean_money(row.get('Quantity', 0))
            price = clean_money(row.get('Last Price', 0))
            today_pct = clean_money(row.get("Today's Gain/Loss Percent", 0))

            accounts[acct_key]['holdings'].append({
                'symbol': symbol,
                'name': str(row.get('Description', symbol)),
                'quantity': quantity,
                'price': price,
                'value': cur_val,
                'today_dollar': today_gl,
                'today_pct': today_pct,
                'total_gl_dollar': total_gl,
                'total_gl_pct': total_gl_pct,
                'cost_basis': cost_basis,
                'pct_of_account': clean_money(row.get('Percent Of Account', 0)),
            })

            accounts[acct_key]['total_value'] += cur_val
            accounts[acct_key]['today_dollar'] += today_gl
            accounts[acct_key]['cost_basis_total'] += cost_basis

        # Build result
        acct_list = []
        total_value = 0
        total_today = 0
        all_holdings = {}

        for key, acct in accounts.items():
            total_value += acct['total_value']
            total_today += acct['today_dollar']
            today_pct = acct['today_dollar'] / (acct['total_value'] - acct['today_dollar']) * 100 if acct['total_value'] > acct['today_dollar'] else 0
            acct_list.append({
                'id': acct['id'],
                'type': f"Fidelity: {acct['type']}",
                'source': 'Fidelity',
                'value': acct['total_value'],
                'today_dollar': acct['today_dollar'],
                'today_pct': round(today_pct, 2),
                'cost_basis': acct['cost_basis_total'],
                'total_gl': round(acct['total_value'] - acct['cost_basis_total'], 2),
                'total_gl_pct': round((acct['total_value'] / acct['cost_basis_total'] - 1) * 100, 2) if acct['cost_basis_total'] > 0 else 0,
                'holdings': acct['holdings'],
            })

            # Merge into consolidated
            for h in acct['holdings']:
                sym = h['symbol']
                if sym not in all_holdings:
                    all_holdings[sym] = {
                        'symbol': sym, 'name': h['name'], 'total_value': 0,
                        'total_amount': 0, 'price': h['price'],
                        'today_pct': h['today_pct'], 'today_dollar': 0,
                        'total_change_pct': h['total_gl_pct'], 'cost_basis': 0,
                        'accounts': [],
                    }
                all_holdings[sym]['total_value'] += h['value']
                all_holdings[sym]['total_amount'] += h['quantity']
                all_holdings[sym]['cost_basis'] += h['cost_basis']
                all_holdings[sym]['today_dollar'] += h['today_dollar']
                all_holdings[sym]['accounts'].append(acct['type'])

        holdings_list = sorted(all_holdings.values(), key=lambda x: -x['total_value'])
        for h in holdings_list:
            h['pct_of_total'] = round(h['total_value'] / total_value * 100, 2) if total_value > 0 else 0

        staleness_hours = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(fpath))).total_seconds() / 3600

        result = {
            'accounts': acct_list,
            'consolidated': {
                'total_value': round(total_value, 2),
                'today_dollar': round(total_today, 2),
                'today_pct': round(total_today / (total_value - total_today) * 100, 2) if total_value > total_today else 0,
                'holdings': holdings_list[:50],
            },
            'file': csvs[0],
            'file_date': file_date,
            'staleness_hours': round(staleness_hours, 1),
            'stale': staleness_hours > 24,
            'ts': datetime.now().isoformat(),
        }
        print(f"  Fidelity: {len(acct_list)} accounts, ${total_value:,.0f} total, {sum(len(a['holdings']) for a in acct_list)} positions, {staleness_hours:.1f}h old")
        return result

    except Exception as e:
        print(f"  Fidelity CSV parse error: {e}")
        import traceback
        traceback.print_exc()
        return None


def _merge_portfolio_sources(composer_data, fidelity_data):
    """Merge Composer API and Fidelity CSV into a unified portfolio view."""
    if not composer_data and not fidelity_data:
        return None

    all_accounts = []
    all_holdings = {}
    total_value = 0
    total_today = 0

    # Stream mapping for Holy Grail
    stream_map = {
        'Equity (1x)': ['SPY','QQQ','SMH','IWM','VOO','RSP','VOOV','VOOG','VTV','QQQE'],
        'Lev Equity': ['UPRO','TQQQ','SOXL','TECL','QLD','SSO','FNGO','HIBL','SPHB'],
        'MF/Alts': ['BTAL','DBMF','KMLM','CTA','RSST'],
        'Gold/Commod': ['GLD','GDX','GDXJ','JNUG','NUGT','SLV','DBC','CPER','UCO','USO','BOIL'],
        'Vol/Hedge': ['UVXY','SVXY','VIXM','TAIL','SH','SQQQ','SOXS'],
        'Bonds': ['TLT','SHY','HYG','LQD','BOND','AGG','FBND','TMF','TMV','BIL'],
        'Currency': ['USDU','UUP','FXE','FXY'],
        'Sector': ['XLP','XLU','XLV','XLE','XLF','XLY','NAIL','CURE','FAS','LABU','DFEN','DRN','TPOR','DUSL'],
        'Cash': ['$USD','SPAXX','FDRXX','FCASH','Pending Activity'],
    }

    def add_holdings(holdings_list, source_label):
        for h in holdings_list:
            sym = h.get('symbol', '')
            if not sym:
                continue
            if sym not in all_holdings:
                all_holdings[sym] = {
                    'symbol': sym, 'name': h.get('name', sym), 'total_value': 0,
                    'total_amount': 0, 'price': h.get('price', 0),
                    'today_pct': h.get('today_pct', 0), 'today_dollar': 0,
                    'total_change_pct': h.get('total_gl_pct', h.get('total_change_pct', 0)),
                    'cost_basis': 0, 'sources': [],
                }
            all_holdings[sym]['total_value'] += h.get('value', h.get('total_value', 0))
            all_holdings[sym]['total_amount'] += h.get('quantity', h.get('total_amount', 0))
            all_holdings[sym]['cost_basis'] += h.get('cost_basis', 0)
            all_holdings[sym]['today_dollar'] += h.get('today_dollar', 0)
            if source_label not in all_holdings[sym]['sources']:
                all_holdings[sym]['sources'].append(source_label)

    # Merge Composer accounts
    if composer_data:
        for a in composer_data.get('accounts', []):
            a['source'] = 'Composer'
            all_accounts.append(a)
            total_value += a.get('value', 0)
            total_today += a.get('today_dollar', 0)
            # Flatten symphony holdings into consolidated
            for s in a.get('symphonies', []):
                for h in s.get('holdings', []):
                    h['value'] = h.get('value', 0)
                    h['quantity'] = h.get('amount', 0)
                    h['today_dollar'] = 0  # not available per-holding from Composer
                    h['cost_basis'] = 0
                add_holdings(s.get('holdings', []), f"Composer:{a.get('type','')}")

    # Merge Fidelity accounts
    fidelity_stale = None
    if fidelity_data:
        fidelity_stale = {
            'file': fidelity_data.get('file', ''),
            'file_date': fidelity_data.get('file_date', ''),
            'staleness_hours': fidelity_data.get('staleness_hours', 0),
            'stale': fidelity_data.get('stale', False),
        }
        for a in fidelity_data.get('accounts', []):
            all_accounts.append(a)
            total_value += a.get('value', 0)
            total_today += a.get('today_dollar', 0)
            add_holdings(a.get('holdings', []), f"Fidelity:{a.get('type','')}")

    # Build consolidated holdings
    holdings_list = sorted(all_holdings.values(), key=lambda x: -x['total_value'])
    for h in holdings_list:
        h['pct_of_total'] = round(h['total_value'] / total_value * 100, 2) if total_value > 0 else 0

    # Stream allocation
    stream_totals = {}
    for h in holdings_list:
        placed = False
        for stream, tickers in stream_map.items():
            if h['symbol'] in tickers:
                stream_totals[stream] = stream_totals.get(stream, 0) + h['total_value']
                placed = True
                break
        if not placed:
            stream_totals['Other'] = stream_totals.get('Other', 0) + h['total_value']

    streams = [{'name': s, 'value': round(v, 2), 'pct': round(v / total_value * 100, 1) if total_value > 0 else 0}
               for s, v in sorted(stream_totals.items(), key=lambda x: -x[1])]

    return {
        'accounts': all_accounts,
        'consolidated': {
            'total_value': round(total_value, 2),
            'today_dollar': round(total_today, 2),
            'today_pct': round(total_today / (total_value - total_today) * 100, 2) if total_value > total_today else 0,
            'holdings': holdings_list[:50],
            'streams': streams,
            'goal_8m_pct': round(total_value / 8_000_000 * 100, 1),
        },
        'fidelity_csv': fidelity_stale,
        'sources': ['Composer'] * bool(composer_data) + ['Fidelity CSV'] * bool(fidelity_data),
        'ts': datetime.now().isoformat(),
    }


# HORMUZ TRANSIT MONITOR (IMF PortWatch)
# ═══════════════════════════════════════════════════════════════════
PORTWATCH_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/"
    "Daily_Chokepoints_Data/FeatureServer/0/query"
)

def compute_hormuz_dashboard():
    """Fetch Hormuz data from HormuzTracker.com API with fallback computation."""
    # Try the API first
    try:
        resp = req_lib.get("https://www.hormuztracker.com/api/data", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        crisis = data.get('crisis', {})
        ships = crisis.get('shipCount', {})
        meta = data.get('meta', {})
        trapped = crisis.get('shipsTrapped', {})
        tl = crisis.get('timeline', [])
        api_result = {
            "day": meta.get('day', 0),
            "updated": meta.get('updated', '')[:16],
            "status": crisis.get('hormuzStatus', 'unknown'),
            "severity": crisis.get('severityScore', 0),
            "current": ships.get('current', 0),
            "baseline": ships.get('baseline', 138),
            "drop_pct": ships.get('dropPercent', 0),
            "verified": ships.get('lastVerified', '')[:10],
            "note": ships.get('note', ''),
            "trapped_gulf": trapped.get('insideGulf', 0),
            "trapped_outside": trapped.get('waitingOutside', 0),
            "seafarers": trapped.get('seafarersStranded', 0),
            "container_ships": trapped.get('containerShipsTrapped', 0),
            "timeline": [{"date": e.get("date",""), "day": e.get("day",0),
                          "event": e.get("event",""), "impact": e.get("impact",""),
                          "type": e.get("type","")} for e in tl[-15:]],
            "source": "HormuzTracker.com (daily, CC BY 4.0)",
        }
        # Check staleness: if day count is more than 2 behind computed day, use fallback
        crisis_start = datetime(2026, 2, 28)
        computed_day = (datetime.now() - crisis_start).days + 1
        if api_result['day'] > 0 and computed_day - api_result['day'] <= 2:
            return api_result
        # API is stale — fall through to fallback
        print(f"  Hormuz API stale: API day={api_result['day']}, computed day={computed_day}. Using fallback.")
    except Exception as e:
        print(f"  Hormuz API error: {e}. Using fallback.")

    # Fallback: compute from known data + scrape main page for energy prices
    crisis_start = datetime(2026, 2, 28)
    computed_day = (datetime.now() - crisis_start).days + 1

    # Try scraping main page for current ship count
    current_ships = 7
    try:
        resp2 = req_lib.get("https://www.hormuztracker.com", timeout=15,
                            headers={"User-Agent": "Mozilla/5.0"})
        text = resp2.text
        # Look for "~N/day" pattern or vessel count
        import re
        m = re.search(r'Vessels detected today[^0-9]*(\d+)', text)
        if m:
            current_ships = int(m.group(1))
        else:
            m2 = re.search(r'Ships:\s*~?(\d+)/day', text)
            if m2:
                current_ships = int(m2.group(1))
    except:
        pass

    # Ceasefire timeline events (manually maintained for accuracy)
    ceasefire_timeline = [
        {"date":"2026-04-03","day":35,"event":"Day 35. WTI surpasses Brent at $112. TTF surges 6%. US gasoline crosses $4.09.","impact":"WTI-Brent inversion reflects acute US refinery demand amid Gulf supply loss.","type":"escalation"},
        {"date":"2026-04-04","day":36,"event":"Bloomberg: Weekly transits reach highest since war began (still <15% of normal). Permission-based corridor transit.","impact":"7-day rolling average ticking up but commercially unviable for most carriers.","type":"response"},
        {"date":"2026-04-05","day":37,"event":"Pakistan delivers 5-point ceasefire proposal. Iran counters with 10-point plan.","impact":"Oil sheds war premium on de-escalation hopes. Brent drops to ~$105.","type":"response"},
        {"date":"2026-04-06","day":38,"event":"Trump threatens to destroy Iranian power plants and bridges if strait not reopened by April 8.","impact":"Brent rebounds. Markets price in binary outcome.","type":"escalation"},
        {"date":"2026-04-07","day":39,"event":"Trump: 'A whole civilization will die tonight' if no deal. Ceasefire framework announced late evening.","impact":"Oil futures collapse in after-hours. WTI drops 16.4% next day.","type":"response"},
        {"date":"2026-04-08","day":40,"event":"2-week ceasefire takes effect. Iran agrees to reopen Hormuz. Israel strikes Lebanon — not included in deal.","impact":"WTI settles $94.41 (-16.4%). Brent $94.75 (-13.3%). Biggest single-day oil drop since Apr 2020.","type":"response"},
        {"date":"2026-04-09","day":41,"event":"Strait remains at virtual standstill. Only 5-9 bulk carriers transited in 24hrs (no tankers). Iran releases mine map. ADNOC CEO: 'Hormuz is NOT open.'","impact":"Iran suspends tanker traffic over Israeli Lebanon strikes. Accuses US of violating ceasefire. 230 loaded tankers waiting inside Gulf.","type":"escalation"},
        {"date":"2026-04-10","day":42,"event":"Day 42. Iran charging $1M+ per transit. Permission-based regime. Ceasefire negotiations set for Islamabad (Vance, Witkoff, Kushner).","impact":"Strait effectively closed. Insurance still withdrawn. Carriers awaiting safety guarantees. EU planning escort mission.","type":"escalation"},
        {"date":"2026-04-11","day":43,"event":"Islamabad talks end without agreement. Iran demands US lift all sanctions before full reopening.","impact":"Ceasefire holding but no progress on strait. Oil rebounds on failure.","type":"escalation"},
        {"date":"2026-04-12","day":44,"event":"Trump announces immediate US Navy blockade of Hormuz via Truth Social. 19 vessels transited Sunday — most since war began — then momentum reversed.","impact":"Markets brace for dual blockade (Iran + US). Oil futures surge in after-hours.","type":"escalation"},
        {"date":"2026-04-13","day":45,"event":"US blockade takes effect 10am ET. Only 4 vessels observed: 1 LPG inbound, 3 small tankers racing out. Shipping collapses again.","impact":"Effective closure by both parties. Bloomberg: transit slumped back to single digits.","type":"escalation"},
    ]

    status = 'restricted'  # Ceasefire but not open
    if current_ships < 15:
        status = 'closed'

    return {
        "day": computed_day,
        "updated": datetime.now().strftime('%Y-%m-%d %H:%M'),
        "status": status,
        "severity": 9,
        "current": current_ships,
        "baseline": 138,
        "drop_pct": round((1 - current_ships/138)*100),
        "verified": datetime.now().strftime('%Y-%m-%d'),
        "note": f"Day {computed_day}. US Navy blockade active (Apr 13). Iran + US dual closure. 4 vessels transited Apr 13. Insurance withdrawn. Ceasefire holding but strait NOT open.",
        "trapped_gulf": 1900,
        "trapped_outside": 300,
        "seafarers": 20000,
        "container_ships": 0,
        "timeline": ceasefire_timeline,
        "source": "Fallback (HormuzTracker.com API stale). Kpler/MarineTraffic/NBC/CNBC.",
        "ceasefire_active": True,
        "ceasefire_start": "2026-04-08",
        "ceasefire_days_remaining": max(0, 14 - (datetime.now() - datetime(2026, 4, 8)).days),
    }

# ═══════════════════════════════════════════════════════════════════
# VIX TERM STRUCTURE
# ═══════════════════════════════════════════════════════════════════
def compute_vix_term_structure(raw):
    """Compute VIX term structure regime from VIX tenor data."""
    tenors = [
        ('^VIX9D', 'VIX9D', 9),
        ('^VIX', 'VIX', 30),
        ('^VIX3M', 'VIX3M', 90),
        ('^VIX6M', 'VIX6M', 180),
        ('^VIX1Y', 'VIX1Y', 365),
    ]
    curve = []
    for ticker, label, days in tenors:
        if ticker in raw and len(raw[ticker]) > 0:
            c = raw[ticker]['Close']
            if isinstance(c, pd.DataFrame): c = c.iloc[:,0]
            val = sf(c.iloc[-1])
            if val > 0:
                curve.append({'label': label, 'days': days, 'value': round(val, 2)})

    if len(curve) < 3:
        return None

    # Classify regime
    vals = {d['label']: d['value'] for d in curve}
    front = vals.get('VIX9D') or vals.get('VIX', 0)
    mid = vals.get('VIX3M', front)
    back = vals.get('VIX1Y') or vals.get('VIX6M', mid)

    if back > 0:
        pct_spread = ((front - back) / back) * 100
    else:
        pct_spread = 0

    if pct_spread < -15: regime = 'STEEP_CONTANGO'
    elif pct_spread < -5: regime = 'MILD_CONTANGO'
    elif pct_spread <= 5: regime = 'FLAT'
    elif pct_spread <= 20: regime = 'MILD_BACKWARDATION'
    else: regime = 'STEEP_BACKWARDATION'

    # Compute spreads
    vix = vals.get('VIX', 0)
    vix3m = vals.get('VIX3M', 0)
    vix9d = vals.get('VIX9D', 0)
    vix6m = vals.get('VIX6M', 0)
    vix1y = vals.get('VIX1Y', 0)
    spreads = {}
    if vix and vix3m: spreads['VIX-VIX3M'] = round(vix - vix3m, 2)
    if vix9d and vix: spreads['VIX9D-VIX'] = round(vix9d - vix, 2)
    if vix and vix6m: spreads['VIX-VIX6M'] = round(vix - vix6m, 2)
    if vix9d and vix1y: spreads['Front/Back'] = round(vix9d / vix1y * 100, 1)

    return {
        'curve': curve,
        'regime': regime,
        'pct_spread': round(pct_spread, 1),
        'spreads': spreads,
        'vix': vix,
    }

def fetch_all():
    raw = {}
    for t in TICKERS:
        try:
            df = yf.download(t, period=HISTORY_PERIOD, progress=False)
            if len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                raw[t] = df
        except:
            pass

    indicators = {}
    for t, df in raw.items():
        if len(df) < 50:
            continue
        try:
            c = df['Close']
            price = sf(c.iloc[-1])
            r10 = sf(rsi_wilder(c, 10).iloc[-1])
            s200 = sf(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else 0
            s50 = sf(c.rolling(50).mean().iloc[-1]) if len(c) >= 50 else 0
            e9 = sf(c.ewm(span=9, adjust=False).mean().iloc[-1])
            e20 = sf(c.ewm(span=20, adjust=False).mean().iloc[-1])
            vs200 = (price / s200 - 1) * 100 if s200 > 0 else None
            chg1d = sf(c.pct_change().iloc[-1]) * 100 if len(c) > 1 else 0
            chg5d = sf((c.iloc[-1]/c.iloc[-5]-1)) * 100 if len(c) > 5 else 0
            # Position vs key moving averages (True = above)
            ab_ema9 = price > e9 if e9 > 0 else None
            ab_ema20 = price > e20 if e20 > 0 else None
            ab_sma50 = price > s50 if s50 > 0 else None
            ab_sma200 = price > s200 if s200 > 0 else None
            vs50 = round((price / s50 - 1) * 100, 1) if s50 > 0 else None
            # DRIF velocity indicators
            cum_ret_5d = round(sf(c.pct_change(5).iloc[-1]) * 100, 2) if len(c) > 5 else 0.0
            cum_ret_7d = round(sf(c.pct_change(7).iloc[-1]) * 100, 2) if len(c) > 7 else 0.0
            rsi_full = rsi_wilder(c, 10)
            rsi_5d_ago = sf(rsi_full.iloc[-6]) if len(rsi_full) > 6 else 50.0
            rsi_velocity = round(r10 - rsi_5d_ago, 1)
            pct_above_sma200 = round((price / s200 - 1) * 100, 1) if s200 > 0 else 0
            indicators[t] = {
                'price': round(price, 2), 'rsi': round(r10, 1),
                'sma200': round(s200, 2), 'sma50': round(s50, 2),
                'ema9': round(e9, 2), 'ema20': round(e20, 2),
                'vsSma200': round(vs200, 1) if vs200 is not None else None,
                'vsSma50': vs50,
                'chg1d': round(chg1d, 2), 'chg5d': round(chg5d, 2),
                'abEma9': ab_ema9, 'abEma20': ab_ema20,
                'abSma50': ab_sma50, 'abSma200': ab_sma200,
                'cumRet5d': cum_ret_5d, 'cumRet7d': cum_ret_7d,
                'rsiVelocity': rsi_velocity,
                'pctAboveSma200': pct_above_sma200,
            }
        except:
            pass

    # Signals
    signals = []
    ind = indicators
    # SPY dip-buys
    spy_r = ind.get('SPY', {}).get('rsi', 50)
    if spy_r < 21: signals.append({'type':'buy','title':'SPY RSI<21 → UPRO','msg':f'SPY RSI={spy_r:.1f} | 87% WR, +8.9% avg 5d'})
    elif spy_r < 25: signals.append({'type':'buy','title':'SPY RSI<25 → UPRO','msg':f'SPY RSI={spy_r:.1f} | 74% WR'})
    elif spy_r < 30: signals.append({'type':'buy','title':'SPY RSI<30 → UPRO','msg':f'SPY RSI={spy_r:.1f} | 69% WR'})
    if spy_r > 85: signals.append({'type':'exit','title':'SPY RSI>85 → Exit UPRO','msg':f'SPY RSI={spy_r:.1f} | Only 36% WR'})
    # QQQ
    qqq_r = ind.get('QQQ', {}).get('rsi', 50)
    if qqq_r < 20: signals.append({'type':'buy','title':'QQQ RSI<20 → TQQQ','msg':f'QQQ RSI={qqq_r:.1f} | 100% WR'})
    if qqq_r > 79: signals.append({'type':'hedge','title':'QQQ RSI>79 → UVXY','msg':f'QQQ RSI={qqq_r:.1f} | 67% WR'})
    # GLD/USDU
    gld_r = ind.get('GLD', {}).get('rsi', 50)
    usdu_r = ind.get('USDU', {}).get('rsi', 50)
    if gld_r > 79 and usdu_r < 25:
        signals.append({'type':'buy','title':'DOUBLE SIGNAL','msg':f'GLD={gld_r:.1f} + USDU={usdu_r:.1f} | TQQQ 88% WR'})
        xlp_r = ind.get('XLP', {}).get('rsi', 0)
        if xlp_r > 65: signals.append({'type':'buy','title':'TRIPLE SIGNAL','msg':f'+ XLP={xlp_r:.1f} | TQQQ 100% WR (n=5)'})
    elif gld_r > 79:
        signals.append({'type':'buy','title':'GLD Overbought','msg':f'GLD RSI={gld_r:.1f} | TQQQ 72% WR'})
    # UCO → TMV
    uco_r = ind.get('UCO', {}).get('rsi', 50)
    if uco_r > 75: signals.append({'type':'buy','title':'UCO>75 → TMV','msg':f'UCO RSI={uco_r:.1f} | Oil→Bond weakness'})
    # SOXS
    smh_r = ind.get('SMH', {}).get('rsi', 50)
    if smh_r > 79 and usdu_r > 70:
        signals.append({'type':'short','title':'SOXS Dollar Squeeze','msg':f'SMH={smh_r:.1f} + USDU={usdu_r:.1f} | 100% WR'})
    # CURE/FAS/LABU
    cure_r = ind.get('CURE', {}).get('rsi', 50)
    if cure_r < 21: signals.append({'type':'buy','title':'CURE RSI<21','msg':f'CURE RSI={cure_r:.1f} | 85% WR, n=33'})
    elif cure_r < 25: signals.append({'type':'buy','title':'CURE RSI<25','msg':f'CURE RSI={cure_r:.1f} | 81% WR'})
    fas_r = ind.get('FAS', {}).get('rsi', 50)
    if fas_r < 30: signals.append({'type':'buy','title':'FAS RSI<30','msg':f'FAS RSI={fas_r:.1f} | 63% WR'})
    # Defensive rotation
    def_ob = any(ind.get(t, {}).get('rsi', 0) > 79 for t in ['XLP','XLU','XLV'])
    if def_ob and spy_r < 79 and qqq_r < 79:
        signals.append({'type':'buy','title':'Defensive Rotation','msg':'Def sector OB, SPY/QQQ not | TQQQ 70% WR 20d'})
    # UVXY B1
    uvxy_r = ind.get('UVXY', {}).get('rsi', 50)
    if uvxy_r > 82: signals.append({'type':'buy','title':'UVXY>82 → SOXL (B1)','msg':f'UVXY RSI={uvxy_r:.1f} | 81% WR 1d'})

    # GLD & Miners (Group 18)
    gdxj_r = ind.get('GDXJ', {}).get('rsi', 50)
    gdx_r = ind.get('GDX', {}).get('rsi', 50)
    if gdxj_r < 21:
        signals.append({'type':'buy','title':'GDXJ RSI<21 → JNUG','msg':f'GDXJ RSI={gdxj_r:.1f} | 59% WR +8.43% avg 1d n=17'})
    elif gdxj_r < 25:
        signals.append({'type':'buy','title':'GDXJ RSI<25 → JNUG','msg':f'GDXJ RSI={gdxj_r:.1f} | 63% WR +3.55% avg 1d n=59'})
    if gdx_r < 21 and gdxj_r >= 25:
        signals.append({'type':'buy','title':'GDX RSI<21 → NUGT','msg':f'GDX RSI={gdx_r:.1f} | 56% WR +1.13% avg 1d n=25'})
    if gdx_r > 85:
        signals.append({'type':'warning','title':'GDX EXTENDED — DO NOT SHORT','msg':f'GDX RSI={gdx_r:.1f} | Miners continue when OB. DUST loses.'})
    if gld_r > 75 and usdu_r > 60:
        signals.append({'type':'hedge','title':'MINER SHORT WINDOW','msg':f'GLD={gld_r:.1f}>75 + USDU={usdu_r:.1f}>60 | JDST 5d: 59% WR n=34'})

    # NAIL RSI<21 signal
    nail_r = ind.get('NAIL', {}).get('rsi', 50)
    if nail_r < 21:
        signals.append({'type':'buy','title':'NAIL RSI<21','msg':f'NAIL RSI={nail_r:.1f} | Oversold'})

    # === v4.3 NEW SIGNALS (Audit Mar 24-25) ===

    # Group 19: HYG Credit Euphoria
    hyg_r = ind.get('HYG', {}).get('rsi', 50)
    # HYG RSI(7) approximated from RSI(10) — exact RSI(7) computed in signal monitor
    if hyg_r > 80:
        hyg_sma200 = ind.get('HYG', {}).get('sma200', 0)
        hyg_price = ind.get('HYG', {}).get('price', 0)
        if hyg_price > hyg_sma200 > 0:
            signals.append({'type':'buy','title':'HYG CREDIT EUPHORIA [T1]','msg':f'HYG RSI≈{hyg_r:.1f} + above SMA200 | TQQQ 1d: 80.6% WR n=36'})

    # Group 20: USMV Overbought
    usmv_r = ind.get('USMV', {}).get('rsi', 50)
    if usmv_r > 82:
        signals.append({'type':'hedge','title':'USMV COMPLACENCY [T1]','msg':f'USMV RSI={usmv_r:.1f}>82 | UVXY 1d: 75% WR +3.2% n=24'})

    # Group 21: UCO+USDU → TMV (enhanced)
    if uco_r > 75 and usdu_r > 55:
        signals.append({'type':'short','title':'OIL+DOLLAR → TMV [T1]','msg':f'UCO={uco_r:.1f}>75 + USDU={usdu_r:.1f}>55 | TMV 10d: 75.4% WR n=57'})

    # Group 23: GLD Oversold
    if gld_r < 20:
        signals.append({'type':'buy','title':'GLD DEEP OVERSOLD [T2]','msg':f'GLD RSI={gld_r:.1f}<20 | TQQQ 10d: 70.6% WR +4.83% | PF=5.99'})
    elif gld_r < 22:
        signals.append({'type':'buy','title':'GLD OVERSOLD','msg':f'GLD RSI={gld_r:.1f}<22 | TQQQ 10d: 73.3% WR +4.94%'})

    # Group 27: SPHB/SPLV ratio (compute from raw data)
    if 'SPHB' in raw and 'SPLV' in raw:
        try:
            sphb_c = raw['SPHB']['Close']
            splv_c = raw['SPLV']['Close']
            if isinstance(sphb_c, pd.DataFrame): sphb_c = sphb_c.iloc[:,0]
            if isinstance(splv_c, pd.DataFrame): splv_c = splv_c.iloc[:,0]
            ratio = sphb_c / splv_c
            ratio_rsi_val = sf(rsi_wilder(ratio, 10).iloc[-1])
            ratio_val = sf(ratio.iloc[-1])
            if ratio_rsi_val < 25:
                signals.append({'type':'buy','title':'RISK ROTATION EXHAUSTION [T2]','msg':f'SPHB/SPLV RSI={ratio_rsi_val:.1f}<25 | TQQQ 10d: 75.5% WR (n=53) | MANUAL'})
        except: pass

    # Group 28: Vol Recovery Alpha
    uvxy_price = ind.get('UVXY', {}).get('price', 0)
    uvxy_sma200 = ind.get('UVXY', {}).get('sma200', 0)
    vixm_price = ind.get('VIXM', {}).get('price', 0)
    vixm_sma50 = ind.get('VIXM', {}).get('sma50', 0)
    if uvxy_sma200 > 0 and vixm_sma50 > 0:
        if uvxy_price > uvxy_sma200 and vixm_price < vixm_sma50:
            signals.append({'type':'buy','title':'VOL RECOVERY ALPHA [T2]','msg':f'UVXY>{uvxy_sma200:.0f}SMA200 + VIXM<{vixm_sma50:.0f}SMA50 | SOXL 10d: 90.3% WR'})
        elif uvxy_price > uvxy_sma200:
            signals.append({'type':'watch','title':'VOL STILL ELEVATED','msg':f'UVXY ${uvxy_price:.2f} > SMA200 — waiting for VIXM normalization'})

    # Group 20: FXY Carry Trade (v4.5 — Wishlist Item 7)
    fxy_r = ind.get('FXY', {}).get('rsi', 50)
    if fxy_r > 75:
        signals.append({'type':'warning','title':'FXY CARRY STRESS [20A]','msg':f'FXY RSI={fxy_r:.1f}>75 | Yen strengthening — carry trade under pressure'})
    if fxy_r > 70:
        tlt_broken = not ind.get('TLT', {}).get('abSma200', True)
        if tlt_broken:
            signals.append({'type':'hedge','title':'FXY+TLT CARRY UNWIND [20B]','msg':f'FXY={fxy_r:.1f}>70 + TLT<SMA200 | BTAL 86.7% WR 1d +0.97% n=15'})
        if usdu_r > 60:
            signals.append({'type':'hedge','title':'DUAL SAFE HAVEN [20D]','msg':f'FXY={fxy_r:.1f}>70 + USDU={usdu_r:.1f}>60 | BTAL 87% WR 1d | Global risk-off'})

    # Group 21: CPER Copper Regime (v4.5 — Wishlist Item 9)
    cper_ae9 = ind.get('CPER', {}).get('abEma9', None)
    spy_ae9 = ind.get('SPY', {}).get('abEma9', True)
    copx_ae9 = ind.get('COPX', {}).get('abEma9', True)
    if cper_ae9 is True and spy_ae9 is False and copx_ae9 is True:
        signals.append({'type':'buy','title':'COPPER REGIME → TQQQ [21A]','msg':f'CPER>EMA9 + SPY<EMA9 + COPX>EMA9 | 40.2% CAGR, 0.23 SPY R | Composer active'})
    elif cper_ae9 is True and copx_ae9 is False:
        signals.append({'type':'warning','title':'COPPER SUPPLY DISRUPTION [21C]','msg':'CPER>EMA9 but COPX<EMA9 | Supply-driven false positive risk'})

    # LABU/SOXL 3x Dip Buy (v4.5 — Wishlist Item 1)
    labu_r2 = ind.get('LABU', {}).get('rsi', 50)
    if labu_r2 < 25:
        pick = 'LABU'
        if 'SOXL' in ind:
            labu_cr = ind.get('LABU', {}).get('chg5d', 0)
            soxl_cr = ind.get('SOXL', {}).get('chg5d', 0)
            pick = 'SOXL' if soxl_cr < labu_cr else 'LABU'
        if labu_r2 < 22:
            signals.append({'type':'buy','title':f'LABU/SOXL CORE DIP [T1]','msg':f'LABU RSI={labu_r2:.1f}<22 → {pick} | 100% WR +12.3% n=11 | SPY R=0.18'})
        else:
            signals.append({'type':'buy','title':f'LABU/SOXL DIP BUY [T1]','msg':f'LABU RSI={labu_r2:.1f}<25 → {pick} | 82% WR +7.0% n=28 | SPY R=0.18'})

    # UVXY SMA200 Cross → SOXL 5d (v4.5 — Wishlist Item 8C)
    if 'UVXY' in raw:
        try:
            uvc = raw['UVXY']['Close']
            if isinstance(uvc, pd.DataFrame): uvc = uvc.iloc[:,0]
            if len(uvc) >= 201:
                uvs = uvc.rolling(200).mean()
                today_ab = float(uvc.iloc[-1]) > float(uvs.iloc[-1])
                yest_ab = float(uvc.iloc[-2]) > float(uvs.iloc[-2])
                if today_ab and not yest_ab:
                    signals.append({'type':'buy','title':'UVXY SMA200 CROSS → SOXL [29]','msg':'UVXY crossed ABOVE SMA200 | Buy SOXL hold 5d | 80% WR +8.3% PF=6.75 n=10 | MANUAL'})
        except: pass

    # Oil Supply Shock EITHER/OR (v4.5 — updated Apr 2 2026)
    uso_r2 = ind.get('USO', {}).get('rsi', 50)
    if (uco_r > 79 or uso_r2 > 79) and usdu_r > 55:
        sig_str = f'UCO={uco_r:.1f}' if uco_r > 79 else f'USO={uso_r2:.1f}'
        conv = ' HIGH CONV' if usdu_r > 60 else ''
        signals.append({'type':'short','title':f'OIL SUPPLY SHOCK → TMV [T1]','msg':f'{sig_str} + USDU={usdu_r:.1f}>55{conv} | TMV 80% WR n=25'})

    # Group 24: ILS Cat Bond Monitoring (v4.5 — Wishlist Item 8D)
    if 'ILS' in ind:
        ils_d = ind['ILS']
        ils_chg5 = ils_d.get('chg5d', 0)
        if ils_chg5 < -3:
            signals.append({'type':'warning','title':'ILS CAT BOND DROP [24A]','msg':f'ILS 5d: {ils_chg5:+.1f}% | DO NOT SELL — drawdowns recover 1-3 months. Check for named storm.'})
        now_month = datetime.now().month
        if 6 <= now_month <= 11:
            signals.append({'type':'watch','title':'HURRICANE SEASON ACTIVE [24B]','msg':f'Month {now_month} — binary risk window open for ILS/cat bonds. Monitor NHC.'})

    # === SMH/IGV DIVERGENCE (v4.6 — restored from Feb 4 analysis) ===
    igv_r = ind.get('IGV', {}).get('rsi', 50)
    smh_igv_spread = round(smh_r - igv_r, 1)
    # Spread > 30 + IGV < 35 → TECL
    if smh_igv_spread > 30 and igv_r < 35:
        signals.append({'type':'buy','title':'SMH/IGV ROTATION → TECL [T2]','msg':f'Spread={smh_igv_spread:+.0f} + IGV RSI={igv_r:.1f}<35 | TECL 10d: 78% WR +13.2% n=13'})
    elif smh_igv_spread > 25 and igv_r < 35:
        signals.append({'type':'buy','title':'SMH/IGV ROTATION → TECL','msg':f'Spread={smh_igv_spread:+.0f} + IGV RSI={igv_r:.1f}<35 | TECL 10d: 75% WR +10.5% n=19'})
    elif smh_igv_spread > 25 and igv_r < 40:
        signals.append({'type':'watch','title':'SMH/IGV SPREAD WIDE','msg':f'Spread={smh_igv_spread:+.0f} + IGV RSI={igv_r:.1f}<40 | TECL 10d: 70% WR +6.9% n=30'})
    # Reverse: Spread < -15 + SMH < 30 → SOXL
    if smh_igv_spread < -15 and smh_r < 30:
        signals.append({'type':'buy','title':'IGV/SMH ROTATION → SOXL [T2]','msg':f'Spread={smh_igv_spread:+.0f} + SMH RSI={smh_r:.1f}<30 | SOXL 10d: 88% WR +14.9% n=34'})
    elif smh_igv_spread < -25 and smh_r < 35:
        signals.append({'type':'buy','title':'IGV/SMH ROTATION → SOXL','msg':f'Spread={smh_igv_spread:+.0f} + SMH RSI={smh_r:.1f}<35 | SOXL 10d: 80% WR +12.1% n=21'})
    # IGV < 30 + GLD > 79 → TECL (best IGV buy signal)
    if igv_r < 30 and gld_r > 79:
        signals.append({'type':'buy','title':'IGV OVERSOLD + GLD OB → TECL [T1]','msg':f'IGV RSI={igv_r:.1f}<30 + GLD={gld_r:.1f}>79 | TECL 5d: 92% WR +13.0% n=15'})

    # === v4.4 NEW SIGNALS ===

    # Group 13: UVXY Vol Regime Shift
    uvxy_vol = compute_uvxy_vol_regime(ind)
    if uvxy_vol:
        pct_ab = uvxy_vol.get('pct_above', -999)
        if pct_ab >= 30:
            signals.append({'type':'buy','title':'VOL REGIME EXTREME','msg':f'UVXY {pct_ab:+.1f}% above SMA(200) | SPY 20d: 94% WR +7.3% | 40d/60d: 100% | n=18'})
        elif pct_ab >= 20:
            signals.append({'type':'buy','title':'VOL REGIME HIGH','msg':f'UVXY {pct_ab:+.1f}% above SMA(200) | SPY 20d: 92% WR +6.2% | 60d: 100% | n=24'})
        elif pct_ab >= 0:
            signals.append({'type':'buy','title':'VOL REGIME SHIFT','msg':f'UVXY {pct_ab:+.1f}% above SMA(200) | SPY 20d: 83% WR +4.3% | 60d: 92% | n=52'})
        elif pct_ab >= -10:
            signals.append({'type':'watch','title':'VOL REGIME APPROACHING','msg':f'UVXY {pct_ab:+.1f}% vs SMA(200) | Threshold: ${uvxy_vol["threshold_signal"]:.2f}'})

    # Group 19: MOVE Index signals
    move_data = compute_move_signals(raw, ind)
    if move_data:
        if move_data.get('19B_active'):
            signals.append({'type':'buy','title':'MOVE EXTREME SPIKE [19B]','msg':f'MOVE 20d Δ={move_data["change_20d_pct"]:+.1f}% | SPY 20d: 86% WR +5.29% (n=69)'})
        elif move_data.get('19A_active'):
            signals.append({'type':'buy','title':'MOVE ELEVATED [19A]','msg':f'MOVE={move_data["price"]:.0f}>115 | SPY 20d: 72% WR +2.07% (n=389d/29 ep)'})
        if move_data.get('19C_active'):
            signals.append({'type':'buy','title':'MOVE VOL CRUSH [19C]','msg':f'MOVE RSI dropped from >79 to {move_data["rsi"]:.1f} | SPY 10d: 92% WR +2.28% (n=24)'})
        # GLD combo: MOVE>100 + SPY RSI<25
        if move_data['price'] > 100 and spy_r < 25:
            signals.append({'type':'buy','title':'MOVE+SPY GLD COMBO','msg':f'MOVE={move_data["price"]:.0f}>100 + SPY RSI={spy_r:.1f}<25 | GLD 20d: 100% WR +6.43% (n=15)'})

    # Group 21b: Multi-Oversold Breadth
    usmv_r2 = ind.get('USMV', {}).get('rsi', 50)
    vtv_r = ind.get('VTV', {}).get('rsi', 50)
    voov_r = ind.get('VOOV', {}).get('rsi', 50)
    upro_r = ind.get('UPRO', {}).get('rsi', 50)
    if all(r < 30 for r in [spy_r, usmv_r2, vtv_r, voov_r, upro_r]):
        hyg_confirm = hyg_r < 30 if hyg_r else False
        msg = f'SPY+USMV+VTV+VOOV+UPRO all RSI<30 | UPRO 5d: 77.8% WR (n=45, 23 ep)'
        if hyg_confirm:
            msg += ' | HYG RSI<30 confirmed'
        signals.append({'type':'buy','title':'MULTI-OVERSOLD BREADTH [21b]','msg':msg})

    # Group 30: DRIF Velocity Filter
    drif_data = compute_drif_signals(ind)
    for t_drif in ['SPY', 'QQQ', 'SMH']:
        d = drif_data.get(t_drif, {})
        if d.get('gate') == 'PASS':
            signals.append({'type':'buy','title':f'DRIF: {t_drif} CONFIRMED','msg':f'{t_drif} RSI={d["rsi"]:.1f} + {d["retField"]}={d["retVal"]:+.1f}% > {d["retGate"]}% | {d["passWr"]} WR (n={d["passN"]})'})
        elif d.get('gate') == 'FAIL':
            signals.append({'type':'warning','title':f'DRIF: {t_drif} FALLING KNIFE','msg':f'{t_drif} RSI={d["rsi"]:.1f} BUT {d["retField"]}={d["retVal"]:+.1f}% < {d["retGate"]}% | Only {d["failWr"]} WR (n={d["failN"]})'})

    # FRED Credit Spread (if API key configured)
    fred_credit = compute_fred_credit()

    # Hormuz transit data
    hormuz_data = compute_hormuz_dashboard()

    # VIX term structure
    vix_structure = compute_vix_term_structure(raw)

    # SMH/IGV spread for display
    smh_igv_data = {}
    if 'SMH' in ind and 'IGV' in ind:
        smh_igv_data = {
            'smh_rsi': ind['SMH'].get('rsi', 0),
            'igv_rsi': ind.get('IGV', {}).get('rsi', 0),
            'spread': round(ind['SMH'].get('rsi', 0) - ind.get('IGV', {}).get('rsi', 0), 1),
        }

    # Inline breadth computation (ZBT, McClellan, %Above50SMA)
    breadth_inline = {}
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Downloading breadth tickers...")
        breadth_raw = {}
        for t in BREADTH_TICKERS:
            try:
                bdf = yf.download(t, period=BREADTH_PERIOD, progress=False)
                if len(bdf) > 0:
                    if isinstance(bdf.columns, pd.MultiIndex): bdf.columns = bdf.columns.get_level_values(0)
                    breadth_raw[t] = bdf
            except: pass
        if breadth_raw:
            # Daily advance/decline
            ref = next(iter(breadth_raw))
            dates = breadth_raw[ref].index
            daily = []
            for i in range(1, len(dates)):
                adv, dec = 0, 0
                for tk, bdf in breadth_raw.items():
                    try:
                        c = bdf['Close']
                        if isinstance(c, pd.DataFrame): c = c.iloc[:,0]
                        td = float(c.iloc[i]) if i < len(c) and not pd.isna(c.iloc[i]) else None
                        yd = float(c.iloc[i-1]) if (i-1) < len(c) and not pd.isna(c.iloc[i-1]) else None
                        if td and yd:
                            if td > yd: adv += 1
                            elif td < yd: dec += 1
                    except: pass
                tot = adv + dec
                ratio = adv / tot if tot > 0 else 0.5
                daily.append({'adv': adv, 'dec': dec, 'ratio': ratio, 'net': adv - dec})
            if daily:
                bdf2 = pd.DataFrame(daily)
                bdf2['zbt_ema'] = bdf2['ratio'].ewm(alpha=0.1, adjust=False).mean()
                bdf2['ema19'] = bdf2['net'].ewm(span=19, adjust=False).mean()
                bdf2['ema39'] = bdf2['net'].ewm(span=39, adjust=False).mean()
                bdf2['mcclellan'] = bdf2['ema19'] - bdf2['ema39']
                zv = float(bdf2['zbt_ema'].iloc[-1])
                mcl = float(bdf2['mcclellan'].iloc[-1])
                mcl_prev = float(bdf2['mcclellan'].iloc[-2]) if len(bdf2) >= 2 else mcl
                # %Above50SMA
                a50, t50 = 0, 0
                for tk, bdf in breadth_raw.items():
                    try:
                        c = bdf['Close']
                        if isinstance(c, pd.DataFrame): c = c.iloc[:,0]
                        if len(c) >= 50:
                            s50 = float(c.rolling(50).mean().iloc[-1])
                            if not pd.isna(s50):
                                t50 += 1
                                if float(c.iloc[-1]) > s50: a50 += 1
                    except: pass
                pct50 = (a50 / t50 * 100) if t50 > 0 else 0
                breadth_inline = {
                    'zbt_ema': round(zv, 4), 'zbt_ratio': round(float(bdf2['ratio'].iloc[-1]), 4),
                    'zbt_zone': 'OVERSOLD' if zv < 0.40 else 'THRUST' if zv >= 0.615 else 'NEUTRAL',
                    'zbt_thrust': zv >= 0.615 and bool((bdf2['zbt_ema'].tail(10) < 0.40).any()),
                    'mcclellan': round(mcl, 1),
                    'mcl_ema19': round(float(bdf2['ema19'].iloc[-1]), 1),
                    'mcl_ema39': round(float(bdf2['ema39'].iloc[-1]), 1),
                    'mcl_direction': 'RISING' if mcl > mcl_prev else 'FALLING',
                    'mcl_zone': 'OVERSOLD' if mcl < -100 else 'OVERBOUGHT' if mcl > 100 else 'POSITIVE' if mcl > 0 else 'NEGATIVE',
                    'pct_above_50sma': round(pct50, 1), 'above50_n': a50, 'above50_total': t50,
                    'adv': int(bdf2['adv'].iloc[-1]), 'dec': int(bdf2['dec'].iloc[-1]),
                }
                print(f"  → ZBT: {zv:.4f} | McClellan: {mcl:+.1f} | %Above50: {pct50:.1f}%")
    except Exception as e:
        print(f"  Breadth computation error: {e}")

    # Fibonacci context levels
    fib_levels = {}
    for sym in ['SPY', 'QQQ', 'SMH']:
        if sym not in raw: continue
        try:
            df_f = raw[sym]
            c_f = df_f['Close']
            if isinstance(c_f, pd.DataFrame): c_f = c_f.iloc[:,0]
            close_f = float(c_f.iloc[-1])
            h_f = df_f['High'] if not isinstance(df_f['High'], pd.DataFrame) else df_f['High'].iloc[:,0]
            l_f = df_f['Low'] if not isinstance(df_f['Low'], pd.DataFrame) else df_f['Low'].iloc[:,0]
            recent30 = df_f.tail(30)
            h30 = float(h_f.tail(30).max()); l30 = float(l_f.tail(30).min())
            diff = h30 - l30; is_up = close_f > (h30 + l30) / 2
            levels = {}
            for pct in [0.236, 0.382, 0.500, 0.618]:
                lvl = (h30 - diff * pct) if is_up else (l30 + diff * pct)
                dist = (close_f - lvl) / close_f * 100
                levels[f'{pct*100:.1f}'] = {'level': round(lvl, 2), 'dist': round(dist, 1), 'near': abs(dist) < 1.5}
            fib_levels[sym] = {'high': round(h30, 2), 'low': round(l30, 2), 'close': round(close_f, 2),
                               'trend': 'UP' if is_up else 'DOWN', 'levels': levels}
        except Exception as e:
            print(f"  Fibonacci error for {sym}: {e}")
    if fred_credit and fred_credit.get('alert'):
        chg_str = f" (20d Δ: {fred_credit['change_20d']:+.2f}%)" if fred_credit.get('change_20d') is not None else ""
        signals.append({'type':'warning','title':f'CREDIT SPREAD {fred_credit["level"]}','msg':f'BB OAS: {fred_credit["current_oas"]}% | Trend: {fred_credit["trend"]}{chg_str} | {fred_credit["alert_reason"]}'})
    elif fred_credit and fred_credit.get('level') == 'COMPLACENT':
        signals.append({'type':'watch','title':'CREDIT: COMPLACENT','msg':f'BB OAS: {fred_credit["current_oas"]}% ({fred_credit["percentile_1y"]:.0f}th pctile) | Watch for widening'})

    # Determine regime for rolling beta weights
    dash_regime = 'UNKNOWN'
    spy_d = ind.get('SPY', {})
    qqq_d = ind.get('QQQ', {})
    if spy_d and qqq_d:
        spy_price = spy_d.get('price', 0)
        spy_sma200 = spy_d.get('sma200', 0)
        spy_ema20 = spy_d.get('ema20', 0)
        above200 = spy_price > spy_sma200 if spy_sma200 > 0 else False
        above_ema20 = spy_price > spy_ema20 if spy_ema20 > 0 else False
        # Vol ratio from raw data
        if 'QQQ' in raw and len(raw['QQQ']) > 50:
            qc = raw['QQQ']['Close']
            if isinstance(qc, pd.DataFrame): qc = qc.iloc[:,0]
            qrets = qc.pct_change()
            s10 = float(qrets.rolling(10).std().iloc[-1]) if len(qrets) > 10 else 0
            s50 = float(qrets.rolling(50).std().iloc[-1]) if len(qrets) > 50 else 0
            vol_exp = (s10/s50 > 1.0) if s50 > 0 else False
        else:
            vol_exp = False
        if above200 and not vol_exp: dash_regime = 'BULL + VOL COMPRESS'
        elif above200 and vol_exp: dash_regime = 'BULL + VOL EXPAND'
        elif not above200 and above_ema20: dash_regime = 'BEAR RECOVERY'
        elif not above200: dash_regime = 'BEAR DEFENSIVE'

    # Rolling betas (compute with Brier, every 10 min)
    rolling_betas = cache.get('rolling_betas', [])

    # Brier scores (compute every 10 min, not every 60s)
    brier_results = cache.get('brier')
    now = time.time()
    if brier_results is None or now - cache.get('brier_ts', 0) > 600:
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Computing Brier scores + rolling betas (regime: {dash_regime})...")
            brier_results = compute_brier(raw)
            cache['brier'] = brier_results
            rolling_betas = compute_rolling_betas(raw, regime=dash_regime)
            cache['rolling_betas'] = rolling_betas
            cache['brier_ts'] = now
            print(f"  → {len(brier_results)} signals scored")
            # Persist Brier to JSON
            persist_brier(brier_results)
        except Exception as e:
            print(f"  Brier error: {e}")
            brier_results = cache.get('brier', [])

    # Compute market breadth regime
    breadth_regime = 'UNKNOWN'
    leadership_gap = 0
    rsp_d = ind.get('RSP', {})
    spy_d2 = ind.get('SPY', {})
    if rsp_d and spy_d2:
        rsp_above = rsp_d.get('price', 0) > rsp_d.get('sma200', 0) if rsp_d.get('sma200', 0) > 0 else False
        spy_above2 = spy_d2.get('price', 0) > spy_d2.get('sma200', 0) if spy_d2.get('sma200', 0) > 0 else False
        if rsp_above and spy_above2: breadth_regime = 'BROAD BULL'
        elif not rsp_above and spy_above2: breadth_regime = 'NARROW LEADERSHIP'
        elif rsp_above and not spy_above2: breadth_regime = 'ROTATION'
        else: breadth_regime = 'BROAD WEAKNESS'
        leadership_gap = round((spy_d2.get('vsSma200', 0) or 0) - (rsp_d.get('vsSma200', 0) or 0), 1)

    # Compute SPHB/SPLV ratio
    ratio_data = {}
    if 'SPHB' in raw and 'SPLV' in raw:
        try:
            sc = raw['SPHB']['Close']
            lc = raw['SPLV']['Close']
            if isinstance(sc, pd.DataFrame): sc = sc.iloc[:,0]
            if isinstance(lc, pd.DataFrame): lc = lc.iloc[:,0]
            r = sc / lc
            ratio_data = {'value': round(sf(r.iloc[-1]), 3), 'rsi': round(sf(rsi_wilder(r, 10).iloc[-1]), 1)}
        except: pass

    # Mid-Month Rotation (Group 25) — Robot James signal
    midmonth_data = {}
    if 'SPY' in raw and 'TLT' in raw:
        try:
            spy_cl = raw['SPY']['Close']
            tlt_cl = raw['TLT']['Close']
            if isinstance(spy_cl, pd.DataFrame): spy_cl = spy_cl.iloc[:,0]
            if isinstance(tlt_cl, pd.DataFrame): tlt_cl = tlt_cl.iloc[:,0]
            today = spy_cl.index[-1]
            cm, cy = today.month, today.year
            spy_m = spy_cl[(spy_cl.index.month == cm) & (spy_cl.index.year == cy)]
            tlt_m = tlt_cl[(tlt_cl.index.month == cm) & (tlt_cl.index.year == cy)]
            if len(spy_m) >= 1 and len(tlt_m) >= 1:
                td_num = len(spy_m)
                spy_mtd = round((float(spy_cl.iloc[-1]) / float(spy_m.iloc[0]) - 1) * 100, 2)
                tlt_mtd = round((float(tlt_cl.iloc[-1]) / float(tlt_m.iloc[0]) - 1) * 100, 2)
                pick = 'TLT' if spy_mtd > tlt_mtd else 'SPY'
                days_to = 15 - td_num
                midmonth_data = {
                    'td': td_num, 'spy_mtd': spy_mtd, 'tlt_mtd': tlt_mtd,
                    'pick': pick, 'days_to_signal': days_to,
                    'is_signal_day': td_num == 15,
                    'is_holding': td_num > 15,
                }
                if td_num == 15:
                    signals.append({'type':'buy','title':'MID-MONTH ROTATION [25] — SIGNAL DAY',
                        'msg':f'Trading day 15! SPY MTD={spy_mtd:+.2f}% vs TLT MTD={tlt_mtd:+.2f}% → Buy {pick} tomorrow, hold through month-end | 63.7% WR, Sharpe 1.03, SPY R=-0.03 | n=281 | MANUAL ONLY'})
                elif td_num == 14:
                    signals.append({'type':'watch','title':'MID-MONTH PREVIEW [25]',
                        'msg':f'Signal fires TOMORROW | SPY MTD={spy_mtd:+.2f}% vs TLT MTD={tlt_mtd:+.2f}% | Leaning: {pick}'})
        except Exception as e:
            print(f"  Mid-month rotation error: {e}")

    return {
        'indicators': indicators,
        'signals': signals,
        'brier': brier_results or [],
        'rolling_betas': rolling_betas or [],
        'breadth_regime': breadth_regime,
        'leadership_gap': leadership_gap,
        'sphb_splv': ratio_data,
        'drif': drif_data,
        'uvxy_vol_regime': uvxy_vol,
        'move_index': move_data,
        'fred_credit': fred_credit,
        'hormuz': hormuz_data,
        'vix_structure': vix_structure,
        'smh_igv': smh_igv_data,
        'midmonth': midmonth_data,
        'breadth_inline': breadth_inline,
        'fibonacci': fib_levels,
        'ts': datetime.now().isoformat(),
        'n_tickers': len(indicators),
    }

# ═══════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════
@app.route('/api/data')
def api_data():
    now = time.time()
    with lock:
        if cache['data'] is None or now - cache['ts'] > CACHE_SECONDS:
            try:
                cache['data'] = fetch_all()
                cache['ts'] = now
            except Exception as e:
                if cache['data'] is None:
                    return jsonify({'error': str(e)}), 500
    return jsonify(cache['data'])

@app.route('/api/composer')
def api_composer():
    now = time.time()
    with lock:
        if cache['composer'] is None or now - cache['composer_ts'] > COMPOSER_CACHE_SECONDS:
            try:
                composer_data = fetch_composer_data()
                fidelity_data = parse_fidelity_csv()
                # Merge: combine accounts, holdings, and totals
                merged = _merge_portfolio_sources(composer_data, fidelity_data)
                cache['composer'] = merged
                cache['composer_ts'] = now
            except Exception as e:
                print(f"  Portfolio fetch error: {e}")
                if cache['composer'] is None:
                    return jsonify({'error': str(e)}), 500
    return jsonify(cache['composer'] or {'error': 'No portfolio data sources configured'})

@app.route('/api/refresh')
def api_refresh():
    with lock:
        cache['data'] = fetch_all()
        cache['ts'] = time.time()
        composer_data = fetch_composer_data() if COMPOSER_KEY_ID else None
        fidelity_data = parse_fidelity_csv()
        if composer_data or fidelity_data:
            cache['composer'] = _merge_portfolio_sources(composer_data, fidelity_data)
            cache['composer_ts'] = time.time()
    return jsonify({'ok': True})

@app.route('/')
def index():
    return Response(HTML, mimetype='text/html')

# ═══════════════════════════════════════════════════════════════════
# DASHBOARD HTML
# ═══════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Signal Monitor v4.5</title>
<style>
:root {
  --bg: #ffffff; --bg2: #f8f9fa; --bg3: #e9ecef;
  --fg: #1a1a2e; --fg2: #6c757d; --fg3: #adb5bd;
  --green: #16a34a; --red: #dc2626; --amber: #d97706;
  --cyan: #0891b2; --blue: #2563eb; --purple: #7c3aed;
  --border: #dee2e6;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--fg); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; font-size:13px; }
a { color:var(--cyan); }

/* Header */
.hdr { display:flex; justify-content:space-between; align-items:center; padding:12px 20px; border-bottom:1px solid var(--border); background:var(--bg2); }
.hdr h1 { font-size:16px; font-weight:700; letter-spacing:1px; color:var(--fg); }
.hdr .meta { color:var(--fg2); font-size:12px; }
.hdr .meta span { margin-left:16px; }
.btn { background:var(--bg); border:1px solid var(--border); color:var(--fg); padding:4px 12px; border-radius:4px; cursor:pointer; font-family:inherit; font-size:12px; }
.btn:hover { background:var(--bg3); }

/* Tabs */
.tabs { display:flex; gap:0; border-bottom:1px solid var(--border); background:var(--bg2); padding:0 20px; }
.tab { padding:10px 20px; cursor:pointer; color:var(--fg2); border-bottom:2px solid transparent; font-size:13px; transition:all .15s; }
.tab:hover { color:var(--fg); }
.tab.active { color:var(--blue); border-bottom-color:var(--blue); font-weight:600; }

/* Layout */
.content { padding:16px 20px; max-width:1400px; margin:0 auto; width:100%; }

/* Signals grid */
.signals-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(380px, 1fr)); gap:8px; margin-bottom:16px; }
.signals-full { grid-column:1 / -1; }

/* Alerts */
.alert-card { background:var(--bg2); border:1px solid var(--border); border-radius:6px; padding:12px 16px; margin-bottom:8px; }
.alert-card.buy { border-left:3px solid var(--green); }
.alert-card.exit, .alert-card.short { border-left:3px solid var(--red); }
.alert-card.hedge, .alert-card.warning, .alert-card.watch { border-left:3px solid var(--amber); }
.alert-title { font-weight:600; font-size:13px; margin-bottom:4px; }
.alert-msg { color:var(--fg2); font-size:12px; white-space:pre-line; }

/* Table */
table { width:100%; border-collapse:collapse; font-size:12px; }
th { text-align:left; padding:8px 10px; color:var(--fg2); font-weight:600; border-bottom:2px solid var(--border); position:sticky; top:0; background:var(--bg); cursor:pointer; user-select:none; white-space:nowrap; }
th:hover { color:var(--fg); background:var(--bg3); }
th .sort-arrow { font-size:10px; margin-left:4px; color:var(--fg3); }
td { padding:6px 10px; border-bottom:1px solid var(--bg3); }
tr:hover { background:var(--bg2); }
tr.ticker-row { cursor:pointer; }
.r { text-align:right; }
.pos { color:var(--green); }
.neg { color:var(--red); }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin:0 2px; }
.dot.up { background:var(--green); }
.dot.dn { background:var(--red); }
.dot.na { background:var(--fg3); }

/* Ticker detail modal */
.modal-overlay { position:fixed; top:0; left:0; right:0; bottom:0; background:rgba(0,0,0,0.4); z-index:100; display:flex; align-items:center; justify-content:center; }
.modal { background:var(--bg); border:1px solid var(--border); border-radius:12px; padding:24px; min-width:420px; max-width:560px; box-shadow:0 8px 32px rgba(0,0,0,0.15); }
.modal h2 { font-size:18px; margin-bottom:16px; display:flex; justify-content:space-between; align-items:center; }
.modal .close-btn { cursor:pointer; font-size:20px; color:var(--fg2); background:none; border:none; }
.modal .close-btn:hover { color:var(--fg); }
.modal-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px 24px; }
.modal-row { display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid var(--bg3); }
.modal-label { color:var(--fg2); font-size:12px; }
.modal-val { font-weight:600; font-size:13px; font-family:'SF Mono',monospace; }

/* RSI bar */
.rsi-bar { display:inline-block; width:60px; height:6px; background:var(--bg3); border-radius:3px; vertical-align:middle; margin-left:6px; position:relative; overflow:hidden; }
.rsi-fill { height:100%; border-radius:3px; position:absolute; left:0; top:0; }

/* Brier section */
.brier-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); gap:12px; margin-top:12px; }
.brier-card { background:var(--bg2); border:1px solid var(--border); border-radius:6px; padding:14px 16px; }
.brier-card.critical { border-left:3px solid var(--red); }
.brier-card.warning { border-left:3px solid var(--amber); }
.brier-card.healthy { border-left:3px solid var(--green); }
.brier-card.insufficient { border-left:3px solid var(--fg3); opacity:0.7; }
.brier-card.active { box-shadow:0 0 0 1px var(--cyan); }
.brier-name { font-weight:600; font-size:13px; display:flex; justify-content:space-between; align-items:center; }
.brier-badge { font-size:10px; padding:2px 8px; border-radius:10px; font-weight:500; }
.bg-critical { background:rgba(239,68,68,.2); color:var(--red); }
.bg-warning { background:rgba(245,158,11,.2); color:var(--amber); }
.bg-healthy { background:rgba(34,197,94,.2); color:var(--green); }
.bg-insufficient { background:rgba(85,85,104,.2); color:var(--fg3); }
.bg-active { background:rgba(6,182,212,.2); color:var(--cyan); }
.brier-stats { display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; margin-top:10px; }
.brier-stat { text-align:center; }
.brier-stat .label { font-size:10px; color:var(--fg3); text-transform:uppercase; letter-spacing:.5px; }
.brier-stat .val { font-size:16px; font-weight:600; margin-top:2px; }
.brier-recent { margin-top:10px; display:flex; gap:4px; flex-wrap:wrap; }
.brier-ep { font-size:10px; padding:2px 6px; border-radius:3px; }
.brier-ep.win { background:rgba(34,197,94,.15); color:var(--green); }
.brier-ep.loss { background:rgba(239,68,68,.15); color:var(--red); }

/* Summary cards */
.summary { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:16px; }
.scard { background:var(--bg2); border:1px solid var(--border); border-radius:6px; padding:12px 16px; text-align:center; }
.scard .label { font-size:10px; color:var(--fg3); text-transform:uppercase; letter-spacing:.5px; }
.scard .val { font-size:22px; font-weight:700; margin-top:4px; }

/* Explainer */
.explainer { background:var(--bg2); border:1px solid var(--border); border-radius:6px; padding:16px 20px; margin-bottom:16px; color:var(--fg2); font-size:12px; line-height:1.7; }
.explainer h3 { color:var(--fg); font-size:14px; margin-bottom:8px; }
.explainer code { background:var(--bg3); padding:1px 5px; border-radius:3px; color:var(--cyan); font-size:11px; }

/* Heatmap */
.hm { display:flex; flex-wrap:wrap; gap:4px; margin-top:8px; }
.hm-chip { padding:3px 8px; border-radius:4px; font-size:11px; cursor:pointer; border:1px solid transparent; }

/* Beta & Miners cards */
.beta-card, .miners-card { background:var(--bg2); border:1px solid var(--border); border-radius:6px; padding:14px 16px; }
.beta-card h3, .miners-card h3 { font-size:13px; font-weight:600; margin-bottom:10px; display:flex; justify-content:space-between; align-items:center; }
.beta-table, .miners-table { width:100%; border-collapse:collapse; font-size:12px; }
.beta-table th, .miners-table th { text-align:right; padding:4px 8px; color:var(--fg3); font-weight:500; border-bottom:1px solid var(--border); font-size:11px; }
.beta-table th:first-child, .miners-table th:first-child { text-align:left; }
.beta-table td, .miners-table td { padding:4px 8px; text-align:right; border-bottom:1px solid var(--border); }
.beta-table td:first-child, .miners-table td:first-child { text-align:left; font-weight:600; }
.beta-table .blend-row { border-top:2px solid var(--fg3); font-weight:700; }
.beta-badge { font-size:10px; padding:2px 8px; border-radius:10px; font-weight:500; }
</style>
</head>
<body>

<div class="hdr">
  <h1>SIGNAL MONITOR v4.6</h1>
  <div class="meta">
    <span id="clock"></span>
    <span id="status">Loading...</span>
    <button class="btn" onclick="forceRefresh()">↻ REFRESH</button>
  </div>
</div>

<div class="tabs">
  <div class="tab active" onclick="setTab('signals')" data-tab="signals">Signals</div>
  <div class="tab" onclick="setTab('portfolio')" data-tab="portfolio">Portfolio</div>
  <div class="tab" onclick="setTab('table')" data-tab="table">All Tickers</div>
  <div class="tab" onclick="setTab('brier')" data-tab="brier">Calibration</div>
  <div class="tab" onclick="setTab('heatmap')" data-tab="heatmap">Heatmap</div>
  <div class="tab" onclick="setTab('hormuz')" data-tab="hormuz">Hormuz</div>
  <div class="tab" onclick="setTab('vix')" data-tab="vix">VIX Structure</div>
  <div class="tab" onclick="setTab('returns')" data-tab="returns">Returns</div>
</div>

<div class="content" id="main"></div>

<script>
let D = null;
let C = null;  // Composer data
let activeTab = 'signals';

async function fetchData() {
  try {
    const r = await fetch('/api/data');
    D = await r.json();
    document.getElementById('status').textContent = `${D.n_tickers} tickers | ${D.ts?.split('T')[1]?.slice(0,8) || ''}`;
    render();
  } catch(e) {
    document.getElementById('status').textContent = 'Error loading data';
  }
}

async function fetchComposer() {
  try {
    const r = await fetch('/api/composer');
    const data = await r.json();
    if (!data.error) C = data;
    if (activeTab === 'portfolio') render();
  } catch(e) {
    console.log('Composer fetch error:', e);
  }
}

function render() {
  if (!D) return;
  const m = document.getElementById('main');
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === activeTab));
  if (activeTab === 'signals') m.innerHTML = renderSignals();
  else if (activeTab === 'portfolio') m.innerHTML = renderPortfolio();
  else if (activeTab === 'table') m.innerHTML = renderTable();
  else if (activeTab === 'brier') m.innerHTML = renderBrier();
  else if (activeTab === 'heatmap') m.innerHTML = renderHeatmap();
  else if (activeTab === 'hormuz') m.innerHTML = renderHormuz();
  else if (activeTab === 'vix') m.innerHTML = renderVixStructure();
  else if (activeTab === 'returns') m.innerHTML = renderReturns();
}

function renderSignals() {
  const sigs = D.signals || [];
  const ind = D.indicators || {};
  const betas = D.rolling_betas || [];
  let h = '';

  // Signal cards in grid
  if (sigs.length) {
    h += '<div class="signals-grid">';
    h += sigs.map(s => `<div class="alert-card ${s.type}"><div class="alert-title">${s.title}</div><div class="alert-msg">${s.msg}</div></div>`).join('');
    h += '</div>';
  } else {
    h += '<div style="padding:20px;color:var(--fg2)">No active signals</div>';
  }

  // Rolling Beta card
  if (betas.length) {
    let blendB63 = null;
    h += '<div class="beta-card signals-full" style="margin:12px 0">';
    h += '<h3>Rolling Beta vs SPY</h3>';
    // Show regime and weights
    const blendRow = betas.find(r => r.is_blend);
    const betaRegime = blendRow ? (blendRow.regime || 'UNKNOWN') : 'UNKNOWN';
    h += `<div style="font-size:11px;color:var(--fg2);margin-bottom:10px">Regime: <b style="color:var(--cyan)">${betaRegime}</b> &mdash; Blend weights adjust to reflect actual Opus positioning (heavy equity only in proven dip-buy/low-vol regimes)</div>`;
    h += '<table class="beta-table"><thead><tr><th style="text-align:left">Group</th><th>63d</th><th>126d</th><th>252d</th><th>Wt</th><th style="text-align:left;padding-left:12px">Note</th></tr></thead><tbody>';
    for (const row of betas) {
      const isBlend = row.is_blend || row.name === 'Est. Blend';
      h += `<tr class="${isBlend ? 'blend-row' : ''}">`;
      h += `<td>${row.name}</td>`;
      for (const k of ['b63','b126','b252']) {
        const v = row[k];
        let color = 'var(--fg)';
        if (v !== null && v !== undefined) {
          if (row.name === 'MF Rotation' || row.name === 'BTAL') {
            color = v < 0 ? 'var(--green)' : 'var(--red)';
          } else if (isBlend) {
            color = v > 2.0 ? 'var(--red)' : v > 1.0 ? 'var(--amber)' : 'var(--green)';
            if (k === 'b63') blendB63 = v;
          }
          h += `<td style="color:${color}">${v >= 0 ? '+' : ''}${v.toFixed(2)}</td>`;
        } else {
          h += '<td style="color:var(--fg3)">N/A</td>';
        }
      }
      // Note cell
      // Weight column
      if (!isBlend) {
        const wt = row.blend_wt != null ? (row.blend_wt * 100).toFixed(0) + '%' : '';
        h += `<td style="text-align:right;color:var(--fg2);font-size:11px">${wt}</td>`;
      } else {
        h += '<td></td>';
      }
      let note = '';
      if (isBlend && row.b63 != null) {
        const tgts = {'BULL + VOL COMPRESS':[0.70,1.00],'BULL + VOL EXPAND':[0.50,0.80],'BEAR RECOVERY':[0.25,0.50],'BEAR DEFENSIVE':[0.00,0.25]};
        const t = tgts[betaRegime];
        if (t && row.b63 >= t[0] && row.b63 <= t[1]) note = 'ON TARGET';
        else if (t && row.b63 < t[0]) note = 'Under';
        else if (t && row.b63 > t[1]) note = 'Over';
        else note = '';
      } else if (row.name === 'MF Rotation' && row.b63 < -0.1) note = '✓ Negative';
      else if (row.name === 'GLD' && row.b63 != null && row.b252 != null && row.b63 > row.b252 + 0.3) note = '↑ Corr rising';
      h += `<td style="text-align:left;padding-left:12px;font-size:11px;color:var(--fg2)">${note}</td>`;
      h += '</tr>';
    }
    h += '</tbody></table>';
    // Regime target assessment
    if (blendB63 !== null) {
      const targets = {
        'BULL + VOL COMPRESS': [0.70, 1.00, 'Capture equity premium, alts dampen vol'],
        'BULL + VOL EXPAND':   [0.50, 0.80, 'Stress rising, start de-risking'],
        'BEAR RECOVERY':       [0.25, 0.50, 'Bouncing but not confirmed, cautious'],
        'BEAR DEFENSIVE':      [0.00, 0.25, 'Full protection, near market-neutral'],
      };
      const tgt = targets[betaRegime];
      if (tgt) {
        const [lo, hi, desc] = tgt;
        let grade, gc;
        if (blendB63 >= lo && blendB63 <= hi) { grade = 'ON TARGET'; gc = 'var(--green)'; }
        else if (blendB63 < lo) { grade = 'UNDER (too defensive)'; gc = 'var(--amber)'; }
        else { grade = 'OVER (too aggressive)'; gc = 'var(--red)'; }
        h += `<div style="margin-top:12px;padding:12px;background:var(--bg2);border-radius:6px;border:1px solid var(--border)">`;
        h += `<div style="display:flex;justify-content:space-between;align-items:center">`;
        h += `<div><span style="font-weight:600">Regime Target:</span> <span style="color:var(--fg2)">${lo.toFixed(2)} to ${hi.toFixed(2)}</span></div>`;
        h += `<div><span style="font-weight:600">Actual (63d):</span> <span style="font-size:16px;font-weight:700;color:${gc}">${blendB63 >= 0 ? '+' : ''}${blendB63.toFixed(2)}</span></div>`;
        h += `<div><span style="background:${gc}18;color:${gc};padding:3px 10px;border-radius:10px;font-weight:600;font-size:12px">${grade}</span></div>`;
        h += `</div>`;
        h += `<div style="margin-top:6px;font-size:11px;color:var(--fg2)">${desc}</div>`;
        h += `<div style="margin-top:8px;font-size:10px;color:var(--fg3)">BULL COMPRESS: 0.70-1.00 | BULL EXPAND: 0.50-0.80 | BEAR RECOVERY: 0.25-0.50 | BEAR DEFENSIVE: 0.00-0.25</div>`;
        h += `</div>`;
      }
    }
    h += '</div>';
  }

  // Miners card
  const minerTickers = ['GLD','GDX','GDXJ','JNUG','NUGT'];
  const hasMiners = minerTickers.some(t => ind[t]);
  if (hasMiners) {
    h += '<div class="miners-card signals-full" style="margin:12px 0">';
    h += '<h3>⛏️ GLD & Miners</h3>';
    h += '<table class="miners-table"><thead><tr><th style="text-align:left">Ticker</th><th>Price</th><th>1d</th><th>RSI(10)</th><th>vs SMA200</th><th style="text-align:left;padding-left:8px">Signal</th></tr></thead><tbody>';
    for (const t of minerTickers) {
      const d = ind[t];
      if (!d) continue;
      const rsiColor = d.rsi < 21 ? 'var(--green)' : d.rsi < 25 ? 'var(--cyan)' : d.rsi > 85 ? 'var(--red)' : d.rsi > 79 ? 'var(--amber)' : 'var(--fg)';
      const chgC = d.chg1d >= 0 ? 'var(--green)' : 'var(--red)';
      const vsC = d.vsSma200 != null ? (d.vsSma200 >= 0 ? 'var(--green)' : 'var(--red)') : 'var(--fg3)';
      let sig = '';
      if (t === 'GDXJ' && d.rsi < 21) sig = '<b style="color:var(--green)">🟢 JNUG BUY</b> 59% +8.4%';
      else if (t === 'GDXJ' && d.rsi < 25) sig = '<b style="color:var(--green)">🟢 JNUG</b> 63% +3.6%';
      else if (t === 'GDX' && d.rsi < 21) sig = '<b style="color:var(--green)">🟢 NUGT</b> 56%';
      else if (t === 'GDX' && d.rsi < 25) sig = '<span style="color:var(--cyan)">Near OS</span>';
      else if ((t==='GDX'||t==='GDXJ') && d.rsi > 85) sig = '<b style="color:var(--amber)">⚠️ DO NOT SHORT</b>';
      else if ((t==='GDX'||t==='GDXJ') && d.rsi > 79) sig = '<span style="color:var(--amber)">Momentum ↑</span>';
      h += `<tr>
        <td><b>${t}</b></td>
        <td>$${d.price < 1000 ? d.price.toFixed(2) : Math.round(d.price).toLocaleString()}</td>
        <td style="color:${chgC}">${d.chg1d >= 0 ? '+' : ''}${d.chg1d.toFixed(1)}%</td>
        <td style="color:${rsiColor}">${d.rsi.toFixed(1)}</td>
        <td style="color:${vsC}">${d.vsSma200 != null ? (d.vsSma200 >= 0 ? '+' : '') + d.vsSma200.toFixed(1) + '%' : '—'}</td>
        <td style="text-align:left;padding-left:8px;font-size:11px">${sig}</td>
      </tr>`;
    }
    h += '</tbody></table></div>';
  }

  // Market Breadth Regime card (v4.3)
  const br = D.breadth_regime;
  if (br && br !== 'UNKNOWN') {
    const brColors = {'BROAD BULL':'var(--green)','NARROW LEADERSHIP':'var(--amber)','ROTATION':'#f97316','BROAD WEAKNESS':'var(--red)'};
    const brGuide = {
      'BROAD BULL':'Breadth healthy. RSI dip-buys at full conviction.',
      'NARROW LEADERSHIP':'Mega-caps carrying, avg stock broken. RSI dip-buys underperform ~25pp. Reduce leverage.',
      'ROTATION':'Equal-weight healthier than SPY/QQQ. Broadening. Favors IWM/RSP.',
      'BROAD WEAKNESS':'Both broken. Crisis mode. GLD/BTAL regime detection should be firing.',
    };
    const gap = D.leadership_gap || 0;
    const rspVs = ind.RSP ? (ind.RSP.vsSma200 != null ? ind.RSP.vsSma200.toFixed(1) : '?') : '?';
    const spyVs = ind.SPY ? (ind.SPY.vsSma200 != null ? ind.SPY.vsSma200.toFixed(1) : '?') : '?';
    const iwmVs = ind.IWM ? (ind.IWM.vsSma200 != null ? ind.IWM.vsSma200.toFixed(1) : '?') : '?';
    h += `<div class="beta-card signals-full" style="margin:12px 0">
      <h3>📊 Market Breadth Regime</h3>
      <div style="display:flex;gap:20px;align-items:center;margin-bottom:10px">
        <span style="font-size:18px;font-weight:700;color:${brColors[br]||'var(--fg)'}">${br}</span>
        <span style="font-size:11px;color:var(--fg2)">Gap (SPY−RSP): <b style="color:${gap>5?'var(--amber)':gap<-5?'#f97316':'var(--fg)'}">${gap>=0?'+':''}${gap.toFixed(1)}pp</b></span>
      </div>
      <div style="display:flex;gap:16px;font-size:12px;margin-bottom:8px">
        <span>RSP vs SMA200: <b style="color:${parseFloat(rspVs)>=0?'var(--green)':'var(--red)'}">${rspVs}%</b></span>
        <span>SPY vs SMA200: <b style="color:${parseFloat(spyVs)>=0?'var(--green)':'var(--red)'}">${spyVs}%</b></span>
        <span>IWM vs SMA200: <b style="color:${parseFloat(iwmVs)>=0?'var(--green)':'var(--red)'}">${iwmVs}%</b></span>
      </div>
      <div style="font-size:11px;color:var(--fg2);padding:8px;background:var(--bg2);border-radius:4px">${brGuide[br]||''}</div>
    </div>`;
  }

  // SPHB/SPLV Risk Appetite card (v4.3)
  const ratio = D.sphb_splv;
  if (ratio && ratio.value) {
    const rsiC = ratio.rsi < 25 ? 'var(--green)' : ratio.rsi < 35 ? 'var(--cyan)' : ratio.rsi > 70 ? 'var(--red)' : 'var(--fg)';
    const label = ratio.rsi < 25 ? 'EXHAUSTION — TQQQ 10d 75.5% WR' : ratio.rsi < 35 ? 'WEAKENING' : ratio.rsi > 70 ? 'STRONG RISK-ON' : 'NEUTRAL';
    h += `<div class="beta-card signals-full" style="margin:12px 0">
      <h3>⚡ Risk Appetite (SPHB/SPLV)</h3>
      <div style="display:flex;gap:20px;align-items:baseline">
        <span style="font-size:12px;color:var(--fg2)">Ratio: <b style="font-size:16px;color:var(--fg)">${ratio.value.toFixed(3)}</b></span>
        <span style="font-size:12px;color:var(--fg2)">RSI(10): <b style="font-size:16px;color:${rsiC}">${ratio.rsi.toFixed(1)}</b></span>
        <span style="font-size:12px;font-weight:600;color:${rsiC}">${label}</span>
      </div>
      ${ratio.rsi < 30 ? '<div style="margin-top:6px;font-size:11px;color:var(--amber)">Manual overlay — same workflow as Double Signal. Check if SPY/QQQ are NOT oversold (unique value).</div>' : ''}
    </div>`;
  }

  // UVXY Vol Regime Shift card (v4.4)
  const vr = D.uvxy_vol_regime;
  if (vr && vr.price) {
    const tierColors = {'EXTREME':'var(--green)','HIGH':'var(--green)','ACTIVE':'var(--cyan)','APPROACHING':'var(--amber)','INACTIVE':'var(--fg2)'};
    const tierStats = {
      'EXTREME':'SPY 20d: 94% WR +7.3% | 40d/60d: 100% WR | n=18',
      'HIGH':'SPY 20d: 92% WR +6.2% | 60d: 100% WR +14.5% | n=24',
      'ACTIVE':'SPY 20d: 83% WR +4.3% | 60d: 92% WR +9.9% | n=52',
      'APPROACHING':'Signal not yet active — monitor for SMA(200) crossover',
      'INACTIVE':'No vol regime shift detected'
    };
    const tc = tierColors[vr.tier] || 'var(--fg2)';
    const pctColor = vr.pct_above >= 20 ? 'var(--green)' : vr.pct_above >= 0 ? 'var(--cyan)' : vr.pct_above >= -10 ? 'var(--amber)' : 'var(--fg2)';
    const barPct = Math.max(0, Math.min(100, ((vr.pct_above + 30) / 80) * 100));
    h += `<div class="beta-card signals-full" style="margin:12px 0">
      <h3>⚡ Vol Regime Shift (UVXY vs SMA200)</h3>
      <div style="display:flex;gap:20px;align-items:baseline;margin-bottom:8px">
        <span style="font-size:18px;font-weight:700;color:${tc}">${vr.tier}</span>
        <span style="font-size:13px;color:var(--fg2)">UVXY: <b style="color:var(--fg)">$${vr.price}</b></span>
        <span style="font-size:13px;color:var(--fg2)">SMA(200): <b style="color:var(--fg)">$${vr.sma200}</b></span>
        <span style="font-size:14px;font-weight:600;color:${pctColor}">${vr.pct_above >= 0 ? '+' : ''}${vr.pct_above}%</span>
      </div>
      <div style="position:relative;height:20px;background:var(--bg2);border-radius:4px;margin:8px 0;overflow:hidden">
        <div style="position:absolute;left:${barPct}%;top:0;bottom:0;width:3px;background:var(--fg);z-index:2"></div>
        <div style="position:absolute;left:${((0+30)/80)*100}%;top:0;bottom:0;width:1px;background:var(--cyan);opacity:0.5"></div>
        <div style="position:absolute;left:${((20+30)/80)*100}%;top:0;bottom:0;width:1px;background:var(--green);opacity:0.5"></div>
        <div style="position:absolute;left:${((30+30)/80)*100}%;top:0;bottom:0;width:1px;background:var(--green);opacity:0.7"></div>
        <div style="position:absolute;left:0;top:0;bottom:0;width:${barPct}%;background:${tc};opacity:0.15;border-radius:4px 0 0 4px"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--fg2);margin-top:-4px;margin-bottom:8px">
        <span>-30%</span><span style="color:var(--cyan)">SMA200</span><span style="color:var(--green)">+20% High</span><span style="color:var(--green)">+30% Extreme</span><span>+50%</span>
      </div>
      <div style="display:flex;gap:16px;font-size:11px;margin-bottom:6px">
        <span>Signal ON: <b style="color:var(--cyan)">$${vr.threshold_signal}</b></span>
        <span>High (20%+): <b style="color:var(--green)">$${vr.threshold_high}</b></span>
        <span>Extreme (30%+): <b style="color:var(--green)">$${vr.threshold_extreme}</b></span>
      </div>
      <div style="font-size:11px;color:var(--fg2);padding:8px;background:var(--bg2);border-radius:4px;margin-top:4px">
        ${tierStats[vr.tier] || ''}
        ${vr.tier === 'ACTIVE' || vr.tier === 'HIGH' || vr.tier === 'EXTREME' ? '<br><span style="color:var(--amber)">Action: Favor UPRO/TQQQ/SOXL. Regime-stable pre/post-2020.</span>' : ''}
      </div>
    </div>`;
  }

  // DRIF Velocity Filter card (v4.4)
  const drif = D.drif;
  if (drif && Object.keys(drif).length) {
    const passes = Object.values(drif).filter(d => d.gate === 'PASS').length;
    const fails = Object.values(drif).filter(d => d.gate === 'FAIL').length;
    let drifBorder = '', drifBadge = 'MONITORING', drifBadgeC = 'var(--fg2)';
    if (passes > 0) { drifBorder = 'border-left:3px solid var(--green);'; drifBadge = passes + ' CONFIRMED'; drifBadgeC = 'var(--green)'; }
    else if (fails > 0) { drifBorder = 'border-left:3px solid var(--amber);'; drifBadge = fails + ' KNIFE'; drifBadgeC = 'var(--amber)'; }
    h += `<div class="beta-card signals-full" style="margin:12px 0;${drifBorder}">
      <h3>🎯 DRIF Velocity Gate <span class="brier-badge" style="background:${drifBadgeC}18;color:${drifBadgeC}">${drifBadge}</span></h3>
      <div style="font-size:11px;color:var(--fg2);margin-bottom:8px">Crash speed filter — separates stabilized dips from falling knives</div>
      <table style="width:100%;font-size:12px;border-collapse:collapse">
      <thead><tr style="border-bottom:2px solid var(--border)"><th style="text-align:left;padding:4px 6px">Ticker</th><th style="padding:4px 6px">RSI</th><th style="padding:4px 6px">5d Ret</th><th style="padding:4px 6px">7d Ret</th><th style="padding:4px 6px">RSI Vel</th><th style="padding:4px 6px">Gate</th><th style="padding:4px 6px">Action</th></tr></thead><tbody>`;
    for (const t of ['SPY','QQQ','SMH']) {
      const d = drif[t];
      if (!d) continue;
      const rc = d.rsi<25?'var(--red)':d.rsi<30?'var(--amber)':'var(--fg)';
      const vc = d.velocity<-25?'var(--red)':d.velocity<-15?'var(--amber)':'var(--fg)';
      let gate='—',action='—';
      if (d.gate==='PASS') { gate='<span style="color:var(--green);font-weight:700">PASS</span>'; action=`<span style="color:var(--green)">${d.lever} (${d.passWr} n=${d.passN})</span>`; }
      else if (d.gate==='FAIL') { gate='<span style="color:var(--red);font-weight:700">FAIL</span>'; action=`<span style="color:var(--red)">WAIT (${d.failWr} n=${d.failN})</span>`; }
      h += `<tr style="border-bottom:1px solid var(--bg3)"><td style="padding:4px 6px;font-weight:600">${t}</td><td style="padding:4px 6px;color:${rc}">${d.rsi.toFixed(1)}</td><td style="padding:4px 6px">${(d.cumRet5d>=0?'+':'')}${d.cumRet5d.toFixed(1)}%</td><td style="padding:4px 6px">${(d.cumRet7d>=0?'+':'')}${d.cumRet7d.toFixed(1)}%</td><td style="padding:4px 6px;color:${vc}">${(d.velocity>=0?'+':'')}${d.velocity.toFixed(0)}</td><td style="padding:4px 6px">${gate}</td><td style="padding:4px 6px">${action}</td></tr>`;
    }
    h += `</tbody></table>
      <div style="margin-top:6px;font-size:10px;color:var(--fg3)">Composer: RSI(10)&lt;25 AND cumulative-return(5d) &gt; -5% | Source: Cakici et al. 2026 DRIF</div>
    </div>`;
  }

  // MOVE Index card (v4.4)
  const mv = D.move_index;
  if (mv && mv.price) {
    const anyActive = mv['19A_active'] || mv['19B_active'] || mv['19C_active'];
    const mvBorder = anyActive ? 'border-left:3px solid var(--green);' : '';
    h += `<div class="beta-card signals-full" style="margin:12px 0;${mvBorder}">
      <h3>📈 MOVE Index (Rates Vol)</h3>
      <div style="display:flex;gap:20px;align-items:baseline;margin-bottom:8px">
        <span style="font-size:16px;font-weight:700;color:var(--fg)">${mv.price}</span>
        <span style="font-size:12px;color:var(--fg2)">RSI: <b style="color:${mv.rsi>79?'var(--red)':mv.rsi<30?'var(--green)':'var(--fg)'}">${mv.rsi}</b></span>
        <span style="font-size:12px;color:var(--fg2)">20d Δ: <b style="color:${mv.change_20d_pct>50?'var(--green)':mv.change_20d_pct>25?'var(--amber)':'var(--fg)'}">${mv.change_20d_pct>=0?'+':''}${mv.change_20d_pct}%</b></span>
        <span style="font-size:12px;color:var(--fg2)">vs SMA200: <b>${mv.pct_above_sma200>=0?'+':''}${mv.pct_above_sma200}%</b></span>
      </div>
      <div style="display:flex;gap:12px;font-size:12px">
        <div style="padding:6px 10px;border-radius:4px;background:${mv['19A_active']?'rgba(22,163,74,.12)':'var(--bg2)'}">
          <span style="color:${mv['19A_active']?'var(--green)':'var(--fg2)'}"><b>19A</b> MOVE>115: ${mv['19A_active']?'ACTIVE':'—'}</span>
          <div style="font-size:10px;color:var(--fg3)">SPY 20d 72% WR +2.1%</div>
        </div>
        <div style="padding:6px 10px;border-radius:4px;background:${mv['19B_active']?'rgba(22,163,74,.12)':'var(--bg2)'}">
          <span style="color:${mv['19B_active']?'var(--green)':'var(--fg2)'}"><b>19B</b> 20d Δ>50%: ${mv['19B_active']?'ACTIVE':'—'}</span>
          <div style="font-size:10px;color:var(--fg3)">SPY 20d 86% WR +5.3%</div>
        </div>
        <div style="padding:6px 10px;border-radius:4px;background:${mv['19C_active']?'rgba(22,163,74,.12)':'var(--bg2)'}">
          <span style="color:${mv['19C_active']?'var(--green)':'var(--fg2)'}"><b>19C</b> RSI>79→<60: ${mv['19C_active']?'ACTIVE':mv['19C_ready']?'PRIMED':'—'}</span>
          <div style="font-size:10px;color:var(--fg3)">SPY 10d 92% WR +2.3%</div>
        </div>
      </div>
      <div style="margin-top:6px;font-size:10px;color:var(--fg3)">Contrarian equity buy — rates vol spikes resolve, equities recover. MOVE>100 + SPY RSI&lt;25 → GLD 100% WR 20d.</div>
    </div>`;
  }

  // FRED Credit Spread card (v4.4)
  const fc = D.fred_credit;
  if (fc && fc.current_oas) {
    const lvlColors = {'COMPLACENT':'var(--amber)','NORMAL':'var(--green)','ELEVATED':'#f97316','CRISIS':'var(--red)'};
    const lvlC = lvlColors[fc.level] || 'var(--fg)';
    const trendC = fc.trend==='SPIKE'?'var(--red)':fc.trend==='DRIFT_UP'?'var(--amber)':fc.trend==='COMPRESSING'?'var(--green)':'var(--fg2)';
    h += `<div class="beta-card signals-full" style="margin:12px 0;${fc.alert?'border-left:3px solid var(--red);':''}">
      <h3>💳 Credit Spread (BB OAS)</h3>
      <div style="display:flex;gap:20px;align-items:baseline;margin-bottom:8px">
        <span style="font-size:18px;font-weight:700;color:${lvlC}">${fc.current_oas}%</span>
        <span style="font-size:14px;font-weight:600;color:${lvlC}">${fc.level}</span>
        <span style="font-size:12px;color:var(--fg2)">Trend: <b style="color:${trendC}">${fc.trend}</b></span>
        ${fc.change_20d?`<span style="font-size:12px;color:var(--fg2)">20d Δ: <b style="color:${trendC}">${fc.change_20d>=0?'+':''}${fc.change_20d}%</b></span>`:''}
        <span style="font-size:12px;color:var(--fg2)">1Y Pctile: <b>${fc.percentile_1y}%</b></span>
      </div>
      ${fc.alert?`<div style="font-size:11px;color:var(--red);padding:6px 8px;background:rgba(220,38,38,.08);border-radius:4px">${fc.alert_reason}</div>`:''}
      <div style="margin-top:6px;font-size:10px;color:var(--fg3)">Source: FRED BAMLH0A1HYBB | As of: ${fc.as_of} | Leads HYG by 1-3 days</div>
    </div>`;
  }

  // FXY Carry Trade card (v4.5 — Group 20)
  const fxy = (D.indicators||{}).FXY;
  if (fxy) {
    const fxyR = fxy.rsi || 50;
    const anyFxyActive = fxyR > 70;
    const fxyBorder = anyFxyActive ? 'border-left:3px solid var(--amber);' : '';
    const tltBroken = (D.indicators||{}).TLT && !(D.indicators.TLT.abSma200);
    h += `<div class="beta-card signals-full" style="margin:12px 0;${fxyBorder}">
      <h3>🇯🇵 FXY Carry Trade Monitor (Group 20)</h3>
      <div style="display:flex;gap:20px;align-items:baseline;margin-bottom:8px">
        <span style="font-size:16px;font-weight:700">$${fxy.price}</span>
        <span style="font-size:12px;color:var(--fg2)">RSI: <b style="color:${fxyR>75?'var(--red)':fxyR>70?'var(--amber)':'var(--fg)'}">${fxyR.toFixed(1)}</b></span>
        <span style="font-size:12px;color:var(--fg2)">vs SMA200: <b>${fxy.vsSma200!=null?(fxy.vsSma200>=0?'+':'')+fxy.vsSma200.toFixed(1)+'%':'N/A'}</b></span>
      </div>
      <div style="display:flex;gap:12px;font-size:12px;flex-wrap:wrap">
        <div style="padding:6px 10px;border-radius:4px;background:${fxyR>75?'rgba(220,38,38,.12)':'var(--bg2)'}">
          <span style="color:${fxyR>75?'var(--red)':'var(--fg2)'}"><b>20A</b> FXY>75: ${fxyR>75?'ACTIVE':'—'}</span>
          <div style="font-size:10px;color:var(--fg3)">Yen strengthening — carry pressure</div>
        </div>
        <div style="padding:6px 10px;border-radius:4px;background:${fxyR>70&&tltBroken?'rgba(245,158,11,.12)':'var(--bg2)'}">
          <span style="color:${fxyR>70&&tltBroken?'var(--amber)':'var(--fg2)'}"><b>20B</b> FXY>70+TLT<SMA200: ${fxyR>70&&tltBroken?'ACTIVE':'—'}</span>
          <div style="font-size:10px;color:var(--fg3)">BTAL 86.7% WR 1d n=15</div>
        </div>
      </div>
      <div style="margin-top:6px;font-size:10px;color:var(--fg3)">Aug 2024: carry unwind → UVXY +35% in 48hrs. Monitor BEFORE it fires. BOJ hiking trend accelerating.</div>
    </div>`;
  }

  // CPER Copper Regime card (v4.5 — Group 21)
  const cper = (D.indicators||{}).CPER;
  if (cper) {
    const cperAE9 = cper.abEma9;
    const spyAE9 = ((D.indicators||{}).SPY||{}).abEma9;
    const copxAE9 = ((D.indicators||{}).COPX||{}).abEma9;
    const contrarian = cperAE9===true && spyAE9===false && copxAE9!==false;
    const supplyRisk = cperAE9===true && copxAE9===false;
    const cBorder = contrarian ? 'border-left:3px solid var(--green);' : supplyRisk ? 'border-left:3px solid var(--amber);' : '';
    h += `<div class="beta-card signals-full" style="margin:12px 0;${cBorder}">
      <h3>🔶 CPER Copper Regime (Group 21)</h3>
      <div style="display:flex;gap:20px;align-items:baseline;margin-bottom:8px">
        <span style="font-size:16px;font-weight:700">$${cper.price}</span>
        <span style="font-size:12px;color:var(--fg2)">EMA9: <b style="color:${cperAE9?'var(--green)':'var(--red)'}">${cperAE9?'ABOVE':'BELOW'}</b></span>
        <span style="font-size:12px;color:var(--fg2)">SPY: <b style="color:${spyAE9?'var(--green)':'var(--red)'}">${spyAE9?'> EMA9':'< EMA9'}</b></span>
        ${(D.indicators||{}).COPX ? `<span style="font-size:12px;color:var(--fg2)">COPX: <b style="color:${copxAE9?'var(--green)':'var(--red)'}">${copxAE9?'> EMA9':'< EMA9'}</b></span>` : ''}
      </div>
      <div style="display:flex;gap:12px;font-size:12px;flex-wrap:wrap">
        <div style="padding:6px 10px;border-radius:4px;background:${contrarian?'rgba(22,163,74,.12)':'var(--bg2)'}">
          <span style="color:${contrarian?'var(--green)':'var(--fg2)'}"><b>21A</b> Contrarian Entry: ${contrarian?'ACTIVE → TQQQ':'—'}</span>
          <div style="font-size:10px;color:var(--fg3)">40.2% CAGR, 0.23 SPY R | Fires ~7%</div>
        </div>
        <div style="padding:6px 10px;border-radius:4px;background:${supplyRisk?'rgba(245,158,11,.12)':'var(--bg2)'}">
          <span style="color:${supplyRisk?'var(--amber)':'var(--fg2)'}"><b>21C</b> Supply Disruption: ${supplyRisk?'WARNING':'Clear'}</span>
          <div style="font-size:10px;color:var(--fg3)">COPX<EMA9 = miner stress. Override 21A.</div>
        </div>
      </div>
      <div style="margin-top:6px;font-size:10px;color:var(--fg3)">Copper reads industrial demand, not equity momentum. 57% independent of SPY trend filters.</div>
    </div>`;
  }

  // Mid-Month Rotation card (v4.6 — Group 25)
  const mm = D.midmonth;
  if (mm && mm.td) {
    const isSignal = mm.is_signal_day;
    const isHolding = mm.is_holding;
    const isPreview = mm.td === 14;
    const border = isSignal ? 'border-left:3px solid var(--green);' : isHolding ? 'border-left:3px solid var(--cyan);' : isPreview ? 'border-left:3px solid var(--amber);' : '';
    const spyWin = mm.spy_mtd > mm.tlt_mtd;
    const daysTo = mm.days_to_signal;
    const sigBanner = isSignal ? '<div style="padding:8px 12px;background:rgba(22,163,74,.12);border-radius:6px;margin-bottom:10px;font-weight:600;color:var(--green)">🟢 SIGNAL DAY — Buy '+mm.pick+' tomorrow, hold through month-end</div>' : '';
    const holdBanner = isHolding ? '<div style="padding:8px 12px;background:rgba(34,211,238,.08);border-radius:6px;margin-bottom:10px;font-weight:600;color:var(--cyan)">Holding: '+mm.pick+' through month-end (signal fired '+Math.abs(daysTo)+' day(s) ago)</div>' : '';
    h += `<div class="beta-card signals-full" style="margin:12px 0;${border}">
      <h3>📅 Mid-Month Rotation (Group 25)</h3>
      ${sigBanner}
      ${holdBanner}
      <div style="display:flex;gap:24px;align-items:baseline;margin-bottom:10px">
        <div>
          <div style="font-size:10px;color:var(--fg3);text-transform:uppercase">Trading Day</div>
          <div style="font-size:22px;font-weight:700;color:${isSignal?'var(--green)':isPreview?'var(--amber)':'var(--fg)'}">${mm.td}</div>
        </div>
        <div>
          <div style="font-size:10px;color:var(--fg3);text-transform:uppercase">SPY MTD</div>
          <div style="font-size:16px;font-weight:600;color:${spyWin?'var(--green)':'var(--red)'}">${mm.spy_mtd>=0?'+':''}${mm.spy_mtd.toFixed(2)}%</div>
        </div>
        <div>
          <div style="font-size:10px;color:var(--fg3);text-transform:uppercase">TLT MTD</div>
          <div style="font-size:16px;font-weight:600;color:${!spyWin?'var(--green)':'var(--red)'}">${mm.tlt_mtd>=0?'+':''}${mm.tlt_mtd.toFixed(2)}%</div>
        </div>
        <div>
          <div style="font-size:10px;color:var(--fg3);text-transform:uppercase">Lean</div>
          <div style="font-size:16px;font-weight:700">Buy ${mm.pick}</div>
        </div>
      </div>
      <div style="display:flex;gap:16px;font-size:12px">
        <div style="padding:6px 10px;border-radius:4px;background:var(--bg2)">
          <b>Signal In:</b> ${daysTo > 0 ? daysTo + ' day(s)' : daysTo === 0 ? '🟢 TODAY' : 'Fired ' + Math.abs(daysTo) + 'd ago'}
        </div>
        <div style="padding:6px 10px;border-radius:4px;background:var(--bg2)">
          <b>Stats:</b> 63.7% WR | Sharpe 1.03 | MaxDD -8.6% | SPY R=-0.03
        </div>
      </div>
      <div style="margin-top:6px;font-size:10px;color:var(--fg3)">Robot James signal. Buy the MTD loser mid-month, hold to month-end. MANUAL only — daily rebalance destroys the -0.03 SPY correlation. n=281 (2002–2025).</div>
    </div>`;
  }

  // Market Internals card (ZBT, McClellan, %Above50SMA) — v4.4
  const bi = D.breadth_inline;
  if (bi && bi.zbt_ema) {
    const zbtC = bi.zbt_zone==='OVERSOLD'?'var(--red)':bi.zbt_zone==='THRUST'?'var(--green)':'var(--fg)';
    const mclC = bi.mcl_zone==='OVERSOLD'?'var(--green)':bi.mcl_zone==='OVERBOUGHT'?'var(--red)':bi.mcclellan>0?'var(--green)':'var(--red)';
    const pctC = bi.pct_above_50sma<25?'var(--red)':bi.pct_above_50sma<40?'var(--amber)':bi.pct_above_50sma>75?'var(--amber)':'var(--green)';
    const pctLabel = bi.pct_above_50sma<25?'WASHED OUT':bi.pct_above_50sma<40?'WEAK':bi.pct_above_50sma>75?'OVEREXTENDED':'HEALTHY';
    const thrustBorder = bi.zbt_thrust ? 'border-left:3px solid var(--green);' : bi.zbt_zone==='OVERSOLD' ? 'border-left:3px solid var(--red);' : '';
    h += `<div class="beta-card signals-full" style="margin:12px 0;${thrustBorder}">
      <h3>📊 Market Internals (ZBT / McClellan / Breadth)</h3>
      ${bi.zbt_thrust ? '<div style="padding:8px 12px;background:rgba(22,163,74,.12);border-radius:6px;margin-bottom:10px;font-weight:600;color:var(--green)">🟢 ZBT THRUST SIGNAL — historically near-100% forward returns at 6 and 12 months</div>' : ''}
      <div style="display:flex;gap:24px;flex-wrap:wrap;margin-bottom:12px">
        <div style="min-width:180px">
          <div style="font-size:10px;color:var(--fg3);text-transform:uppercase;letter-spacing:0.5px">Zweig Breadth Thrust</div>
          <div style="font-size:20px;font-weight:700;color:${zbtC}">${bi.zbt_ema.toFixed(4)}</div>
          <div style="font-size:11px;color:${zbtC}">${bi.zbt_zone}${bi.zbt_zone==='NEUTRAL'&&bi.zbt_ema<0.45?' (near oversold)':''}</div>
          <div style="font-size:10px;color:var(--fg3)">Oversold &lt;0.40 | Thrust &gt;0.615</div>
        </div>
        <div style="min-width:180px">
          <div style="font-size:10px;color:var(--fg3);text-transform:uppercase;letter-spacing:0.5px">McClellan Oscillator</div>
          <div style="font-size:20px;font-weight:700;color:${mclC}">${bi.mcclellan>=0?'+':''}${bi.mcclellan.toFixed(1)}</div>
          <div style="font-size:11px;color:var(--fg2)">${bi.mcl_zone} (${bi.mcl_direction})</div>
          <div style="font-size:10px;color:var(--fg3)">19d: ${bi.mcl_ema19>=0?'+':''}${bi.mcl_ema19} | 39d: ${bi.mcl_ema39>=0?'+':''}${bi.mcl_ema39}</div>
        </div>
        <div style="min-width:180px">
          <div style="font-size:10px;color:var(--fg3);text-transform:uppercase;letter-spacing:0.5px">% Above 50d SMA</div>
          <div style="font-size:20px;font-weight:700;color:${pctC}">${bi.pct_above_50sma.toFixed(1)}%</div>
          <div style="font-size:11px;color:${pctC}">${pctLabel}</div>
          <div style="font-size:10px;color:var(--fg3)">${bi.above50_n}/${bi.above50_total} stocks</div>
        </div>
        <div style="min-width:120px">
          <div style="font-size:10px;color:var(--fg3);text-transform:uppercase;letter-spacing:0.5px">Today</div>
          <div style="font-size:14px;color:var(--fg)"><span style="color:var(--green)">${bi.adv} ▲</span> / <span style="color:var(--red)">${bi.dec} ▼</span></div>
          <div style="font-size:11px;color:var(--fg2)">Ratio: ${bi.zbt_ratio.toFixed(3)}</div>
        </div>
      </div>
      <!-- %Above50 bar -->
      <div style="position:relative;height:16px;background:var(--bg3);border-radius:8px;overflow:hidden;margin-bottom:4px">
        <div style="position:absolute;left:0;top:0;bottom:0;width:${Math.min(100,bi.pct_above_50sma)}%;background:${pctC};opacity:0.3;border-radius:8px 0 0 8px"></div>
        <div style="position:absolute;left:25%;top:0;bottom:0;width:1px;background:var(--red);opacity:0.4"></div>
        <div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--fg3);opacity:0.3"></div>
        <div style="position:absolute;left:75%;top:0;bottom:0;width:1px;background:var(--amber);opacity:0.4"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:9px;color:var(--fg3);margin-bottom:8px">
        <span>0%</span><span style="color:var(--red)">25% Washed</span><span>50%</span><span style="color:var(--amber)">75%</span><span>100%</span>
      </div>
    </div>`;
  }

  // Fibonacci Context card — v4.4
  const fib = D.fibonacci;
  if (fib && Object.keys(fib).length) {
    h += `<div class="beta-card signals-full" style="margin:12px 0">
      <h3>📐 Fibonacci Context Levels (30d)</h3>
      <table style="width:100%;font-size:12px;border-collapse:collapse">
      <thead><tr style="border-bottom:2px solid var(--border)">
        <th style="text-align:left;padding:4px 8px">Ticker</th><th style="padding:4px 8px">Trend</th>
        <th style="padding:4px 8px">Close</th><th style="padding:4px 8px">High</th><th style="padding:4px 8px">Low</th>
        <th style="padding:4px 8px">23.6%</th><th style="padding:4px 8px">38.2%</th><th style="padding:4px 8px">50%</th><th style="padding:4px 8px">61.8%</th>
      </tr></thead><tbody>`;
    for (const sym of ['SPY','QQQ','SMH']) {
      const f = fib[sym];
      if (!f) continue;
      const trendC = f.trend==='UP'?'var(--green)':'var(--red)';
      h += `<tr style="border-bottom:1px solid var(--bg3)">
        <td style="padding:4px 8px;font-weight:600">${sym}</td>
        <td style="padding:4px 8px;text-align:center;color:${trendC}">${f.trend}</td>
        <td style="padding:4px 8px;text-align:center">$${f.close.toFixed(2)}</td>
        <td style="padding:4px 8px;text-align:center;color:var(--fg2)">$${f.high.toFixed(2)}</td>
        <td style="padding:4px 8px;text-align:center;color:var(--fg2)">$${f.low.toFixed(2)}</td>`;
      for (const pct of ['23.6','38.2','50.0','61.8']) {
        const lvl = f.levels[pct];
        if (!lvl) { h += '<td></td>'; continue; }
        const nc = lvl.near ? 'font-weight:700;color:var(--cyan)' : 'color:var(--fg2)';
        h += `<td style="padding:4px 8px;text-align:center;${nc}" title="${lvl.dist>=0?'+':''}${lvl.dist}%${lvl.near?' NEAR':''}">$${lvl.level.toFixed(0)}</td>`;
      }
      h += '</tr>';
    }
    h += `</tbody></table>
      <div style="margin-top:6px;font-size:10px;color:var(--fg3)">Bold/cyan = within 1.5% of level | Hover for distance | Levels from 30-day high/low range</div>
    </div>`;
  }

  return h;
}

function renderTable() {
  const ind = D.indicators || {};
  const cats = {
    'Core':['SPY','QQQ','SMH','IWM'],
    'Defensive':['XLP','XLU','XLV','XLY'],
    'Macro':['GLD','TLT','HYG','USDU','UCO','DBC','BOIL','TMV'],
    'Volatility':['UVXY','SVXY','VIXM'],
    '3x Lev':['NAIL','CURE','FAS','LABU','TQQQ','SOXL','TECL','UPRO','DRN','DFEN'],
    'MF/Alts':['BTAL','DBMF','KMLM','CTA'],
    'Style':['VOOV','VOOG','VTV','QQQE','VOX','USMV'],
    'Intl':['EDC','YINN','KORU','EURL','INDL'],
    'Miners':['GDX','GDXJ','JNUG','NUGT'],
    'Other':['AMD','NVDA','BTC-USD','FNGO','SLV','CPER','COPX','UUP','ILS','RSP','SPHB','SPLV','DFEN','FXY','^MOVE'],
  };
  const catLookup = {};
  for (const [cat, syms] of Object.entries(cats)) { for (const s of syms) catLookup[s] = cat; }

  let tickers = Object.keys(ind);
  // Sort
  const col = window._sortCol || 'rsi';
  const asc = window._sortAsc != null ? window._sortAsc : true;
  tickers.sort((a,b) => {
    let va, vb;
    if (col === 'ticker') { va = a; vb = b; return asc ? va.localeCompare(vb) : vb.localeCompare(va); }
    if (col === 'cat') { va = catLookup[a]||'ZZZ'; vb = catLookup[b]||'ZZZ'; return asc ? va.localeCompare(vb) : vb.localeCompare(va); }
    if (col === 'price') { va = ind[a].price; vb = ind[b].price; }
    else if (col === 'chg1d') { va = ind[a].chg1d; vb = ind[b].chg1d; }
    else if (col === 'chg5d') { va = ind[a].chg5d||0; vb = ind[b].chg5d||0; }
    else if (col === 'rsi') { va = ind[a].rsi; vb = ind[b].rsi; }
    else if (col === 'vs200') { va = ind[a].vsSma200||0; vb = ind[b].vsSma200||0; }
    else { va = ind[a].rsi; vb = ind[b].rsi; }
    return asc ? va - vb : vb - va;
  });

  const arrow = (c) => col === c ? (asc ? ' ▲' : ' ▼') : '';
  const srt = (c) => `onclick="window._sortCol='${c}';window._sortAsc=(window._sortCol==='${c}'?!window._sortAsc:true);render()"`;

  let h = '<table><thead><tr>';
  h += `<th ${srt('ticker')}>Ticker${arrow('ticker')}</th>`;
  h += `<th ${srt('cat')}>Cat${arrow('cat')}</th>`;
  h += `<th class="r" ${srt('price')}>Price${arrow('price')}</th>`;
  h += `<th class="r" ${srt('chg1d')}>1d${arrow('chg1d')}</th>`;
  h += `<th class="r" ${srt('chg5d')}>5d${arrow('chg5d')}</th>`;
  h += `<th class="r" ${srt('rsi')}>RSI${arrow('rsi')}</th>`;
  h += '<th>RSI</th>';
  h += '<th style="text-align:center">E9</th><th style="text-align:center">E20</th><th style="text-align:center">S50</th><th style="text-align:center">S200</th>';
  h += `<th class="r" ${srt('vs200')}>vs200${arrow('vs200')}</th>`;
  h += '</tr></thead><tbody>';
  for (const t of tickers) {
    const d = ind[t];
    const rc = d.rsi < 25 ? 'pos' : d.rsi > 79 ? 'neg' : '';
    const c1 = d.chg1d >= 0 ? 'pos' : 'neg';
    const c5 = (d.chg5d||0) >= 0 ? 'pos' : 'neg';
    const vc = d.vsSma200 != null ? (d.vsSma200 >= 0 ? 'pos' : 'neg') : '';
    const rsiColor = d.rsi > 79 ? 'var(--red)' : d.rsi > 70 ? 'var(--amber)' : d.rsi < 21 ? 'var(--green)' : d.rsi < 30 ? '#22c55e' : 'var(--fg3)';
    const cat = catLookup[t] || '';
    h += `<tr class="ticker-row" onclick="showDetail('${t}')">`;
    h += `<td><b>${t}</b></td>`;
    h += `<td style="font-size:10px;color:var(--fg2)">${cat}</td>`;
    h += `<td class="r">$${d.price < 1000 ? d.price.toFixed(2) : Math.round(d.price).toLocaleString()}</td>`;
    h += `<td class="r ${c1}">${d.chg1d >= 0 ? '+' : ''}${d.chg1d.toFixed(1)}%</td>`;
    h += `<td class="r ${c5}">${(d.chg5d||0) >= 0 ? '+' : ''}${(d.chg5d||0).toFixed(1)}%</td>`;
    h += `<td class="r ${rc}">${d.rsi.toFixed(1)}</td>`;
    h += `<td><div class="rsi-bar"><div class="rsi-fill" style="width:${d.rsi}%;background:${rsiColor}"></div></div></td>`;
    h += `<td style="text-align:center"><span class="dot ${d.abEma9===true?'up':d.abEma9===false?'dn':'na'}"></span></td>`;
    h += `<td style="text-align:center"><span class="dot ${d.abEma20===true?'up':d.abEma20===false?'dn':'na'}"></span></td>`;
    h += `<td style="text-align:center"><span class="dot ${d.abSma50===true?'up':d.abSma50===false?'dn':'na'}"></span></td>`;
    h += `<td style="text-align:center"><span class="dot ${d.abSma200===true?'up':d.abSma200===false?'dn':'na'}"></span></td>`;
    h += `<td class="r ${vc}">${d.vsSma200 != null ? (d.vsSma200 >= 0 ? '+' : '') + d.vsSma200.toFixed(1) + '%' : ''}</td>`;
    h += '</tr>';
  }
  h += '</tbody></table>';
  h += '<div style="margin-top:8px;font-size:11px;color:var(--fg2)"><span class="dot up"></span> Above &nbsp;<span class="dot dn"></span> Below &nbsp; E9=EMA(9) E20=EMA(20) S50=SMA(50) S200=SMA(200) &nbsp;|&nbsp; Click ticker for details</div>';
  return h;
}

function showDetail(t) {
  const d = (D.indicators||{})[t];
  if (!d) return;
  const ov = document.createElement('div');
  ov.className = 'modal-overlay';
  ov.onclick = (e) => { if (e.target === ov) ov.remove(); };
  const rsiC = d.rsi < 25 ? 'var(--green)' : d.rsi > 79 ? 'var(--red)' : 'var(--fg)';
  const dot = (v) => v === true ? '<span class="dot up"></span> Above' : v === false ? '<span class="dot dn"></span> Below' : 'N/A';
  const pct = (v) => v != null ? (v >= 0 ? '+' : '') + v.toFixed(1) + '%' : 'N/A';
  ov.innerHTML = `<div class="modal">
    <h2><span>${t} — $${d.price < 1000 ? d.price.toFixed(2) : Math.round(d.price).toLocaleString()}</span><button class="close-btn" onclick="this.closest('.modal-overlay').remove()">×</button></h2>
    <div class="modal-grid">
      <div><div class="modal-row"><span class="modal-label">RSI(10)</span><span class="modal-val" style="color:${rsiC}">${d.rsi.toFixed(1)}</span></div></div>
      <div><div class="modal-row"><span class="modal-label">1d Change</span><span class="modal-val ${d.chg1d>=0?'pos':'neg'}">${d.chg1d>=0?'+':''}${d.chg1d.toFixed(2)}%</span></div></div>
      <div><div class="modal-row"><span class="modal-label">5d Change</span><span class="modal-val ${(d.chg5d||0)>=0?'pos':'neg'}">${(d.chg5d||0)>=0?'+':''}${(d.chg5d||0).toFixed(2)}%</span></div></div>
      <div><div class="modal-row"><span class="modal-label">vs SMA(200)</span><span class="modal-val">${pct(d.vsSma200)}</span></div></div>
    </div>
    <div style="margin-top:16px;border-top:1px solid var(--bg3);padding-top:12px">
      <div style="font-weight:600;margin-bottom:8px">Moving Averages</div>
      <div class="modal-grid">
        <div><div class="modal-row"><span class="modal-label">EMA(9)</span><span class="modal-val">$${d.ema9.toFixed(2)} ${dot(d.abEma9)}</span></div></div>
        <div><div class="modal-row"><span class="modal-label">EMA(20)</span><span class="modal-val">$${d.ema20.toFixed(2)} ${dot(d.abEma20)}</span></div></div>
        <div><div class="modal-row"><span class="modal-label">SMA(50)</span><span class="modal-val">$${d.sma50.toFixed(2)} ${dot(d.abSma50)}</span></div></div>
        <div><div class="modal-row"><span class="modal-label">SMA(200)</span><span class="modal-val">$${d.sma200.toFixed(2)} ${dot(d.abSma200)}</span></div></div>
      </div>
    </div>
    <div style="margin-top:16px;border-top:1px solid var(--bg3);padding-top:12px">
      <div style="font-weight:600;margin-bottom:8px">Distance from Key Levels</div>
      <div class="modal-grid">
        <div><div class="modal-row"><span class="modal-label">vs EMA(9)</span><span class="modal-val">${d.ema9>0?pct((d.price/d.ema9-1)*100):'N/A'}</span></div></div>
        <div><div class="modal-row"><span class="modal-label">vs EMA(20)</span><span class="modal-val">${d.ema20>0?pct((d.price/d.ema20-1)*100):'N/A'}</span></div></div>
        <div><div class="modal-row"><span class="modal-label">vs SMA(50)</span><span class="modal-val">${pct(d.vsSma50)}</span></div></div>
        <div><div class="modal-row"><span class="modal-label">vs SMA(200)</span><span class="modal-val">${pct(d.vsSma200)}</span></div></div>
      </div>
    </div>
  </div>`;
  document.body.appendChild(ov);
}

function renderBrier() {
  const bs = D.brier || [];
  if (!bs.length) return '<div style="padding:40px;text-align:center;color:var(--fg2)">Computing calibration data...</div>';

  // Summary stats
  const healthy = bs.filter(b => b.health === 'healthy').length;
  const warning = bs.filter(b => b.health === 'warning').length;
  const critical = bs.filter(b => b.health === 'critical').length;
  const active = bs.filter(b => b.active).length;

  let h = `
  <div class="explainer">
    <h3>What is Signal Calibration?</h3>
    Every signal predicts a probability — "SPY RSI&lt;30 → UPRO: 69% win rate" means we expect a 69% chance of profit over 5 days.
    The <b>Brier Score</b> measures how accurate those predictions are over time.<br><br>
    <code>Brier = avg(predicted_probability - actual_outcome)²</code><br><br>
    <b>0.00</b> = perfect predictions &nbsp;|&nbsp; <b>0.25</b> = coin flip &nbsp;|&nbsp; Lower is better<br><br>
    The <b>Brier Skill Score (BSS)</b> compares each signal to a naive baseline (just guessing the unconditional win rate).
    <b>Positive BSS</b> = signal adds value. <b>Negative BSS</b> = signal is worse than guessing.<br><br>
    <b>Trailing 20 WR</b> is the win rate of the last 20 firings — the early warning for degradation.
    If this drops well below the historical WR, the market structure may have changed (like 0DTE compression killing UVXY multi-day holds post-2020).
  </div>

  <div class="summary">
    <div class="scard"><div class="label">Healthy</div><div class="val" style="color:var(--green)">${healthy}</div></div>
    <div class="scard"><div class="label">Warning</div><div class="val" style="color:var(--amber)">${warning}</div></div>
    <div class="scard"><div class="label">Critical</div><div class="val" style="color:var(--red)">${critical}</div></div>
    <div class="scard"><div class="label">Active Now</div><div class="val" style="color:var(--cyan)">${active}</div></div>
  </div>

  <div class="brier-grid">`;

  // Sort: critical first, then warning, then active, then healthy, then insufficient
  const order = {critical:0, warning:1, healthy:2, insufficient:3};
  const sorted = [...bs].sort((a,b) => {
    if (a.active && !b.active) return -1;
    if (!a.active && b.active) return 1;
    return (order[a.health]||9) - (order[b.health]||9);
  });

  for (const b of sorted) {
    const hc = b.health;
    const wrColor = b.actual_wr >= b.hist_wr - 0.05 ? 'var(--green)' : b.actual_wr >= b.hist_wr - 0.15 ? 'var(--amber)' : 'var(--red)';
    const bssColor = b.bss > 0 ? 'var(--green)' : b.bss > -0.1 ? 'var(--amber)' : 'var(--red)';
    const brierColor = b.brier < 0.20 ? 'var(--green)' : b.brier < 0.25 ? 'var(--amber)' : 'var(--red)';

    h += `<div class="brier-card ${hc} ${b.active ? 'active' : ''}">
      <div class="brier-name">
        <span>${b.name}</span>
        <span>
          ${b.active ? '<span class="brier-badge bg-active">ACTIVE</span> ' : ''}
          <span class="brier-badge bg-${hc}">${hc === 'insufficient' ? 'LOW N' : hc.toUpperCase()}</span>
        </span>
      </div>
      <div class="brier-stats" style="grid-template-columns:1fr 1fr 1fr 1fr">
        <div class="brier-stat"><div class="label">Actual WR</div><div class="val" style="color:${wrColor}">${(b.actual_wr*100).toFixed(0)}%</div></div>
        <div class="brier-stat"><div class="label">Brier</div><div class="val" style="color:${brierColor}">${b.brier.toFixed(3)}</div></div>
        <div class="brier-stat"><div class="label">BSS</div><div class="val" style="color:${bssColor}">${b.bss > 0 ? '+' : ''}${b.bss.toFixed(3)}</div></div>
        <div class="brier-stat"><div class="label">Bayes Kelly</div><div class="val" style="color:${b.bk>70?'var(--green)':b.bk>40?'var(--amber)':'var(--red)'}">${b.bk||0}%</div></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:11px;color:var(--fg2)">
        <span>Hist: ${(b.hist_wr*100).toFixed(0)}% | Trail${b.trail_n}: ${(b.trail_wr*100).toFixed(0)}%</span>
        <span>BK/FK: ${b.bk_ratio||0} | n=${b.n} | Tier ${b.tier}</span>
      </div>
      <div class="brier-recent">
        ${(b.recent||[]).map(e => `<span class="brier-ep ${e.win ? 'win' : 'loss'}">${e.win ? 'W' : 'L'} ${e.ret > 0 ? '+' : ''}${e.ret.toFixed(1)}%</span>`).join('')}
      </div>
    </div>`;
  }

  h += '</div>';
  return h;
}

function renderHormuz() {
  const hz = D.hormuz;
  if (!hz || !hz.status) return '<div style="padding:40px;text-align:center;color:var(--fg2)">Hormuz data unavailable</div>';

  const rc = hz.status==='closed' ? 'var(--red)' : hz.status==='restricted' ? 'var(--amber)' : 'var(--green)';
  const statusLabel = hz.status==='closed' ? 'CLOSED' : hz.status==='restricted' ? 'RESTRICTED' : 'OPEN';
  const tl = hz.timeline || [];

  let h = `<div class="beta-card" style="margin-bottom:16px">
    <h3>🚢 Strait of Hormuz — Day ${hz.day}</h3>`;

  // Ceasefire banner
  if (hz.ceasefire_active) {
    const daysLeft = hz.ceasefire_days_remaining || 0;
    h += `<div style="padding:10px 14px;border-radius:6px;background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.3);margin-bottom:12px;font-size:13px">
      <b style="color:#3b82f6">⚠️ CEASEFIRE ACTIVE</b> (started Apr 8) — <b>${daysLeft} days remaining</b> of 2-week window<br>
      <span style="font-size:11px;color:var(--fg2)">Strait NOT reopened. Permission-based transit only. Iran charging $1M+/transit. Insurance still withdrawn. 230 loaded tankers waiting. Negotiations in Islamabad (Vance/Witkoff/Kushner).</span>
    </div>`;
  }

  h += `<div style="display:flex;gap:24px;align-items:baseline;margin-bottom:12px;flex-wrap:wrap">
      <span style="font-size:22px;font-weight:700;color:${rc}">${statusLabel}</span>
      <span style="font-size:14px;color:var(--fg2)">Severity: <b style="color:${rc}">${hz.severity}/10</b></span>
      <span style="font-size:14px;color:var(--fg2)">Ships: <b style="color:${rc}">~${hz.current}/day</b> (baseline: ${hz.baseline})</span>
      <span style="font-size:14px;color:var(--fg2)">↓${hz.drop_pct}%</span>
      <span style="font-size:11px;color:var(--fg3)">Verified: ${hz.verified}</span>
    </div>
    <div style="display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap">
      <div style="padding:8px 14px;border-radius:6px;background:rgba(220,38,38,.08);border:1px solid rgba(220,38,38,.2)">
        <div style="font-size:12px;font-weight:600;color:var(--red)">~${(hz.trapped_gulf||0).toLocaleString()} trapped in Gulf</div>
        <div style="font-size:10px;color:var(--fg3)">${(hz.seafarers||0).toLocaleString()} seafarers stranded</div>
      </div>
      <div style="padding:8px 14px;border-radius:6px;background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.2)">
        <div style="font-size:12px;font-weight:600;color:var(--amber)">${hz.container_ships||0} container ships affected</div>
        <div style="font-size:10px;color:var(--fg3)">${(hz.trapped_outside||0).toLocaleString()} waiting outside</div>
      </div>
      <div style="padding:8px 14px;border-radius:6px;background:${hz.current>=70?'rgba(22,163,74,.12)':'var(--bg2)'};border:1px solid ${hz.current>=70?'rgba(22,163,74,.3)':'var(--border)'}">
        <div style="font-size:12px;font-weight:600;color:${hz.current>=70?'var(--green)':'var(--fg2)'}">${hz.current>=70?'🟢 Kill-switch TRIGGERED':'❌ Kill-switch inactive'}</div>
        <div style="font-size:10px;color:var(--fg3)">Threshold: >70 vessels/day × 5 days</div>
      </div>
    </div>`;
  if (hz.current < 20) {
    h += `<div style="padding:10px 14px;border-radius:6px;background:rgba(220,38,38,.06);border-left:3px solid var(--red);margin-bottom:16px;font-size:12px;color:var(--fg2)">
      <b style="color:var(--red)">Dead-hand framework:</b> Insurance/reinsurance withdrawal keeps strait commercially closed even if military situation improves. P&I clubs need weeks to reinstate coverage. Hormuz positions (UNG/SLV/VLO/STNG/CF/MOS) remain structurally supported until insurance normalizes.</div>`;
  } else if (hz.current < 70) {
    h += `<div style="padding:10px 14px;border-radius:6px;background:rgba(245,158,11,.06);border-left:3px solid var(--amber);margin-bottom:16px;font-size:12px;color:var(--fg2)">
      <b style="color:var(--amber)">Partial reopening:</b> Permission-based transit only. Insurance still withdrawn. Brent stays elevated until P&I coverage reinstated.</div>`;
  }
  if (hz.note) {
    h += `<div style="padding:8px 14px;border-radius:6px;background:var(--bg2);margin-bottom:12px;font-size:12px;color:var(--fg2)">${hz.note}</div>`;
  }
  h += `<div style="font-weight:600;margin-bottom:8px;font-size:13px">Timeline</div><div style="max-height:400px;overflow-y:auto">`;
  for (const evt of tl.slice().reverse()) {
    const tc2 = evt.type==='response' ? 'var(--cyan)' : evt.type==='escalation' ? 'var(--red)' : 'var(--fg)';
    h += `<div style="padding:6px 0;border-bottom:1px solid var(--bg3);font-size:12px">
      <span style="color:var(--fg3)">Day ${evt.day} (${evt.date})</span>
      <div style="color:${tc2};margin-top:2px">${evt.event}</div>
      ${evt.impact ? '<div style="color:var(--fg3);font-size:11px;margin-top:1px">'+evt.impact+'</div>' : ''}
    </div>`;
  }
  h += `</div>`;
  h += `<div style="margin-top:8px;font-size:10px;color:var(--fg3)">Source: ${hz.source||'HormuzTracker.com'} | Updated: ${hz.updated||''}</div>`;
  h += `</div>`;
  return h;
}

function renderVixStructure() {
  const vix = D.vix_structure;
  if (!vix) return '<div style="padding:40px;text-align:center;color:var(--fg2)">VIX term structure data unavailable</div>';

  const regimeInfo = {
    'STEEP_CONTANGO': {name:'Steep Contango',icon:'▲',color:'#22c55e',bg:'rgba(34,197,94,0.08)',
      summary:'Complacency — front-end fear low relative to back-end.',
      leverage:'Excellent — SOXL/TQQQ/UPRO thrive. Vol suppression means 3x compounding works in your favor. Best risk-adjusted regime for leveraged equity.',
      leverageColor:'#22c55e',
      hedge:'Terrible — UVXY bleeds 5-10%/mo from negative roll yield. Only justified as tail insurance.',
      hedgeColor:'#ef4444',
      actions:['SVXY / short VIXM carry trade ON','VIXM<25→HIBL signal likely active','UVXY decay accelerated — avoid longs','Watch VIX9D spike as early warning']},
    'MILD_CONTANGO': {name:'Normal Contango',icon:'△',color:'#86efac',bg:'rgba(134,239,172,0.05)',
      summary:'Healthy risk appetite — normal term structure (~80% of time).',
      leverage:'Good — bread-and-butter regime. Vol drag moderate, offset by equity drift. Expected Sharpe 0.5-0.8.',
      leverageColor:'#86efac',
      hedge:'Poor — UVXY decays 3-5%/mo. BTAL near flat. Hedges are cost center.',
      hedgeColor:'#f97316',
      actions:['Normal operations — equity RSI signals dominate','Modest SVXY carry available','No hedging urgency']},
    'FLAT': {name:'Flat',icon:'—',color:'#facc15',bg:'rgba(250,204,21,0.05)',
      summary:'Transition zone — regime change may be imminent.',
      leverage:'Neutral — watch for direction. If flattening FROM contango, reduce 3x. If FROM backwardation, cautiously add.',
      leverageColor:'#facc15',
      hedge:'Mixed — BTAL most useful here. UVXY decay slows but still negative.',
      hedgeColor:'#facc15',
      actions:['Watch VIX9D for direction signal','Reduce position sizes vs trending regimes','BTAL as primary hedge']},
    'MILD_BACKWARDATION': {name:'Mild Backwardation',icon:'▽',color:'#fb923c',bg:'rgba(251,146,60,0.05)',
      summary:'Elevated near-term fear — market pricing imminent risk.',
      leverage:'Caution — vol drag intensifies. 3x leverage underperforms vs 1x on risk-adjusted basis. Reduce SOXL/TQQQ sizing 30-50%.',
      leverageColor:'#fb923c',
      hedge:'Good — UVXY roll yield turns positive. BTAL outperforms. Hedges become profit centers.',
      hedgeColor:'#22c55e',
      actions:['Reduce 3x leverage sizing','UVXY roll yield turning positive','BTAL outperforming','Watch for Vol Recovery Alpha signal']},
    'STEEP_BACKWARDATION': {name:'Steep Backwardation',icon:'▼',color:'#ef4444',bg:'rgba(239,68,68,0.08)',
      summary:'Crisis mode — extreme near-term fear.',
      leverage:'Dangerous — 3x leverage in crisis amplifies drawdowns catastrophically. SOXL can lose 20-40% in days. Cut to minimum.',
      leverageColor:'#ef4444',
      hedge:'Excellent — UVXY surging. BTAL at peak alpha. This is what you hold hedges FOR.',
      hedgeColor:'#22c55e',
      actions:['UVXY>82→SOXL signal may fire (contrarian)','Cut 3x equity to minimum','BTAL peak performance','Look for Vol Recovery Alpha entry']},
  };

  const r = regimeInfo[vix.regime] || regimeInfo['FLAT'];
  const curve = vix.curve || [];
  const spreads = vix.spreads || {};

  let h = `<div class="beta-card" style="margin-bottom:16px">
    <h3>📈 VIX Term Structure — ${r.icon} ${r.name}</h3>
    <div style="font-size:14px;color:var(--fg2);margin-bottom:8px">${r.summary}</div>
    <div style="padding:10px 14px;border-radius:6px;background:${r.bg};border:1px solid var(--border);margin-bottom:12px">
      <div style="font-size:11px;font-weight:700;color:var(--fg);margin-bottom:4px">📘 What This Regime Means</div>
      <div style="font-size:11px;color:var(--fg2);line-height:1.6">
        ${vix.regime==='STEEP_CONTANGO'?'The market expects <b>less volatility now</b> than in the future. This is the best environment for leveraged equity — vol decay works in your favor, 3x compounding is efficient, and UVXY bleeds heavily. Your Opus/Sleuth/Dispersion symphonies should outperform. <b>Maximize equity allocation.</b>':''}
        ${vix.regime==='MILD_CONTANGO'?'Normal market conditions — the term structure slopes upward slightly, which is typical ~80% of the time. Leveraged equity works fine but without the tailwind of steep contango. <b>Standard operations — let your symphony signals drive decisions.</b>':''}
        ${vix.regime==='FLAT'?'The curve is flat — the market can\'t decide if risk is imminent or not. This is a <b>transition zone</b> and often precedes a regime shift. If flattening from contango, risk is rising. If from backwardation, risk is falling. <b>Reduce position sizes and watch VIX9D for direction.</b>':''}
        ${vix.regime==='MILD_BACKWARDATION'?'Near-term fear exceeds long-term fear — the market is pricing in <b>imminent risk</b>. Vol drag on 3x ETFs intensifies (SOXL/TQQQ lose more to daily rebalancing). UVXY roll yield turns positive — hedges start <b>making</b> money instead of costing it. <b>Reduce 3x sizing 30-50%. Increase BTAL/CTA/KMLM allocation.</b>':''}
        ${vix.regime==='STEEP_BACKWARDATION'?'<b>Crisis mode.</b> Extreme near-term fear — VIX9D far above VIX6M. This is when 3x leverage destroys capital (SOXL can lose 20-40% in days). UVXY surging, BTAL at peak crisis alpha. <b>Cut 3x equity to minimum. This is what you hold hedges for.</b> Watch for Vol Recovery Alpha (UVXY>SMA200 + VIXM<SMA50 → SOXL) as the contrarian re-entry signal.':''}
      </div>
    </div>
    <div style="font-size:12px;color:var(--fg3);margin-bottom:16px">Front/Back spread: <b style="color:${r.color}">${vix.pct_spread}%</b> | VIX: ${vix.vix}</div>`;

  // Curve visualization as bar chart
  if (curve.length > 0) {
    const maxVal = Math.max(...curve.map(c=>c.value));
    h += '<div style="display:flex;gap:12px;align-items:flex-end;height:140px;margin-bottom:16px;padding:12px;background:var(--bg2);border-radius:8px">';
    for (const pt of curve) {
      const pct = (pt.value / maxVal) * 100;
      const barColor = pt.label === 'VIX9D' || pt.label === 'VIX' ? r.color : 'var(--fg3)';
      h += `<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:4px">
        <div style="font-size:11px;font-weight:600;color:var(--fg)">${pt.value.toFixed(1)}</div>
        <div style="width:100%;max-width:48px;height:${pct}%;background:${barColor};border-radius:4px 4px 0 0;min-height:8px"></div>
        <div style="font-size:10px;color:var(--fg3)">${pt.label}</div>
        <div style="font-size:9px;color:var(--fg3)">${pt.days}d</div>
      </div>`;
    }
    h += '</div>';
  }

  // Spreads
  h += '<div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap">';
  for (const [k,v] of Object.entries(spreads)) {
    const sv = typeof v === 'number' ? v.toFixed(2) : v;
    const sc = typeof v === 'number' ? (v < 0 ? 'var(--green)' : v > 0 ? 'var(--red)' : 'var(--fg)') : 'var(--fg)';
    h += `<div style="padding:6px 12px;border-radius:6px;background:var(--bg2);border:1px solid var(--border)">
      <div style="font-size:10px;color:var(--fg3)">${k}</div>
      <div style="font-size:14px;font-weight:600;color:${sc}">${sv}</div>
    </div>`;
  }
  h += '</div>';

  // Leverage & Hedge assessment
  h += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px">';
  h += `<div style="padding:12px;border-radius:8px;background:var(--bg2);border:1px solid var(--border)">
    <div style="font-size:11px;color:var(--fg3);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">3x Leverage Environment</div>
    <div style="font-size:13px;font-weight:700;color:${r.leverageColor};margin-bottom:6px">${r.leverage.split('—')[0].trim()}</div>
    <div style="font-size:11px;color:var(--fg2)">${r.leverage}</div>
  </div>`;
  h += `<div style="padding:12px;border-radius:8px;background:var(--bg2);border:1px solid var(--border)">
    <div style="font-size:11px;color:var(--fg3);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">Hedge Effectiveness</div>
    <div style="font-size:13px;font-weight:700;color:${r.hedgeColor};margin-bottom:6px">${r.hedge.split('—')[0].trim()}</div>
    <div style="font-size:11px;color:var(--fg2)">${r.hedge}</div>
  </div>`;
  h += '</div>';

  // Actions
  h += '<div style="padding:12px;border-radius:8px;background:' + r.bg + ';border:1px solid var(--border);margin-bottom:16px">';
  h += '<div style="font-size:11px;color:var(--fg3);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">Regime Actions</div>';
  for (const a of r.actions) {
    h += `<div style="font-size:12px;color:var(--fg);padding:3px 0">• ${a}</div>`;
  }
  h += '</div>';

  // SMH/IGV spread card
  const sig = D.smh_igv || {};
  if (sig.spread !== undefined) {
    const sc = Math.abs(sig.spread) > 25 ? 'var(--amber)' : 'var(--fg)';
    h += `<div style="padding:12px;border-radius:8px;background:var(--bg2);border:1px solid var(--border)">
      <div style="font-size:11px;color:var(--fg3);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">SMH/IGV Rotation Spread</div>
      <div style="display:flex;gap:24px;align-items:baseline">
        <span style="font-size:18px;font-weight:700;color:${sc}">${sig.spread > 0 ? '+' : ''}${sig.spread}</span>
        <span style="font-size:12px;color:var(--fg2)">SMH RSI: ${sig.smh_rsi?.toFixed(1) || '?'}</span>
        <span style="font-size:12px;color:var(--fg2)">IGV RSI: ${sig.igv_rsi?.toFixed(1) || '?'}</span>
      </div>
      <div style="font-size:10px;color:var(--fg3);margin-top:4px">>30 + IGV<35 → TECL 78% WR | <-15 + SMH<30 → SOXL 88% WR</div>
    </div>`;
  }

  h += '</div>';
  return h;
}

function renderReturns() {
  // Return expectations calibrated to Composer equal-weighted backtest
  // Source: April 5 2026 analysis — 97.1% CAGR, 26.2% vol, 2.72 Sharpe, -13.2% MaxDD, 0.58 SPY R
  const aum = 2200000;  // $2.2M — update when AUM changes significantly
  const daily = [
    {pct:'P5 (bad day)',ret:'-2.13%',dlr:'−$'+Math.round(aum*0.0213/1000)+'K'},
    {pct:'P25',ret:'-0.57%',dlr:'−$'+Math.round(aum*0.0057/1000)+'K'},
    {pct:'P50 (typical)',ret:'+0.27%',dlr:'+$'+Math.round(aum*0.0027/1000)+'K'},
    {pct:'P75',ret:'+1.12%',dlr:'+$'+Math.round(aum*0.0112/1000)+'K'},
    {pct:'P95 (great day)',ret:'+2.67%',dlr:'+$'+Math.round(aum*0.0267/1000)+'K'},
  ];
  const bigMoves = [
    {move:'+3% day',freq:'9.3x/yr',every:'~27 trading days'},
    {move:'+5% day',freq:'2.0x/yr',every:'~6 months'},
    {move:'-3% day',freq:'5.8x/yr',every:'~2 months'},
    {move:'-5% day',freq:'1.4x/yr',every:'~8.5 months'},
  ];
  const horizons = [
    {h:'Weekly',ret:'+1.34%',dlr:'+$29K',wr:'66%'},
    {h:'Monthly',ret:'+5.60%',dlr:'+$123K',wr:'78%'},
    {h:'Quarterly',ret:'+17.70%',dlr:'+$389K',wr:'90%'},
    {h:'Annual',ret:'+91.5%',dlr:'+$2.01M',wr:'99%'},
  ];
  const badRef = [
    {e:'Bad month (P5)',mag:'−$144K',freq:'~2x/year'},
    {e:'Bad week (P5)',mag:'−$96K',freq:'~2-3x/year'},
    {e:'Median worst annual DD',mag:'−$282K (−12.8%)',freq:'Most years'},
    {e:'P95 annual DD',mag:'−$523K (−23.8%)',freq:'1 in 20 years'},
  ];
  const target = [
    {t:'2 years',prob:'34%',med:'$7.1M'},
    {t:'3 years',prob:'91%',med:'$12.7M'},
    {t:'4 years',prob:'99%',med:'$22.4M'},
  ];

  let h = `<div class="explainer">
    <h3>Return Expectations — Calibrated to Composer Backtest</h3>
    Source: 18-strategy equal-weighted blend (Jun 2024 – Apr 2026). 97.1% CAGR, 26.2% vol, 2.72 Sharpe, −13.2% MaxDD, 0.58 SPY R.<br>
    Fat-tailed Monte Carlo (Student-t df=3.5, 1M simulated days). AUM: $${(aum/1e6).toFixed(1)}M.<br><br>
    <b>Purpose:</b> "Is this move normal?" reference. Prevents panic on −$100K weeks and euphoria on +$150K weeks.
  </div>`;

  // Daily distribution
  h += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">';
  h += '<div class="beta-card"><h3>📊 Daily Distribution (on $' + (aum/1e6).toFixed(1) + 'M)</h3><table style="width:100%;font-size:12px;border-collapse:collapse"><thead><tr style="border-bottom:2px solid var(--border)"><th style="text-align:left;padding:4px 8px">Percentile</th><th style="padding:4px 8px">Return</th><th style="padding:4px 8px">Dollar</th></tr></thead><tbody>';
  for (const d of daily) {
    const c = d.ret.startsWith('-') ? 'var(--red)' : 'var(--green)';
    h += `<tr style="border-bottom:1px solid var(--bg3)"><td style="padding:4px 8px">${d.pct}</td><td style="padding:4px 8px;color:${c}">${d.ret}</td><td style="padding:4px 8px;color:${c}">${d.dlr}</td></tr>`;
  }
  h += '</tbody></table></div>';

  // Big move frequency
  h += '<div class="beta-card"><h3>⚡ Big Move Frequency</h3><table style="width:100%;font-size:12px;border-collapse:collapse"><thead><tr style="border-bottom:2px solid var(--border)"><th style="text-align:left;padding:4px 8px">Move</th><th style="padding:4px 8px">Per Year</th><th style="padding:4px 8px">~Every</th></tr></thead><tbody>';
  for (const m of bigMoves) {
    const c = m.move.startsWith('-') ? 'var(--red)' : 'var(--green)';
    h += `<tr style="border-bottom:1px solid var(--bg3)"><td style="padding:4px 8px;color:${c};font-weight:600">${m.move}</td><td style="padding:4px 8px">${m.freq}</td><td style="padding:4px 8px;color:var(--fg2)">${m.every}</td></tr>`;
  }
  h += '</tbody></table><div style="margin-top:8px;font-size:11px;color:var(--fg2)">Asymmetry: 1.73:1 at ±2%, 1.60:1 at ±3% — positive convexity by design</div></div>';
  h += '</div>';

  // Multi-horizon expectations
  h += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">';
  h += '<div class="beta-card"><h3>📈 Median Return Expectations</h3><table style="width:100%;font-size:12px;border-collapse:collapse"><thead><tr style="border-bottom:2px solid var(--border)"><th style="text-align:left;padding:4px 8px">Horizon</th><th style="padding:4px 8px">Return</th><th style="padding:4px 8px">Dollar</th><th style="padding:4px 8px">Win Rate</th></tr></thead><tbody>';
  for (const hz of horizons) {
    h += `<tr style="border-bottom:1px solid var(--bg3)"><td style="padding:4px 8px;font-weight:600">${hz.h}</td><td style="padding:4px 8px;color:var(--green)">${hz.ret}</td><td style="padding:4px 8px;color:var(--green)">${hz.dlr}</td><td style="padding:4px 8px">${hz.wr}</td></tr>`;
  }
  h += '</tbody></table></div>';

  // Normal bad reference
  h += '<div class="beta-card"><h3>📉 "Normal Bad" Reference</h3><table style="width:100%;font-size:12px;border-collapse:collapse"><thead><tr style="border-bottom:2px solid var(--border)"><th style="text-align:left;padding:4px 8px">Event</th><th style="padding:4px 8px">Magnitude</th><th style="padding:4px 8px">Frequency</th></tr></thead><tbody>';
  for (const b of badRef) {
    h += `<tr style="border-bottom:1px solid var(--bg3)"><td style="padding:4px 8px">${b.e}</td><td style="padding:4px 8px;color:var(--red)">${b.mag}</td><td style="padding:4px 8px;color:var(--fg2)">${b.freq}</td></tr>`;
  }
  h += '</tbody></table></div>';
  h += '</div>';

  // $8M target
  h += '<div class="beta-card" style="margin-bottom:16px"><h3>🎯 $8M Target Probability (with $75K/yr contributions)</h3><table style="width:100%;font-size:12px;border-collapse:collapse"><thead><tr style="border-bottom:2px solid var(--border)"><th style="text-align:left;padding:4px 8px">Timeline</th><th style="padding:4px 8px">P(reach $8M)</th><th style="padding:4px 8px">Median Value</th></tr></thead><tbody>';
  for (const t of target) {
    const pc = parseInt(t.prob) >= 90 ? 'var(--green)' : parseInt(t.prob) >= 50 ? 'var(--amber)' : 'var(--fg)';
    h += `<tr style="border-bottom:1px solid var(--bg3)"><td style="padding:4px 8px;font-weight:600">${t.t}</td><td style="padding:4px 8px;color:${pc};font-weight:700">${t.prob}</td><td style="padding:4px 8px">${t.med}</td></tr>`;
  }
  h += '</tbody></table><div style="margin-top:8px;font-size:10px;color:var(--fg3)">Recalibrate when: allocation changes >5%, AUM crosses $3M/$4M, or after 1 year of live data.</div></div>';

  // Q1 2026 validation
  h += `<div class="explainer">
    <h3>Q1 2026 Validation</h3>
    Actual: <b style="color:var(--green)">+9.0%</b> ($2,274K → $2,478K) while SPY −4%, QQQ −6%.<br>
    Model median for SPY −4% quarter: +6.9%. Actual percentile rank: P72 (conditional on SPY −4%).<br>
    Outperformance: +13pp vs SPY. Q1 has zero predictive power for rest of year.<br><br>
    <b>SPY-conditional alpha:</b> Positive in EVERY regime. Increases as SPY improves.<br>
    SPY &lt;−10%: portfolio +19.8% (α +35.5%) | SPY +15-20%: portfolio +63.8% (α +46.4%) | SPY &gt;+30%: portfolio +95.6% (α +55.1%)
  </div>`;

  return h;
}

function renderHeatmap() {
  const ind = D.indicators || {};
  const groups = {
    'Core': ['SPY','QQQ','SMH','IWM'],
    'Defensive': ['XLP','XLU','XLV','XLY'],
    'Macro': ['GLD','TLT','HYG','USDU','UCO','DBC','BOIL','TMV'],
    'Volatility': ['UVXY','SVXY','VIXM'],
    '3x Leveraged': ['NAIL','CURE','FAS','LABU','TQQQ','SOXL','TECL','UPRO','DRN'],
    'MF/Alts': ['BTAL','DBMF','KMLM','CTA'],
    'Style': ['VOOV','VOOG','VTV','QQQE','VOX','USMV'],
    'Intl': ['EDC','YINN','KORU','EURL','INDL'],
    'Other': ['AMD','NVDA','BTC-USD','FNGO','SLV','CPER','COPX','UUP','ILS','RSP','FXY','IGV','^MOVE'],
    'Gold/Miners': ['GLD','GDX','GDXJ','JNUG','NUGT'],
    'Risk Factors': ['SPHB','SPLV','DFEN'],
  };
  let h = '';
  for (const [name, tickers] of Object.entries(groups)) {
    const chips = tickers.filter(t => ind[t]).map(t => {
      const d = ind[t];
      // Font color: green for oversold, red for overbought, dark gray neutral
      const fc = d.rsi <= 21 ? 'var(--green)' : d.rsi <= 30 ? '#22c55e' : d.rsi >= 79 ? 'var(--red)' : d.rsi >= 70 ? 'var(--amber)' : 'var(--fg)';
      const fw = (d.rsi <= 25 || d.rsi >= 79) ? 'font-weight:700;' : '';
      // Background intensity based on extremity
      const bg = d.rsi <= 21 ? 'rgba(22,163,74,0.12)' : d.rsi <= 30 ? 'rgba(22,163,74,0.06)' : d.rsi >= 79 ? 'rgba(220,38,38,0.12)' : d.rsi >= 70 ? 'rgba(217,119,6,0.06)' : 'var(--bg2)';
      return `<div class="hm-chip" style="background:${bg};color:${fc};${fw}border:1px solid var(--border)" title="${t}: RSI=${d.rsi.toFixed(1)} | ${d.vsSma200!=null?(d.vsSma200>=0?'+':'')+d.vsSma200.toFixed(1)+'%':''}">
        ${t} <small>${d.rsi.toFixed(0)}</small></div>`;
    }).join('');
    if (chips) h += `<div style="margin-bottom:14px"><div style="color:var(--fg2);font-size:11px;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px;font-weight:600">${name}</div><div class="hm">${chips}</div></div>`;
  }
  return h;
}

function renderPortfolio() {
  if (!C) return '<div style="padding:40px;text-align:center;color:var(--fg2)">Loading portfolio data... <br><span style="font-size:11px">(Composer: 5-min cache | Fidelity: CSV drop folder)</span></div>';
  let h = '';
  const con = C.consolidated || {};
  const accts = C.accounts || [];
  const fmt = (v) => v != null ? '$' + Math.abs(v).toLocaleString('en-US', {minimumFractionDigits:0, maximumFractionDigits:0}) : 'N/A';
  const fmtD = (v) => v != null ? '$' + v.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}) : 'N/A';
  const fmtP = (v) => v != null ? (v>=0?'+':'') + v.toFixed(2) + '%' : 'N/A';
  const pc = (v) => v>=0 ? 'var(--green)' : 'var(--red)';
  const sources = C.sources || [];

  // Fidelity CSV staleness warning
  const fcsv = C.fidelity_csv;
  if (fcsv) {
    const staleC = fcsv.stale ? 'var(--red)' : fcsv.staleness_hours > 12 ? 'var(--amber)' : 'var(--green)';
    const staleLabel = fcsv.stale ? '⚠️ STALE (>24h)' : fcsv.staleness_hours > 12 ? '⏳ Getting old' : '✅ Fresh';
    h += `<div class="beta-card" style="margin-bottom:12px;padding:8px 16px;border-left:3px solid ${staleC}">
      <span style="font-size:12px"><b>Fidelity CSV:</b> ${fcsv.file} — modified ${fcsv.file_date} — <span style="color:${staleC}">${staleLabel} (${fcsv.staleness_hours}h ago)</span></span>
    </div>`;
  }

  // === ACCOUNT SUMMARY CARDS ===
  h += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin-bottom:16px">';
  for (const a of accts) {
    const src = a.source || 'Unknown';
    const srcBadge = src === 'Composer' ? '<span style="font-size:9px;background:#8b5cf6;color:#fff;padding:1px 6px;border-radius:3px;margin-left:6px">COMPOSER</span>' : src === 'Fidelity' ? '<span style="font-size:9px;background:#22c55e;color:#fff;padding:1px 6px;border-radius:3px;margin-left:6px">FIDELITY</span>' : '';
    const label = a.type.includes('roth') || a.type.toLowerCase().includes('roth') ? '🟢 Roth IRA' : a.type.includes('traditional') || a.type.toLowerCase().includes('trad') ? '🔵 Traditional IRA' : a.type;
    h += `<div class="beta-card">
      <h3>${label}${srcBadge}</h3>
      <div style="font-size:28px;font-weight:800">${fmt(a.value)}</div>
      <div style="font-size:14px;color:${pc(a.today_dollar)};margin-top:4px">${a.today_dollar>=0?'+':''}${fmtD(a.today_dollar)} (${fmtP(a.today_pct)})</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px;font-size:11px">
        ${a.twr!=null?'<div><span style="color:var(--fg3)">TWR:</span> <b>'+fmtP(a.twr*100)+'</b></div>':''}
        ${a.cash!=null?'<div><span style="color:var(--fg3)">Cash:</span> <b>'+fmt(a.cash)+'</b></div>':''}
        ${a.net_deposits!=null?'<div><span style="color:var(--fg3)">Net Deposits:</span> <b>'+fmt(a.net_deposits)+'</b></div>':''}
        ${a.symphonies?'<div><span style="color:var(--fg3)">Symphonies:</span> <b>'+a.symphonies.length+'</b></div>':''}
        ${a.cost_basis?'<div><span style="color:var(--fg3)">Cost Basis:</span> <b>'+fmt(a.cost_basis)+'</b></div>':''}
        ${a.total_gl!=null?'<div><span style="color:var(--fg3)">Total G/L:</span> <b style="color:'+pc(a.total_gl)+'">'+fmtP(a.total_gl_pct)+'</b></div>':''}
      </div>
    </div>`;
  }
  // Consolidated card
  const goalPct = con.goal_8m_pct || 0;
  const goalColor = goalPct >= 75 ? 'var(--green)' : goalPct >= 50 ? 'var(--amber)' : 'var(--fg2)';
  h += `<div class="beta-card" style="border-left:3px solid var(--cyan)">
    <h3>📊 Consolidated</h3>
    <div style="font-size:28px;font-weight:800">${fmt(con.total_value)}</div>
    <div style="font-size:14px;color:${pc(con.today_dollar)};margin-top:4px">${con.today_dollar>=0?'+':''}${fmtD(con.today_dollar)} (${fmtP(con.today_pct)})</div>
    <div style="margin-top:12px">
      <div style="font-size:10px;color:var(--fg3);text-transform:uppercase;letter-spacing:.5px">$8M Goal Progress</div>
      <div style="background:var(--bg3);border-radius:4px;height:10px;margin-top:4px;overflow:hidden">
        <div style="background:${goalColor};height:100%;width:${Math.min(goalPct,100)}%;border-radius:4px;transition:width 0.5s"></div>
      </div>
      <div style="font-size:12px;font-weight:700;color:${goalColor};margin-top:4px">${goalPct.toFixed(1)}%</div>
    </div>
  </div>`;
  h += '</div>';

  // === HOLY GRAIL STREAM ALLOCATION ===
  const streams = con.streams || [];
  if (streams.length) {
    h += '<div class="beta-card" style="margin-bottom:16px"><h3>🏆 Holy Grail Stream Allocation</h3>';
    h += '<div style="display:flex;gap:3px;height:28px;border-radius:4px;overflow:hidden;margin-bottom:12px">';
    const streamColors = {'Equity (1x)':'#3b82f6','Lev Equity':'#8b5cf6','MF/Alts':'#22c55e','Gold/Commod':'#eab308','Vol/Hedge':'#ef4444','Bonds':'#06b6d4','Currency':'#f97316','Sector':'#ec4899','Cash':'#6b7280','Other':'#a3a3a3'};
    for (const s of streams) {
      if (s.pct < 0.5) continue;
      const c = streamColors[s.name] || '#888';
      h += `<div style="background:${c};flex:${s.pct};min-width:2px" title="${s.name}: ${s.pct}%"></div>`;
    }
    h += '</div>';
    h += '<table style="width:100%;font-size:12px;border-collapse:collapse"><thead><tr style="border-bottom:2px solid var(--border)"><th style="text-align:left;padding:4px 8px">Stream</th><th style="text-align:right;padding:4px 8px">Value</th><th style="text-align:right;padding:4px 8px">%</th><th style="text-align:left;padding:4px 8px;width:120px">Bar</th></tr></thead><tbody>';
    for (const s of streams) {
      const c = streamColors[s.name] || '#888';
      h += `<tr style="border-bottom:1px solid var(--bg3)"><td style="padding:4px 8px;font-weight:600"><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${c};margin-right:6px"></span>${s.name}</td><td style="text-align:right;padding:4px 8px">${fmt(s.value)}</td><td style="text-align:right;padding:4px 8px;font-weight:600">${s.pct.toFixed(1)}%</td><td style="padding:4px 8px"><div style="background:${c};height:8px;border-radius:2px;width:${Math.min(s.pct*2,100)}%"></div></td></tr>`;
    }
    h += '</tbody></table></div>';
  }

  // === SYMPHONY TABLE ===
  for (const a of accts) {
    const syms = a.symphonies || [];
    if (!syms.length) continue;
    const label = a.type.toLowerCase().includes('roth') ? 'Roth IRA' : a.type.toLowerCase().includes('trad') ? 'Traditional IRA' : a.type;
    h += `<div class="beta-card" style="margin-bottom:16px"><h3>🎵 Symphonies — ${label}</h3>`;
    h += '<div style="overflow-x:auto"><table style="width:100%;font-size:11px;border-collapse:collapse"><thead><tr style="border-bottom:2px solid var(--border);font-size:10px"><th style="text-align:left;padding:6px 8px">Symphony</th><th style="text-align:right;padding:6px 8px">Value</th><th style="text-align:right;padding:6px 8px">% Acct</th><th style="text-align:right;padding:6px 8px">Today</th><th style="text-align:right;padding:6px 8px">Ann. Return</th><th style="text-align:right;padding:6px 8px">Sharpe</th><th style="text-align:right;padding:6px 8px">Max DD</th><th style="text-align:right;padding:6px 8px">Holdings</th><th style="text-align:left;padding:6px 8px">Last Rebal</th></tr></thead><tbody>';
    const sorted = [...syms].sort((a,b) => (b.value||0) - (a.value||0));
    for (const s of sorted) {
      const annRet = s.annualized_return != null ? fmtP(s.annualized_return * 100) : '—';
      const annC = s.annualized_return != null ? pc(s.annualized_return) : 'var(--fg2)';
      const sharpe = s.sharpe != null ? s.sharpe.toFixed(2) : '—';
      const sharpeC = s.sharpe != null ? (s.sharpe > 1.5 ? 'var(--green)' : s.sharpe > 0.8 ? 'var(--fg)' : 'var(--red)') : 'var(--fg2)';
      const mdd = s.max_dd != null ? (s.max_dd * 100).toFixed(1) + '%' : '—';
      const mddC = s.max_dd != null ? (s.max_dd > -0.15 ? 'var(--green)' : s.max_dd > -0.30 ? 'var(--amber)' : 'var(--red)') : 'var(--fg2)';
      const todayStr = s.last_dollar_change != null ? (s.last_dollar_change>=0?'+':'') + '$' + Math.abs(s.last_dollar_change).toFixed(0) : '—';
      const todayPctStr = s.last_pct_change != null ? fmtP(s.last_pct_change * 100) : '';
      const holdTickers = (s.holdings||[]).filter(h=>h.ticker!=='$USD').map(h=>h.ticker).join(', ');
      const rebal = s.last_rebalance ? new Date(s.last_rebalance).toLocaleDateString('en-US',{month:'short',day:'numeric'}) : '—';
      const skipBadge = s.skip_rebalance_today ? ' <span style="font-size:9px;background:var(--amber);color:#000;padding:1px 4px;border-radius:3px">SKIP</span>' : '';
      h += `<tr style="border-bottom:1px solid var(--bg3)">
        <td style="padding:6px 8px;font-weight:600"><span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:${s.color||'#888'};margin-right:6px"></span>${s.name}${skipBadge}</td>
        <td style="text-align:right;padding:6px 8px;font-weight:600">${fmt(s.value)}</td>
        <td style="text-align:right;padding:6px 8px">${s.pct_of_account}%</td>
        <td style="text-align:right;padding:6px 8px;color:${s.last_dollar_change!=null?pc(s.last_dollar_change):'var(--fg2)'}">${todayStr}<br><span style="font-size:10px">${todayPctStr}</span></td>
        <td style="text-align:right;padding:6px 8px;color:${annC};font-weight:600">${annRet}</td>
        <td style="text-align:right;padding:6px 8px;color:${sharpeC}">${sharpe}</td>
        <td style="text-align:right;padding:6px 8px;color:${mddC}">${mdd}</td>
        <td style="text-align:right;padding:6px 8px;font-size:10px;color:var(--fg2);max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${holdTickers}">${holdTickers||'—'}</td>
        <td style="padding:6px 8px;font-size:10px">${rebal}</td>
      </tr>`;
    }
    h += '</tbody></table></div></div>';
  }

  // === CONSOLIDATED HOLDINGS TABLE ===
  const holdings = con.holdings || [];
  if (holdings.length) {
    h += '<div class="beta-card" style="margin-bottom:16px"><h3>📋 Consolidated Holdings (Top 50)</h3>';
    h += '<div style="overflow-x:auto"><table style="width:100%;font-size:11px;border-collapse:collapse"><thead><tr style="border-bottom:2px solid var(--border);font-size:10px"><th style="text-align:left;padding:6px 8px">Ticker</th><th style="text-align:left;padding:6px 8px">Name</th><th style="text-align:right;padding:6px 8px">Value</th><th style="text-align:right;padding:6px 8px">% Total</th><th style="text-align:right;padding:6px 8px">Shares</th><th style="text-align:right;padding:6px 8px">Price</th><th style="text-align:right;padding:6px 8px">Today %</th><th style="text-align:right;padding:6px 8px">Total P&L %</th><th style="text-align:right;padding:6px 8px">Cost Basis</th></tr></thead><tbody>';
    for (const h2 of holdings) {
      const isCash = h2.symbol === '$USD';
      const concRisk = h2.pct_of_total > 15 && !isCash;
      const rowStyle = concRisk ? 'background:rgba(239,68,68,.06);' : '';
      h += `<tr style="border-bottom:1px solid var(--bg3);${rowStyle}">
        <td style="padding:5px 8px;font-weight:700">${h2.symbol}${concRisk?' ⚠️':''}</td>
        <td style="padding:5px 8px;color:var(--fg2);max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${h2.name||''}</td>
        <td style="text-align:right;padding:5px 8px;font-weight:600">${fmt(h2.total_value)}</td>
        <td style="text-align:right;padding:5px 8px;font-weight:${h2.pct_of_total>5?'700':'400'}">${h2.pct_of_total.toFixed(1)}%</td>
        <td style="text-align:right;padding:5px 8px">${h2.total_amount!=null?h2.total_amount.toFixed(h2.total_amount<10?4:2):'—'}</td>
        <td style="text-align:right;padding:5px 8px">${h2.price?'$'+h2.price.toFixed(2):'—'}</td>
        <td style="text-align:right;padding:5px 8px;color:${h2.today_pct!=null?pc(h2.today_pct):'var(--fg2)'}">${h2.today_pct!=null?fmtP(h2.today_pct):'—'}</td>
        <td style="text-align:right;padding:5px 8px;color:${h2.total_change_pct!=null?pc(h2.total_change_pct):'var(--fg2)'}">${h2.total_change_pct!=null?fmtP(h2.total_change_pct):'—'}</td>
        <td style="text-align:right;padding:5px 8px;color:var(--fg2)">${h2.cost_basis?fmt(h2.cost_basis):'—'}</td>
      </tr>`;
    }
    h += '</tbody></table></div></div>';
  }

  // === PORTFOLIO HISTORY CHART (simple ASCII sparkline from account histories) ===
  for (const a of accts) {
    if (!a.history || !a.history.values || a.history.values.length < 10) continue;
    const vals = a.history.values;
    const dates = a.history.dates;
    const label = a.type.toLowerCase().includes('roth') ? 'Roth IRA' : a.type.toLowerCase().includes('trad') ? 'Traditional IRA' : a.type;
    const minV = Math.min(...vals);
    const maxV = Math.max(...vals);
    const range = maxV - minV || 1;
    const W = 600, H = 120;
    const pts = vals.map((v,i) => `${(i/(vals.length-1))*W},${H - ((v-minV)/range)*H}`).join(' ');
    const lastV = vals[vals.length-1];
    const firstV = vals[0];
    const totalRet = ((lastV/firstV)-1)*100;
    // Running max DD
    let peak = 0, maxDD = 0;
    for (const v of vals) { if (v > peak) peak = v; const dd = (v-peak)/peak; if (dd < maxDD) maxDD = dd; }

    h += `<div class="beta-card" style="margin-bottom:16px">
      <h3>📈 ${label} — Equity Curve</h3>
      <div style="display:flex;gap:24px;margin-bottom:8px;font-size:11px">
        <span>Start: <b>${fmt(firstV)}</b> (${dates[0]})</span>
        <span>Current: <b>${fmt(lastV)}</b></span>
        <span>Return: <b style="color:${pc(totalRet)}">${fmtP(totalRet)}</b></span>
        <span>Max DD: <b style="color:var(--red)">${(maxDD*100).toFixed(1)}%</b></span>
      </div>
      <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:${H}px;overflow:visible">
        <polyline points="${pts}" fill="none" stroke="var(--cyan)" stroke-width="1.5" />
        <line x1="0" y1="${H}" x2="${W}" y2="${H}" stroke="var(--border)" stroke-width="0.5" />
      </svg>
    </div>`;
  }

  h += `<div style="font-size:10px;color:var(--fg3);text-align:center;margin-top:8px">Sources: ${(C.sources||[]).join(' + ') || 'None'} | Last: ${C.ts?.split('T')[1]?.slice(0,8)||'N/A'}${fcsv?' | Fidelity CSV: '+fcsv.staleness_hours+'h old':''}</div>`;
  return h;
}

function setTab(t) {
  activeTab = t;
  if (t === 'portfolio' && !C) fetchComposer();
  render();
}
async function forceRefresh() {
  document.querySelector('.btn').textContent = '⏳';
  await fetch('/api/refresh');
  await fetchData();
  if (C || activeTab === 'portfolio') await fetchComposer();
  document.querySelector('.btn').textContent = '↻ REFRESH';
}

setInterval(() => {
  const c = document.getElementById('clock');
  if (c) c.textContent = new Date().toLocaleTimeString();
}, 1000);

setInterval(fetchData, 60000);
setInterval(fetchComposer, 300000);  // 5 min
fetchData();
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Signal Monitor — Dashboard Server v4.6")
    print("=" * 60)
    print(f"  Tickers: {len(TICKERS)}")
    print(f"  Refresh: every {CACHE_SECONDS}s")
    print(f"  Brier: recomputed every 10 min (persists to {BRIER_JSON_PATH})")
    print(f"  FRED API: {'configured' if FRED_API_KEY else 'not set (credit spreads disabled)'}")
    print(f"  Composer API: {'configured' if COMPOSER_KEY_ID else 'not set (portfolio tab disabled)'}")
    print(f"  Fidelity CSV: {FIDELITY_CSV_DIR} ({'exists' if os.path.isdir(FIDELITY_CSV_DIR) else 'not found — create folder to enable'})")
    print(f"  History: {HISTORY_PERIOD}")
    print()
    print("  → Open http://localhost:5052 in your browser")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5052, debug=False)
