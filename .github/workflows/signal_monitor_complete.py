#!/usr/bin/env python3
"""
CHF Signal Monitor — Dashboard Server v4.2
============================================
Self-hosted real-time market signal dashboard with Brier Score calibration,
rolling beta vs SPY, and gold miners signal group.

Usage:
    pip install flask yfinance pandas numpy
    python chf_dashboard_server.py

Then open http://localhost:5050 in your browser.

v4.2: Rolling beta vs SPY, GLD & miners (Group 18), CSS display fix
v4.0: Signal Calibration tab with rolling Brier scores
"""

from flask import Flask, jsonify, Response
import yfinance as yf
import pandas as pd
import numpy as np
import json
import time
import threading
from datetime import datetime

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
    'VOOV','VOOG','VTV','QQQE','VOX',
    'BTAL','DBMF','KMLM','CTA',
    'FNGO',
    'UUP','SLV','CPER',
    # Gold Miners (Group 18)
    'GDX','GDXJ','JNUG','NUGT',
]

CACHE_SECONDS = 60
HISTORY_PERIOD = '2y'

cache = {'data': None, 'ts': 0, 'brier': None, 'brier_ts': 0, 'rolling_betas': None}
lock = threading.Lock()

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
            'Gold/Commod':0.07,'Vol/Hedge':0.06,'Bonds/Cash':0.12,
        },
        'BULL + VOL EXPAND': {
            'Equity 1x':0.48,'Lev Equity':0.12,'MF/Alts':0.12,
            'Gold/Commod':0.08,'Vol/Hedge':0.10,'Bonds/Cash':0.10,
        },
        'BEAR RECOVERY': {
            'Equity 1x':0.42,'Lev Equity':0.10,'MF/Alts':0.20,
            'Gold/Commod':0.10,'Vol/Hedge':0.13,'Bonds/Cash':0.05,
        },
        'BEAR DEFENSIVE': {
            'Equity 1x':0.35,'Lev Equity':0.08,'MF/Alts':0.22,
            'Gold/Commod':0.10,'Vol/Hedge':0.20,'Bonds/Cash':0.05,
        },
    }
    default_weights = {
        'Equity 1x':0.45,'Lev Equity':0.12,'MF/Alts':0.13,
        'Gold/Commod':0.08,'Vol/Hedge':0.10,'Bonds/Cash':0.12,
    }
    blend_w = regime_weights.get(regime, default_weights)

    groups = [
        ('Equity 1x',   [('SPY',1.0)]),
        ('Lev Equity',  [('UPRO',1.0)]),
        ('MF/Alts',     [('CTA',0.25),('DBMF',0.25),('BTAL',0.30),('KMLM',0.20)]),
        ('Gold/Commod', [('GLD',0.85),('DBC',0.15)]),
        ('Vol/Hedge',   [('UVXY',0.35),('TMV',0.35),('BTAL',0.30)]),
        ('Bonds/Cash',  [('TLT',0.5),('SHY',0.5)]),
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
                        snap[tk] = {'rsi10': float(v)}
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

        results.append({
            'id': sig['id'], 'name': sig['name'], 'tier': sig['tier'],
            'hist_wr': sig['wr'], 'actual_wr': round(np.mean(outcomes), 3),
            'n': n, 'brier': round(brier, 4), 'bss': round(bss, 4),
            'trail_n': len(recent), 'trail_wr': round(trail_wr, 3),
            'trail_brier': round(trail_brier, 4), 'health': health,
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
                snap_now[tk] = {'rsi10': float(v)}
    for r in results:
        sig = next((s for s in BRIER_SIGNALS if s['id'] == r['id']), None)
        if sig:
            try:
                r['active'] = sig['cond'](snap_now)
            except:
                pass

    return results

# ═══════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════
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
            indicators[t] = {
                'price': round(price, 2), 'rsi': round(r10, 1),
                'sma200': round(s200, 2), 'sma50': round(s50, 2),
                'ema9': round(e9, 2), 'ema20': round(e20, 2),
                'vsSma200': round(vs200, 1) if vs200 is not None else None,
                'chg1d': round(chg1d, 2),
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
        except Exception as e:
            print(f"  Brier error: {e}")
            brier_results = cache.get('brier', [])

    return {
        'indicators': indicators,
        'signals': signals,
        'brier': brier_results or [],
        'rolling_betas': rolling_betas or [],
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

@app.route('/api/refresh')
def api_refresh():
    with lock:
        cache['data'] = fetch_all()
        cache['ts'] = time.time()
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
<title>Signal Monitor v4.2</title>
<style>
:root {
  --bg: #0a0a0f; --bg2: #12121a; --bg3: #1a1a26;
  --fg: #e0e0e8; --fg2: #8888a0; --fg3: #555568;
  --green: #22c55e; --red: #ef4444; --amber: #f59e0b;
  --cyan: #06b6d4; --blue: #3b82f6; --purple: #a855f7;
  --border: #2a2a3a;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--fg); font-family:'SF Mono','Fira Code',monospace; font-size:13px; }
a { color:var(--cyan); }

/* Header */
.hdr { display:flex; justify-content:space-between; align-items:center; padding:12px 20px; border-bottom:1px solid var(--border); background:var(--bg2); }
.hdr h1 { font-size:16px; font-weight:600; letter-spacing:1px; }
.hdr .meta { color:var(--fg2); font-size:12px; }
.hdr .meta span { margin-left:16px; }
.btn { background:var(--bg3); border:1px solid var(--border); color:var(--fg); padding:4px 12px; border-radius:4px; cursor:pointer; font-family:inherit; font-size:12px; }
.btn:hover { background:var(--border); }

/* Tabs */
.tabs { display:flex; gap:0; border-bottom:1px solid var(--border); background:var(--bg2); padding:0 20px; }
.tab { padding:10px 20px; cursor:pointer; color:var(--fg2); border-bottom:2px solid transparent; font-size:13px; transition:all .15s; }
.tab:hover { color:var(--fg); }
.tab.active { color:var(--cyan); border-bottom-color:var(--cyan); }

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
th { text-align:left; padding:8px 10px; color:var(--fg2); font-weight:500; border-bottom:1px solid var(--border); position:sticky; top:0; background:var(--bg); }
td { padding:6px 10px; border-bottom:1px solid var(--border); }
tr:hover { background:var(--bg2); }
.r { text-align:right; }
.pos { color:var(--green); }
.neg { color:var(--red); }

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
  <h1>SIGNAL MONITOR v4.2</h1>
  <div class="meta">
    <span id="clock"></span>
    <span id="status">Loading...</span>
    <button class="btn" onclick="forceRefresh()">↻ REFRESH</button>
  </div>
</div>

<div class="tabs">
  <div class="tab active" onclick="setTab('signals')" data-tab="signals">Signals</div>
  <div class="tab" onclick="setTab('table')" data-tab="table">All Tickers</div>
  <div class="tab" onclick="setTab('brier')" data-tab="brier">Calibration</div>
  <div class="tab" onclick="setTab('heatmap')" data-tab="heatmap">Heatmap</div>
</div>

<div class="content" id="main"></div>

<script>
let D = null;
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

function render() {
  if (!D) return;
  const m = document.getElementById('main');
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === activeTab));
  if (activeTab === 'signals') m.innerHTML = renderSignals();
  else if (activeTab === 'table') m.innerHTML = renderTable();
  else if (activeTab === 'brier') m.innerHTML = renderBrier();
  else if (activeTab === 'heatmap') m.innerHTML = renderHeatmap();
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
    h += '<h3><span>Rolling Beta vs SPY</span><span id="beta-badge" class="beta-badge"></span></h3>';
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
        if (row.b63 > 2.0) note = 'HIGH LEVERAGE';
        else if (row.b63 > 1.5) note = 'Elevated';
        else if (row.b63 < 1.0) note = 'Diversified';
        else note = 'Moderate';
      } else if (row.name === 'MF Rotation' && row.b63 < -0.1) note = '✓ Negative';
      else if (row.name === 'GLD' && row.b63 != null && row.b252 != null && row.b63 > row.b252 + 0.3) note = '↑ Corr rising';
      h += `<td style="text-align:left;padding-left:12px;font-size:11px;color:var(--fg2)">${note}</td>`;
      h += '</tr>';
    }
    h += '</tbody></table>';
    // Blend status badge
    if (blendB63 !== null) {
      const bc = blendB63 > 2.0 ? 'var(--red)' : blendB63 > 1.5 ? 'var(--amber)' : blendB63 < 1.0 ? 'var(--green)' : 'var(--cyan)';
      const bl = blendB63 > 2.0 ? 'HIGH' : blendB63 > 1.5 ? 'ELEVATED' : blendB63 < 1.0 ? 'DIVERSIFIED' : 'OK';
      h += `<script>document.getElementById("beta-badge").textContent="${bl}";document.getElementById("beta-badge").style.background="${bc}22";document.getElementById("beta-badge").style.color="${bc}";</script>`;
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

  return h;
}

function renderTable() {
  const ind = D.indicators || {};
  const tickers = Object.keys(ind).sort((a,b) => ind[a].rsi - ind[b].rsi);
  let h = '<table><thead><tr><th>Ticker</th><th class="r">Price</th><th class="r">Chg</th><th class="r">RSI(10)</th><th>RSI</th><th class="r">vs SMA200</th><th class="r">SMA50</th></tr></thead><tbody>';
  for (const t of tickers) {
    const d = ind[t];
    const rc = d.rsi < 25 ? 'pos' : d.rsi > 79 ? 'neg' : '';
    const cc = d.chg1d >= 0 ? 'pos' : 'neg';
    const vc = d.vsSma200 != null ? (d.vsSma200 >= 0 ? 'pos' : 'neg') : '';
    const rsiColor = d.rsi > 79 ? 'var(--red)' : d.rsi > 70 ? 'var(--amber)' : d.rsi < 21 ? 'var(--green)' : d.rsi < 30 ? 'var(--cyan)' : 'var(--fg3)';
    h += `<tr>
      <td><b>${t}</b></td>
      <td class="r">$${d.price < 1000 ? d.price.toFixed(2) : Math.round(d.price).toLocaleString()}</td>
      <td class="r ${cc}">${d.chg1d >= 0 ? '+' : ''}${d.chg1d.toFixed(1)}%</td>
      <td class="r ${rc}">${d.rsi.toFixed(1)}</td>
      <td><div class="rsi-bar"><div class="rsi-fill" style="width:${d.rsi}%;background:${rsiColor}"></div></div></td>
      <td class="r ${vc}">${d.vsSma200 != null ? (d.vsSma200 >= 0 ? '+' : '') + d.vsSma200.toFixed(1) + '%' : '—'}</td>
      <td class="r">${d.sma50 > 0 ? '$'+d.sma50.toFixed(2) : '—'}</td>
    </tr>`;
  }
  h += '</tbody></table>';
  return h;
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
      <div class="brier-stats">
        <div class="brier-stat"><div class="label">Actual WR</div><div class="val" style="color:${wrColor}">${(b.actual_wr*100).toFixed(0)}%</div></div>
        <div class="brier-stat"><div class="label">Brier</div><div class="val" style="color:${brierColor}">${b.brier.toFixed(3)}</div></div>
        <div class="brier-stat"><div class="label">BSS</div><div class="val" style="color:${bssColor}">${b.bss > 0 ? '+' : ''}${b.bss.toFixed(3)}</div></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:11px;color:var(--fg2)">
        <span>Hist: ${(b.hist_wr*100).toFixed(0)}% | Trail${b.trail_n}: ${(b.trail_wr*100).toFixed(0)}%</span>
        <span>n=${b.n} | Tier ${b.tier}</span>
      </div>
      <div class="brier-recent">
        ${(b.recent||[]).map(e => `<span class="brier-ep ${e.win ? 'win' : 'loss'}">${e.win ? 'W' : 'L'} ${e.ret > 0 ? '+' : ''}${e.ret.toFixed(1)}%</span>`).join('')}
      </div>
    </div>`;
  }

  h += '</div>';
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
    'Style': ['VOOV','VOOG','VTV','QQQE','VOX'],
    'Intl': ['EDC','YINN','KORU','EURL','INDL'],
    'Other': ['AMD','NVDA','BTC-USD','FNGO','SLV','CPER','UUP'],
    'Gold/Miners': ['GLD','GDX','GDXJ','JNUG','NUGT'],
  };
  let h = '';
  for (const [name, tickers] of Object.entries(groups)) {
    const chips = tickers.filter(t => ind[t]).map(t => {
      const d = ind[t];
      const c = d.rsi >= 79 ? 'var(--red)' : d.rsi >= 70 ? 'var(--amber)' : d.rsi <= 21 ? 'var(--green)' : d.rsi <= 30 ? 'var(--cyan)' : 'var(--blue)';
      const int = d.rsi >= 79 || d.rsi <= 21 ? 0.6 : d.rsi >= 70 || d.rsi <= 30 ? 0.35 : 0.12;
      const a = Math.round(int*255).toString(16).padStart(2,'0');
      const fc = int > 0.3 ? '#fff' : c;
      return `<div class="hm-chip" style="background:${c}${a};color:${fc}" title="${t}: RSI=${d.rsi} | ${d.vsSma200!=null?(d.vsSma200>=0?'+':'')+d.vsSma200.toFixed(1)+'%':''}">
        ${t} <small>${d.rsi.toFixed(0)}</small></div>`;
    }).join('');
    if (chips) h += `<div style="margin-bottom:12px"><div style="color:var(--fg2);font-size:11px;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px">${name}</div><div class="hm">${chips}</div></div>`;
  }
  return h;
}

function setTab(t) { activeTab = t; render(); }
async function forceRefresh() {
  document.querySelector('.btn').textContent = '⏳';
  await fetch('/api/refresh');
  await fetchData();
  document.querySelector('.btn').textContent = '↻ REFRESH';
}

setInterval(() => {
  const c = document.getElementById('clock');
  if (c) c.textContent = new Date().toLocaleTimeString();
}, 1000);

setInterval(fetchData, 60000);
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
    print("  Signal Monitor — Dashboard Server v4.2")
    print("=" * 60)
    print(f"  Tickers: {len(TICKERS)}")
    print(f"  Refresh: every {CACHE_SECONDS}s")
    print(f"  Brier: recomputed every 10 min")
    print(f"  History: {HISTORY_PERIOD}")
    print()
    print("  → Open http://localhost:5050 in your browser")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5050, debug=False)
