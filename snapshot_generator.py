#!/usr/bin/env python3
"""
Real-Time Signal Snapshot Generator v4.0
=========================================
Generates a JSON snapshot of all signal conditions for consumption by Claude
and other tools. Matches signal_monitor_complete.py calculations exactly.

v4.0 ADDITIONS (March 2026):
- Oil Supply Shock (UCO/USO + USDU combo with tiered alerts)
- Dispersion Regime (sector RSI spread + defensive rotation)
- Commodity Playbook (PALL, PDBC manual hold alerts)
- KOLD Oil Reversal (XLE>79 + USDU amplifier)
- AND Combination Signals (UVXY+XLU, BTAL+QQQ, VIXM+SPY)
- Managed Futures dashboard (CTA/DBMF/KMLM)

USAGE:
  python snapshot_generator.py              # Full snapshot
  python snapshot_generator.py --compact    # Minimal output

OUTPUT:
  data/snapshot.json
"""

import os
import sys
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "snapshot.json"

TICKERS = [
    # Core Indices
    'SPY', 'QQQ', 'SMH', 'IWM',
    # Defensive Sectors
    'XLP', 'XLU', 'XLV', 'XLF', 'XLE', 'XLY',
    # Safe Havens & Macro
    'GLD', 'TLT', 'HYG', 'LQD', 'TMV', 'USDU', 'BND',
    # Commodities / Oil
    'UCO', 'USO', 'BOIL', 'DBC',
    # Commodity Playbook
    'PALL', 'PDBC',
    # Volatility
    'UVXY', 'SVXY', 'VIXY', 'VIXM',
    # 3x Leveraged ETFs
    'TQQQ', 'SOXL', 'SOXS', 'TECL', 'FAS', 'UPRO',
    'NAIL', 'CURE', 'LABU', 'DRN', 'FNGO', 'HIBL',
    # Inverse / Hedge
    'SQQQ',
    # International
    'EDC', 'YINN', 'KORU', 'EURL', 'INDL',
    # Crypto
    'BTC-USD',
    # Individual Stocks
    'AMD', 'NVDA',
    # Style/Factor
    'VOOV', 'VOOG', 'VTV', 'QQQE',
    # Managed Futures / Alternatives
    'KMLM', 'DBMF', 'CTA', 'BTAL',
    # Software / Dispersion
    'IGV',
]

SMH_LEVELS = {'trim': 30, 'warn': 35, 'sell': 40}

def calculate_rsi_wilder(prices, period):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def safe_float(value):
    if isinstance(value, pd.Series):
        return float(value.iloc[-1]) if len(value) > 0 else None
    elif isinstance(value, np.ndarray):
        return float(value[-1]) if len(value) > 0 else None
    elif pd.isna(value):
        return None
    else:
        return float(value)

def compute_indicators(df):
    if len(df) < 200:
        return None
    try:
        close = df['Close']
        price = safe_float(close.iloc[-1])
        prev_close = safe_float(close.iloc[-2]) if len(close) > 1 else price
        rsi10 = safe_float(calculate_rsi_wilder(close, 10).iloc[-1])
        sma50 = safe_float(close.rolling(50).mean().iloc[-1])
        sma200 = safe_float(close.rolling(200).mean().iloc[-1])
        ema9 = safe_float(close.ewm(span=9, adjust=False).mean().iloc[-1])
        ema20 = safe_float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = safe_float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        ema200 = safe_float(close.ewm(span=200, adjust=False).mean().iloc[-1])
        ret_1d = (price / prev_close - 1) * 100 if prev_close else None
        ret_5d = safe_float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) > 6 else None
        ret_10d = safe_float((close.iloc[-1] / close.iloc[-11] - 1) * 100) if len(close) > 11 else None
        ret_20d = safe_float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) > 21 else None
        vs_sma200 = ((price / sma200) - 1) * 100 if sma200 and sma200 > 0 else None
        vs_sma50 = ((price / sma50) - 1) * 100 if sma50 and sma50 > 0 else None
        vs_ema9 = ((price / ema9) - 1) * 100 if ema9 and ema9 > 0 else None
        vs_ema20 = ((price / ema20) - 1) * 100 if ema20 and ema20 > 0 else None
        ema_bull = ema9 > ema20 if (ema9 and ema20) else None
        return {
            'price': round(price, 2) if price else None,
            'change_pct': round(ret_1d, 2) if ret_1d else None,
            'rsi10': round(rsi10, 1) if rsi10 else None,
            'ema9': round(ema9, 2) if ema9 else None,
            'ema20': round(ema20, 2) if ema20 else None,
            'ema50': round(ema50, 2) if ema50 else None,
            'ema200': round(ema200, 2) if ema200 else None,
            'sma50': round(sma50, 2) if sma50 else None,
            'sma200': round(sma200, 2) if sma200 else None,
            'ema_cross': 'BULL' if ema_bull else ('BEAR' if ema_bull is not None else None),
            'above_sma200': price > sma200 if (price and sma200) else None,
            'vs_sma200': round(vs_sma200, 1) if vs_sma200 is not None else None,
            'vs_sma50': round(vs_sma50, 1) if vs_sma50 is not None else None,
            'vs_ema9': round(vs_ema9, 2) if vs_ema9 is not None else None,
            'vs_ema20': round(vs_ema20, 2) if vs_ema20 is not None else None,
            'ret_1d': round(ret_1d, 2) if ret_1d is not None else None,
            'ret_5d': round(ret_5d, 2) if ret_5d is not None else None,
            'ret_10d': round(ret_10d, 2) if ret_10d is not None else None,
            'ret_20d': round(ret_20d, 2) if ret_20d is not None else None,
            'above_ema9': price > ema9 if (price and ema9) else None,
            'above_ema20': price > ema20 if (price and ema20) else None,
            'above_ema50': price > ema50 if (price and ema50) else None,
        }
    except Exception as e:
        print(f"  Error computing indicators: {e}")
        return None

def evaluate_signals(indicators):
    signals = {}
    def rsi(ticker):
        return indicators.get(ticker, {}).get('rsi10')

    gld_rsi = rsi('GLD')
    usdu_rsi = rsi('USDU')
    xlp_rsi = rsi('XLP')
    spy_rsi = rsi('SPY')
    qqq_rsi = rsi('QQQ')
    smh_rsi = rsi('SMH')
    xlf_rsi = rsi('XLF')
    uvxy_rsi = rsi('UVXY')
    vixm_rsi = rsi('VIXM')

    # --- Playbook conditions ---
    signals['playbook'] = {
        'GLD_RSI_gt_79': {'value': gld_rsi, 'threshold': 79, 'active': gld_rsi > 79 if gld_rsi else False},
        'USDU_RSI_lt_25': {'value': usdu_rsi, 'threshold': 25, 'active': usdu_rsi < 25 if usdu_rsi else False},
        'XLP_RSI_gt_65': {'value': xlp_rsi, 'threshold': 65, 'active': xlp_rsi > 65 if xlp_rsi else False},
        'XLP_RSI_gt_75': {'value': xlp_rsi, 'threshold': 75, 'active': xlp_rsi > 75 if xlp_rsi else False},
        'SPY_RSI_gt_79': {'value': spy_rsi, 'threshold': 79, 'active': spy_rsi > 79 if spy_rsi else False},
        'QQQ_RSI_gt_79': {'value': qqq_rsi, 'threshold': 79, 'active': qqq_rsi > 79 if qqq_rsi else False},
        'SMH_RSI_gt_79': {'value': smh_rsi, 'threshold': 79, 'active': smh_rsi > 79 if smh_rsi else False},
        'XLF_RSI_gt_70': {'value': xlf_rsi, 'threshold': 70, 'active': xlf_rsi > 70 if xlf_rsi else False},
        'UVXY_RSI_gt_82': {'value': uvxy_rsi, 'threshold': 82, 'active': uvxy_rsi > 82 if uvxy_rsi else False},
        'VIXM_RSI_lt_25': {'value': vixm_rsi, 'threshold': 25, 'active': vixm_rsi < 25 if vixm_rsi else False},
    }

    # --- Combo signals ---
    gld_active = gld_rsi and gld_rsi > 79
    usdu_active = usdu_rsi and usdu_rsi < 25
    xlp_65_active = xlp_rsi and xlp_rsi > 65
    xlp_75_active = xlp_rsi and xlp_rsi > 75

    signals['combos'] = {
        'double_signal': {
            'active': bool(gld_active and usdu_active),
            'description': 'GLD RSI>79 + USDU RSI<25 → TQQQ buy',
            'components': {'GLD': gld_active, 'USDU': usdu_active},
        },
        'triple_signal': {
            'active': bool(gld_active and usdu_active and xlp_65_active),
            'description': 'Double + XLP RSI>65 → TQQQ high conviction',
            'components': {'GLD': gld_active, 'USDU': usdu_active, 'XLP': xlp_65_active},
        },
        'xlp_cascade': {
            'active': bool(xlp_75_active),
            'description': 'XLP RSI>75 → UVXY 1-day hold in Composer',
        },
    }

    # --- Bond momentum ---
    tlt_ind = indicators.get('TLT', {})
    bnd_ind = indicators.get('BND', {})
    tlt_ret10 = tlt_ind.get('ret_10d')
    bnd_ret10 = bnd_ind.get('ret_10d')
    bonds_rising = tlt_ret10 > 0 if tlt_ret10 is not None else None
    uvxy_conviction = None
    if bonds_rising is True:
        uvxy_conviction = 'MODERATE'
    elif bonds_rising is False:
        uvxy_conviction = 'HIGH'
    signals['bond_momentum'] = {
        'direction': 'RISING' if bonds_rising else ('FALLING' if bonds_rising is False else 'UNKNOWN'),
        'tlt_ret_10d': round(tlt_ret10, 2) if tlt_ret10 is not None else None,
        'bnd_ret_10d': round(bnd_ret10, 2) if bnd_ret10 is not None else None,
        'uvxy_conviction': uvxy_conviction,
    }

    # --- SMH / SOXL levels ---
    smh_ind = indicators.get('SMH', {})
    smh_vs200 = smh_ind.get('vs_sma200')
    smh_sma200 = smh_ind.get('sma200')
    smh_price = smh_ind.get('price')
    signals['smh_levels'] = {
        'price': smh_price, 'sma200': smh_sma200,
        'pct_above': round(smh_vs200, 1) if smh_vs200 is not None else None,
        'trim_level': round(smh_sma200 * 1.30, 2) if smh_sma200 else None,
        'warn_level': round(smh_sma200 * 1.35, 2) if smh_sma200 else None,
        'sell_level': round(smh_sma200 * 1.40, 2) if smh_sma200 else None,
    }

    # --- Contrarian weakness setups ---
    contrarian = {}
    for ticker in ['FAS', 'TECL', 'FNGO', 'LABU', 'NAIL']:
        ind = indicators.get(ticker, {})
        r = ind.get('rsi10')
        below_200 = ind.get('above_sma200') is False
        bear_ema = ind.get('ema_cross') == 'BEAR'
        if r is not None and below_200 and r < 40:
            status = 'ACTIVE'
        elif r is not None and below_200 and r < 50:
            status = 'WATCH'
        else:
            status = 'INACTIVE'
        contrarian[ticker] = {'rsi10': r, 'below_sma200': below_200, 'bear_ema': bear_ema, 'status': status}
    signals['contrarian'] = contrarian

    # --- Extended / overbought warnings ---
    extended = {}
    for ticker in ['SOXL', 'KORU', 'EDC', 'HIBL', 'LABU', 'SMH']:
        ind = indicators.get(ticker, {})
        vs200 = ind.get('vs_sma200')
        r = ind.get('rsi10')
        if vs200 is not None and vs200 > 50:
            extended[ticker] = {'vs_sma200': vs200, 'rsi10': r, 'warning': 'EXTENDED' if vs200 > 100 else 'ELEVATED'}
    signals['extended'] = extended

    # =====================================================================
    # v4.0 ADDITIONS
    # =====================================================================

    # --- Oil Supply Shock ---
    uco_rsi_val = rsi('UCO')
    uso_rsi_val = rsi('USO')
    usdu_rsi_val = rsi('USDU')

    oil_signal_level = 'INACTIVE'
    oil_hedge_rec = ''
    if uco_rsi_val and usdu_rsi_val:
        if uco_rsi_val >= 85 and usdu_rsi_val > 60:
            oil_signal_level = 'EXTREME'
            oil_hedge_rec = '33% UVXY / 33% BTAL / 33% SQQQ (5d hold). 3/3 episodes, +7.78% avg'
        elif uco_rsi_val > 82 and usdu_rsi_val > 55:
            oil_signal_level = 'STRONG'
            oil_hedge_rec = 'Avoid adding TQQQ/SOXL/UPRO. Consider hedge basket.'
        elif uco_rsi_val > 79 and usdu_rsi_val > 55:
            oil_signal_level = 'WARNING'
            oil_hedge_rec = 'Caution on leveraged equity adds. Monitor for escalation.'
        elif uco_rsi_val > 75 and usdu_rsi_val > 50:
            oil_signal_level = 'WATCH'
            oil_hedge_rec = 'Approaching supply shock zone.'
        elif uco_rsi_val > 79 and usdu_rsi_val < 40:
            oil_signal_level = 'DEMAND_BULLISH'
            oil_hedge_rec = 'Oil up + Dollar weak = demand-driven. QQQ 85% WR 5d.'

    signals['oil_supply_shock'] = {
        'signal_level': oil_signal_level,
        'uco_rsi': uco_rsi_val,
        'uso_rsi': uso_rsi_val,
        'usdu_rsi': usdu_rsi_val,
        'hedge_recommendation': oil_hedge_rec,
        'duration_playbook': {
            'gte_1d': {'spy_wr': '30%', 'btal_wr': '78%', 'uvxy_avg': '+3.93%'},
            'gte_2d': {'spy_wr': '8%', 'btal_wr': '92%', 'uvxy_avg': '+9.34%'},
            'gte_3d': {'spy_wr': '11%', 'btal_wr': '100%', 'uvxy_avg': '+9.95%'},
        },
        'exit_rule': 'Cut when UCO RSI <79 or basket negative after D+2',
    }

    # --- Dispersion Regime ---
    sector_rsis = {}
    for t in ['XLP', 'XLU', 'XLV', 'XLF', 'XLE', 'XLY', 'SMH']:
        r = rsi(t)
        if r is not None:
            sector_rsis[t] = r

    disp_range = None
    disp_leader = disp_laggard = None
    smh_xlp_gap_val = None
    def_rotation = False

    if len(sector_rsis) >= 5:
        rsi_vals = list(sector_rsis.values())
        disp_range = round(max(rsi_vals) - min(rsi_vals), 1)
        disp_leader = max(sector_rsis, key=sector_rsis.get)
        disp_laggard = min(sector_rsis, key=sector_rsis.get)
        if 'SMH' in sector_rsis and 'XLP' in sector_rsis:
            smh_xlp_gap_val = round(sector_rsis['SMH'] - sector_rsis['XLP'], 1)
        if sector_rsis.get('XLP', 0) > 65 and sector_rsis.get('SMH', 100) < 50:
            def_rotation = True

    signals['dispersion'] = {
        'sector_rsi_range': disp_range,
        'leader': f"{disp_leader} ({sector_rsis.get(disp_leader, '')})" if disp_leader else None,
        'laggard': f"{disp_laggard} ({sector_rsis.get(disp_laggard, '')})" if disp_laggard else None,
        'smh_xlp_gap': smh_xlp_gap_val,
        'extreme_dispersion': bool(disp_range and disp_range > 45),
        'defensive_rotation_active': def_rotation,
        'sector_rsis': {k: round(v, 1) for k, v in sector_rsis.items()},
    }

    # --- IGV Mean Reversion ---
    igv_rsi_val = rsi('IGV')
    signals['igv_mean_reversion'] = {
        'rsi10': igv_rsi_val,
        'active': bool(igv_rsi_val and igv_rsi_val < 25),
        'description': 'IGV RSI<25: 83% win 10d, +4.96% avg | n=12',
    }

    # --- Commodity Playbook ---
    pall_ind = indicators.get('PALL', {})
    pdbc_ind = indicators.get('PDBC', {})
    pall_rsi_val = rsi('PALL')
    pdbc_rsi_val = rsi('PDBC')
    pall_above = pall_ind.get('above_sma200', False)
    pdbc_above = pdbc_ind.get('above_sma200', False)

    signals['commodity_playbook'] = {
        'PALL': {
            'rsi10': pall_rsi_val, 'price': pall_ind.get('price'),
            'above_sma200': pall_above, 'vs_sma200': pall_ind.get('vs_sma200'),
            'signal': 'BUY_10D' if (pall_rsi_val and pall_rsi_val < 25 and pall_above) else
                      'WATCH' if (pall_rsi_val and pall_rsi_val < 30 and pall_above) else 'INACTIVE',
        },
        'PDBC': {
            'rsi10': pdbc_rsi_val, 'price': pdbc_ind.get('price'),
            'above_sma200': pdbc_above, 'vs_sma200': pdbc_ind.get('vs_sma200'),
            'signal': 'BUY_10D' if (pdbc_rsi_val and pdbc_rsi_val < 30 and pdbc_above) else 'INACTIVE',
        },
    }

    # --- KOLD Oil Reversal ---
    xle_rsi_val = rsi('XLE')
    xlf_rsi_v = rsi('XLF')
    kold_active = bool(xle_rsi_val and xle_rsi_val > 79)
    usdu_amp = bool(usdu_rsi_val and usdu_rsi_val > 60)

    signals['kold_signal'] = {
        'xle_rsi': xle_rsi_val,
        'active': kold_active,
        'usdu_amplifier': usdu_amp,
        'cyclical_euphoria_top': bool(kold_active and xlf_rsi_v and xlf_rsi_v > 79),
    }

    # --- AND Combination Signals ---
    xlu_rsi_val = rsi('XLU')
    btal_rsi_val = rsi('BTAL')

    signals['and_combos'] = {
        'crisis_recovery': {
            'active': bool(uvxy_rsi and uvxy_rsi > 79 and xlu_rsi_val and xlu_rsi_val < 30),
            'uvxy_rsi': uvxy_rsi, 'xlu_rsi': xlu_rsi_val,
            'action': 'SOXL: 91.7% WR, +13.11% avg | 6 episodes',
        },
        'quality_to_growth': {
            'active': bool(btal_rsi_val and btal_rsi_val > 75 and qqq_rsi and qqq_rsi < 30),
            'btal_rsi': btal_rsi_val, 'qqq_rsi': qqq_rsi,
            'action': 'SOXL: n=40, 18 episodes',
        },
        'calm_market_hedge': {
            'active': bool(vixm_rsi and vixm_rsi < 30 and spy_rsi and spy_rsi > 79),
            'vixm_rsi': vixm_rsi, 'spy_rsi': spy_rsi,
            'action': 'UVXY: calm vol + overbought = snap-back risk',
        },
    }

    # --- Managed Futures Dashboard ---
    mf_data = {}
    for t in ['CTA', 'DBMF', 'KMLM', 'BTAL']:
        ind = indicators.get(t, {})
        mf_data[t] = {
            'price': ind.get('price'), 'rsi10': ind.get('rsi10'),
            'vs_sma200': ind.get('vs_sma200'), 'ema_cross': ind.get('ema_cross'),
        }
    signals['managed_futures'] = mf_data

    # --- Active alerts summary ---
    active_alerts = []
    if signals['combos']['double_signal']['active']:
        active_alerts.append('🟢🔥 DOUBLE SIGNAL: GLD/USDU → TQQQ buy')
    if signals['combos']['triple_signal']['active']:
        active_alerts.append('🟢🔥🔥 TRIPLE SIGNAL: GLD/USDU/XLP → TQQQ high conviction')
    if signals['combos']['xlp_cascade']['active']:
        active_alerts.append('🟡 XLP CASCADE: RSI>75 → UVXY 1-day hold')
    for ticker, c in contrarian.items():
        if c['status'] == 'ACTIVE':
            active_alerts.append(f'🟢 {ticker} CONTRARIAN: RSI<40 + below SMA200')
        elif c['status'] == 'WATCH':
            active_alerts.append(f'⚠️ {ticker} WATCH: below SMA200, approaching oversold')
    for ticker, e in extended.items():
        active_alerts.append(f'🔴 {ticker} {e["warning"]}: {e["vs_sma200"]:+.0f}% above SMA200')

    # v4 alerts
    if oil_signal_level == 'EXTREME':
        active_alerts.append(f'🔴🔴 OIL SUPPLY SHOCK EXTREME: UCO RSI={uco_rsi_val:.0f} + USDU RSI={usdu_rsi_val:.0f} → HEDGE NOW')
    elif oil_signal_level == 'STRONG':
        active_alerts.append(f'🔴 OIL SUPPLY SHOCK STRONG: UCO RSI={uco_rsi_val:.0f} + USDU RSI={usdu_rsi_val:.0f}')
    elif oil_signal_level == 'WARNING':
        active_alerts.append(f'🟡 OIL SUPPLY WARNING: UCO RSI={uco_rsi_val:.0f} + USDU RSI={usdu_rsi_val:.0f}')
    elif oil_signal_level == 'DEMAND_BULLISH':
        active_alerts.append(f'🟢 OIL DEMAND (BULLISH): UCO RSI={uco_rsi_val:.0f} + weak dollar')

    if signals['dispersion']['extreme_dispersion']:
        active_alerts.append(f'🟡 EXTREME DISPERSION: Range {disp_range:.0f}')
    if signals['dispersion']['defensive_rotation_active']:
        active_alerts.append(f'🟡 DEFENSIVE ROTATION: XLP>65 + SMH<50')
    if signals['igv_mean_reversion']['active']:
        active_alerts.append(f'🟢 IGV MEAN REVERSION: RSI={igv_rsi_val:.0f} <25')
    if signals['commodity_playbook']['PALL']['signal'] == 'BUY_10D':
        active_alerts.append(f'🟢 PALL OVERSOLD (10d hold): RSI={pall_rsi_val:.0f}')
    if signals['commodity_playbook']['PDBC']['signal'] == 'BUY_10D':
        active_alerts.append(f'🟢 PDBC PULLBACK (10d hold): RSI={pdbc_rsi_val:.0f}')
    if kold_active:
        amp = ' + USDU amplifier' if usdu_amp else ''
        active_alerts.append(f'🟡 KOLD SIGNAL: XLE RSI={xle_rsi_val:.0f}{amp}')
    if signals['kold_signal'].get('cyclical_euphoria_top'):
        active_alerts.append(f'🔴 CYCLICAL EUPHORIA TOP: XLE + XLF both >79')
    for combo_name, combo in signals['and_combos'].items():
        if combo['active']:
            active_alerts.append(f'🟢🔥 {combo_name.upper()}: {combo["action"]}')

    signals['active_alerts'] = active_alerts
    return signals


def main():
    compact = '--compact' in sys.argv
    print(f"Generating signal snapshot v4.0 at {datetime.now()}")
    print(f"Downloading data for {len(TICKERS)} tickers...")

    data = {}
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, period='2y', progress=False)
            if len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
        except Exception as e:
            print(f"  Error downloading {ticker}: {e}")

    print(f"Downloaded data for {len(data)} tickers")

    indicators = {}
    for ticker, df in data.items():
        ind = compute_indicators(df)
        if ind:
            indicators[ticker] = ind
    print(f"Computed indicators for {len(indicators)} tickers")

    signals = evaluate_signals(indicators)

    now_utc = datetime.now(timezone.utc)
    et_offset = timedelta(hours=-5)
    now_et = now_utc + et_offset

    snapshot = {
        'meta': {
            'generated_utc': now_utc.isoformat(),
            'generated_et': now_et.strftime('%Y-%m-%d %H:%M:%S ET'),
            'ticker_count': len(indicators),
            'version': '4.0',
        },
        'signals': signals,
    }

    if not compact:
        snapshot['indicators'] = indicators
    else:
        key_set = {'SPY', 'QQQ', 'SMH', 'GLD', 'USDU', 'XLP', 'TLT', 'UVXY',
                    'SVXY', 'VIXM', 'TQQQ', 'SOXL', 'UPRO', 'FAS', 'TECL',
                    'FNGO', 'KMLM', 'BTAL', 'BND', 'BTC-USD',
                    'UCO', 'USO', 'PALL', 'PDBC', 'IGV', 'XLE', 'XLY'}
        snapshot['indicators'] = {k: v for k, v in indicators.items() if k in key_set}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(snapshot, f, indent=2, default=str)

    print(f"\nSnapshot written to {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size:,} bytes")

    print(f"\n{'='*60}")
    print(f"SNAPSHOT SUMMARY — {now_et.strftime('%Y-%m-%d %H:%M ET')}")
    print(f"{'='*60}")

    if signals['active_alerts']:
        for alert in signals['active_alerts']:
            print(f"  {alert}")
    else:
        print("  No active signals")

    pb = signals['playbook']
    print(f"\n  Playbook Conditions:")
    for key, val in pb.items():
        status = '🟢' if val['active'] else '○'
        print(f"    {status} {key}: {val['value']}")

    bm = signals['bond_momentum']
    print(f"\n  Bond Momentum: {bm['direction']} (TLT 10d: {bm['tlt_ret_10d']}%)")
    print(f"  UVXY Conviction: {bm['uvxy_conviction']}")

    oil = signals['oil_supply_shock']
    print(f"\n  Oil Supply Shock: {oil['signal_level']} (UCO RSI={oil['uco_rsi']}, USDU RSI={oil['usdu_rsi']})")
    if oil['hedge_recommendation']:
        print(f"    → {oil['hedge_recommendation']}")

    disp = signals['dispersion']
    print(f"  Dispersion: range={disp['sector_rsi_range']} leader={disp['leader']} laggard={disp['laggard']}")


if __name__ == '__main__':
    main()
