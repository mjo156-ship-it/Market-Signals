#!/usr/bin/env python3
"""
Comprehensive Market Signal Monitor v3.0
==========================================
Email generator that produces the full-content signal report restored from the
4/21/2026 baseline, plus additions:
- Composer Rotation Forecast (NEW): predicts which leaf each symphony will trade into
- VIX Term Structure bounce/pullback zones (Group 13, see group13 module)
- All restored sections from v2.x: Intramonth Cycle, Mid-Month, Crisis Alpha,
  Rolling Beta, GLD & Miners, UVXY Vol Regime, FXY, CPER, DRIF, MOVE, Fibonacci,
  Portfolio Performance & Win Rates per account.

DATA STRATEGY
-------------
Tries 3 data sources in order of preference:
  1. Live dashboard at DASHBOARD_URL (default localhost:5052) — full data, fastest
  2. Local import of chf_dashboard_server (fetch_all() directly) — same data, slower
  3. Standalone yfinance fallback — partial data only (basic indicators)

Run with:
    DASHBOARD_URL=http://localhost:5052 python signal_monitor_v3.py [preclose]

ENV VARS (existing):
    SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL, PHONE_EMAIL
"""

import os
import sys
import json
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
import numpy as np
import requests as req_lib

# Local module — must be co-located
try:
    from composer_dry_run import (
        fetch_dry_run_preview,
        parse_dry_run_response,
        format_dry_run_for_email,
    )
    DRY_RUN_AVAILABLE = True
except ImportError:
    print("WARN: composer_dry_run not importable; rebalance preview disabled")
    DRY_RUN_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════
SENDER_EMAIL = os.environ.get('SENDER_EMAIL', '')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD', '')
RECIPIENT_EMAIL = os.environ.get('RECIPIENT_EMAIL', '')
PHONE_EMAIL = os.environ.get('PHONE_EMAIL', '')
DASHBOARD_URL = os.environ.get('DASHBOARD_URL', 'http://localhost:5052')

IS_PRECLOSE = len(sys.argv) > 1 and sys.argv[1] == 'preclose'

# Emoji shortcuts (keep in one place — these were corrupted in v2.x)
GREEN = "\U0001F7E2"   # 🟢
RED = "\U0001F534"     # 🔴
YELLOW = "\U0001F7E1"  # 🟡
SHIP = "\U0001F6A2"    # 🚢
CHART = "\U0001F4CA"   # 📊
COMPASS = "\U0001F9ED" # 🧭


# ════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════
def load_dashboard_data():
    """Try the 3 data sources in order. Returns (data_dict, raw_data, source_label)."""
    # 1. Live dashboard via HTTP — use /api/email-state which bundles
    #    indicators + signals + composer portfolio + dry-run preview
    try:
        r = req_lib.get(f"{DASHBOARD_URL}/api/email-state", timeout=30)
        if r.status_code == 200:
            data = r.json()
            print(f"Loaded data from dashboard at {DASHBOARD_URL}/api/email-state")
            return data, None, 'dashboard-email-state'
    except Exception as e:
        print(f"Dashboard /api/email-state unavailable ({e}); trying /api/data")

    # 1b. Older dashboards may not have /api/email-state — try /api/data + /api/composer
    try:
        r = req_lib.get(f"{DASHBOARD_URL}/api/data", timeout=30)
        if r.status_code == 200:
            data = r.json()
            print(f"Loaded data from {DASHBOARD_URL}/api/data (legacy endpoint)")
            # Try to add composer portfolio + dry-run separately
            try:
                cr = req_lib.get(f"{DASHBOARD_URL}/api/composer", timeout=30)
                if cr.status_code == 200:
                    data['composer'] = cr.json()
            except Exception:
                pass
            try:
                dr = req_lib.get(f"{DASHBOARD_URL}/api/composer-dry-run", timeout=30)
                if dr.status_code == 200:
                    dr_data = dr.json()
                    # Reconstruct raw response from parsed rotations isn't easy,
                    # so pass the parsed structure under a new key the formatter reads
                    data['composer_dry_run_parsed'] = dr_data
            except Exception:
                pass
            return data, None, 'dashboard-http-legacy'
    except Exception as e:
        print(f"Dashboard HTTP unavailable ({e}); trying local import")

    # 2. Local import
    try:
        from chf_dashboard_server import fetch_all
        result = fetch_all()
        if isinstance(result, tuple) and len(result) == 2:
            data, raw = result
        else:
            data = result
            raw = None
        print("Loaded data via direct import of chf_dashboard_server")
        return data, raw, 'dashboard-local'
    except Exception as e:
        print(f"Local import unavailable ({e}); falling back to direct fetches")

    # 3. Standalone fallback — basic indicators + direct Composer dry-run + Hormuz
    return _fallback_full(), None, 'standalone-fallback'


def _fallback_full():
    """
    Standalone fallback used when the dashboard is unreachable.
    Pulls what it can directly:
      - Basic indicators from yfinance (universe of ~50 tickers)
      - Hormuz transit data from public Windward snapshot URL (if available)
      - Composer dry-run preview from API (if COMPOSER_KEY_ID is set)
      - Composer portfolio (account-level summary) from /accounts endpoint

    What this can't reconstruct without the dashboard:
      - calendar_cycle, midmonth, breadth_regime, rolling_betas, drif,
        move_index, fibonacci, uvxy_vol_regime, vix_term_structure
      - Per-symphony Sharpe/MaxDD/CAGR (those need the dashboard's portfolio history)

    Those sections will simply not appear in the email — better to omit than to
    fabricate placeholder data.
    """
    data = {'signals': []}

    # ── A. yfinance basic indicators ──
    data['indicators'] = _fetch_yfinance_indicators()

    # ── B. Hormuz from public Windward snapshot (if available) ──
    try:
        hormuz = _fetch_hormuz_snapshot()
        if hormuz:
            data['hormuz'] = hormuz
    except Exception as e:
        print(f"  Hormuz fetch failed: {e}")

    # ── C. Composer dry-run + portfolio (if credentials available) ──
    if os.environ.get('COMPOSER_KEY_ID'):
        try:
            from composer_dry_run import (fetch_dry_run_preview,
                                           parse_dry_run_response,
                                           aggregate_net_trade_flow,
                                           compute_concentration_warnings)
            # Need account UUIDs first — pull from /accounts
            portfolio = _fetch_composer_portfolio_basic()
            if portfolio:
                data['composer'] = portfolio
                account_uuids = [a.get('id') or a.get('account_uuid')
                                  for a in portfolio.get('accounts', [])
                                  if (a.get('id') or a.get('account_uuid'))]
                if account_uuids:
                    raw = fetch_dry_run_preview(account_uuids=account_uuids)
                    if raw:
                        data['composer_dry_run'] = raw
        except Exception as e:
            print(f"  Composer fallback failed: {e}")
    else:
        print("  COMPOSER_KEY_ID not set — skipping dry-run preview in fallback")

    return data


def _fetch_yfinance_indicators():
    """Basic yfinance fetch — price, RSI(10), SMA200 for ~50 tickers.

    Returns dict in the same shape fetch_all() produces (uses `rsi`,
    `pctAboveSma200`, `vsSma200`, etc.) so downstream formatters work
    identically across data sources."""
    import yfinance as yf

    tickers = [
        'SMH','SPY','QQQ','IWM','XLP','XLU','XLV','XLE','XLF',
        'GLD','TLT','HYG','LQD','TMV','USDU','UCO','BOIL',
        'UVXY','VIXM','BTC-USD','AMD','NVDA',
        'NAIL','CURE','FAS','LABU','TQQQ','SOXL','TECL','DRN',
        'VOOV','VOOG','VTV','QQQE','EURL','YINN','KORU','INDL','EDC',
        'BTAL','DBMF','KMLM','CTA','CPER','SPHB','MNA','GDX','GDXJ','JNUG','NUGT'
    ]
    indicators = {}
    for t in tickers:
        try:
            df = yf.download(t, period='1y', progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) < 200:
                continue
            close = df['Close']
            price = float(close.iloc[-1])
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            ag = gain.ewm(alpha=1/10, adjust=False).mean()
            al = loss.ewm(alpha=1/10, adjust=False).mean()
            rs = ag / al
            rsi = float((100 - 100/(1+rs)).iloc[-1])
            sma200 = float(close.rolling(200).mean().iloc[-1])
            sma50 = float(close.rolling(50).mean().iloc[-1])
            pct_above = (price/sma200 - 1) * 100 if sma200 > 0 else 0
            indicators[t] = {
                'price': round(price, 2),
                'rsi': round(rsi, 1),
                'sma200': round(sma200, 2),
                'sma50': round(sma50, 2),
                'pctAboveSma200': round(pct_above, 1),
                'vsSma200': round(pct_above, 1),
            }
        except Exception:
            continue
    return indicators


def _fetch_hormuz_snapshot():
    """
    Fetch Hormuz transit data from the public GitHub snapshot.
    Returns a dict in the shape the email formatter expects, or None.

    The dashboard server's compute_hormuz() function reads the same snapshot
    and adds derived fields. This is a simplified version for the fallback path.
    """
    snapshot_url = os.environ.get(
        'HORMUZ_SNAPSHOT_URL',
        'https://raw.githubusercontent.com/mjo156-ship-it/Market-Signals/refs/heads/main/data/snapshot.json'
    )
    try:
        r = req_lib.get(snapshot_url, timeout=15)
        if r.status_code != 200:
            return None
        snap = r.json()
        # Try common nesting patterns — the snapshot schema may have varied
        h = (snap.get('hormuz')
             or snap.get('signals', {}).get('hormuz')
             or snap.get('signals', {}).get('hormuz_windward'))
        if not h:
            return None

        # Normalize to the dict shape fmt_hormuz expects
        return {
            'latest_date': h.get('latest_date') or h.get('date'),
            'blockade_active': h.get('blockade_active', False),
            'transits': h.get('transits') or {},
            'vessels_in_gulf': h.get('vessels_in_gulf', '—'),
            'dark_activity': h.get('dark_activity', '—'),
            'dark_change_str': h.get('dark_change_str', ''),
            'attacks_total': h.get('attacks_total', '—'),
            'attacks_change_str': h.get('attacks_change_str', 'unchanged'),
            'iran_flagged': h.get('iran_flagged', '—'),
            'risk_tiers_str': h.get('risk_tiers_str', '—'),
            'summary': h.get('summary') or h.get('daily_summary'),
        }
    except Exception as e:
        print(f"  Hormuz snapshot fetch error: {e}")
        return None


def _fetch_composer_portfolio_basic():
    """
    Fetch the user's account list from Composer's /accounts endpoint, returning
    the minimal portfolio dict needed for dry-run filtering and totals.

    Returns dict like {'accounts': [{'account_uuid': str, 'account_type': str,
                                      'portfolio_value': float}, ...]} or None.
    """
    key_id = os.environ.get('COMPOSER_KEY_ID', '')
    key_secret = os.environ.get('COMPOSER_KEY_SECRET', '')
    if not (key_id and key_secret):
        return None
    base = 'https://api.composer.trade/api/v0.1'
    try:
        r = req_lib.get(
            f"{base}/accounts",
            headers={'x-api-key-id': key_id,
                      'authorization': f'Bearer {key_secret}'},
            timeout=15,
        )
        r.raise_for_status()
        accts_raw = r.json()
    except Exception as e:
        print(f"  Composer /accounts error: {e}")
        return None

    # /accounts returns a list with broker_account_uuid, broker, account_type, etc.
    accounts = []
    for a in (accts_raw if isinstance(accts_raw, list) else accts_raw.get('accounts', [])):
        uuid = a.get('broker_account_uuid') or a.get('account_uuid')
        if not uuid:
            continue

        # Fetch the holdings to get portfolio_value
        portfolio_value = 0
        try:
            hr = req_lib.get(
                f"{base}/accounts/{uuid}/holdings",
                headers={'x-api-key-id': key_id,
                          'authorization': f'Bearer {key_secret}'},
                timeout=15,
            )
            if hr.status_code == 200:
                hjson = hr.json()
                portfolio_value = (hjson.get('total_value')
                                    or hjson.get('account_value')
                                    or sum(h.get('market_value', 0)
                                            for h in hjson.get('holdings', [])))
        except Exception:
            pass

        accounts.append({
            'account_uuid': uuid,
            'account_type': a.get('account_type') or a.get('account_name') or 'Account',
            'account_name': a.get('account_type') or a.get('account_name') or 'Account',
            'portfolio_value': portfolio_value,
        })

    return {'accounts': accounts} if accounts else None


def _ind_get(ind, *keys, default=0):
    """Safe lookup across possible key spellings.
    Dashboard uses: 'rsi', 'pctAboveSma200', 'sma200', 'cumRet5d', etc.
    Fallback yfinance builder uses: 'rsi10', 'pct_above_sma200', 'sma200'.
    This helper tries each key in order until one returns a non-None value."""
    if not ind:
        return default
    for k in keys:
        v = ind.get(k)
        if v is not None:
            return v
    return default


def _rsi(ind):
    return _ind_get(ind, 'rsi', 'rsi10', default=0)


def _pct_above_sma200(ind):
    return _ind_get(ind, 'pctAboveSma200', 'pct_above_sma200', 'vsSma200', default=0)


# ════════════════════════════════════════════════════════════════════
# SECTION FORMATTERS
# ════════════════════════════════════════════════════════════════════
def fmt_header(title, char='='):
    return f"\n{char*70}\n{title}\n{char*70}\n"


def fmt_calendar_banner(data):
    """Top banner: intramonth cycle position with dip-buy boost note.

    Reads from data['calendar_cycle'] which has keys:
      trading_days_left, zone ('early'/'window'/'late'), desc, color,
      in_window, dip_buy_boost, window_dates, month.
    """
    cc = data.get('calendar_cycle')
    if not cc:
        return ''

    desc = cc.get('desc') or ''
    if not desc:
        return ''

    out = f"\n{YELLOW} INTRAMONTH CYCLE: {desc}\n"
    if cc.get('in_window') or cc.get('dip_buy_boost'):
        out += "  Dip-buy conviction: BOOSTED (buying forced institutional selling)\n"
        out += "  TQQQ avg +0.08%/day in window vs +0.26%/day outside\n"
    return out


def fmt_alerts(data):
    """Buy / Exit / Warning alerts from signals list."""
    signals = data.get('signals', [])
    if not signals:
        return "No signals triggered today.\n\n"

    buy = [s for s in signals if s.get('type') == 'buy']
    exit_ = [s for s in signals if s.get('type') in ('exit', 'short')]
    warn = [s for s in signals if s.get('type') in ('warning', 'hedge', 'watch', 'OB ALERT')]

    out = ''
    if buy:
        out += f"{GREEN} BUY SIGNALS:\n" + "-"*50 + "\n"
        for s in buy:
            out += f"{s.get('title','')}\n{s.get('msg','')}\n\n"
    if exit_:
        out += f"{RED} EXIT/SHORT SIGNALS:\n" + "-"*50 + "\n"
        for s in exit_:
            out += f"{s.get('title','')}\n{s.get('msg','')}\n\n"
    if warn:
        out += f"{YELLOW} WARNINGS/WATCH:\n" + "-"*50 + "\n"
        for s in warn:
            out += f"{s.get('title','')}\n{s.get('msg','')}\n\n"
    return out


def fmt_hormuz(data):
    """Strait of Hormuz Windward intelligence header.

    data['hormuz'] shape:
      {
        'current': int (total transits today),
        'baseline': int (138),
        'updated': str date,
        'source': str,
        'windward': {
          'vessels_in_gulf': {'value', 'sub', 'delta'},
          'inbound': {'value', 'sub'},
          'outbound': {'value', 'sub'},
          'dark_activity': {'value', 'sub', 'delta'},
          'attacks': {'value', 'sub'},
          'total_transits': int,
          'intel_summary': str,
          'risk': {...},
          'fleet': {...},
          'flags': [{flag, count}, ...],
          'blockade_active': bool,
          'iran_flagged': int,
          'as_of': str,
        }
      }
    """
    h = data.get('hormuz')
    if not h:
        return ''
    w = h.get('windward') or {}

    blockade = w.get('blockade_active', False)
    status_emoji = RED if blockade else GREEN
    status_label = "BLOCKADE ACTIVE" if blockade else "OPERATIONAL"
    as_of = w.get('as_of') or h.get('updated', 'N/A')

    out = fmt_header(f"{SHIP} STRAIT OF HORMUZ — WINDWARD INTELLIGENCE")
    out += f"Data as of: {as_of}   |   {status_emoji} {status_label}\n\n"

    # Transits
    inbound = w.get('inbound') or {}
    outbound = w.get('outbound') or {}
    total_t = w.get('total_transits') or h.get('current')
    if total_t is not None:
        breakdown = ''
        if inbound.get('value') is not None or outbound.get('value') is not None:
            in_v = inbound.get('value', 0) or 0
            out_v = outbound.get('value', 0) or 0
            breakdown = f' ({in_v} in / {out_v} out)'
        out += f" Total transits:   {total_t}{breakdown}   (baseline ~{h.get('baseline', 138)})\n"

    # Vessels in Gulf
    vig = w.get('vessels_in_gulf') or {}
    if vig.get('value') is not None:
        sub = vig.get('sub') or ''
        delta = vig.get('delta') or ''
        suffix = f" {sub}".strip() if sub else ''
        suffix += f" ({delta})" if delta else ''
        out += f" Vessels in Gulf:  {vig['value']}{suffix}\n"

    # Dark activity
    dark = w.get('dark_activity') or {}
    if dark.get('value') is not None:
        delta = dark.get('delta') or ''
        suffix = f" ({delta})" if delta else ''
        out += f" Dark activity:    {dark['value']}{suffix}\n"

    # Attacks
    attacks = w.get('attacks') or {}
    if attacks.get('value') is not None:
        sub = attacks.get('sub') or ''
        suffix = f" ({sub})" if sub else ''
        out += f" Attacks (total):  {attacks['value']}{suffix}\n"

    # Iran-flagged
    iran = w.get('iran_flagged')
    if iran is not None:
        out += f" Iran-flagged:     {iran}\n"

    # Risk tiers
    risk = w.get('risk') or {}
    if risk:
        # risk is a dict like {'high': N, 'moderate': M, 'low': L}
        parts = []
        if 'high' in risk: parts.append(f"{risk['high']} High")
        if 'moderate' in risk: parts.append(f"{risk['moderate']} Mod")
        if 'low' in risk: parts.append(f"{risk['low']} Low")
        if parts:
            out += f" Risk tiers:       {' / '.join(parts)}\n"

    # Intel summary
    summary = w.get('intel_summary')
    if summary:
        out += f"\n Daily Intelligence Summary:\n   {summary}\n"
    out += f"\n Source: {h.get('source', 'insights.windward.ai')}\n"
    return out


def fmt_indicators_table(data, tickers, title):
    indicators = data.get('indicators', {})
    out = fmt_header(title)
    out += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}\n"
    out += "-" * 50 + "\n"
    for t in tickers:
        ind = indicators.get(t)
        if not ind:
            continue
        price = ind.get('price', 0)
        price_str = f"${price:,.0f}" if price >= 1000 else f"${price:.2f}"
        rsi = _rsi(ind)
        pct = _pct_above_sma200(ind)
        out += f"{t:<10} {price_str:>12} {rsi:>10.1f} {pct:>+11.1f}%\n"
    return out


def fmt_3x_leveraged(data):
    indicators = data.get('indicators', {})
    out = fmt_header("3x LEVERAGED ETFs")
    out += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}  Signal\n"
    out += "-" * 65 + "\n"
    for t in ['NAIL', 'CURE', 'FAS', 'LABU', 'TQQQ', 'SOXL', 'TECL', 'DRN']:
        ind = indicators.get(t)
        if not ind:
            continue
        price = ind.get('price', 0)
        rsi = _rsi(ind)
        pct = _pct_above_sma200(ind)
        if rsi < 21:
            sig = f"{GREEN} OVERSOLD"
        elif rsi < 30:
            sig = f"{GREEN} Watch"
        elif rsi > 85:
            sig = f"{RED} OVERBOUGHT"
        elif rsi > 79:
            sig = f"{YELLOW} Extended"
        else:
            sig = ""
        out += f"{t:<10} ${price:>10.2f} {rsi:>10.1f} {pct:>+11.1f}%  {sig}\n"
    return out


def fmt_smh_levels(data):
    ind = data.get('indicators', {}).get('SMH')
    if not ind:
        return ''
    sma200 = ind.get('sma200', 0)
    out = fmt_header("SMH/SOXL LEVELS")
    out += f"Current Price:    ${ind.get('price', 0):.2f}\n"
    out += f"SMA(200):         ${sma200:.2f}\n"
    out += f"% Above SMA200:   {_pct_above_sma200(ind):+.1f}%\n"
    out += "\nKey Levels:\n"
    out += f"  30% (Trim):     ${sma200 * 1.30:.2f}\n"
    out += f"  35% (Warning):  ${sma200 * 1.35:.2f}\n"
    out += f"  40% (Sell):     ${sma200 * 1.40:.2f}\n"
    return out


def fmt_crisis_alpha_regime(data):
    """Market breadth + regime block.

    data['breadth_regime'] is a STRING label: 'BROAD BULL', 'NARROW LEADERSHIP',
      'ROTATION', 'BROAD WEAKNESS', or 'UNKNOWN'.
    data['breadth_inline'] keys: zbt_ema, zbt_ratio, zbt_zone, zbt_thrust,
      mcclellan, mcl_ema19, mcl_ema39, mcl_direction, mcl_zone,
      pct_above_50sma, above50_n, above50_total, adv, dec.
    data['leadership_gap'], data['sphb_splv'] also available if relevant.
    """
    regime = data.get('breadth_regime')
    inline = data.get('breadth_inline')
    if not regime and not inline:
        return ''

    label = regime if isinstance(regime, str) else (
        regime.get('label', 'UNKNOWN') if isinstance(regime, dict) else 'UNKNOWN'
    )
    out = fmt_header(f"BREADTH REGIME: {label}")
    if inline:
        zbt = inline.get('zbt_ema')
        zbt_zone = inline.get('zbt_zone')
        mcl = inline.get('mcclellan')
        mcl_zone = inline.get('mcl_zone')
        pct50 = inline.get('pct_above_50sma')
        adv = inline.get('adv')
        dec = inline.get('dec')
        thrust_mark = ' [THRUST]' if inline.get('zbt_thrust') else ''
        if zbt is not None:
            out += f"ZBT EMA: {zbt:.4f} [{zbt_zone}]{thrust_mark}\n"
        if mcl is not None:
            out += f"McClellan: {mcl:+.1f} [{mcl_zone}] | EMA19/39: {inline.get('mcl_ema19','?')}/{inline.get('mcl_ema39','?')} | {inline.get('mcl_direction','')}\n"
        if pct50 is not None:
            out += f"%Above 50-SMA: {pct50:.1f}% ({inline.get('above50_n','?')}/{inline.get('above50_total','?')})\n"
        if adv is not None and dec is not None:
            out += f"Advances/Declines: {adv}/{dec}\n"

    # SPHB/SPLV risk appetite if present
    sphb = data.get('sphb_splv')
    if sphb and isinstance(sphb, dict):
        out += f"\nSPHB/SPLV ratio: {sphb.get('ratio', '?')} [{sphb.get('label', '')}]\n"

    return out


def fmt_rolling_beta(data):
    """Rolling beta vs SPY table.

    data['rolling_betas'] is a list of rows with keys:
      name (group name), b63, b126, b252, c63, c126, c252, blend_wt.
    The last row has is_blend=True and represents 'Est. Blend'.
    """
    betas = data.get('rolling_betas', [])
    if not betas:
        return ''
    out = fmt_header("ROLLING BETA vs SPY")
    out += f"{'Group':<26} {'63d':>8} {'126d':>8} {'252d':>8}\n"
    out += "-" * 52 + "\n"
    for row in betas:
        name = row.get('name', '?')
        b63 = row.get('b63')
        b126 = row.get('b126')
        b252 = row.get('b252')
        b63s = f"{b63:+.2f}" if isinstance(b63, (int, float)) else '?'
        b126s = f"{b126:+.2f}" if isinstance(b126, (int, float)) else '?'
        b252s = f"{b252:+.2f}" if isinstance(b252, (int, float)) else '?'
        # Visual separator before blend row
        if row.get('is_blend'):
            out += "-" * 52 + "\n"
        out += f"{name:<26} {b63s:>8} {b126s:>8} {b252s:>8}\n"
    return out


def fmt_gold_miners(data):
    """GLD & miners block."""
    indicators = data.get('indicators', {})
    miners = ['GLD', 'GDX', 'GDXJ', 'JNUG', 'NUGT']
    available = [t for t in miners if t in indicators]
    if not available:
        return ''
    out = fmt_header("GLD & MINERS")
    for t in miners:
        ind = indicators.get(t)
        if not ind:
            continue
        out += (f"  {t:>4} ${ind.get('price', 0):>9.2f} RSI={_rsi(ind):.1f} "
                f"{_pct_above_sma200(ind):+.1f}%\n")
    return out


def fmt_uvxy_vol_regime(data):
    """UVXY Vol Regime Shift block.

    data['uvxy_vol_regime'] keys: price, sma200, pct_above, tier, tier_color,
    threshold_signal, threshold_high, threshold_extreme.
    """
    uv = data.get('uvxy_vol_regime')
    if not uv:
        return ''
    out = fmt_header("UVXY VOL REGIME SHIFT")
    out += (f"UVXY: ${uv.get('price', 0):.2f} | SMA200: ${uv.get('sma200', 0):.2f} | "
            f"{uv.get('pct_above', 0):+.1f}%\n")
    tier = uv.get('tier')
    if tier:
        out += f"Tier: {tier}"
        if uv.get('threshold_high'):
            out += f" (HIGH threshold: ${uv['threshold_high']:.2f}, EXTREME: ${uv.get('threshold_extreme', 0):.2f})"
        out += "\n"
    return out


def fmt_vix_term_structure(data):
    """VIX Term Structure block.

    data['vix_structure'] keys: curve (list), regime, pct_spread, spreads, vix.
    """
    vts = data.get('vix_structure') or data.get('vix_term_structure')
    if not vts:
        return ''
    out = fmt_header("VIX TERM STRUCTURE")
    regime = vts.get('regime', '?').replace('_', ' ')
    out += f"Regime: {regime}\n"
    if vts.get('vix') is not None:
        out += f"VIX: {vts['vix']:.2f}\n"
    if vts.get('pct_spread') is not None:
        out += f"Front/back spread: {vts['pct_spread']:+.1f}%\n"

    # Curve
    curve = vts.get('curve', [])
    if curve:
        curve_str = ' | '.join(f"{p['label']}={p['value']:.2f}" for p in curve)
        out += f"Curve: {curve_str}\n"

    # Bounce zone (Group 13 — if present)
    if 'ratio' in vts and vts.get('ratio') is not None:
        out += f"VIX3M/VIX ratio: {vts['ratio']:.2f}"
        if vts.get('zone'):
            out += f" ({vts['zone'].upper()})"
        out += "\n"

    return out


def fmt_midmonth(data):
    """Mid-Month Rotation block (Group 25)."""
    mm = data.get('midmonth')
    if not mm:
        return ''
    out = fmt_header("MID-MONTH ROTATION (Group 25)")
    out += f"Trading Day:  {mm.get('td','?')} of month\n"
    out += f"SPY MTD:      {mm.get('spy_mtd', 0):+.2f}%\n"
    out += f"TLT MTD:      {mm.get('tlt_mtd', 0):+.2f}%\n"
    out += f"Current Lean: Buy {mm.get('pick','?')} (the MTD loser)\n"
    if mm.get('is_signal_day'):
        out += f"Signal:       FIRES TODAY — execute MOC at Fidelity\n"
    elif mm.get('is_holding'):
        out += f"Status:       HOLDING through month-end\n"
    else:
        out += f"Signal In:    {mm.get('days_to_signal','?')} trading day(s)\n"
    return out


def fmt_drif(data):
    """DRIF Velocity Filter table.

    data['drif'] is a dict keyed by ticker (SPY, QQQ, SMH) with values:
      {ticker, lever, rsi, velocity, cumRet5d, cumRet7d, gate, label,
       level, passWr, passN, failWr, failN, retField, retGate, retVal, hold}
    """
    drif = data.get('drif')
    if not drif or not isinstance(drif, dict):
        return ''
    out = fmt_header("DRIF VELOCITY FILTER")
    out += f"{'Ticker':<8} {'RSI':>6} {'5d Ret':>8} {'7d Ret':>8} {'Vel':>6}  {'Gate':>5}  Status\n"
    out += "-" * 70 + "\n"
    for ticker in ['SPY', 'QQQ', 'SMH']:
        r = drif.get(ticker)
        if not r:
            continue
        out += (f"{r.get('ticker', ticker):<8} "
                f"{r.get('rsi', 0):>6.1f} "
                f"{r.get('cumRet5d', 0):>+7.1f}% "
                f"{r.get('cumRet7d', 0):>+7.1f}% "
                f"{r.get('velocity', 0):>+6.0f}  "
                f"{r.get('gate', '---'):>5}  "
                f"{r.get('label', '')}\n")
    return out


def fmt_move_index(data):
    """MOVE Index block.

    data['move_index'] keys: price, rsi, sma200, pct_above_sma200,
    change_20d_pct, 19A_active, 19B_active, 19C_active, 19C_ready.
    """
    mv = data.get('move_index')
    if not mv:
        return ''
    out = fmt_header("MOVE INDEX")
    out += (f"Price: {mv.get('price', 0):.2f} | RSI: {mv.get('rsi', 0):.1f} | "
            f"20d: {mv.get('change_20d_pct', 0):+.1f}%\n")
    out += (f"19A(>115):{'ON' if mv.get('19A_active') else '-'} | "
            f"19B(20d>50%):{'ON' if mv.get('19B_active') else '-'} | "
            f"19C(crush):{'ON' if mv.get('19C_active') else '-'}\n")
    return out


def fmt_fibonacci(data):
    """Fibonacci retracement levels for SPY/QQQ/SMH.

    data['fibonacci'] keyed by 'SPY'/'QQQ'/'SMH' with shape:
      {high, low, close, trend ('UP'/'DOWN'),
       levels: {'23.6': {level, dist, near}, '38.2': ..., '50.0': ..., '61.8': ...}}
    """
    fib = data.get('fibonacci')
    if not fib:
        return ''
    out = fmt_header("FIBONACCI CONTEXT")
    for sym in ['SPY', 'QQQ', 'SMH']:
        f = fib.get(sym)
        if not f:
            continue
        out += (f"\n{sym} (30d): H={f.get('high', 0):.2f} L={f.get('low', 0):.2f} "
                f"C={f.get('close', 0):.2f} [{f.get('trend', '?')}]\n")
        levels = f.get('levels') or {}
        for pct_str in ['23.6', '38.2', '50.0', '61.8']:
            entry = levels.get(pct_str)
            if not entry:
                continue
            lvl = entry.get('level')
            dist = entry.get('dist')
            if lvl is not None and dist is not None:
                near_mark = ' ★' if entry.get('near') else ''
                out += f"   {pct_str:>5}%: ${lvl:>8.2f} ({dist:+.1f}%){near_mark}\n"
    return out


def fmt_portfolio_performance(data):
    """Per-account portfolio performance with win rates and per-symphony stats.

    data['composer']['accounts'] is a list with keys:
      type, source, value, today_dollar, today_pct, win_rates (dict),
      symphonies (list).

    Each win_rates dict has keys: daily_20d, daily_60d, daily_all,
      weekly_all, weekly_12w, monthly_all, streak_days, streak_direction, max_dd.

    Each symphony has keys:
      id, name, value, pct_of_account, twr_annualized, sharpe, max_dd,
      last_pct_change, next_rebalance, may_rebalance_today, holdings.
    """
    composer = data.get('composer') or data.get('portfolio')
    if not composer:
        return ''
    accounts = composer.get('accounts', [])
    if not accounts:
        return ''

    out = fmt_header("PORTFOLIO PERFORMANCE & WIN RATES")
    for acct in accounts:
        name = acct.get('type') or acct.get('account_type') or 'Account'
        value = acct.get('value', 0)
        today_pct = acct.get('today_pct', 0)
        source = acct.get('source', '')
        source_label = f" [{source}]" if source else ''
        out += f"\n {name}{source_label}: ${value:,.0f} | Today: {today_pct:+.2f}%\n"

        wr = acct.get('win_rates') or {}
        if wr:
            wr_parts = []
            if 'daily_20d' in wr: wr_parts.append(f"20d:{wr['daily_20d']:.0f}%")
            if 'daily_60d' in wr: wr_parts.append(f"60d:{wr['daily_60d']:.0f}%")
            if 'daily_all' in wr: wr_parts.append(f"All:{wr['daily_all']:.0f}%")
            if 'weekly_12w' in wr: wr_parts.append(f"Wk(12w):{wr['weekly_12w']:.0f}%")
            if 'monthly_all' in wr: wr_parts.append(f"Mo:{wr['monthly_all']:.0f}%")
            streak = wr.get('streak')
            if streak is not None:
                streak_dir = 'winning' if streak > 0 else ('losing' if streak < 0 else 'none')
                wr_parts.append(f"Streak: {streak:+d}d ({streak_dir})")
            if wr_parts:
                out += f" Win Rates: {' | '.join(wr_parts)}\n"

        symphonies = acct.get('symphonies', [])
        if symphonies:
            out += f" {'Symphony':<32} {'Value':>11} {'Today':>8} {'Ann.Ret':>9} {'Sharpe':>7} {'MaxDD':>7}\n"
            out += " " + "-" * 78 + "\n"
            for s in symphonies:
                sname = (s.get('name', ''))[:30]
                sv = s.get('value', 0) or 0
                spct = (s.get('last_pct_change') or 0) * 100 if s.get('last_pct_change') is not None else 0
                # twr_annualized is in decimal form (0.42 = 42%); convert to %
                ann = s.get('twr_annualized')
                ar_str = f"{ann*100:+.1f}%" if isinstance(ann, (int, float)) else '—'
                sh = s.get('sharpe')
                sh_str = f"{sh:.2f}" if isinstance(sh, (int, float)) else '—'
                # max_dd from Composer is already a decimal % (e.g. -0.115 = -11.5%)
                dd = s.get('max_dd')
                dd_str = f"{dd*100:+.1f}%" if isinstance(dd, (int, float)) else '—'
                out += f" {sname:<32} ${sv:>10,.0f} {spct:>+7.2f}% {ar_str:>9} {sh_str:>7} {dd_str:>7}\n"

    consolidated = composer.get('consolidated', {})
    if consolidated:
        total = consolidated.get('total_value', 0)
        today_pct = consolidated.get('today_pct', 0)
        goal_pct = consolidated.get('goal_8m_pct')
        out += f"\n CONSOLIDATED TOTAL: ${total:,.0f} | Today: {today_pct:+.2f}%"
        if goal_pct is not None:
            out += f" | $8M goal: {goal_pct:.1f}%"
        out += "\n"
    return out


def _format_preparsed_dry_run(pre_parsed):
    """
    Render dry-run preview from already-parsed data returned by /api/composer-dry-run.
    Unlike the raw-response path, this gives us pre-computed warnings (with the
    dashboard's authoritative total_portfolio_value denominator) and pre-computed
    net_flow — we just need to format them.
    """
    from composer_dry_run import (format_concentration_warnings,
                                    aggregate_net_trade_flow)
    from collections import defaultdict

    rotations = pre_parsed.get('rotations', [])
    warnings = pre_parsed.get('warnings')
    pre_net_flow = pre_parsed.get('net_flow', [])

    if not rotations:
        return ("\n" + "=" * 70 + "\n"
                "🧭 COMPOSER NEXT-REBALANCE PREVIEW\n"
                "=" * 70 + "\n"
                "No symphonies returned from dashboard dry-run.\n")

    # Reuse the main formatter's layout but inject pre-computed warnings/flow.
    # Easiest path: call format_dry_run_for_email with rotations alone (it will
    # recompute warnings using rotation_sum fallback), then if we have better
    # pre-computed warnings, swap that section.
    body = format_dry_run_for_email(rotations)

    if warnings:
        # Replace the locally-computed warning block with the dashboard's
        # authoritative version (uses real total_portfolio_value denominator)
        formatted_warnings = format_concentration_warnings(warnings)
        # Find and replace the existing warning block in body
        import re
        pattern = re.compile(
            r'\n=+\n\*\*\* RISK CONCENTRATION WARNINGS \*\*\*\n=+\n.*?(?=\n── PORTFOLIO NET TRADE FLOW)',
            re.DOTALL,
        )
        if formatted_warnings:
            body = pattern.sub(formatted_warnings + '\n', body, count=1)
        else:
            # No warnings — strip any warning block that was locally computed
            body = pattern.sub('', body, count=1)

    return body


def fmt_composer_rebalance_preview(data):
    """
    Real Composer dry-run preview — calls /api/v0.1/dry-run for the EXACT trades
    each symphony will execute at next rebalance.

    Source priority (in order):
      1. data['composer_dry_run'] — pre-computed by dashboard server (preferred)
      2. data['composer_dry_run_parsed'] — from legacy /api/composer-dry-run
      3. Local fetch via composer_dry_run.fetch_dry_run_preview() — direct API
         using COMPOSER_KEY_ID/SECRET env vars

    Total portfolio value (for concentration % calculations) is read from
    data['composer'] when available — sums 'value' across all accounts.
    """
    # Compute total portfolio value for concentration warnings
    total_pv = None
    composer = data.get('composer') or data.get('portfolio')
    if composer and composer.get('accounts'):
        total_pv = sum(a.get('value', 0) or 0 for a in composer['accounts'])
        if total_pv <= 0:
            total_pv = None

    # 1. Pre-computed and pre-parsed by dashboard's /api/composer-dry-run
    #    (legacy path when /api/email-state isn't available)
    pre_parsed = data.get('composer_dry_run_parsed')
    if pre_parsed and pre_parsed.get('rotations'):
        return _format_preparsed_dry_run(pre_parsed)

    # 2. Pre-computed by dashboard (raw API response in composer_dry_run)
    pre = data.get('composer_dry_run')
    if pre:
        parsed = parse_dry_run_response(pre)
        return format_dry_run_for_email(parsed, total_portfolio_value=total_pv)

    # 3. Local fetch (direct Composer API)
    if not DRY_RUN_AVAILABLE:
        return ''
    if not os.environ.get('COMPOSER_KEY_ID'):
        return ''

    # Pull Composer account IDs only (Fidelity/E*Trade accounts don't have one)
    account_uuids = None
    if composer and composer.get('accounts'):
        account_uuids = [a.get('id') for a in composer['accounts']
                         if a.get('id') and a.get('source') == 'Composer']
        if not account_uuids:
            account_uuids = None

    raw = fetch_dry_run_preview(account_uuids=account_uuids)
    if raw is None:
        return ("\n" + "=" * 70 + "\n"
                "🧭 COMPOSER NEXT-REBALANCE PREVIEW\n"
                "=" * 70 + "\n"
                "Dry-run fetch failed — check Composer API credentials and rate limit.\n")
    parsed = parse_dry_run_response(raw)
    return format_dry_run_for_email(parsed, total_portfolio_value=total_pv)


# ════════════════════════════════════════════════════════════════════
# MAIN EMAIL ASSEMBLY
# ════════════════════════════════════════════════════════════════════
def format_email(data, source_label, is_preclose=False):
    now = datetime.now()
    timing = "PRE-CLOSE PREVIEW (3:15 PM)" if is_preclose else "MARKET CLOSE CONFIRMATION (4:05 PM)"

    body = ''

    # Hormuz first (matches the pasted email order)
    body += fmt_hormuz(data)

    body += fmt_header(f"MARKET SIGNAL MONITOR - {timing}")
    body += f"{now.strftime('%Y-%m-%d %H:%M')} ET\n"
    body += "=" * 70 + "\n"

    body += fmt_calendar_banner(data)
    body += "\n"
    body += fmt_alerts(data)

    # NEW: Composer Next-Rebalance Preview from real dry-run API
    body += fmt_composer_rebalance_preview(data)

    # Indicator tables
    key_tickers = ['SPY', 'QQQ', 'SMH', 'GLD', 'USDU', 'XLP', 'TLT', 'HYG',
                   'XLF', 'UVXY', 'BTC-USD', 'AMD', 'NVDA']
    body += fmt_indicators_table(data, key_tickers, "CURRENT INDICATOR STATUS")
    body += fmt_3x_leveraged(data)

    other = ['XLV', 'XLU', 'XLE', 'TMV', 'VOOV', 'VOOG', 'VTV', 'QQQE',
             'BOIL', 'EURL', 'YINN', 'KORU', 'INDL', 'EDC']
    body += fmt_indicators_table(data, other, "OTHER ETFs")

    body += fmt_smh_levels(data)
    body += fmt_crisis_alpha_regime(data)
    body += fmt_rolling_beta(data)
    body += fmt_gold_miners(data)
    body += fmt_uvxy_vol_regime(data)
    body += fmt_vix_term_structure(data)
    body += fmt_midmonth(data)
    body += fmt_drif(data)
    body += fmt_move_index(data)
    body += fmt_portfolio_performance(data)
    body += fmt_fibonacci(data)

    if is_preclose:
        body += fmt_header("NOTE")
        body += "PRE-CLOSE preview. Signals may change by close. Final email at 4:05 PM ET.\n"

    # Footer with data source for debugging
    body += f"\n{'─'*70}\nData source: {source_label}\n"
    return body


def send_email(subject, body):
    if not (SENDER_EMAIL and SENDER_PASSWORD and RECIPIENT_EMAIL):
        print("Email not configured — printing to console:")
        print(f"Subject: {subject}\n")
        print(body)
        return False
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(SENDER_EMAIL, SENDER_PASSWORD)
            s.send_message(msg)
        if PHONE_EMAIL:
            short = MIMEMultipart()
            short['From'] = SENDER_EMAIL
            short['To'] = PHONE_EMAIL
            short['Subject'] = subject
            short.attach(MIMEText(subject + '\n\n' + body[:500], 'plain'))
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
                s.login(SENDER_EMAIL, SENDER_PASSWORD)
                s.send_message(short)
        print(f"Email sent: {subject}")
        return True
    except Exception as e:
        print(f"Email send failed: {e}")
        return False


def main():
    data, raw, source = load_dashboard_data()
    if raw is not None:
        data['_raw_data'] = raw

    # Subject construction
    signals = data.get('signals', [])
    buy = sum(1 for s in signals if s.get('type') == 'buy')
    exit_ = sum(1 for s in signals if s.get('type') in ('exit', 'short'))

    if exit_ > 0:
        emoji, urgency = RED, "EXIT SIGNALS"
    elif buy > 0:
        emoji, urgency = GREEN, "BUY SIGNALS"
    elif signals:
        emoji, urgency = YELLOW, "WATCH"
    else:
        emoji, urgency = CHART, "No Alerts"

    timing = "PRE-CLOSE" if IS_PRECLOSE else "CLOSE"
    if signals:
        subject = f"{emoji} [{timing}] Market Signals: {len(signals)} Alert(s) - {urgency}"
    else:
        subject = f"{emoji} [{timing}] Market Signals: No Alerts"

    body = format_email(data, source, IS_PRECLOSE)
    send_email(subject, body)
    print(f"\n{len(signals)} signal(s) detected")


if __name__ == '__main__':
    main()
