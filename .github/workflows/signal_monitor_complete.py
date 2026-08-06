#!/usr/bin/env python3
"""
Comprehensive Market Signal Monitor v5.1
========================================
Monitors all backtested trading signals and sends alerts.
+ Robot James module (Ensemble 50/50 mom+below-SMA base + 4-way EO strict overlay)

SCHEDULE: Three emails daily (weekdays)
- 9:45 AM ET:  Post-open snapshot (open mode)
- 11:00 AM ET: Mid-day preview (preclose mode)
- 4:05 PM ET:  Market close confirmation (close mode)

Robot James decision points:
- TD1 close:  Set 60/40 base (30% mom pick + 30% below-SMA pick + 40% GLD)
- TD15 close: Rotate 100% into 4-way EO lagger (TMF/TQQQ/UPRO/SOXL) or strict skip
- EOM close:  Rotate back to cash; next TD1 sets new base
Execution: Market-on-Close (MOC) by 3:50 PM ET. After-hours fallback if missed.
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
import json
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

# Mode detection: three modes (open, preclose, close)
# - 'open':     post-open snapshot at 9:45 AM ET
# - 'preclose': mid-day preview at 11:00 AM ET
# - (default):  market close confirmation at 4:05 PM ET
_MODE_ARG = sys.argv[1] if len(sys.argv) > 1 else ''
IS_OPEN = _MODE_ARG == 'open'
IS_PRECLOSE = _MODE_ARG == 'preclose'
IS_CLOSE = not (IS_OPEN or IS_PRECLOSE)

# Backwards-compatible flag used throughout: any pre-close run (open OR preclose)
# behaves like the old "preclose" mode for body-level rendering decisions
IS_PRECLOSE_LIKE = IS_OPEN or IS_PRECLOSE

# Robot James state file (written after each run, read by dashboard)
RJ_STATE_FILE = os.environ.get('ROBOT_JAMES_STATE', '/tmp/robot_james_state.json')

# =============================================================================
# ROBOT JAMES MODULE
# =============================================================================
# Strategy: Ensemble 50/50 mom+below-SMA base + 4-way EO strict overlay
#   Base (TD1 -> TD15 close):
#     - Momentum-60d pick (highest trailing 60-day return from SPY/QQQ/SMH)
#     - Below-SMA50 pick (most-below-SMA50 from SPY/QQQ/SMH)
#     - Rules agree -> 60% that equity + 40% GLD
#     - Rules disagree -> 30% pick_A + 30% pick_B + 40% GLD
#   Overlay (TD15 -> EOM close):
#     - 4-way EO strict: biggest MTD lagger among SPY/QQQ/SMH/TLT at TD14
#     - TLT lagger -> TMF (no gate)
#     - Equity lagger -> leveraged ETF if below SMA50; else SKIP (stay in base)
# Execution: MOC by 3:50 PM ET (Fidelity), after-hours fallback if missed.
# =============================================================================
RJ_BASE_POOL = ['SPY', 'QQQ', 'SMH']
RJ_OVERLAY_POOL = ['SPY', 'QQQ', 'SMH', 'TLT']
RJ_LEV_MAP = {'SPY': 'UPRO', 'QQQ': 'TQQQ', 'SMH': 'SOXL', 'TLT': 'TMF'}
RJ_GLD = 'GLD'
RJ_HEADER_BAR = "=" * 70


def rj_compute_td_info(ref_date, trading_days):
    """Compute TD number info for reference date against calendar of trading days."""
    ref = pd.Timestamp(ref_date).normalize()
    month_start = ref.replace(day=1)
    tds_past = [d for d in trading_days
                if pd.Timestamp(d).normalize() >= month_start
                and pd.Timestamp(d).normalize() <= ref]
    if not tds_past:
        return None
    current_td = len(tds_past)
    td1_date = pd.Timestamp(tds_past[0]).normalize()
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
    return {
        'current_td': current_td,
        'total_tds': total_tds,
        'is_td1': current_td == 1,
        'is_td14': current_td == 14,
        'is_td15': current_td == 15,
        'is_eom': ref == eom,
        'td1_date': td1_date,
        'td14_date': td14,
        'td15_date': td15,
        'eom_date': eom,
        'all_tds_month': all_tds,
    }


def rj_compute_base_weights(data, decision_date):
    """Compute base rule picks at decision_date close. Returns dict for rendering."""
    decision_date = pd.Timestamp(decision_date).normalize()
    mom60, below_sma = {}, {}
    for t in RJ_BASE_POOL:
        if t not in data: continue
        df = data[t]
        df_cut = df[df.index <= decision_date]
        if len(df_cut) < 61: continue
        close = df_cut['Close']
        mom60[t] = float(close.iloc[-1] / close.iloc[-61] - 1)
        sma50 = close.rolling(50).mean()
        if pd.notna(sma50.iloc[-1]) and sma50.iloc[-1] > 0:
            below_sma[t] = float(close.iloc[-1] / sma50.iloc[-1] - 1)
    mom_pick = max(mom60, key=mom60.get) if mom60 else 'SPY'
    sma_pick = min(below_sma, key=below_sma.get) if below_sma else 'SPY'
    agree = (mom_pick == sma_pick)
    weights = {mom_pick: 0.60} if agree else {mom_pick: 0.30, sma_pick: 0.30}
    return {
        'mom_pick': mom_pick,
        'below_sma_pick': sma_pick,
        'weights': weights,
        'gld_weight': 0.40,
        'rules_agree': agree,
        'agree_ticker': mom_pick if agree else None,
        'mom60_values': mom60,
        'below_sma_values': below_sma,
    }


def rj_compute_overlay(data, td1_date, td14_date, td15_date):
    """Compute 4-way EO strict overlay decision."""
    td1_date = pd.Timestamp(td1_date).normalize()
    td14_date = pd.Timestamp(td14_date).normalize()
    td15_date = pd.Timestamp(td15_date).normalize()
    mtd = {}
    for t in RJ_OVERLAY_POOL:
        if t not in data: continue
        df = data[t]
        idx = df.index.normalize()
        td1_sub = df[idx <= td1_date]
        td14_sub = df[idx <= td14_date]
        if len(td1_sub) == 0 or len(td14_sub) == 0: continue
        mtd[t] = float(td14_sub['Close'].iloc[-1] / td1_sub['Close'].iloc[-1] - 1)
    if not mtd:
        return None
    lagger = min(mtd, key=mtd.get)
    gate_status = {}
    strict_skip = False
    skip_reason = None
    signal_asset = None
    if lagger == 'TLT':
        signal_asset = 'TMF'
    else:
        df = data[lagger]
        df_cut = df[df.index.normalize() <= td15_date]
        close = df_cut['Close']
        sma50 = close.rolling(50).mean()
        if pd.notna(sma50.iloc[-1]) and sma50.iloc[-1] > 0:
            pct = float(close.iloc[-1] / sma50.iloc[-1] - 1)
            gate_status[lagger] = pct
            if close.iloc[-1] < sma50.iloc[-1]:
                signal_asset = RJ_LEV_MAP[lagger]
            else:
                strict_skip = True
                skip_reason = f"{lagger} is {pct*100:+.1f}% vs SMA50 (above trend) - gate FAIL"
    for t in ['SPY', 'QQQ', 'SMH']:
        if t not in gate_status and t in data:
            df = data[t]
            df_cut = df[df.index.normalize() <= td15_date]
            close = df_cut['Close']
            sma50 = close.rolling(50).mean()
            if pd.notna(sma50.iloc[-1]) and sma50.iloc[-1] > 0:
                gate_status[t] = float(close.iloc[-1] / sma50.iloc[-1] - 1)
    return {
        'lagger': lagger,
        'mtd_values': mtd,
        'signal_asset': signal_asset,
        'gate_status': gate_status,
        'strict_skip': strict_skip,
        'skip_reason': skip_reason,
    }


def rj_determine_action(td_info):
    current_td = td_info['current_td']
    if td_info['is_td1']:
        atype, severity = 'TD1_SET_BASE', 'action_required'
    elif td_info['is_td15']:
        atype, severity = 'TD15_OVERLAY', 'action_required'
    elif td_info['is_eom']:
        atype, severity = 'EOM_ROTATE_BACK', 'action_required'
    else:
        atype, severity = 'HOLD', 'none'
    next_date, next_type = None, None
    if current_td < 15 and td_info['td15_date'] is not None:
        next_date, next_type = td_info['td15_date'], 'TD15_OVERLAY'
    elif current_td < td_info['total_tds'] and td_info['eom_date'] is not None:
        next_date, next_type = td_info['eom_date'], 'EOM_ROTATE_BACK'
    days_until = None
    if next_date is not None:
        today_ts = pd.Timestamp.today().normalize()
        days_until = (next_date - today_ts).days
    return {
        'type': atype,
        'severity': severity,
        'next_decision_date': str(next_date.date()) if next_date is not None else None,
        'next_decision_type': next_type,
        'days_until_next_decision': days_until,
    }


def rj_compute_state(data, ref_date=None):
    """Compute full Robot James state dictionary."""
    ref = pd.Timestamp(ref_date).normalize() if ref_date else pd.Timestamp.today().normalize()
    if 'SPY' not in data:
        return None
    trading_days = sorted(data['SPY'].index.normalize().unique().tolist())
    td_info = rj_compute_td_info(ref, trading_days)
    if td_info is None:
        return None
    state = {
        'ref_date': str(ref.date()),
        'computed_at': datetime.now().isoformat(),
        'current_td': td_info['current_td'],
        'total_tds_this_month': td_info['total_tds'],
        'td1_date': str(td_info['td1_date'].date()) if td_info['td1_date'] is not None else None,
        'td14_date': str(td_info['td14_date'].date()) if td_info['td14_date'] is not None else None,
        'td15_date': str(td_info['td15_date'].date()) if td_info['td15_date'] is not None else None,
        'eom_date': str(td_info['eom_date'].date()) if td_info['eom_date'] is not None else None,
        'is_td1': td_info['is_td1'],
        'is_td14': td_info['is_td14'],
        'is_td15': td_info['is_td15'],
        'is_eom': td_info['is_eom'],
    }
    latest_date = trading_days[-1]
    state['projected_base'] = rj_compute_base_weights(data, latest_date)
    if td_info['current_td'] >= 15 and td_info['td15_date'] is not None:
        state['actual_overlay'] = rj_compute_overlay(
            data, td_info['td1_date'], td_info['td14_date'], td_info['td15_date'])
    yesterday = trading_days[-2] if len(trading_days) >= 2 else trading_days[-1]
    state['projected_overlay'] = rj_compute_overlay(
        data, td_info['td1_date'], yesterday, latest_date)
    state['action'] = rj_determine_action(td_info)
    return state


def rj_write_state(state, path=None):
    path = path or RJ_STATE_FILE
    try:
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        print(f"[RJ] Could not write state: {e}")


def _rj_fmt_pct(v, sign=True):
    if v is None: return "n/a"
    return f"{v*100:+.2f}%" if sign else f"{v*100:.2f}%"


def rj_render_block(state):
    """Return (subject_prefix, body_block). If HOLD day, subject_prefix is empty."""
    if state is None:
        return "", ""
    atype = state['action']['type']
    if atype == 'TD1_SET_BASE':
        return _rj_render_td1(state)
    if atype == 'TD15_OVERLAY':
        return _rj_render_td15(state)
    if atype == 'EOM_ROTATE_BACK':
        return _rj_render_eom(state)
    return _rj_render_hold(state)


def _rj_render_td1(state):
    pb = state['projected_base']
    mom_pick = pb['mom_pick']
    sma_pick = pb['below_sma_pick']
    agree = pb['rules_agree']
    if agree:
        subj = f"[ACTION REQUIRED] Robot James TD1 - SET BASE 60% {mom_pick} + 40% GLD by 3:50 PM"
    else:
        subj = f"[ACTION REQUIRED] Robot James TD1 - SET BASE {mom_pick}/{sma_pick}/GLD by 3:50 PM"
    L = [RJ_HEADER_BAR,
         "ROBOT JAMES - TD1: SET BASE WEIGHTS TODAY (MOC by 3:50 PM ET)",
         RJ_HEADER_BAR, "",
         "Rule inputs:",
         "  Momentum-60d (pick HIGHEST trailing 60d return):"]
    for t, v in sorted(pb['mom60_values'].items(), key=lambda x: -x[1]):
        marker = " <- PICK" if t == mom_pick else ""
        L.append(f"    {t}: {_rj_fmt_pct(v)}{marker}")
    L.append("  Most-below-SMA50 (pick LOWEST % vs SMA50):")
    for t, v in sorted(pb['below_sma_values'].items(), key=lambda x: x[1]):
        marker = " <- PICK" if t == sma_pick else ""
        L.append(f"    {t}: {_rj_fmt_pct(v)}{marker}")
    L.append("")
    if agree:
        L.append(f"Rules AGREE on {mom_pick}")
        L.append("")
        L.append("EXECUTE AT TODAY'S CLOSE (MOC):")
        L.append(f"  -> 60% {mom_pick}")
        L.append(f"  -> 40% GLD")
    else:
        L.append("Rules DISAGREE - split equity allocation 30/30")
        L.append("")
        L.append("EXECUTE AT TODAY'S CLOSE (MOC):")
        L.append(f"  -> 30% {mom_pick}  (momentum-60d pick)")
        L.append(f"  -> 30% {sma_pick}  (most-below-SMA50 pick)")
        L.append(f"  -> 40% GLD")
    L.append("")
    L.append("Submit MOC orders before 3:50 PM ET (Fidelity).")
    L.append("Fallback: after-hours fill if missed; avoid next-day open.")
    L.append("")
    L.append(f"Next decision: TD15 ({state.get('td15_date')}) - 4-way EO overlay rotation")
    L.append(RJ_HEADER_BAR)
    return subj, "\n".join(L)


def _rj_render_td15(state):
    ov = state.get('projected_overlay') or state.get('actual_overlay')
    if ov is None:
        return "", ""
    lagger = ov['lagger']
    mtd = ov['mtd_values']
    sig = ov['signal_asset']
    skip = ov['strict_skip']
    gate = ov['gate_status']
    reason = ov['skip_reason']
    if skip:
        subj = "[ACTION: NONE] Robot James TD15 - STRICT SKIP, stay in base"
    else:
        subj = f"[ACTION REQUIRED] Robot James TD15 - Rotate 100% -> {sig} by 3:50 PM"
    L = [RJ_HEADER_BAR,
         "ROBOT JAMES - TD15: OVERLAY ROTATION TODAY (MOC by 3:50 PM ET)",
         RJ_HEADER_BAR, "",
         f"MTD at TD14 close ({state.get('td14_date')}):"]
    for t, v in sorted(mtd.items(), key=lambda x: x[1]):
        marker = " <- LAGGER" if t == lagger else ""
        L.append(f"  {t}: {_rj_fmt_pct(v)}{marker}")
    L.append("")
    if lagger == 'TLT':
        L.append(f"Lagger: TLT -> Signal asset: TMF (no gate needed)")
        L.append("")
        L.append("EXECUTE AT TODAY'S CLOSE (MOC):")
        L.append(f"  -> SELL ALL base positions")
        L.append(f"  -> BUY 100% TMF")
    else:
        L.append(f"Gate check - equity lagger vs SMA50:")
        for t in ['SPY', 'QQQ', 'SMH']:
            if t in gate:
                v = gate[t]
                status = "BELOW SMA50 (PASS)" if v < 0 else "ABOVE SMA50 (FAIL)"
                marker = " <- LAGGER" if t == lagger else ""
                L.append(f"  {t}: {_rj_fmt_pct(v)} ({status}){marker}")
        L.append("")
        if skip:
            L.append(f"STRICT SKIP: {reason}")
            L.append("")
            L.append(f"NO ACTION TODAY. Continue holding base through EOM ({state.get('eom_date')}).")
        else:
            L.append(f"Lagger {lagger} passes SMA50 gate -> Signal asset: {sig}")
            L.append("")
            L.append("EXECUTE AT TODAY'S CLOSE (MOC):")
            L.append(f"  -> SELL ALL base positions (equities + GLD)")
            L.append(f"  -> BUY 100% {sig}")
    L.append("")
    if not skip:
        L.append(f"Hold {sig} until EOM ({state.get('eom_date')}), then rotate back to base.")
        L.append("")
        L.append("Submit MOC orders before 3:50 PM ET. Fallback: after-hours if missed.")
    L.append(RJ_HEADER_BAR)
    return subj, "\n".join(L)


def _rj_render_eom(state):
    subj = "[ACTION REQUIRED] Robot James EOM - Rotate back to base by 3:50 PM"
    L = [RJ_HEADER_BAR,
         "ROBOT JAMES - EOM: ROTATE BACK TO BASE (MOC by 3:50 PM ET)",
         RJ_HEADER_BAR, ""]
    actual = state.get('actual_overlay') or state.get('projected_overlay')
    if actual and actual.get('strict_skip'):
        L.append("No overlay was active this month (strict skip at TD15).")
        L.append("You were in base all month. NO ACTION REQUIRED AT EOM.")
        L.append("")
        L.append("Tomorrow is next month's TD1 - new base weights will fire.")
    else:
        held = actual.get('signal_asset') if actual else 'signal asset'
        L.append(f"Currently holding: 100% {held} (since TD15)")
        L.append("")
        L.append("EXECUTE AT TODAY'S CLOSE (MOC):")
        L.append(f"  -> SELL ALL {held}")
        L.append(f"  -> Sit in cash overnight; next month's TD1 sets new base")
    L.append("")
    L.append("Submit MOC orders before 3:50 PM ET. Fallback: after-hours if missed.")
    L.append(RJ_HEADER_BAR)
    return subj, "\n".join(L)


def _rj_render_hold(state):
    current_td = state['current_td']
    total_tds = state['total_tds_this_month']
    next_date = state['action']['next_decision_date']
    next_type = state['action']['next_decision_type']
    days_until = state['action']['days_until_next_decision']
    in_overlay = (current_td > 15)
    pb = state.get('projected_base', {})
    ov = state.get('actual_overlay') or state.get('projected_overlay', {})
    if in_overlay:
        if ov and ov.get('strict_skip'):
            pos = "BASE (strict skip this month)"
        elif ov and ov.get('signal_asset'):
            pos = f"100% {ov['signal_asset']}"
        else:
            pos = "BASE"
    else:
        if pb.get('rules_agree'):
            pos = f"60% {pb.get('agree_ticker')} + 40% GLD"
        elif pb.get('mom_pick') and pb.get('below_sma_pick'):
            pos = f"30% {pb['mom_pick']} + 30% {pb['below_sma_pick']} + 40% GLD"
        else:
            pos = "BASE"
    L = [RJ_HEADER_BAR,
         f"ROBOT JAMES - HOLDING (TD {current_td} of {total_tds})",
         RJ_HEADER_BAR,
         f"Current position: {pos}", ""]
    if not in_overlay and ov:
        L.append("If TD15 were today (projection):")
        for t, v in sorted(ov.get('mtd_values', {}).items(), key=lambda x: x[1]):
            marker = " <- current lagger" if t == ov.get('lagger') else ""
            L.append(f"  {t} MTD: {_rj_fmt_pct(v)}{marker}")
        if ov.get('strict_skip'):
            L.append(f"  -> Would STRICT SKIP: {ov.get('skip_reason')}")
        elif ov.get('signal_asset'):
            L.append(f"  -> Would rotate to: {ov.get('signal_asset')}")
        L.append("")
    elif in_overlay and state.get('actual_overlay'):
        a = state['actual_overlay']
        L.append(f"Overlay triggered TD15 ({state.get('td15_date')}):")
        L.append(f"  Lagger was: {a.get('lagger')} -> {a.get('signal_asset') or 'SKIP'}")
        L.append("")
    if next_date:
        L.append(f"Next decision: {next_type} on {next_date} ({days_until} days)")
    L.append(RJ_HEADER_BAR)
    return "", "\n".join(L)


def rj_render_postclose(state, close_prices):
    """Post-close confirmation block (4:05 PM email only, only on action days)."""
    if state is None:
        return ""
    atype = state['action']['type']
    if atype == 'HOLD':
        return ""
    L = [RJ_HEADER_BAR,
         f"ROBOT JAMES - POST-CLOSE CONFIRMATION ({atype})",
         RJ_HEADER_BAR, ""]
    if atype == 'TD1_SET_BASE':
        pb = state['projected_base']
        L.append("Base weights should now be:")
        for t, w in pb['weights'].items():
            cp = close_prices.get(t)
            px = f"${cp:.2f}" if cp else "n/a"
            L.append(f"  {t}: {w*100:.0f}%  (close {px})")
        gp = close_prices.get('GLD')
        gldpx = f"${gp:.2f}" if gp else "n/a"
        L.append(f"  GLD: {pb['gld_weight']*100:.0f}%  (close {gldpx})")
    elif atype == 'TD15_OVERLAY':
        ov = state.get('projected_overlay') or state.get('actual_overlay')
        if ov and not ov.get('strict_skip'):
            sa = ov['signal_asset']
            cp = close_prices.get(sa)
            px = f"${cp:.2f}" if cp else "n/a"
            L.append(f"Should now hold 100% {sa}")
            L.append(f"  {sa} close today: {px}")
        else:
            L.append("Strict skip confirmed - still in base.")
    elif atype == 'EOM_ROTATE_BACK':
        L.append("Position rotated out of overlay. Next TD1 sets new base.")
    L.append("")
    L.append("Verify your actual fills vs above close prices.")
    L.append(RJ_HEADER_BAR)
    return "\n".join(L)


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

def detrended_zscore(numerator, denominator, lookback=504):
    """Detrended z-score of a price ratio against a rolling linear-regression channel.

    Fits a linear regression to the trailing `lookback` ratio values, projects today's
    value, divides by residual sigma. Matches Andrei Sota's "trend channel" framing —
    secular trend in the ratio is removed before measuring stretch.

    Used by SIGNAL GROUP 13 (Z-Score Ratio Signals).
    """
    df = pd.concat([numerator, denominator], axis=1).dropna()
    if len(df) < lookback + 1:
        return pd.Series(index=df.index, dtype=float)
    ratio = df.iloc[:, 0] / df.iloc[:, 1]

    n = lookback
    x = np.arange(n)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    z_series = pd.Series(index=ratio.index, dtype=float)
    arr = ratio.values

    for i in range(n, len(ratio)):
        y = arr[i-n:i]
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / x_var
        intercept = y_mean - slope * x_mean
        pred_today = intercept + slope * n
        residuals = y - (intercept + slope * x)
        sd = residuals.std()
        z_series.iloc[i] = (arr[i] - pred_today) / sd if sd > 0 else np.nan

    return z_series

def download_data(tickers, period='2y'):
    """Download data for multiple tickers"""
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False)
            if len(df) > 0:
                # Flatten multi-index columns if present
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
        if len(df) < 200:
            continue
        
        try:
            close = df['Close']
            
            # Get latest values as scalars
            price = safe_float(close.iloc[-1])
            rsi10 = safe_float(calculate_rsi_wilder(close, 10).iloc[-1])
            rsi50 = safe_float(calculate_rsi_wilder(close, 50).iloc[-1])
            sma200 = safe_float(close.rolling(window=200).mean().iloc[-1])
            sma50 = safe_float(close.rolling(window=50).mean().iloc[-1])
            ema21 = safe_float(close.ewm(span=21, adjust=False).mean().iloc[-1])
            
            indicators[ticker] = {
                'price': price,
                'rsi10': rsi10,
                'rsi50': rsi50,
                'sma200': sma200,
                'sma50': sma50,
                'ema21': ema21,
            }
            
            # Calculate % above SMA200
            if sma200 > 0:
                indicators[ticker]['pct_above_sma200'] = (price / sma200 - 1) * 100
            else:
                indicators[ticker]['pct_above_sma200'] = 0
                
        except Exception as e:
            print(f"Error calculating indicators for {ticker}: {e}")
            continue
    
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
        if smh['sma50'] < smh['sma200'] and smh['sma200'] > 0:
            alerts.append(('🔴 DEATH CROSS', f"SMH SMA(50) below SMA(200) - Bearish trend", 'exit'))
        
        # BUY Signals - Days below SMA200
        if 'SMH' in data:
            smh_df = data['SMH']
            close = smh_df['Close']
            sma200_series = close.rolling(window=200).mean()
            
            # Count consecutive days below
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
        
        # Individual GLD overbought
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
    
    # =========================================================================
    # SIGNAL GROUP 9: NAIL (3x Homebuilders) Signals
    # =========================================================================
    if 'NAIL' in indicators:
        nail = indicators['NAIL']
        
        # GLD/USDU combo for NAIL (with XLF filter)
        if 'GLD' in indicators and 'USDU' in indicators and 'XLF' in indicators:
            gld = indicators['GLD']
            usdu = indicators['USDU']
            xlf = indicators['XLF']
            
            if gld['rsi10'] > 79 and usdu['rsi10'] < 25 and xlf['rsi10'] < 70:
                alerts.append(('🟢 NAIL SIGNAL', 
                    f"GLD>{gld['rsi10']:.0f} + USDU<{usdu['rsi10']:.0f} + XLF<{xlf['rsi10']:.0f}\n"
                    f"   → Long NAIL: 90% win, +4.9% avg (5d), +14.4% avg (10d) | n=10", 'buy'))
            
            # Warning: XLF strong + USDU weak = danger for NAIL
            if xlf['rsi10'] > 70 and usdu['rsi10'] < 25:
                alerts.append(('🔴 NAIL DANGER', 
                    f"XLF RSI={xlf['rsi10']:.1f} > 70 + USDU < 25 = Historically BAD for NAIL\n"
                    f"   → 11% win, -11.5% avg (5d) | Consider exit", 'exit'))
        
        # NAIL overbought/oversold
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
        
        if cure['rsi10'] > 79:
            alerts.append(('🔴 CURE OVERBOUGHT', 
                f"CURE RSI={cure['rsi10']:.1f} > 79 → Exit CURE: Only 40% win (5d) | n=95", 'exit'))
        elif cure['rsi10'] > 85:
            alerts.append(('🔴 CURE SELL', 
                f"CURE RSI={cure['rsi10']:.1f} > 85 → Sell CURE: Only 33% win (5d) | n=15", 'exit'))
    
    # =========================================================================
    # SIGNAL GROUP 11: FAS (3x Financials) Signals
    # =========================================================================
    if 'FAS' in indicators:
        fas = indicators['FAS']
        
        # FAS responds to GLD/USDU signal
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
        
        if fas['rsi10'] > 82:
            alerts.append(('🔴 FAS OVERBOUGHT', 
                f"FAS RSI={fas['rsi10']:.1f} > 82 → Exit FAS: Only 38% win (5d) | n=40", 'exit'))
        elif fas['rsi10'] > 85:
            alerts.append(('🔴 FAS SELL', 
                f"FAS RSI={fas['rsi10']:.1f} > 85 → Sell FAS: Only 8% win! (5d) | n=12", 'exit'))
    
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
        
        # LABU extreme extension warning
        if labu.get('pct_above_sma200', 0) > 80:
            alerts.append(('🟡 LABU EXTREME',
                f"LABU {labu['pct_above_sma200']:.0f}% above SMA(200) → Very extended, consider profits", 'warning'))

    # =========================================================================
    # SIGNAL GROUP 13: Z-Score Ratio Signals (Tier 2, manual execution)
    # =========================================================================
    # Detrended 504-day trend-channel z-score on price ratios.
    # Validated 2026-05-13 — stress test P(>QQQ) = 98.2% (regime-conditional bootstrap, 3000 trials).
    # MANUAL EXECUTION only (no Composer automation: SMA-based proxies failed in current regime).
    # Default holding when no signal active: 100% QQQ. On fire, rotate to 100% TQQQ for 20 trading days.
    # Sleeve sizing: 5-8% initial, scale to 12-15% after 2-3 confirming live fires.
    zscore_status = {}
    for ratio_name, num_ticker, den_ticker, threshold, direction, action_text in [
        ('QQQ_SPY',  'QQQ',  'SPY', 1.5,  'ge', 'Long TQQQ 20d (vs QQQ default): +7.30pp edge, Sharpe 1.06, MDD parity | n=45 ep'),
        ('QQQ_RSP',  'QQQ',  'RSP', -1.5, 'le', 'Long TQQQ 20d (vs QQQ default): +11.47pp edge, MDD -52% | n=35 ep'),
        ('QQQE_QQQ', 'QQQE', 'QQQ', -2.5, 'le', 'Long TQQQ 20d (vs QQQ default): borderline Tier 3, regime-concentrated | n=14 ep'),
    ]:
        if num_ticker not in data or den_ticker not in data:
            continue
        try:
            num_close = data[num_ticker]['Close']
            den_close = data[den_ticker]['Close']
            if isinstance(num_close, pd.DataFrame):
                num_close = num_close.iloc[:, 0]
            if isinstance(den_close, pd.DataFrame):
                den_close = den_close.iloc[:, 0]
            num_close = num_close.dropna()
            den_close = den_close.dropna()
            # Need 504-day regression window plus at least 1 valid output day
            if len(num_close) < 510 or len(den_close) < 510:
                continue

            z_series = detrended_zscore(num_close, den_close, lookback=504)
            z_clean = z_series.dropna()
            if z_clean.empty:
                continue

            z_today = safe_float(z_clean.iloc[-1])
            if direction == 'ge':
                fire_series = z_clean >= threshold
            else:
                fire_series = z_clean <= threshold
            fired_today = bool(fire_series.iloc[-1])

            fire_dates = z_clean.index[fire_series]
            last_fire_str = 'never'
            days_since = -1
            if len(fire_dates) > 0:
                last_fire = fire_dates[-1]
                last_fire_str = last_fire.strftime('%Y-%m-%d')
                days_since = (z_clean.index[-1] - last_fire).days

            zscore_status[ratio_name] = {
                'z_today': z_today,
                'threshold': threshold,
                'direction': direction,
                'fired_today': fired_today,
                'last_fire': last_fire_str,
                'days_since_fire': days_since,
            }

            arrow = '≥' if direction == 'ge' else '≤'
            if fired_today:
                alerts.append((
                    f'🟢 Z-SCORE {ratio_name} FIRED',
                    f"{num_ticker}/{den_ticker} detrended z = {z_today:+.2f}σ {arrow} {threshold}σ\n"
                    f"   → {action_text}\n"
                    f"   → Hold 20 trading days from entry, then return to QQQ default\n"
                    f"   → Last fire: {last_fire_str} ({days_since}d ago)",
                    'buy'
                ))
            else:
                approaching = (direction == 'ge' and z_today >= threshold - 0.2) or \
                              (direction == 'le' and z_today <= threshold + 0.2)
                if approaching:
                    gap = z_today - threshold
                    alerts.append((
                        f'🟡 Z-SCORE {ratio_name} APPROACHING',
                        f"{num_ticker}/{den_ticker} z = {z_today:+.2f}σ (threshold {arrow} {threshold}σ, gap {gap:+.2f}σ)",
                        'watch'
                    ))
        except Exception as e:
            print(f"Error computing z-score for {ratio_name}: {e}")
            continue

    status['zscore_signals'] = zscore_status

    # =========================================================================
    # SIGNAL GROUP 14: IBS DIP GATE + CONVICTION (manual/IRA; shared ibs_engine)
    # =========================================================================
    try:
        ibs_alerts, ibs_status = ibs_group14(data, indicators)
        alerts.extend(ibs_alerts)
        if ibs_status:
            status['ibs'] = ibs_status
    except Exception as e:
        print(f"[IBS] Group 14 skipped (non-fatal): {e}")

    return alerts, status

# =============================================================================
# SIGNAL GROUP 14: IBS DIP GATE + CONVICTION  (helper)
# =============================================================================
# Manual/IRA swing dip-buy on SPY/QQQ/SMH/XLK -> SPY(1x)/TQQQ/SOXL/TECL.
# Reuses the OHLC already in `data` (download_data keeps full frames) and
# indicators[t]['rsi10']. Shares ibs_engine.py / ibs_tracker.py (repo root,
# duplicated in chf-dashboard). Emits (title, message, category) 3-tuples so it
# flows through the existing subject/section/state machinery unchanged.
def ibs_group14(data, indicators):
    """Returns (alerts, ibs_status). Close-mode only emits buy/exit/resize and
    persists ibs_tracker.json; open/preclose emit a provisional watch (IBS is
    only final at the close). The caller wraps this in try/except."""
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(os.path.dirname(_here))   # repo root (ibs_engine.py lives here)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    import ibs_engine as eng
    import ibs_tracker as itr

    und = eng.BREADTH_UNIVERSE   # SPY, QQQ, SMH, XLK
    if not all(u in data and u in indicators for u in und):
        return [], {}

    rsi_by, ibs3_by = {}, {}
    for u in und:
        rsi_by[u] = indicators[u].get('rsi10')
        ibs3_by[u] = safe_float(eng.ibs_sma3(data[u]).iloc[-1])
    breadth = eng.breadth_count(rsi_by)

    state = itr.load_state()
    prior_state = state.get('gate_state', {})
    prior_mult = state.get('current_multiplier', {})

    alerts = []
    new_state, new_mult = {}, {}
    for u in und:
        veh = eng.SLEEVE_MAP[u]
        s3 = ibs3_by[u] if ibs3_by[u] == ibs3_by[u] else float('nan')   # NaN-safe hold
        st = eng.gate_state(s3, prior_state.get(u, 'OUT'))
        mult = eng.conviction_multiplier(u, rsi_by[u], breadth) if st == 'IN' else 0.0
        new_state[u], new_mult[u] = st, mult
        pv_st = prior_state.get(u, 'OUT')
        pv_m = prior_mult.get(veh, 0.0)
        entry_price = safe_float(data[veh]['Close'].iloc[-1]) if veh in data else None
        rsi_txt = f"{rsi_by[u]:.0f}" if rsi_by[u] is not None else "n/a"
        ctx = f"{u} IBS {ibs3_by[u]:.2f}, RSI {rsi_txt}, breadth {breadth}"
        rsi_r = round(rsi_by[u], 1) if rsi_by[u] is not None else None

        if IS_PRECLOSE_LIKE:
            if st == 'IN':
                alerts.append(('🟡 IBS WATCH (provisional)',
                               f"{veh} {mult:.1f}x ({ctx}) — provisional; IBS final at close", 'watch'))
            continue

        if st == 'IN' and pv_st != 'IN':
            alerts.append((f'🟢 IBS BUY: {veh} {mult:.1f}x',
                           f"Gate IN ({ctx}) → buy {veh} at {mult:.1f}x base", 'buy'))
            itr.log_firing(state, data[u].index[-1].strftime('%Y-%m-%d'), veh,
                           'gate_entry', 1.0, u, round(ibs3_by[u], 4), rsi_r, breadth, entry_price)
            _ibs_log_conviction(itr, eng, state, u, veh, mult, breadth, data, ibs3_by, rsi_r, entry_price)
        elif st == 'IN' and pv_st == 'IN' and mult != pv_m:
            arrow = '↑' if mult > pv_m else '↓'
            alerts.append((f'🟡 IBS RESIZE {arrow}: {veh}',
                           f"{pv_m:.1f}x → {mult:.1f}x ({ctx})", 'watch'))
            _ibs_log_conviction(itr, eng, state, u, veh, mult, breadth, data, ibs3_by, rsi_r, entry_price)
        elif st == 'OUT' and pv_st == 'IN':
            alerts.append((f'🔴 IBS EXIT: {veh}',
                           f"Gate OUT ({ctx}) → sell {veh}, go flat", 'exit'))

    # close-mode: mature forward returns, refresh rolling, persist tracker
    if IS_CLOSE:
        vprices = {v: data[v]['Close'] for v in set(eng.SLEEVE_MAP.values()) if v in data}
        itr.mature_results(state, lambda tk: vprices.get(tk))
        itr.snapshot(state, new_state, new_mult, breadth)
        itr.save_state(state)

    # faltering tiers -> prepend a warning alert (investigate/suspend by hand)
    falt = itr.faltering_flags(state)
    for key, stt in falt.items():
        if stt == 'FALTERING':
            base = state.get('baseline', {}).get(key, {})
            roll = state.get('rolling', {}).get(key, {})
            alerts.insert(0, ('⚠️ IBS FALTERING',
                f"{key} trailing WR {roll.get('trailing_wr', 0)*100:.0f}% vs baseline "
                f"{base.get('wr', 0)*100:.0f}% [lo {base.get('wilson_lo', 0)*100:.0f}%], "
                f"n={roll.get('trailing_n', 0)} — investigate/suspend this tier by hand", 'warning'))

    ibs_status = {
        'gate_state': new_state,
        'current_multiplier': {eng.SLEEVE_MAP[u]: (new_mult[u] if new_state[u] == 'IN' else 0.0)
                               for u in und},
        'breadth': breadth,
        'ibs_sma3': {u: (round(ibs3_by[u], 4) if ibs3_by[u] == ibs3_by[u] else None) for u in und},
        'rsi10': {u: (round(rsi_by[u], 1) if rsi_by[u] is not None else None) for u in und},
        'equity_budget': eng.EQUITY_BUDGET,
        'faltering': falt,
        'as_of': datetime.now().strftime('%Y-%m-%d'),
    }
    return alerts, ibs_status


def _ibs_log_conviction(itr, eng, state, u, veh, mult, breadth, data, ibs3_by, rsi_r, entry_price):
    """Log the conviction-tier firing when an elevated tier is entered."""
    if mult not in (eng.MULT_SINGLE, eng.MULT_BREADTH):
        return
    if breadth >= 3:
        stype = 'conviction_2.0x_breadth'
    elif eng.SLEEVE_MAP[u] == 'SOXL':
        stype = 'conviction_2.0x_single'
    else:
        stype = 'conviction_1.5x'
    itr.log_firing(state, data[u].index[-1].strftime('%Y-%m-%d'), veh, stype, mult, u,
                   round(ibs3_by[u], 4), rsi_r, breadth, entry_price)


# =============================================================================
# HORMUZ INTELLIGENCE (Windward)
# =============================================================================
# Lightweight Hormuz fetcher — mirrors the one in chf_dashboard_server.py
# but writes to the same shared history file so both systems stay in sync.
# Provides a compact email block + day-over-day attack-jump alert.
HORMUZ_HISTORY_PATH = os.environ.get('HORMUZ_HISTORY_PATH', '/tmp/hormuz_history.json')


def fetch_hormuz_windward():
    """Pull structured Hormuz data from insights.windward.ai. Returns dict
    with total_transits, vessels_in_gulf, dark_activity, attacks, intel
    summary, etc. Returns None on any failure — never raises."""
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
            "as_of": as_of,
            "total_transits": total,
            "inbound": kpis["inbound"],
            "outbound": kpis["outbound"],
            "vessels_in_gulf": kpis["vessels_in_gulf"],
            "dark_activity": kpis["dark_activity"],
            "attacks": kpis["attacks"],
            "risk": risk,
            "iran_flagged": iran_count,
            "blockade_active": blockade,
            "intel_summary": intel,
        }
    except Exception as e:
        print(f"[Hormuz] fetch error: {e}")
        return None


def hormuz_load_history():
    """Load the rolling history file. Returns [] if missing/corrupt."""
    try:
        if os.path.exists(HORMUZ_HISTORY_PATH):
            with open(HORMUZ_HISTORY_PATH, 'r') as fh:
                return json.load(fh)
    except Exception:
        pass
    return []


def hormuz_save_history(history):
    try:
        with open(HORMUZ_HISTORY_PATH, 'w') as fh:
            json.dump(history, fh, indent=2)
    except Exception as e:
        print(f"[Hormuz] save history failed: {e}")


def hormuz_upsert_history(hz):
    """Append/update today's row; return (prev_row, today_row) for delta logic."""
    history = hormuz_load_history()
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
    # Find the most recent row from a DIFFERENT as_of date for delta comparison
    today_key = row.get("as_of") or row["ts"][:10]
    prev_row = None
    for r in reversed(history):
        r_key = r.get("as_of") or (r.get("ts") or "")[:10]
        if r_key and r_key != today_key:
            prev_row = r
            break

    # Dedupe by as_of
    seen = {}
    for r in history:
        k = r.get("as_of") or r.get("ts")
        seen[k] = r
    seen[row.get("as_of") or row["ts"]] = row
    hormuz_save_history(list(seen.values())[-60:])
    return prev_row, row


def hormuz_build_alerts(hz, prev_row, today_row):
    """Return list of (title, msg, kind) tuples for Hormuz-related alerts
    that should bubble up into the main alerts panel."""
    alerts = []

    # Attack-count jump — each new attack has historically been an equity vol event
    today_attacks = today_row.get("attacks")
    prev_attacks = (prev_row or {}).get("attacks")
    if today_attacks is not None and prev_attacks is not None:
        delta = today_attacks - prev_attacks
        if delta >= 1:
            alerts.append((
                '🚨 HORMUZ ATTACK(S)',
                f"Windward reports {delta} new vessel attack(s) today "
                f"(cumulative: {today_attacks}, was {prev_attacks}). "
                f"Historically each jump is an equity vol event — watch UVXY/oil names.",
                'warning'
            ))

    # Dark activity spike — >30% day-over-day increase
    today_dark = today_row.get("dark_activity")
    prev_dark = (prev_row or {}).get("dark_activity")
    if today_dark is not None and prev_dark and prev_dark >= 20:
        if today_dark > prev_dark * 1.3:
            pct = (today_dark / prev_dark - 1) * 100
            alerts.append((
                '⚠️ HORMUZ DARK ACTIVITY SPIKE',
                f"Dark-activity events {prev_dark} → {today_dark} (+{pct:.0f}%). "
                f"Leading indicator for sanctions-evasion and interdictions.",
                'warning'
            ))

    # Blockade status change
    today_blockade = today_row.get("blockade_active")
    prev_blockade = (prev_row or {}).get("blockade_active")
    if prev_blockade is not None and today_blockade != prev_blockade:
        if today_blockade:
            alerts.append((
                '🔴 HORMUZ BLOCKADE ACTIVATED',
                f"Windward banner shows US blockade now ACTIVE (was inactive). "
                f"UNG/SLV/VLO/STNG/CF/MOS structurally supported.",
                'warning'
            ))
        else:
            alerts.append((
                '🟢 HORMUZ BLOCKADE LIFTED',
                f"Windward banner shows US blockade INACTIVE (was active). "
                f"Watch for oil/energy mean reversion.",
                'buy'
            ))

    # Transit collapse — sustained low transit count
    today_transits = today_row.get("total_transits")
    if today_transits is not None and today_transits < 15:
        alerts.append((
            '🚢 HORMUZ TRANSIT LOW',
            f"Only {today_transits} transits today (baseline ~138). "
            f"Dead-hand framework still in force — insurance withdrawn.",
            'watch'
        ))

    return alerts


def format_hormuz_block(hz, prev_row, today_row):
    """Compact Hormuz section for the email body."""
    if not hz:
        return ""

    lines = [
        "=" * 70,
        "🚢 STRAIT OF HORMUZ — WINDWARD INTELLIGENCE",
        "=" * 70,
    ]

    as_of = hz.get("as_of") or "unknown"
    blockade_txt = "🔴 BLOCKADE ACTIVE" if hz.get("blockade_active") else "BLOCKADE INACTIVE"
    lines.append(f"Data as of: {as_of}   |   {blockade_txt}")
    lines.append("")

    # KPI row
    def _fmt(kpi):
        if not kpi or kpi.get("value") is None:
            return "—"
        val = kpi["value"]
        delta = kpi.get("delta", "")
        return f"{val:,}" + (f" ({delta})" if delta else "")

    transits = hz.get("total_transits")
    transit_str = f"{transits}" if transits is not None else "—"
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
        lines.append(f"  Risk tiers:       {risk.get('high','—')} High / {risk.get('moderate','—')} Mod / {risk.get('low','—')} Low")

    # Day-over-day deltas for history context
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
            lines.append(f"  Δ vs {prev_as_of}: {', '.join(dow_deltas)}")

    # Intel summary (wrap at ~90 chars)
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
# EMAIL FUNCTIONS
# =============================================================================
def format_email(alerts, status, mode='close'):
    """Format the email body.

    mode: one of 'open' (9:45 AM), 'preclose' (11:00 AM), or 'close' (4:05 PM).
    Backwards compatible: if a boolean is passed (legacy is_preclose=True),
    it's interpreted as 'preclose'.
    """
    # Backwards compatibility: accept legacy boolean is_preclose
    if isinstance(mode, bool):
        mode = 'preclose' if mode else 'close'

    now = datetime.now()

    if mode == 'open':
        timing = "POST-OPEN SNAPSHOT (9:45 AM)"
    elif mode == 'preclose':
        timing = "MID-DAY PREVIEW (11:00 AM)"
    else:
        timing = "MARKET CLOSE CONFIRMATION (4:05 PM)"

    is_preclose_like = mode in ('open', 'preclose')
    
    body = f"""
{'='*70}
MARKET SIGNAL MONITOR - {timing}
{now.strftime('%Y-%m-%d %H:%M')} ET
{'='*70}

"""
    
    if alerts:
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
    
    body += f"""
{'='*70}
CURRENT INDICATOR STATUS
{'='*70}

"""
    
    indicators = status.get('indicators', {})
    
    key_tickers = ['SPY', 'QQQ', 'SMH', 'GLD', 'USDU', 'XLP', 'TLT', 'HYG', 'XLF', 'UVXY', 'BTC-USD', 'AMD', 'NVDA']
    body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}\n"
    body += "-"*50 + "\n"
    
    for ticker in key_tickers:
        if ticker in indicators:
            ind = indicators[ticker]
            price = f"${ind['price']:.2f}" if ind['price'] < 1000 else f"${ind['price']:,.0f}"
            rsi = f"{ind['rsi10']:.1f}"
            pct = f"{ind.get('pct_above_sma200', 0):+.1f}%"
            body += f"{ticker:<10} {price:>12} {rsi:>10} {pct:>12}\n"
    
    # Add 3x Leveraged ETFs Section
    body += f"""
{'='*70}
3x LEVERAGED ETFs
{'='*70}
"""
    body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}  Signal\n"
    body += "-"*65 + "\n"
    
    leveraged_tickers = ['NAIL', 'CURE', 'FAS', 'LABU', 'TQQQ', 'SOXL', 'TECL', 'DRN']
    for ticker in leveraged_tickers:
        if ticker in indicators:
            ind = indicators[ticker]
            price = f"${ind['price']:.2f}"
            rsi = f"{ind['rsi10']:.1f}"
            pct = f"{ind.get('pct_above_sma200', 0):+.1f}%"
            
            # Signal status
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
    
    # Add International/Other ETFs
    body += f"""
{'='*70}
OTHER ETFs
{'='*70}
"""
    body += f"{'Ticker':<10} {'Price':>12} {'RSI(10)':>10} {'vs SMA200':>12}\n"
    body += "-"*50 + "\n"
    
    other_tickers = ['XLV', 'XLU', 'XLE', 'TMV', 'VOOV', 'VOOG', 'VTV', 'QQQE', 'BOIL', 'EURL', 'YINN', 'KORU', 'INDL', 'EDC']
    for ticker in other_tickers:
        if ticker in indicators:
            ind = indicators[ticker]
            price = f"${ind['price']:.2f}" if ind['price'] < 1000 else f"${ind['price']:,.0f}"
            rsi = f"{ind['rsi10']:.1f}"
            pct = f"{ind.get('pct_above_sma200', 0):+.1f}%"
            body += f"{ticker:<10} {price:>12} {rsi:>10} {pct:>12}\n"

    # Z-Score Ratio Signals block (Group 13, Tier 2 manual execution)
    zscore_signals = status.get('zscore_signals', {})
    if zscore_signals:
        body += f"""
{'='*70}
Z-SCORE RATIO SIGNALS (Tier 2, manual execution)
{'='*70}

"""
        for ratio_name, info in zscore_signals.items():
            arrow = '≥' if info['direction'] == 'ge' else '≤'
            fire_marker = '★ FIRED' if info['fired_today'] else 'inactive'
            body += (
                f"{ratio_name:<12} z = {info['z_today']:+.2f}σ  "
                f"(trigger {arrow} {info['threshold']}σ)  {fire_marker}  "
                f"| last fire: {info['last_fire']} "
                f"({info['days_since_fire']}d ago)\n"
            )
        body += "\n"

    # IBS Dip Gate + Conviction block (Group 14, manual/IRA)
    ibs = status.get('ibs') if isinstance(status, dict) else None
    if ibs:
        sleeve_map = {'SPY': 'SPY', 'QQQ': 'TQQQ', 'SMH': 'SOXL', 'XLK': 'TECL'}
        body += f"""
{'='*70}
IBS DIP GATE + CONVICTION (Group 14, manual/IRA)
{'='*70}
"""
        gs = ibs.get('gate_state', {})
        cm = ibs.get('current_multiplier', {})
        i3 = ibs.get('ibs_sma3', {})
        r10 = ibs.get('rsi10', {})
        for u in ['SPY', 'QQQ', 'SMH', 'XLK']:
            veh = sleeve_map[u]
            st = gs.get(u, 'OUT')
            m = cm.get(veh, 0.0)
            mtxt = f"{m:.1f}x" if st == 'IN' else "--"
            s3 = i3.get(u)
            rr = r10.get(u)
            s3txt = f"{s3:.2f}" if s3 is not None else "n/a"
            rrtxt = f"{rr:.0f}" if rr is not None else "n/a"
            body += f"  {u:<4}→{veh:<5} {st:<3} {mtxt:<5} (IBS {s3txt}, RSI {rrtxt})\n"
        body += f"  Breadth (RSI10<30 of SPY/QQQ/SMH/XLK): {ibs.get('breadth', 0)}\n"
        falt = ibs.get('faltering', {})
        if falt:
            body += "  ⚠️ Faltering tiers: " + ", ".join(f"{k}={v}" for k, v in falt.items()) + "\n"

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
    
    if mode == 'open':
        body += f"""
{'='*70}
NOTE: This is a POST-OPEN snapshot. Signals reflect early trading and may
shift materially before the close. Mid-day update at 11:00 AM ET.
{'='*70}
"""
    elif mode == 'preclose':
        body += f"""
{'='*70}
NOTE: This is a MID-DAY preview. Signals may change by market close.
Final confirmation email will be sent at 4:05 PM ET.
{'='*70}
"""
    
    return body


def write_state_json(alerts, status, mode, hormuz_data=None, rj_state=None,
                     composer_summary=None, path='dashboard_state.json'):
    """
    Serialize current signal state to JSON for external consumers.

    Writes to ./dashboard_state.json in the current working directory.
    A separate workflow step copies this file to a public mirror repo
    so it's accessible at a stable raw.githubusercontent.com URL.
    """
    indicators = status.get('indicators', {}) or {}

    state = {
        'generated_at_local': datetime.now().isoformat(),
        'generated_at_utc': datetime.utcnow().isoformat() + 'Z',
        'mode': mode,
        'summary': {
            'total_alerts': len(alerts),
            'buy_count': len([a for a in alerts if a[2] == 'buy']),
            'exit_count': len([a for a in alerts if a[2] in ('exit', 'short')]),
            'warning_count': len([a for a in alerts if a[2] in ('warning', 'hedge', 'watch')]),
        },
        'alerts': [
            {'title': str(t), 'message': str(m), 'category': str(c)}
            for (t, m, c) in alerts
        ],
        'indicators': {},
        'smh_days_below_sma200': int(status.get('smh_days_below_sma200', 0) or 0),
    }

    # Flatten indicators to JSON-safe values, drop NaN
    def _safe_num(v, places=2):
        try:
            f = float(v)
            if f != f:  # NaN check
                return None
            return round(f, places)
        except (TypeError, ValueError):
            return None

    for ticker, ind in indicators.items():
        state['indicators'][ticker] = {
            'price':              _safe_num(ind.get('price'), 2),
            'rsi10':              _safe_num(ind.get('rsi10'), 1),
            'rsi50':              _safe_num(ind.get('rsi50'), 1),
            'sma200':             _safe_num(ind.get('sma200'), 2),
            'sma50':              _safe_num(ind.get('sma50'), 2),
            'ema21':              _safe_num(ind.get('ema21'), 2),
            'pct_above_sma200':   _safe_num(ind.get('pct_above_sma200'), 2),
        }

    # Robot James snapshot (subset of full state, enough for dashboard reads)
    if rj_state is not None:
        try:
            state['robot_james'] = {
                'current_td': rj_state.get('current_td'),
                'total_tds_this_month': rj_state.get('total_tds_this_month'),
                'action_type': (rj_state.get('action') or {}).get('type'),
                'phase': rj_state.get('phase'),
            }
        except Exception:
            pass

    # Hormuz snapshot (top-line numbers only)
    if hormuz_data is not None:
        try:
            state['hormuz'] = {
                'as_of': hormuz_data.get('as_of'),
                'total_transits': hormuz_data.get('total_transits'),
                'attacks': (hormuz_data.get('attacks') or {}).get('value'),
            }
        except Exception:
            pass

    # Composer dry-run preview summary (top-line counts + total $)
    if composer_summary is not None:
        try:
            state['composer_dry_run'] = {
                'symphonies_evaluated':   int(composer_summary.get('symphonies_evaluated', 0)),
                'symphonies_rebalancing': int(composer_summary.get('symphonies_rebalancing', 0)),
                'total_pv':               _safe_num(composer_summary.get('total_pv'), 2),
            }
        except Exception:
            pass

    # Group 13 z-score ratio signals (per-ratio current state)
    zscore_signals = status.get('zscore_signals') if isinstance(status, dict) else None
    if zscore_signals:
        try:
            def _int_or_neg1(v):
                # 0 is a valid count (fired today), so `int(v or -1)` is wrong
                if v is None:
                    return -1
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return -1
            state['zscore_signals'] = {
                name: {
                    'z_today':         _safe_num(info.get('z_today'), 3),
                    'threshold':       _safe_num(info.get('threshold'), 2),
                    'direction':       info.get('direction'),
                    'fired_today':     bool(info.get('fired_today')),
                    'last_fire':       info.get('last_fire'),
                    'days_since_fire': _int_or_neg1(info.get('days_since_fire')),
                }
                for name, info in zscore_signals.items()
            }
        except Exception:
            pass

    # Group 14 IBS Dip Gate + Conviction (current gate state + multipliers + flags)
    if isinstance(status, dict) and status.get('ibs'):
        try:
            state['ibs'] = status['ibs']
        except Exception:
            pass

    # Retirement projection snapshot (v5.17 CHF dashboard writes retirement_state.json;
    # the monitor merges it here if it's available in this environment, and omits the
    # block gracefully if not — the monitor has no portfolio access of its own).
    try:
        ret_candidates = [
            os.environ.get('RETIREMENT_STATE_JSON'),
            'retirement_state.json',
            os.path.join(os.environ.get('PORTFOLIO_SNAPSHOT_REPO', ''), 'data',
                         'retirement_state.json') if os.environ.get('PORTFOLIO_SNAPSHOT_REPO') else None,
            os.path.expanduser('~/portfolio-snapshot/data/retirement_state.json'),
        ]
        for rp in ret_candidates:
            if rp and os.path.isfile(rp):
                with open(rp) as rf:
                    state['retirement'] = json.load(rf)
                print(f"[STATE] Merged retirement block from {rp}")
                break
    except Exception as e:
        print(f"[STATE] Retirement block skipped: {e}")

    try:
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        print(f"[STATE] Wrote {path}")
    except Exception as e:
        print(f"[STATE] Failed to write {path}: {e}")

    return path


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
    if IS_OPEN:
        _mode_label = "OPEN (9:45 AM)"
    elif IS_PRECLOSE:
        _mode_label = "MID-DAY (11:00 AM)"
    else:
        _mode_label = "CLOSE (4:05 PM)"
    print(f"Mode: {_mode_label}")
    
    tickers = [
        # Core Indices
        'SMH', 'SPY', 'QQQ', 'IWM',
        # Tech sector (IBS Group 14 breadth + TECL sleeve)
        'XLK',
        # Defensive Sectors
        'XLP', 'XLU', 'XLV',
        # Safe Havens & Macro
        'GLD', 'TLT', 'HYG', 'LQD', 'TMV',
        'USDU', 'UCO', 'BOIL',
        # Volatility
        'UVXY',
        # International
        'EDC', 'YINN', 'KORU', 'EURL', 'INDL',
        # Crypto
        'BTC-USD',
        # Individual Stocks
        'AMD', 'NVDA',
        # 3x Leveraged ETFs
        'NAIL', 'CURE', 'FAS', 'LABU',
        'TQQQ', 'SOXL', 'TECL', 'DRN', 'UPRO',
        # Style/Factor ETFs
        'VOOV', 'VOOG', 'VTV', 'QQQE',
        # Energy
        'XLE', 'XLF',
        # Robot James overlay (leveraged bonds)
        'TMF',
        # Equal-weight S&P 500 (for SIGNAL GROUP 13 z-score ratios)
        'RSP',
    ]

    print("Downloading market data...")
    data = download_data(tickers)
    print(f"Downloaded data for {len(data)} tickers")

    # SIGNAL GROUP 13 z-score ratios need 504-day regression window + extra
    # history to locate last-fire dates several years back. Refetch the four
    # ratio tickers at 5y and overwrite the 2y entries in `data`.
    try:
        zscore_5y = download_data(['QQQ', 'SPY', 'RSP', 'QQQE'], period='5y')
        for t, df in zscore_5y.items():
            data[t] = df
        print(f"[Group13] Refetched 5y history for {len(zscore_5y)} z-score tickers")
    except Exception as e:
        print(f"[Group13] Failed to refetch 5y history: {e}")

    alerts, status = check_signals(data)

    # =========================================================================
    # HORMUZ INTELLIGENCE (Windward)
    # =========================================================================
    hormuz_data = None
    hormuz_block = ""
    hormuz_alerts = []
    try:
        hormuz_data = fetch_hormuz_windward()
        if hormuz_data:
            prev_row, today_row = hormuz_upsert_history(hormuz_data)
            hormuz_alerts = hormuz_build_alerts(hormuz_data, prev_row, today_row)
            hormuz_block = format_hormuz_block(hormuz_data, prev_row, today_row)
            print(f"[Hormuz] as_of={hormuz_data.get('as_of')} "
                  f"transits={hormuz_data.get('total_transits')} "
                  f"attacks={(hormuz_data.get('attacks') or {}).get('value')} "
                  f"hormuz_alerts={len(hormuz_alerts)}")
            # Prepend Hormuz alerts to main alerts list so they surface in subject + panel
            alerts = hormuz_alerts + alerts
    except Exception as e:
        print(f"[Hormuz] unexpected error: {e}")
    
    # =========================================================================
    # ROBOT JAMES STATE
    # =========================================================================
    rj_state = None
    rj_subj_override = ""
    rj_block = ""
    rj_postclose_block = ""
    try:
        rj_state = rj_compute_state(data)
        if rj_state is not None:
            rj_write_state(rj_state)
            if IS_PRECLOSE_LIKE:
                rj_subj_override, rj_block = rj_render_block(rj_state)
            else:
                # Post-close: confirmation block on action days only
                close_prices = {}
                for t in ['TMF', 'TQQQ', 'UPRO', 'SOXL', 'SPY', 'QQQ', 'SMH', 'GLD']:
                    if t in data and len(data[t]) > 0:
                        close_prices[t] = safe_float(data[t]['Close'].iloc[-1])
                rj_postclose_block = rj_render_postclose(rj_state, close_prices)
            print(f"[RJ] TD {rj_state['current_td']}/{rj_state['total_tds_this_month']} "
                  f"action={rj_state['action']['type']}")
    except Exception as e:
        print(f"[RJ] Error computing state: {e}")

    # =========================================================================
    # COMPOSER DRY-RUN PREVIEW
    # =========================================================================
    # Asks Composer's API what each symphony will trade at next rebalance.
    # Self-contained: no Composer creds → graceful skip with empty block.
    composer_block = ""
    composer_summary = None
    try:
        # Importable because composer_dry_run.py sits next to this script
        import sys as _sys
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in _sys.path:
            _sys.path.insert(0, _here)
        from composer_dry_run import (
            fetch_dry_run_preview,
            parse_dry_run_response,
            format_dry_run_for_email,
        )

        raw = fetch_dry_run_preview()  # account_uuids=None → all accounts on the API key
        if raw is None:
            print("[Composer] dry-run unavailable (no API key, network error, or auth failure)")
        else:
            parsed = parse_dry_run_response(raw)
            if not parsed:
                print(f"[Composer] dry-run returned 0 symphonies "
                      f"(accounts in response: {len(raw)})")
            else:
                will_rebalance = sum(1 for r in parsed if r.get('will_rebalance'))
                total_pv = sum((r.get('symphony_value') or 0) for r in parsed)
                composer_summary = {
                    'symphonies_evaluated': len(parsed),
                    'symphonies_rebalancing': will_rebalance,
                    'total_pv': round(float(total_pv), 2),
                }
                composer_block = format_dry_run_for_email(parsed,
                                                          total_portfolio_value=total_pv)
                print(f"[Composer] dry-run: {len(parsed)} symphonies, "
                      f"{will_rebalance} rebalancing, total_pv=${total_pv:,.0f}")
    except Exception as e:
        print(f"[Composer] error: {e}")

    # Mode label for subject line
    if IS_OPEN:
        timing_label = "OPEN"
        mode_str = 'open'
    elif IS_PRECLOSE:
        timing_label = "MID-DAY"
        mode_str = 'preclose'
    else:
        timing_label = "CLOSE"
        mode_str = 'close'

    # Subject line
    if rj_subj_override:
        # RJ action day -- RJ subject wins
        subject = rj_subj_override
    elif alerts:
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
        subject = f"{emoji} [{timing_label}] Market Signals: {len(alerts)} Alert(s) - {urgency}"
    else:
        subject = f"📊 [{timing_label}] Market Signals: No Alerts"

    # Body: prepend RJ block (preclose action OR hold-day info) or postclose confirmation
    # Then Hormuz intel block (if available)
    body_prefix = ""
    if rj_block:
        body_prefix = rj_block + "\n\n"
    elif rj_postclose_block:
        body_prefix = rj_postclose_block + "\n\n"
    if hormuz_block:
        body_prefix += hormuz_block + "\n"
    if composer_block:
        body_prefix += composer_block + "\n"
    body = body_prefix + format_email(alerts, status, mode_str)

    # Write machine-readable state for external consumers (dashboard mirror, etc.)
    try:
        write_state_json(alerts, status, mode_str,
                         hormuz_data=hormuz_data,
                         rj_state=rj_state,
                         composer_summary=composer_summary)
    except Exception as e:
        print(f"[STATE] Unexpected error writing state: {e}")

    send_email(subject, body)

    print(f"\n{len(alerts)} signal(s) detected")
    for title, msg, _ in alerts:
        print(f"  {title}")

    # =========================================================================
    # PRICE STORE -- refresh the self-updating historical price store on the
    # post-close (4:05 PM ET) run only. Fully isolated: this is the LAST thing
    # main() does and is double-wrapped (here + inside update_price_store) so a
    # failure can NEVER prevent the signal snapshot/email above from completing.
    # =========================================================================
    if IS_CLOSE:
        try:
            # os/sys are module-level imports; do NOT re-import them here — a
            # local `import os` would make `os` function-local to main() and
            # break the earlier Composer dry-run block (UnboundLocalError).
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))  # repo root (price_store.py lives there)
            from price_store import update_price_store
            print("\n[PRICE STORE] post-close update starting...")
            update_price_store()
        except Exception as e:
            print(f"[PRICE STORE] update failed (suppressed, snapshot already written): {e}")

        # Daily signal-research scan over the freshly-updated store. Separate
        # guard so a research failure can't undo the price-store update above.
        try:
            # os/sys are module-level; see note above — no local re-import.
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))  # repo root (signal_research.py lives there)
            from signal_research import run_signal_research
            print("\n[RESEARCH] daily signal-research scan starting...")
            run_signal_research(write=True)
        except Exception as e:
            print(f"[RESEARCH] scan failed (suppressed): {e}")

if __name__ == "__main__":
    main()
