"""
ibs_engine.py — IBS Dip Gate + Conviction Multiplier (shared, pure functions).

CANONICAL COPY — KEEP IN SYNC across BOTH repos (no shared package spans them):
    ~/Market-Signals/ibs_engine.py   → imported by the signal monitor (TIER 10)
    ~/chf-dashboard/ibs_engine.py    → imported by chf_dashboard_server.compute_ibs_signals()
Edit one, edit the other, or they drift.

Strategy (see ibs_conviction_strategy_and_build.md + ibs_build_addendum.md, 2026-07-22):
  - Per sleeve, an IBS_sma3 gate with hysteresis (0.40 entry / 0.60 exit) sizes a
    base position.
  - A conviction multiplier (1.0 / 1.5 / 2.0x) scales the base on deep-oversold
    (own RSI10 < 30) or broad-washout (breadth >= 3) days. SOXL uses 2.0x at the
    single-name tier. SPY is the "IBS-alone" sleeve (no multiplier).

Pure functions only. NO network, NO file I/O, NO clock.
"""
import numpy as np
import pandas as pd

# --- Update 5: SPY-sleeve vehicle is configurable ----------------------------
# "SPY" (default): SPY 1x, the IBS-alone sleeve.
# "UPRO": leveraged broad-market sleeve — raises the SPY-sleeve return AND risk.
SPY_SLEEVE_VEHICLE = "SPY"

SLEEVE_MAP = {
    "SPY": SPY_SLEEVE_VEHICLE,   # Update 5: single source of truth for the SPY vehicle
    "QQQ": "TQQQ",
    "SMH": "SOXL",
    "XLK": "TECL",
}
BREADTH_UNIVERSE = ["SPY", "QQQ", "SMH", "XLK"]

ENTRY, EXIT = 0.40, 0.60          # IBS_sma3 gate thresholds (hysteresis band)
MULT_SINGLE, MULT_BREADTH = 1.5, 2.0
SOXL_SINGLE = 2.0                 # SOXL single-name (RSI10<30) tier override
NO_MULTIPLIER = {"SPY"}           # SPY is the IBS-alone sleeve — base gate only

# Base ("1.0x") notional per sleeve as a fraction of the equity budget. Four
# sleeves x 0.25 = 1.00 gross when all are IN at 1.0x (before the cap). This is
# a sizing knob, not a signal parameter.
BASE_WEIGHT = 0.25

# --- Update 1: gross-exposure cap (no-margin safety) -------------------------
# Stacked multipliers push gross equity >100% on ~3% of days (max ~140%) when
# several sleeves hit high conviction at once (breadth >= 3). The IRA can't use
# margin, so cap gross. 0.80 if running 20% BTAL; set 1.00 if no BTAL / no cash
# reserved.
EQUITY_BUDGET = 0.80


def apply_exposure_cap(positions: dict, budget=EQUITY_BUDGET):
    """Scale sleeve weights down proportionally if gross exposure exceeds budget."""
    gross = sum(positions.values())
    if gross > budget:
        scale = budget / gross
        return {k: v * scale for k, v in positions.items()}
    return positions


def compute_ibs(df):
    """IBS = (Close - Low) / (High - Low). df needs High, Low, Close columns."""
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    return (df["Close"] - df["Low"]) / rng


def ibs_sma3(df):
    """3-day SMA of IBS (the gate input)."""
    return compute_ibs(df).rolling(3).mean()


def gate_state(ibs_sma3_today, prev_state):
    """Hysteresis gate. prev_state in {"IN","OUT"}. NaN safely holds state."""
    if prev_state != "IN" and ibs_sma3_today < ENTRY:
        return "IN"
    if prev_state == "IN" and ibs_sma3_today > EXIT:
        return "OUT"
    return prev_state


def breadth_count(rsi_by_ticker):
    """How many of {SPY,QQQ,SMH,XLK} have RSI(10) < 30. dict {ticker: rsi10}."""
    return sum(1 for t in BREADTH_UNIVERSE
               if rsi_by_ticker.get(t) is not None and rsi_by_ticker[t] < 30)


def conviction_multiplier(underlying, own_rsi10, breadth):
    """First matching tier wins: breadth>=3 -> 2.0x; own RSI10<30 -> 1.5x
    (2.0x for SOXL); else 1.0x. SPY (NO_MULTIPLIER) is always 1.0x."""
    if underlying in NO_MULTIPLIER:
        return 1.0
    if breadth >= 3:
        return MULT_BREADTH
    if own_rsi10 is not None and own_rsi10 < 30:
        return SOXL_SINGLE if SLEEVE_MAP[underlying] == "SOXL" else MULT_SINGLE
    return 1.0


def sleeve_positions(gate_states, multipliers,
                     base_weight=BASE_WEIGHT, budget=EQUITY_BUDGET):
    """Build per-sleeve gross weights (keyed by underlying) with the Update-1 cap
    applied. IN sleeves get base_weight * multiplier; OUT sleeves get 0.0.

    gate_states: {underlying: "IN"/"OUT"}   multipliers: {underlying: float}
    Returns {underlying: capped_weight} for every underlying in SLEEVE_MAP.

    Both callers (dashboard compute_ibs_signals and monitor TIER 10) MUST use
    this so the cap is applied identically and the two never drift.
    """
    raw = {u: (base_weight * multipliers.get(u, 1.0)) if gate_states.get(u) == "IN" else 0.0
           for u in SLEEVE_MAP}
    active = {u: w for u, w in raw.items() if w > 0}
    capped = apply_exposure_cap(active, budget)
    return {u: capped.get(u, 0.0) for u in SLEEVE_MAP}
