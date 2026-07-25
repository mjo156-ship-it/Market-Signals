"""
synth_splice.py — calibrated pre-inception splices for baseline priming.

Used ONLY by ibs_tracker.prime_history() to build the full-cycle (Update 2)
baseline. NEVER used live. The pre-inception segments are ILLUSTRATIVE: the
synthetic 3x understates the choppy-crash decay of the real leveraged ETF, and
the BTAL proxy understates real crash response — so any pre-inception baseline
number the dashboard shows must be flagged as such.

CANONICAL COPY — keep in sync with ~/chf-dashboard/synth_splice.py.
"""
import pandas as pd


def synth_splice(underlying_ret, real_lev_ret):
    """Splice real leveraged-ETF daily returns with calibrated synthetic-3x
    before the ETF's inception (Update 3).

    underlying_ret : daily returns of the 1x underlying (QQQ / SMH / XLK), full
                     history back to ~2000.
    real_lev_ret   : daily returns of the real leveraged ETF (TQQQ / SOXL /
                     TECL); NaN before inception.

    Returns one spliced daily-return series: 3*underlying - drag before
    inception, real ETF after. `drag` is fit to the real ETF over the overlap so
    the synth CAGR matches within 1-2 pts (validated: overlap corr 0.999 / 0.982
    / 0.995 for TQQQ / SOXL / TECL).

    Underlyings -> vehicles: QQQ->TQQQ, SMH->SOXL, XLK->TECL. The SPY sleeve is
    real 1x (no splice) unless SPY_SLEEVE_VEHICLE == "UPRO".
    """
    ov = underlying_ret.index.intersection(real_lev_ret.dropna().index)
    drag = (3 * underlying_ret.reindex(ov) - real_lev_ret.reindex(ov)).mean()  # fit to real ETF
    synth = 3 * underlying_ret - drag
    pre = synth.index.difference(real_lev_ret.dropna().index)
    return pd.concat([synth.reindex(pre), real_lev_ret.dropna()]).sort_index()


def btal_anti_beta_proxy(sector_rets):
    """Pre-2011 BTAL proxy (Update 4, baseline only, NEVER live):

        0.39 * (mean[XLP, XLU, XLV] - mean[XLK, SMH])

    sector_rets : DataFrame with daily-return columns XLP, XLU, XLV, XLK, SMH.
    Calibrated to real BTAL (daily corr ~0.35). It UNDERSTATES the real crash
    response, so treat proxy-era BTAL contribution as a CONSERVATIVE FLOOR.
    """
    longs = sector_rets[["XLP", "XLU", "XLV"]].mean(axis=1)
    shorts = sector_rets[["XLK", "SMH"]].mean(axis=1)
    return 0.39 * (longs - shorts)


def btal_splice(sector_rets, real_btal_ret):
    """Splice the pre-2011 anti-beta proxy with real BTAL from 2011-09 forward.

    Returns one daily-return series: proxy before BTAL inception, real after.
    """
    real = real_btal_ret.dropna()
    proxy = btal_anti_beta_proxy(sector_rets)
    pre = proxy.index.difference(real.index)
    return pd.concat([proxy.reindex(pre), real]).sort_index()
