#!/usr/bin/env python3
"""
composer_oos_watchlist.py — in-sample vs out-of-sample split for watched symphonies.

For each watchlist symphony whose logic was frozen >= 6 months ago (its OOS
inception = Composer's last_semantic_update_at), backtest the full history with
the CURRENT (frozen) definition, split the equity curve at the freeze date, and
compute stats on each side:

    in-sample  = backtest over dates BEFORE the freeze (the fitted history)
    out-sample = backtest over dates ON/AFTER the freeze (genuine forward test)

The gap (OOS CAGR - in-sample CAGR) is the decay/robustness signal. Read-only:
uses composer_oos.backtest (the one permitted POST) + _curve_stats. Commits
data/composer_oos_watchlist.jsonl and prints a ranked report.
"""
from __future__ import annotations
import os, sys, json, time
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos as oos

EARLY_START = "2015-01-01"
OUT = os.environ.get("COMPOSER_OOS_WL_PATH", "data/composer_oos_watchlist.jsonl")

# (id, oos-freeze date, short name) for every watchlist symphony frozen >= 6mo
# ago, extracted from the watchlist response (last_semantic_update_at).
WATCH = [
    ("0DgISaXxMYwUMnaF4uFM", "2025-07-02", "B- DC trip 99/13"),
    ("0TMH5QEjuNag5HL4wSAr", "2025-06-29", "D- Base 2 TECL 383/38"),
    ("1gHhOZ0K5NHbSMyh1Po0", "2023-06-05", "Copy of Ease Up on the Gas V2a"),
    ("1i91IlJvdSl18BrXgLvW", "2024-11-02", "Copy of Wash Sale 3 by 5d MaxDD"),
    ("22nOCrE8hEqzqxu9kPzn", "2025-06-07", "Mag7 Winners & Buy Dip 211/17"),
    ("28m5GEDrcKoHGe6nfuHX", "2023-05-16", "Commodities/Bonds - Weekly [shared]"),
    ("2lTE6fQMAuyAL4dAM86a", "2024-10-24", "Industrials (No VIX) 237/26"),
    ("2NBBTwkY9q7P57jOXsZg", "2025-02-05", "Monthly Dividend 48/23"),
    ("3gALwork9mg3zQTmjxmO", "2025-10-29", "D- Sisyphus (250/11/0) VIX"),
    ("3W80K6PVgou3IF93Un0N", "2025-07-11", "The AI 500 [shared]"),
    ("48bXsZqImB7fFMrvqIV7", "2025-05-20", "D - Semis then sort w/ TAIL 601/15"),
    ("4f43290YwG1ocYGTSbAp", "2025-07-28", "Consumer Staples (No VIX)"),
    ("4R1sB8GmUga5LQD0DU2C", "2025-06-27", "B- AI & Big Data +TAIL/TQQQ 43/19"),
    ("4y37EiFDfW8Rb3Q9OsMD", "2025-10-31", "B- Survival Bots 104/11/.32"),
    ("74eyvSoZXhw4ouk2fuPU", "2025-08-18", "C- Avoid Volatility (VIX) 89/28"),
    ("8ilUSeGVgqiWLCi3WNjq", "2025-10-13", "TQQQ during dips (no VIX)"),
    ("a5dJAzPC2tz5QOeUfcUh", "2025-10-29", "Pelosi's Chips Modified 131/46"),
    ("AknCRuUPkmu9Bq7L6VhR", "2025-04-11", "C- Bulls w/ Blue Chip Back-Up 177/21"),
    ("AxrUstLwZz9GrglAZD9D", "2025-02-17", "B - PUNCS VIXM filter 60/22"),
    ("B1OR4bs3NMgjaz1TJl4e", "2025-05-30", "South Korea (No VIX)"),
    ("b4rQcJPcXvi3YfnDP9e7", "2025-06-28", "B- PFIX/BITX/USMV/Metals 35/23"),
    ("b8JXr9dSjY8dJKahbBre", "2025-05-20", "Emerging Markets (No VIX)"),
    ("bEItOV1Os2iBkkrTTcl4", "2025-07-13", "SIMPLICITY, SPMO 46/33"),
    ("BkSoFieJXCYgRzaq5HbT", "2022-09-12", "Copy of Apple Long Term Investing"),
    ("BRBNlVzndKRdqyQTGiBr", "2025-09-25", "Old Economy2"),
    ("bZu9LeaC4lvJztDUPYqM", "2025-07-25", "C- SIMPLE yet Uncorrelated R2 .25 42/7"),
    ("c1bWgoNRaiff8ZZwBGD0", "2022-02-13", "Stoic: Inflation Spiral Hedge [shared]"),
    ("Cid5eqKgassokbnqDiXR", "2025-01-20", "Overheat, 19/25 BACK TO 2008"),
    ("CLWnmpUkjArmrAJP2A38", "2025-07-24", "Copy of Newish Short End Bond"),
    ("CXUAXghXz92ln6ylUSGA", "2025-01-29", "Pharma Bull 126/25 (UVXY)"),
    ("dh16jpkc2lENdpE6xXmw", "2025-02-16", "VIXM filter 62/27 to 2012"),
    ("dHpFRQigYCLjXhQV1bCF", "2025-05-20", "C - QQQ/TAIL Diverge VIXM 103/25"),
    ("DvDegddskNkpVqqx33RX", "2025-10-31", "D - Top Sort Leverage (138/11)"),
    ("DyUm8IplFB2tQWRgEbOF", "2025-08-24", "C- Ex-US Markets 136/18"),
    ("EaBXOYOQAHod3o86bRgR", "2025-05-27", "Catch-Up Trade (TNA,MIDU,DUSL,LABU)"),
    ("EC3mH97dI0EszW4FQ2GW", "2025-07-12", "C- SIMPLICITY, UPRO 80/34"),
    ("EeeiIGPX8ZELFTsfcyKs", "2025-05-03", "India 22/13 (BTAL)"),
    ("EqDy5AUydosWUnDMGVZA", "2025-05-10", "D - SOXL mostly simple (238/21)"),
    ("es21H6YhrliF5k4KGLw1", "2025-06-28", "B- Blue Chip Backup 34/15 (no VIX)"),
    ("es9fE4NgF1k0mjUzvL1g", "2025-06-24", "Semiconductors #1 751/26"),
    ("f0znz9eW3tl0JxT0tE4i", "2025-07-01", "Copy of Simons KMLM switcher"),
    ("FezxTWwSjfOuLeJWRZBL", "2025-06-29", "D- Mag7 Hybrid (UVIX) 248/40"),
    ("fjL2dOwIBlrbZAgojfnj", "2025-07-24", "Copy of N1"),
    ("FOdFSlCAKDKOrVaCezx3", "2025-05-06", "C - Four Corners 156/19"),
    ("goy4idUzTfOXoGemr4Ov", "2025-07-20", "SHANE SIMPLICITY TQQQ/UPRO/SPMO 81/24"),
    ("gRUZY4Wgflcgl8qN4ZHs", "2024-03-22", "Base 2 - LABU Version"),
    ("gUYcMk9mUrR8YSpVuNcB", "2026-01-06", "C- WW Growth (UVXY) 113/14"),
    ("h39G48Umhln3Yn1q5q2a", "2025-12-24", "C- SIMPLICITY VOO QQQ TQQQ 81/24"),
    ("HAmZLe9eZpdCNONzhKmi", "2025-08-11", "D- Bitcoin, Gold, TECL & TAIL 193/20"),
    ("HcePoNlW5OGwA8fUq2Pl", "2025-04-07", "Insurance Bull 169/23 (UVXY)"),
    ("HgK8mCeBnH4fQFNcfZ7q", "2022-07-28", "Inside Nancy Pelosi's Chips V3 [shared]"),
    ("hnY2FmKgz3CowjrPpTp4", "2025-09-11", "Old Economy"),
    ("hRSa4zuQtstfj6DO5Sbr", "2024-10-10", "Semiconductors #3 875/23"),
    ("IbiH4hBpMDbLolfDBVDf", "2025-06-24", "FNGU"),
    ("iHFLIhls4pa2sOclgRSo", "2025-05-30", "Master Balancer"),
    ("IIxLbW1WcWb80lz5ZzSo", "2025-07-20", "D- SIMPLICITY+RATES QQQ 205/34"),
    ("INoh16ZWtXzayakJ7Zul", "2025-10-31", "D - Battle Bots 192/15/.22"),
    ("IOcJQgrFTUA44WJuej15", "2022-02-13", "Copy of Buy the Dips: Nasdaq 100"),
    ("iOkxIU2ruIVWGH0f2zqb", "2025-06-16", "Nat Gas 20-Day EMA 864/32"),
    ("ipvxht5UbCPvNdYR9lW5", "2025-04-04", "Holy Grail plus Frontrunner 252/13"),
    ("IS9h9L3EbaFc4jzcL3kD", "2025-08-18", "C- SIMPLICITY TOP 3TECH"),
    ("J3eDjCGZ2NEKRtMR24fV", "2025-05-25", "Copy of Copy of Holy Grail | KMLM"),
    ("JW49FgJ2mZKASSpbtpfO", "2025-06-29", "D- FNGU w TAIL/PFIX 136/18"),
    ("K3YAdutrF96KSxVTpuoU", "2025-07-23", "SPMO as Proxy 60/22"),
    ("k9D8PYPa0MzaoUxMYHLG", "2025-10-31", "A- messin around 28/14"),
    ("KbnDRwd2amwAE7c0C4si", "2025-05-20", "Mexico (No VIX)"),
    ("KfIsAC2FEPyPrQdMMBBd", "2025-08-24", "Real Estate (Some VIX) 338/32"),
    ("kHcoGXd7034USpBBcDhN", "2025-09-29", "Gold (No VIX)"),
    ("KIZkZ4Jq4aC09ObGELmy", "2024-10-01", "Gasoline (UVXY)"),
    ("krQf2nkztcGoFICgmHso", "2025-07-29", "A- Super Trend no VIX (77/16/.51)"),
    ("KTstjsvPIkU5alZUjJu8", "2025-06-26", "D- Top 5 Tech (UVXY)"),
    ("KUksUkhpNOJdiqMoss9r", "2025-09-16", "D- Top of Tech or Super Trend/QQQ dip"),
    ("L7KJCM7HZCIJF07PcvHI", "2025-08-01", "*- High Yield Bond / Income 18/8"),
    ("LFz7Ps2NwcrbwNfcK21o", "2025-06-09", "FNGU (No VIX) 533/38"),
    ("ljdkhTQX8c0TLZoith61", "2024-03-31", "Copy of Base 2 - High Beta"),
    ("LY1kRX0zMarzOQR93P9g", "2025-07-02", "Diversification, Lagging Sectors"),
    ("M3QpEsKtFFsoSpdgLtph", "2025-05-20", "Healthcare (Some VIX) 234/23"),
    ("MjdSVGCm7sIO7ziUFkpw", "2025-05-20", "10d RSI SPY v CORP [shared]"),
    ("MOIK2ZETLOjwKCQqQ5NQ", "2025-06-07", "Biotech (Some VIX) 352/35"),
    ("NBGy93LePACWUqlRYpSA", "2024-08-06", "D- Base - SOXL"),
    ("NxfBdt0Kokogj6h7RFBG", "2025-01-11", "equal weighted leverage"),
    ("OghR4VqAfN0qjKrstQK8", "2025-09-16", "C- All things in balance 157/15"),
    ("ohSQn2Tjgfx6plvIwPdG", "2024-03-21", "UVXY Hedge"),
    ("OiCSKPW7fECY4hQNNHVR", "2025-06-06", "Extended Beta Baller (SOXL long)"),
    ("OU8UQTPGvy2CnmFLZvF3", "2025-06-22", "C- Tech Bull 146/13 (UVXY)"),
    ("P7UeIapdg2Yqi0uaQOPY", "2025-06-29", "D- EqWt 224/27 SPY/QQQ/SMH/VTV"),
    ("PkfK83gPWFTyuv9bQI5Y", "2025-05-09", "Copy of Consumer Staples (No VIX)"),
    ("qz8qwrhCrkDZwXwHy4d8", "2025-05-24", "C- FAS 20-Day EMA 164/28"),
    ("RJ5wy9GhYERZeckJtRe1", "2025-05-27", "B- QQQ enhanced 69/25"),
    ("rjhBONNBaA5qodjPNvoi", "2025-01-29", "Healthcare Bull 137/23 (UVXY)"),
    ("s794yZ3rZR2hT6aRk2Hz", "2025-06-29", "D- TECL Bull 292/38 (UVXY)"),
    ("sbrkjgeP7fYxwbqbHosw", "2025-08-10", "NDX Top 100, no vix 30/24"),
    ("scayu5ONQOgSJfp8IPz2", "2025-10-06", "B- Tech & SIN 140/12"),
    ("t73IMAAMjCgomuN6LngF", "2025-05-20", "Diversified Commodities 497/18"),
    ("TGJQs49qL6sOq2U6sWVD", "2025-05-23", "RSI Sorter"),
    ("U5D1j9XERt0iol68BCj3", "2025-07-11", "B - QQQ/TAIL Diverge, no VIX 32/14"),
    ("UCUOpRsvpy0AbNtVv3R4", "2025-05-05", "C - Energy or SPY (BTAL) + Insurance"),
    ("UjvVX9WI16zCT7G6LY2Q", "2025-05-15", "Copy of (A) 20%/3% MDD"),
    ("UQSiaPcyb0kp7GpW1mzG", "2025-05-20", "D - Top Sort by 20d return 124/20"),
    ("vaS4q8LplmqsoxzAk0wI", "2025-05-21", "Energy (UVXY) 658/22"),
    ("VF81LpW8Ro5hwgP6bxAH", "2025-05-20", "D- Dips & Vix 450/18 (UVXY)"),
    ("VrrYo0iqezW5BZi0Bd5R", "2025-07-07", "D- QQQ LESS SIMPLICITY 111/24"),
    ("VwcPcYfP7ZiLMK2vbbpZ", "2025-06-20", "D- Tech, TAIL & VIXM Hedge 133/9"),
    ("W7WT305mhZB3UHLHKvNz", "2023-12-15", "Tech or Sector Momentum [shared]"),
    ("wQiJcF0aADrKVDaxPLDJ", "2022-09-27", "Slow and Steady Growth (UUP/TLT Mod)"),
    ("wRB3FMRwr30GZ0g1Simr", "2025-05-14", "Copy of Nuclear v2"),
    ("Xk6wYgPK8RggrPAAn0Iq", "2026-01-17", "Best of QQQ 75/15/.67"),
    ("xN5Hi5Hv94gRHZynUTj5", "2025-07-06", "SIMPLICITY, SPY 36/27"),
    ("xXe1F2vjfzWkOViIpNJX", "2025-06-10", "C - EQ WT, Risk Averse 99/17"),
    ("yHRXt40v63jzODUHvTkn", "2025-07-14", "C- SIMPLICITY, QUANTUM 216/37"),
    ("YNEcGCD8kceDvWytUjVF", "2022-12-08", "TQQQ For The Long Term V4.1 [shared]"),
    ("yoqqOgvzkoFLyOqZoYQd", "2025-05-20", "Utilities (No VIX) 202/41"),
    ("YszEXaJnydfKj2GLNR44", "2025-07-20", "C- LUKE SIMPLICITY UPRO/TQQQ 108/32"),
    ("yufIIgEQ7Gi8wJtrk5Cp", "2025-07-03", "A- SIN 31/14"),
    ("YuXxG3yXDQQzOaWf9hAc", "2025-10-31", "D - High Beta (Low VIX) 126/19"),
    ("Z5VmkFJNegQsavTwNbKo", "2025-06-11", "D- FNGU to the Third 413/44"),
    ("ziRhQASn0OdLZEdHtFxZ", "2024-09-07", "UPRO FTLT (Azqato Longer BT) [shared]"),
    ("ZlRSaY5tcXO9hMU5fbWM", "2025-06-16", "Diversified Laggard (779/21)"),
    ("zNAFpyOkgaQvddIrazsM", "2026-01-17", "D - Best of the Best 84/24/.65"),
    ("ZrC4sdNzB9stQemijK3i", "2025-05-20", "Yippe Kaye 332/16 (UVXY)"),
    ("zU3PN537XLaI4Wvumebp", "2025-07-02", "D- Diversification #2 305/43"),
]


def _cs(c):
    return oos._curve_stats(c)


def _ext(curve):
    """Extra risk metrics from an equity curve, to complement _curve_stats:
        calmar  = CAGR / |MaxDD|
        sortino = annualized mean / downside deviation (rf=0, target 0, sqrt(252))
        ulcer   = Ulcer Index = RMS of the percentage drawdown path
        martin  = Martin ratio = annualized return % / Ulcer Index
    Returns {} for curves too short to be meaningful."""
    if len(curve) < 3:
        return {}
    ds = sorted(curve)
    vals = [curve[d] for d in ds]
    rets = [vals[i] / vals[i - 1] - 1 for i in range(1, len(vals)) if vals[i - 1]]
    if len(rets) < 2:
        return {}
    n = len(rets)
    mean = sum(rets) / n
    cum = vals[-1] / vals[0] - 1
    yrs = n / 252
    cagr = ((1 + cum) ** (1 / yrs) - 1) if (yrs > 0.08 and (1 + cum) > 0) else None
    dd_dev = (sum(min(r, 0.0) ** 2 for r in rets) / n) ** 0.5
    sortino = round((mean * 252) / (dd_dev * 252 ** 0.5), 2) if dd_dev > 0 else None
    peak, mdd, sq = vals[0], 0.0, 0.0
    for v in vals:
        peak = max(peak, v)
        d = v / peak - 1
        mdd = min(mdd, d)
        sq += (d * 100.0) ** 2
    ulcer = (sq / len(vals)) ** 0.5
    calmar = round(cagr / abs(mdd), 2) if (cagr is not None and mdd < 0) else None
    martin = round((cagr * 100) / ulcer, 2) if (cagr is not None and ulcer > 0) else None
    return {"calmar": calmar, "sortino": sortino,
            "ulcer": round(ulcer, 2), "martin": martin}


def _backtest_resilient(sid, start, end, tries=5):
    """oos.backtest with extra backoff on the backtest endpoint's burst rate
    limit. oos._request already retries 429 four times (<=8s total); running
    121 backtests back-to-back still tripped a burst cap in groups of ~3. On a
    'rate limited after retries' RuntimeError, wait longer and retry."""
    for k in range(tries):
        try:
            return oos.backtest(sid, start, end)
        except RuntimeError as e:
            if 'rate limited' in str(e).lower() and k < tries - 1:
                time.sleep(6 * (k + 1))   # 6, 12, 18, 24s
                continue
            raise


def main():
    asof = date.today().isoformat()
    rows = []
    for i, (sid, oosdate, name) in enumerate(WATCH, 1):
        time.sleep(0.5)                       # gentle pacing to avoid burst 429s
        try:
            curve, _ = _backtest_resilient(sid, EARLY_START, asof)
        except Exception as e:
            print(f"[{i}/{len(WATCH)}] {name}: FAIL {type(e).__name__}: {str(e)[:80]}",
                  file=sys.stderr)
            continue
        if not curve:
            print(f"[{i}/{len(WATCH)}] {name}: empty curve", file=sys.stderr)
            continue
        oos_curve = {d: v for d, v in curve.items() if d >= oosdate}
        ins = _cs({d: v for d, v in curve.items() if d < oosdate})
        out = _cs(oos_curve)
        oute = _ext(oos_curve)          # Calmar / Sortino / Ulcer / Martin (OOS side)
        row = {
            "date": asof, "scope": "oos_split", "sym_id": sid, "name": name,
            "oos_date": oosdate,
            "bt_start": min(curve), "bt_end": max(curve),
            "is_days": ins.get("n_days"), "oos_days": out.get("n_days"),
            "is_cagr_pct": ins.get("cagr_pct"), "oos_cagr_pct": out.get("cagr_pct"),
            "is_sharpe": ins.get("sharpe"), "oos_sharpe": out.get("sharpe"),
            "is_maxdd_pct": ins.get("maxdd_pct"), "oos_maxdd_pct": out.get("maxdd_pct"),
            "oos_cum_pct": out.get("cum_pct"),
            "oos_calmar": oute.get("calmar"), "oos_sortino": oute.get("sortino"),
            "oos_ulcer": oute.get("ulcer"), "oos_martin": oute.get("martin"),
        }
        if ins.get("cagr_pct") is not None and out.get("cagr_pct") is not None:
            row["cagr_gap_pct"] = round(out["cagr_pct"] - ins["cagr_pct"], 2)
        else:
            row["cagr_gap_pct"] = None
        rows.append(row)
        print(f"[{i}/{len(WATCH)}] {name}: OOS {out.get('oos_days') or out.get('n_days')}d "
              f"cagr {out.get('cagr_pct')} sharpe {out.get('sharpe')}")

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Ranked report: by OOS length (longest first), then show the split.
    rows.sort(key=lambda r: r.get("oos_days") or 0, reverse=True)
    hdr = (f"{'oosDays':>7} {'IScagr':>7} {'OOScagr':>7} {'gap':>7} "
           f"{'ISsh':>5} {'OOSsh':>5} {'OOSdd':>7}  name")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r.get('oos_days') or 0:7d} {r.get('is_cagr_pct') or 0:7.1f} "
              f"{r.get('oos_cagr_pct') or 0:7.1f} {r.get('cagr_gap_pct') or 0:7.1f} "
              f"{r.get('is_sharpe') or 0:5.2f} {r.get('oos_sharpe') or 0:5.2f} "
              f"{r.get('oos_maxdd_pct') or 0:7.1f}  {(r.get('name') or '')[:46]}")
    print(f"\n[oos-split] {len(rows)}/{len(WATCH)} symphonies -> {OUT}")


if __name__ == "__main__":
    main()
