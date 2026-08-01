#!/usr/bin/env python3
"""
composer_oos_watchlist.py — in-sample vs out-of-sample split for watched symphonies.

For every watchlisted symphony (every strategy followed, regardless of how
recently it was frozen — its OOS inception = Composer's last_semantic_update_at),
backtest the full history with the CURRENT (frozen) definition, split the equity
curve at the freeze date, and compute stats on each side:

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

WATCH = [
    # All watchlisted symphonies (every strategy followed), extracted from the
    # watchlist response (last_semantic_update_at = OOS freeze/inception). Short-OOS
    # ones are kept too — the dashboard grades them by a confidence tier.
    ("c1bWgoNRaiff8ZZwBGD0", "2022-02-13", "Stoic Finance Presents: Inflation Spiral Hedge"),
    ("IOcJQgrFTUA44WJuej15", "2022-02-13", "Copy of Buy the Dips: Nasdaq 100"),
    ("HgK8mCeBnH4fQFNcfZ7q", "2022-07-28", "Inside Nancy Pelosi's Chips- V3"),
    ("BkSoFieJXCYgRzaq5HbT", "2022-09-12", "Copy of Apple Long Term Investing"),
    ("wQiJcF0aADrKVDaxPLDJ", "2022-09-27", "Copy of Slow and Steady Growth 9.6%/20.2% DD (Backtest to Feb 2007) | UUP/TLT Mod"),
    ("YNEcGCD8kceDvWytUjVF", "2022-12-08", "TQQQ For The Long Term V4.1 | Garen Mod | https://www.investorcollab.com"),
    ("28m5GEDrcKoHGe6nfuHX", "2023-05-16", "Commodities/Bonds - Weekly"),
    ("1gHhOZ0K5NHbSMyh1Po0", "2023-06-05", "Copy of Ease Up on the Gas V2a (add a little nitro)"),
    ("W7WT305mhZB3UHLHKvNz", "2023-12-15", "Tech or Sector Momentum"),
    ("ohSQn2Tjgfx6plvIwPdG", "2024-03-21", "UVXY Hedge"),
    ("gRUZY4Wgflcgl8qN4ZHs", "2024-03-22", "Base 2 - LABU Version"),
    ("ljdkhTQX8c0TLZoith61", "2024-03-31", "Copy of Base 2 - High Beta (UVXY, SPHB, no 3x for backtest)"),
    ("NBGy93LePACWUqlRYpSA", "2024-08-06", "D- Base - SOXL"),
    ("6Thdl4MTHtNAYQD43BoU", "2024-09-05", "Feaver Frontrunner V3 655/40"),
    ("ziRhQASn0OdLZEdHtFxZ", "2024-09-07", "UPRO FTLT (Azqato's Longer Backtest)"),
    ("KIZkZ4Jq4aC09ObGELmy", "2024-10-01", "Gasoline (UVXY)"),
    ("hRSa4zuQtstfj6DO5Sbr", "2024-10-10", "Semiconductors #3 875/23 (UVIX/UVXY)"),
    ("2lTE6fQMAuyAL4dAM86a", "2024-10-24", "Industrials (No VIX w/zip) 237/26"),
    ("1i91IlJvdSl18BrXgLvW", "2024-11-02", "Copy of Wash Sale 3 Sorted by 5 day Max DD WM 74"),
    ("NxfBdt0Kokogj6h7RFBG", "2025-01-11", "equal weighted leverage"),
    ("Cid5eqKgassokbnqDiXR", "2025-01-20", "Overheat, 19/25 BACK TO 2008"),
    ("CXUAXghXz92ln6ylUSGA", "2025-01-29", "Pharma Bull 126/25 (UVXY)"),
    ("rjhBONNBaA5qodjPNvoi", "2025-01-29", "Healthcare Bull 137/23 (UVXY)"),
    ("2NBBTwkY9q7P57jOXsZg", "2025-02-05", "Monthly Dividend 48/23"),
    ("dh16jpkc2lENdpE6xXmw", "2025-02-16", "VIXM filter 62/27 to 2012"),
    ("AxrUstLwZz9GrglAZD9D", "2025-02-17", "B - PUNCS VIXM filter 60/22 "),
    ("ipvxht5UbCPvNdYR9lW5", "2025-04-04", "Holy Grail plus Frontrunner (some VIX) 252/13"),
    ("HcePoNlW5OGwA8fUq2Pl", "2025-04-07", "Insurance Bull 169/23 (UVXY), no leverage"),
    ("AknCRuUPkmu9Bq7L6VhR", "2025-04-11", "C- Bulls with Blue Chip Back-Up Strategy 177/21  (UVXY)"),
    ("EeeiIGPX8ZELFTsfcyKs", "2025-05-03", "India 22/13 (BTAL)"),
    ("UCUOpRsvpy0AbNtVv3R4", "2025-05-05", "C - Energy or SPY (BTAL) + Insurance (UVXY) 100/17"),
    ("FOdFSlCAKDKOrVaCezx3", "2025-05-06", "C - \"Four Corners\" 156/19"),
    ("PkfK83gPWFTyuv9bQI5Y", "2025-05-09", "Copy of Consumer Staples (No VIX)"),
    ("EqDy5AUydosWUnDMGVZA", "2025-05-10", "D - SOXL mostly simple (238/21)"),
    ("wRB3FMRwr30GZ0g1Simr", "2025-05-14", "Copy of Nuclear v2"),
    ("UjvVX9WI16zCT7G6LY2Q", "2025-05-15", "Copy of (A) 20%/3% MDD "),
    ("48bXsZqImB7fFMrvqIV7", "2025-05-20", "D - Semis then sort w/ TAIL 601/15 (Some UVXY)"),
    ("b8JXr9dSjY8dJKahbBre", "2025-05-20", "Emerging Markets (No VIX)"),
    ("dHpFRQigYCLjXhQV1bCF", "2025-05-20", "C - QQQ/TAIL Divergence w/ VIXM Filter (Low Vix) 103/25"),
    ("KbnDRwd2amwAE7c0C4si", "2025-05-20", "Mexico (No VIX)"),
    ("M3QpEsKtFFsoSpdgLtph", "2025-05-20", "Healthcare (Some VIX) 234/23"),
    ("MjdSVGCm7sIO7ziUFkpw", "2025-05-20", "10d RSI SPY v CORP"),
    ("t73IMAAMjCgomuN6LngF", "2025-05-20", "Diversified Commodities 497/18 (Light VIX)"),
    ("UQSiaPcyb0kp7GpW1mzG", "2025-05-20", "D - Top Sort by 20d return 124/20"),
    ("VF81LpW8Ro5hwgP6bxAH", "2025-05-20", "D- Dips & Vix 450/18 (UVXY)"),
    ("yoqqOgvzkoFLyOqZoYQd", "2025-05-20", "Utilities (No VIX) 202/41"),
    ("ZrC4sdNzB9stQemijK3i", "2025-05-20", "Yippe Kaye 332/16 (UVXY), w leverage"),
    ("vaS4q8LplmqsoxzAk0wI", "2025-05-21", "Energy (UVXY) 658/22 since '22"),
    ("TGJQs49qL6sOq2U6sWVD", "2025-05-23", "RSI Sorter"),
    ("qz8qwrhCrkDZwXwHy4d8", "2025-05-24", "C- FAS 20-Day EMA (no VIX) 164/28 (Buy Copy)"),
    ("J3eDjCGZ2NEKRtMR24fV", "2025-05-25", "Copy of Copy of Holy Grail | KMLM"),
    ("EaBXOYOQAHod3o86bRgR", "2025-05-27", "Catch-Up Trade (TNA, MIDU, DUSL, LABU) w/ VIXM"),
    ("RJ5wy9GhYERZeckJtRe1", "2025-05-27", "B- QQQ enhanced 69/25"),
    ("B1OR4bs3NMgjaz1TJl4e", "2025-05-30", "South Korea (No VIX)"),
    ("iHFLIhls4pa2sOclgRSo", "2025-05-30", "Master Balancer"),
    ("OiCSKPW7fECY4hQNNHVR", "2025-06-06", "Copy of Extended history Inflation Protected Simple Beta Baller Signal | IEF>IBTK (Other IBT*, SPTI ,SCHQ are related) & SHY>SBND (IBTF, ITBG, ITBE are related)"),
    ("22nOCrE8hEqzqxu9kPzn", "2025-06-07", "Mag7 Winners & Buy Dip in Tech (no VIX) 211/17"),
    ("MOIK2ZETLOjwKCQqQ5NQ", "2025-06-07", "Biotech (Some VIX) 352/35"),
    ("LFz7Ps2NwcrbwNfcK21o", "2025-06-09", "FNGU (No VIX) 533/38"),
    ("xXe1F2vjfzWkOViIpNJX", "2025-06-10", "C - EQ WT, Risk Averse 99/17 (VIXY)"),
    ("Z5VmkFJNegQsavTwNbKo", "2025-06-11", "D- FNGU to the Third (VXX & UVXY) 413/44"),
    ("iOkxIU2ruIVWGH0f2zqb", "2025-06-16", "Nat Gas 20-Day EMA (UVXY) 864/32 since '22"),
    ("ZlRSaY5tcXO9hMU5fbWM", "2025-06-16", "Diversified Laggard (779/21)"),
    ("VwcPcYfP7ZiLMK2vbbpZ", "2025-06-20", "D- Tech, TAIL & VIXM Hedge 133/9"),
    ("OU8UQTPGvy2CnmFLZvF3", "2025-06-22", "C- Tech Bull 146/13 (UVXY)"),
    ("es9fE4NgF1k0mjUzvL1g", "2025-06-24", "Semiconductors #1 751/26 (Some UVXY)"),
    ("IbiH4hBpMDbLolfDBVDf", "2025-06-24", "FNGU"),
    ("KTstjsvPIkU5alZUjJu8", "2025-06-26", "D- Top 5 Tech (UVXY)"),
    ("4R1sB8GmUga5LQD0DU2C", "2025-06-27", "B- AI & Big Data +TAIL/TQQQ 43/19"),
    ("b4rQcJPcXvi3YfnDP9e7", "2025-06-28", "B- PFIX/BITX/USMV/Metals 35/23/.16"),
    ("es21H6YhrliF5k4KGLw1", "2025-06-28", "B- Blue Chip Backup Strategy 34/15 (no VIX)"),
    ("0TMH5QEjuNag5HL4wSAr", "2025-06-29", "D- Base 2 TECL 383/38"),
    ("FezxTWwSjfOuLeJWRZBL", "2025-06-29", "D- Mag7 Hybrid (UVIX) 248/40"),
    ("JW49FgJ2mZKASSpbtpfO", "2025-06-29", "D- FNGU w TAIL/PFIX (VXX & UVXY) 136/18"),
    ("P7UeIapdg2Yqi0uaQOPY", "2025-06-29", "D- EqWt 224/27 SPY/QQQ,/SMH/VTV & 3x (VXX)"),
    ("s794yZ3rZR2hT6aRk2Hz", "2025-06-29", "D- TECL Bull 292/38 (UVXY)"),
    ("f0znz9eW3tl0JxT0tE4i", "2025-07-01", "Copy of Simons KMLM switcher (single pops)| BT 4/13/22  (Invest Copy)"),
    ("0DgISaXxMYwUMnaF4uFM", "2025-07-02", "B- DC trip 99/13"),
    ("LY1kRX0zMarzOQR93P9g", "2025-07-02", "Diversification, Lagging Sectors "),
    ("zU3PN537XLaI4Wvumebp", "2025-07-02", "D- Diversification #2 (Light VIX) 305/43"),
    ("yufIIgEQ7Gi8wJtrk5Cp", "2025-07-03", "A- SIN 31/14 "),
    ("xN5Hi5Hv94gRHZynUTj5", "2025-07-06", "SIMPLICITY, SPY 36/27"),
    ("VrrYo0iqezW5BZi0Bd5R", "2025-07-07", "D- QQQ, LESS SIMPLICITY + Enhanced QQQ (UVXY) 111/24"),
    ("3W80K6PVgou3IF93Un0N", "2025-07-11", "The AI 500"),
    ("U5D1j9XERt0iol68BCj3", "2025-07-11", "B - QQQ/TAIL Diverge, no VIX 32/14"),
    ("EC3mH97dI0EszW4FQ2GW", "2025-07-12", "C- SIMPLICITY, UPRO 80/34"),
    ("bEItOV1Os2iBkkrTTcl4", "2025-07-13", "SIMPLICITY, SPMO 46/33"),
    ("yHRXt40v63jzODUHvTkn", "2025-07-14", "C- SIMPLICITY, QUANTUM 216/37"),
    ("goy4idUzTfOXoGemr4Ov", "2025-07-20", "\"SHANE\" SIMPLICITY, TQQQ, UPRO & SPMO 81/24"),
    ("IIxLbW1WcWb80lz5ZzSo", "2025-07-20", "D- SIMPLICITY+RATES, QQQ (UVXY) 205/34"),
    ("YszEXaJnydfKj2GLNR44", "2025-07-20", "C- LUKE SIMPLICITY, UPRO/TQQQ 108/32"),
    ("K3YAdutrF96KSxVTpuoU", "2025-07-23", "SPMO as Proxy 60/22"),
    ("CLWnmpUkjArmrAJP2A38", "2025-07-24", "Copy of Newish Short End Bond Symphony"),
    ("fjL2dOwIBlrbZAgojfnj", "2025-07-24", "Copy of N1"),
    ("bZu9LeaC4lvJztDUPYqM", "2025-07-25", "C- SIMPLE yet Uncorrelated R2 .25 42/7"),
    ("4f43290YwG1ocYGTSbAp", "2025-07-28", "Consumer Staples (No VIX)"),
    ("krQf2nkztcGoFICgmHso", "2025-07-29", "A- Super Trend no VIX (77/16/.51)"),
    ("L7KJCM7HZCIJF07PcvHI", "2025-08-01", "*- High Yield Bond / Income Strategy 18/8"),
    ("sbrkjgeP7fYxwbqbHosw", "2025-08-10", "NDX Top 100, no vix 30/24"),
    ("HAmZLe9eZpdCNONzhKmi", "2025-08-11", "D- Bitcoin, Gold, TECL & TAIL (193/20)"),
    ("74eyvSoZXhw4ouk2fuPU", "2025-08-18", "C- Avoid Volatility (VIX) 89/28/.17"),
    ("IS9h9L3EbaFc4jzcL3kD", "2025-08-18", "C- SIMPLICITY TOP 3TECH"),
    ("DyUm8IplFB2tQWRgEbOF", "2025-08-24", "C- Ex-US Markets (Some Vix & Leverage) 136/18"),
    ("KfIsAC2FEPyPrQdMMBBd", "2025-08-24", "Real Estate (Some VIX) 338/32"),
    ("hnY2FmKgz3CowjrPpTp4", "2025-09-11", "Old Economy"),
    ("KUksUkhpNOJdiqMoss9r", "2025-09-16", "D- Top of Tech or Super Trend / buy QQQ dip (VIX)"),
    ("OghR4VqAfN0qjKrstQK8", "2025-09-16", "C- All things in balance (Some VIX) 157/15/.03"),
    ("BRBNlVzndKRdqyQTGiBr", "2025-09-25", "Old Economy2"),
    ("kHcoGXd7034USpBBcDhN", "2025-09-29", "Gold (No VIX) "),
    ("scayu5ONQOgSJfp8IPz2", "2025-10-06", "B- Tech & SIN 140/12 "),
    ("8ilUSeGVgqiWLCi3WNjq", "2025-10-13", "TQQQ during dips (no VIX)"),
    ("3gALwork9mg3zQTmjxmO", "2025-10-29", "D- Sisyphus (250/11/0) VIX"),
    ("a5dJAzPC2tz5QOeUfcUh", "2025-10-29", "Pelosi's Chips Modified 131/46/.13"),
    ("4y37EiFDfW8Rb3Q9OsMD", "2025-10-31", "B- Survival Bots 104/11/.32"),
    ("DvDegddskNkpVqqx33RX", "2025-10-31", "D - Top Sort Leverage (138/11) (Buy Copy)"),
    ("INoh16ZWtXzayakJ7Zul", "2025-10-31", "D - Battle Bots 192/15/.22"),
    ("k9D8PYPa0MzaoUxMYHLG", "2025-10-31", "A- messin around 28/14"),
    ("YuXxG3yXDQQzOaWf9hAc", "2025-10-31", "D - High Beta (Low VIX) 126/19"),
    ("h39G48Umhln3Yn1q5q2a", "2025-12-24", "C- SIMPLICITY, VOO QQQ TQQQ 81/24/.32"),
    ("gUYcMk9mUrR8YSpVuNcB", "2026-01-06", "C- WW Growth (UVXY) 113/14"),
    ("Xk6wYgPK8RggrPAAn0Iq", "2026-01-17", "Best of QQQ 75/15/.67"),
    ("zNAFpyOkgaQvddIrazsM", "2026-01-17", "D - Best of the Best 84/24/.65"),
    ("MhrFjP1901YybhhNQPGL", "2026-02-03", "Claude Optimized 58/19/.24"),
    ("e4a4Lf3zqDPQHbczzIgc", "2026-02-04", "Materials (No VIX) 280/39"),
    ("AbILZXFPtCrqobtzGTYZ", "2026-02-11", "C - Eq Wt SPY/QQQ/SMH (some VIX) 32/18"),
    ("gHcPN3LB5oRBuAF1qDfl", "2026-02-18", "Claude Hedge Fund 63/16/.23"),
    ("P6kOKMNFumsxCPAIOj2V", "2026-02-18", "CTA/DBMF"),
    ("YeLsQvAMxyfO0jczAiSx", "2026-02-27", "Hedgapotamus (18/10/.1) no commodity arm"),
    ("fDYTeyE4O1qxDp75f8wx", "2026-03-01", "Dollar & TLT Tell (59/19/.33)"),
    ("OPhNeS2qlkkqpURDrG2T", "2026-03-01", "VOO or GLD (24/21/.31)"),
    ("D14YSirlyqUjNNAofHrF", "2026-03-02", "Negative Correlation (17/30/.07)"),
    ("LTznli5K7fSdLupw2qRi", "2026-03-10", "Manual trade for Oil & Dollar Spike"),
    ("5sjkBA4PicWI3FPMeeO8", "2026-03-13", "DroneOn"),
    ("zMpyulxeqtK6WEdi0bfp", "2026-03-13", "C- Top of Tech or Super Trend simplified (VIX)"),
    ("Q03sqXqZb74HO4grhOc7", "2026-03-16", "Bonds & Commodities PLUS (37/10/.28)"),
    ("CgLJpsJEV947r6IkiMAf", "2026-03-18", "Opus 4.6 67/14/.55"),
    ("mT45iQNFGZrMG4Wq8nYO", "2026-03-19", "Claude Sleuth (167/19/.1) v2"),
    ("DMbxS6X4GsHVHLUr5pTt", "2026-03-24", "Top of Tech Simplified"),
    ("H1ka2fJCl9RdINyFkDlP", "2026-03-24", "Bonds & Bombs 48/11/.69"),
    ("kVwbt03HASQ1a454P0QH", "2026-03-27", "Boom, Bust & URA! 155/30/.27)"),
    ("YzbPNKqA6Qb4bBt9zN4t", "2026-03-27", "UVXY >200d"),
    ("q8WsPUEsiHHGdx4Zpe75", "2026-03-29", "HGER/GLD/DBA/SHY 16/24/.13"),
    ("GlHCJSaA7F8jPAndH7NL", "2026-04-04", "BTC Momentum (16/12/.04)"),
    ("nJvtWFfnhkQNkGPlKa6P", "2026-04-04", "SOXL/TECL 2d RSI (68/30/.39)"),
    ("peGMSDq3NIZX3xhWwCrE", "2026-04-04", "Current Blend March 14"),
    ("RyUirf1C0SnHhkAALDdU", "2026-04-04", "SPHB/XLP Pairs or Uncorrelated (40/10/.19)"),
    ("kpgsFTLjupe1SOcGo5yG", "2026-04-05", "CPER Shines B4 SPY (40/20/.22)"),
    ("5P9AFECG5fi5BmxrTTMz", "2026-04-08", "Dispersion Trade Filter & Uncorrelated 30/15/.05"),
    ("6Iofaw3q4QBhDUF3Zz3W", "2026-04-13", "Multi-Asset Rotation with Vol Hedges (39/7/.34)"),
    ("XYFoP0Q8fnigzSxODWgy", "2026-04-14", "Copy of Current Blend April 4"),
    ("eiednkTAB2594jz4KbSe", "2026-04-20", "B- ExUS Tech (no Vix) 63/28"),
    ("Emxm8QTDxsEWWL29bgZQ", "2026-04-22", "SemiSniper (131/31/.49) no VIX"),
    ("J8tz7o6LOjRu7jNjjWHI", "2026-04-27", "Dip Buys and EnerGoldTech (134/27/.39)"),
    ("KrkWQbxwnThhAPxii3sJ", "2026-04-27", "Claude Novel Signals (92/23/.31)"),
    ("wrTVljHuHaKNuP9HuQSh", "2026-04-27", "C- Super Trend simplified w VIX (57/21/.78)"),
    ("paIoADQuAmpkLnIsJZeY", "2026-05-03", "SPY or SHY with XLK dip buy 15/16/.6 back to '99"),
    ("POLgfXcBQpCjWFmJ4ufC", "2026-05-03", "TM Hedge + 5% CAOS (Regime-Gated)"),
    ("2e5NabfK5pUpg0qgzISM", "2026-05-05", "Vol Drag + CPER + SPHB/XLP (93/20.46)"),
    ("1cvawM4TAQoMz9OJC7c9", "2026-05-06", "Citrini AI Infrastructure"),
    ("UjO0bDBZvI94EG8I1Cev", "2026-05-06", "Best of Times, Worst of Times (351/25/.12)"),
    ("9xt5o3MBpaqYX1YgjaT1", "2026-05-11", "D- QLD/QQQ back to 2006 (78/24/.53)"),
    ("shlMZMVMeGmYuxZNTMiD", "2026-05-11", "Equity Mom or MFers w VIX(32/10/.22)"),
    ("W8vRLgyhm1fT8rVwit7b", "2026-05-11", "Top of Tech Simplified (20% UVXY)"),
    ("sr0LciMhfVLZmA4eLhrc", "2026-05-17", "A - HYG v TMV & PFIX/TMF (47/12/.1)"),
    ("1pspEPz40iba7upR7X8k", "2026-05-19", "HORMUZ Manual Strategy"),
    ("0ppklRczquy3W5QpwyZY", "2026-05-21", "A - GRNY w/o Super Trend / No VIX"),
    ("KtKZhQhnqN8DMB6Zhqjz", "2026-05-21", "multi gates 25/17/.76"),
    ("FcUhITY6qqBiVxNU8GDw", "2026-05-23", "BTC Momentum (16/12/.04) (Buy Copy)"),
    ("hvziIwcTO6dW1TumoNCI", "2026-05-26", "Meme Funds (AI, Space, Quantum, Robotics)"),
    ("PtLsi4FPJLH4W2l2Skt0", "2026-06-02", "Equity or Hedgapotamus v3 (Adapted with Claude) 35/15/.49"),
    ("5M93cJHDpc3OXMHEaxgh", "2026-06-04", "GOLD Jerry (16/35/-.02)"),
    ("rEXDYjpFFtY00EW9qUtn", "2026-06-13", "Light Speed Stack - Names Not Yet Held (Equal Weight)"),
    ("0f138Q9uScGkDmjJBiao", "2026-06-15", "Dip Buy with Bond Stress Alpha (32/8/.24), no VIX"),
    ("wCT7tmigbfid8uAvbQ0r", "2026-06-20", "Tech Rotation + Dip/OB Overlays (49/30/.55) (Buy Copy)"),
    ("v9IXBkon7st0hklGQZES", "2026-06-24", "Triple Overlay (no VIX) - HYG-TLT + VIX-Fade + BTAL Modulator (21/23/.8)"),
    ("GW3DQjphnuc8lb88AQZ5", "2026-06-29", "MFers as Risk On Signal 18/10/.44 no vix"),
    ("ycWEWdacxO31zDXEQdJl", "2026-06-29", "MF-Weakness Risk-On Overlay (27/18/.36) no vix"),
    ("Ht9KCukebWItSncV4GaQ", "2026-07-02", "FANG+/QQQ/XLK Oversold Plus MF Default (58/13/.5)"),
    ("a65craateyUXt4lV5wMe", "2026-07-10", "SIMPLICITY Sector Rotate 61/20/.5 (w VIX)"),
    ("2FgtJjbd2blT3rRQxTIx", "2026-07-12", "New US/INTL Dip Harvest (SHY/USDU/GLD default) 50/18/.32"),
    ("64TX7qYpzjgadBDYyMd7", "2026-07-12", "International Dip Harvest (SHY/USDU/GLD default) 21/22/.2"),
    ("e5FnuxB4Bsmq7mDf8Ubs", "2026-07-12", "US Dip Harvest (SHY/USDU/GLD default) 50/18/.32"),
    ("3ytMgOEcunUXu8QMgp2v", "2026-07-20", "SOXL Gated by SMA & Vol (32/15/.65) no Vix"),
    ("FKq1FbXSP3dBfQOY5OXf", "2026-07-25", "Semis w/ filter (49/37/.62)"),
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
    cagr = ((1 + cum) ** (1 / yrs) - 1) if (yrs >= oos.MIN_ANNUALIZE_YEARS and (1 + cum) > 0) else None
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


def main(rewrite=False):
    asof = date.today().isoformat()
    history = oos.load_jsonl(OUT)          # prior runs, for run-over-run deltas
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
        conf, conf_rank, sharpe_se = oos._confidence(out.get("n_days"))
        if ins.get("cagr_pct") is not None and out.get("cagr_pct") is not None:
            gap = round(out["cagr_pct"] - ins["cagr_pct"], 2)
        else:
            gap = None
        extra = oos.oos_extra_fields(oos_curve, out, ins.get("cagr_pct"), gap)
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
            "oos_conf": conf, "oos_conf_rank": conf_rank, "sharpe_se": sharpe_se,
            "cagr_gap_pct": gap,
            **extra,
        }
        rows.append(row)
        print(f"[{i}/{len(WATCH)}] {name}: OOS {out.get('oos_days') or out.get('n_days')}d "
              f"cagr {out.get('cagr_pct')} sharpe {out.get('sharpe')}")

    # P1-4: stamp run-over-run Sharpe deltas, then append (keeping prior runs)
    # with dedup on (sym_id, date). --rewrite restores the old truncate behaviour.
    oos.stamp_sharpe_deltas(rows, history, asof)
    kept, added = oos.append_history(OUT, rows, asof, rewrite=rewrite)
    print(f"[oos-split] {'rewrote' if rewrite else 'appended'}: "
          f"{added} new rows + {kept} historical -> {OUT}")

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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrite", action="store_true",
                    help="truncate the ledger instead of appending history")
    main(rewrite=ap.parse_args().rewrite)
