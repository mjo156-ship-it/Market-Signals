# Complete Trading Signal Playbook
## Mike's Quantitative Strategy Reference
### Updated: January 26, 2026

---

# Statistical Confidence Guide

| Confidence | Sample Size | Interpretation |
|------------|-------------|----------------|
| 🟢 **High** | n ≥ 30 | Statistically robust |
| 🟡 **Moderate** | n = 10-29 | Directionally reliable |
| 🔴 **Low** | n < 10 | Use with caution |

**Always verify sample size before sizing positions.**

---

# Table of Contents

1. [Active Signals - Current Market](#active-signals---current-market)
2. [High-Conviction Combo Signals](#high-conviction-combo-signals)
3. [SOXL/SMH Long-Term Signals](#soxlsmh-long-term-signals)
4. [UPRO Entry/Exit Signals](#upro-entryexit-signals)
5. [SOXS Short Signals](#soxs-short-signals)
6. [Volatility Hedge Signals](#volatility-hedge-signals)
7. [BTC Signals](#btc-signals)
8. [AMD/NVDA Signals](#amdnvda-signals)
9. [Defensive Rotation Signals](#defensive-rotation-signals)
10. [Dollar-Based Signals](#dollar-based-signals)
11. [Signal Monitor Schedule](#signal-monitor-schedule)

---

# Active Signals - Current Market

**As of January 25, 2026:**

| Indicator | Current Value | Signal Status |
|-----------|---------------|---------------|
| GLD RSI(10) | 82.2 | ✅ > 79 ACTIVE |
| USDU RSI(10) | 21.5 | ✅ < 25 ACTIVE |
| XLP RSI(10) | 74.1 | ✅ > 65 ACTIVE |
| SPY RSI(10) | 56.4 | Neutral |
| QQQ RSI(10) | 50.7 | Neutral |
| SMH RSI(10) | 68.0 | Neutral |
| BTC RSI(10) | 37.3 | Approaching oversold |
| AMD RSI(10) | 82.3 | ⚠️ Extended |
| NVDA RSI(10) | 55.5 | Neutral |

**🔥 TRIPLE SIGNAL ACTIVE:** GLD > 79 + USDU < 25 + XLP > 65
- Long TQQQ: **100% win rate**, +11.6% avg (5d)
- Long UPRO: **85% win rate**, +5.2% avg (5d)
- Long AMD/NVDA: **86% win rate**, +5-8% avg (5d)

---

# High-Conviction Combo Signals

## 🔥🔥 Triple Signal (Highest Conviction)
**GLD RSI(10) > 79 AND USDU RSI(10) < 25 AND XLP RSI(10) > 65**

| Hold | Win Rate | Avg Return | Sample |
|------|----------|------------|--------|
| 5 days | **100%** | +11.6% | 🔴 n=5 days, 2 episodes |

**Trade History:** Every occurrence was a winner
- 2018-01-24: +1.6%
- 2020-07-27: +10.6%
- 2020-07-28: +16.4%
- 2020-07-29: +13.4%
- 2020-07-30: +15.9%

**⚠️ Sample Size Warning:** Only 2 independent market environments (2018, 2020). Fundamentally sound but statistically limited.

**Optimal Hold Strategy:**
- Enter on Day 1 of signal
- Hold while signal remains active
- Exit 5 trading days after signal ends (GLD drops < 79 OR USDU rises > 25)
- Avg return improves from +6.9% (fixed 5d) to +9.2% (hold through + 5d)

**Pyramiding:** Not recommended. Day 2+ shows better returns but sample too small (n=5). Enter full position on Day 1.

**Action:** Long TQQQ immediately

---

## 🔥 Double Signal (High Conviction)
**GLD RSI(10) > 79 AND USDU RSI(10) < 25**

| Asset | 5d Win | 5d Avg | 10d Win | 10d Avg | Sample |
|-------|--------|--------|---------|---------|--------|
| TQQQ | **88%** | +7.0% | 100% | +11.7% | 🔴 n=7 episodes |
| UPRO | **85%** | +5.2% | 92% | +7.5% | 🔴 n=7 episodes |
| AMD | **86%** | +7.7% | 79% | +5.8% | 🔴 n=7 episodes |
| NVDA | **86%** | +5.0% | 86% | +5.5% | 🔴 n=7 episodes |

**Duration Effect:** Signal improves with consecutive days
| GLD Consecutive Days | 5d Win | 5d Avg |
|---------------------|--------|--------|
| ≥ 1 day | 93% | +8.3% |
| ≥ 2 days | 89% | +9.0% |
| ≥ 5 days | 100% | +13.3% |

**Logic:** Gold strong + Dollar weak = Global liquidity expansion / risk-on rotation

**Action:** Long TQQQ, UPRO, AMD, NVDA

---

## GLD Overbought Alone
**GLD RSI(10) > 79**

| Hold | Win Rate | Avg Return | Sample |
|------|----------|------------|--------|
| 5 days | 72% | +3.2% | 🟡 n=97 days |
| 10 days | 75% | +5.6% | 🟡 n=97 days |

**Action:** Long TQQQ (weaker than combo signals)

---

# SOXL/SMH Long-Term Signals

## 🟢 BUY Signals (Accumulation)

### Best Signal: 100+ Days Below SMA(200) + RSI(50) < 45
| Metric | Value |
|--------|-------|
| Win Rate | **97%** |
| Avg Return (6mo) | **+81%** |
| Sample | 🟢 n=33 trades |

### Standard Signal: 100+ Days Below SMA(200)
| Metric | Value |
|--------|-------|
| Win Rate | **85%** |
| Avg Return (6mo) | **+54%** |
| Sample | 🟢 n=52 trades |

**Action:** Aggressive SOXL accumulation

---

## 🔴 EXIT Signals

| Signal | Action | Notes |
|--------|--------|-------|
| SMH **40%+ above SMA(200)** | 🔴 SELL ALL SOXL | Historical top zone |
| SMH **35-40% above SMA(200)** | 🟡 WARNING | Approaching danger |
| SMH **30-35% above SMA(200)** | 🟡 TRIM 25-50% | Reduce exposure |
| Death Cross (SMA50 < SMA200) | 🔴 EXIT | Trend reversal |

**Current SMH Status:**
- Price: $402.82
- SMA(200): ~$302
- % Above: +33.3% (TRIM zone)

---

# UPRO Entry/Exit Signals

## 🔴 EXIT Signals

| Signal | Win Rate | 5d Avg | 10d Avg | Sample | Action |
|--------|----------|--------|---------|--------|--------|
| SPY RSI(10) > **85** | 36% | -3.5% | -11.2% | 🔴 n=11 | 🔴 SELL/TRIM |
| SPY RSI(10) > 82 | 49% | -1.0% | -3.9% | 🟡 n=35 | ⚠️ Caution |
| SPY RSI(10) > 79 | 46% | -1.0% | -2.1% | 🟢 n=68 | ⚠️ Watch |

**SPY > 85 Trade History (all losers):**
- 2018-01-26: RSI 89.4 → UPRO -11.4%
- 2020-09-02: RSI 86.6 → UPRO -19.4%

---

## 🟢 BUY Signals

| Signal | Win Rate | 5d Avg | Sample | Action |
|--------|----------|--------|--------|--------|
| SPY RSI(10) < **21** | **94%** | +8.9% | 🟡 n=18 | 🟢 STRONG BUY |
| SPY RSI(10) < 25 | 74% | +3.9% | 🟢 n=42 | 🟢 Buy |
| SPY RSI(10) < 30 | 69% | +4.3% | 🟢 n=108 | 🟢 Consider |
| SPY < 30 AND HYG < 30 | **77%** | +5.3% | 🟢 n=65 | 🟢 Buy (credit stress) |

**SPY < 21 Trade History (17/18 winners):**
- 2015-08-24: RSI 13.5 → +11.6%
- 2018-12-24: RSI 13.4 → +20.2%
- 2022-01-26: RSI 19.6 → +17.0%
- 2025-04-08: RSI 17.5 → +23.0%

---

# SOXS Short Signals

## 🔴 Signal 1: Dollar Squeeze (100% Win Rate)
**SMH RSI(10) > 79 AND USDU RSI(10) > 70**

| Hold | Win Rate | Avg Return | Sample |
|------|----------|------------|--------|
| 5 days | **100%** | +9.5% | 🔴 n=8 trades |

**Trade History (all winners):**
| Date | SMH RSI | USDU RSI | SOXS 5d |
|------|---------|----------|---------|
| 2014-09-08 | 80.5 | 73.1 | +9.8% |
| 2014-12-05 | 81.0 | 71.5 | +13.8% |
| 2024-06-13 | 80.4 | 70.4 | +3.8% |
| 2024-06-18 | 85.4 | 73.0 | +15.6% |

**Logic:** Strong dollar = global liquidity tightening → semis reverse

---

## 🔴 Signal 2: Narrow Leadership (86% Win Rate)
**SMH RSI(10) > 79 AND IWM RSI(10) < 50**

| Hold | Win Rate | Avg Return | Sample |
|------|----------|------------|--------|
| 5 days | **86%** | +6.9% | 🔴 n=7 trades |

**Logic:** Semis ripping while small caps lag = fragile, narrow leadership

---

# Volatility Hedge Signals

## QQQ Overbought → Long UVXY
**QQQ RSI(10) > 79**

| Hold | Win Rate | Avg Return | CAGR |
|------|----------|------------|------|
| 5 days | 67% | +8.2% | +33% |

**Best combo:** QQQ > 79 AND SPY not > 79 (divergence)
- 5d Win: 72%
- 5d Avg: +10.8%

---

## SPY Overbought → Long UVXY
**SPY RSI(10) > 79**

| Hold | Win Rate | Avg Return | CAGR |
|------|----------|------------|------|
| 5 days | 64% | +5.9% | +41% |

---

## Dual Overbought → Long UVXY
**SPY RSI(10) > 79 AND QQQ RSI(10) > 79**

| Hold | Win Rate | Avg Return |
|------|----------|------------|
| 5 days | 76% | +9.0% |

---

# BTC Signals

## Key Finding: BTC Overbought = MOMENTUM (Not a Sell!)

Unlike stocks, overbought BTC keeps running:

| BTC RSI | 5d Win | 5d Avg | 10d Win | Sample |
|---------|--------|--------|---------|--------|
| Baseline | 55% | +1.4% | 56% | - |
| **> 79** | **67%** | **+5.2%** | **70%** | 🟢 n=215 days |
| > 82 | 68% | +5.7% | 72% | 🟢 n=157 days |
| > 85 | 67% | +5.4% | 71% | 🟢 n=99 days |

**Action:** Hold/Add BTC when RSI > 79

---

## 🟢 BTC Dip Buy Signal
**BTC RSI(10) < 30 AND UVXY RSI(10) < 40**

| Hold | Win Rate | Avg Return | Sample |
|------|----------|------------|--------|
| 5 days | **77%** | +4.1% | 🟢 n=30 trades |
| 10 days | 63% | +5.6% | 🟢 n=30 trades |

**Logic:** BTC oversold + low volatility = safe dip buy (not capitulation)

---

## BTC as Leading Indicator

| BTC Signal | TQQQ 5d Win | TQQQ 10d Win | Sample |
|------------|-------------|--------------|--------|
| BTC > 85 | **65%** | **70%** | 🟢 n=99 |
| Baseline | 59% | 63% | - |

**When BTC rips, equities follow**

---

# AMD/NVDA Signals

## When SMH is Overbought

| When SMH > 79 | AMD 5d | NVDA 5d |
|---------------|--------|---------|
| Win Rate | 46% ⚠️ | 55% |
| Avg Return | -0.88% | -0.20% |

**AMD underperforms significantly** when sector is extended. NVDA holds up better.

---

## Current GLD/USDU Signal → Bullish for AMD/NVDA

| Asset | 5d Win | 5d Avg | 10d Win |
|-------|--------|--------|---------|
| AMD | **86%** | +7.7% | 79% |
| NVDA | **86%** | +5.0% | 86% |

---

## Exit Warnings

| Signal | Action |
|--------|--------|
| AMD RSI(10) > 85 | ⚠️ Take profits |
| NVDA RSI(10) > 85 | ⚠️ Take profits |
| SMH > 79 + USDU > 70 | 🔴 Trim semis |

---

# Defensive Rotation Signals

## XLP/XLU/XLV Overbought → Long TQQQ
**Any defensive sector RSI(10) > 79 AND SPY/QQQ RSI(10) < 79**

| Hold | Win Rate | Avg Return |
|------|----------|------------|
| 20 days | **70%** | +5.0% |

**Logic:** Money rotating into defensives while indices not overbought = broad strength, TQQQ catches up

---

# Dollar-Based Signals

## Dollar Extremely Weak (USDU < 21) → Risk Off!

**Counter-intuitive:** Extreme dollar weakness is NOT bullish for risk assets

| Asset | 5d Win | 10d Avg |
|-------|--------|---------|
| TQQQ | 41% | -4.1% |
| EDC (EM) | 37% | -5.9% |
| GLD | **68%** | +1.1% |
| UVXY | 51% | +17.4% |

**Action:** Long GLD, consider UVXY, avoid TQQQ/EM

---

## Dollar Strong (USDU > 79) → Low Volatility

| Signal | UVXY 5d |
|--------|---------|
| Win Rate | 30% (UVXY loses) |
| Avg Loss | -4.9% |

**Action:** Short volatility (or long SVXY) when dollar is strong

---

# Signal Monitor Schedule

## Daily Emails

| Time (ET) | Type | Purpose |
|-----------|------|---------|
| **3:15 PM** | Pre-close | See signals setting up |
| **4:05 PM** | Close | Final confirmation |

## Tickers Monitored

**Indices:** SPY, QQQ, SMH, IWM
**Defensives:** XLP, XLU, XLV
**Safe Havens:** GLD, TLT, HYG, LQD
**Macro:** USDU, UCO
**Volatility:** UVXY
**EM:** EDC, YINN
**Crypto:** BTC-USD
**Individual:** AMD, NVDA

---

# NAIL (3x Homebuilders) Signals

## Best Buy Signals
| Signal | Win Rate | 5d Avg | Sample | Confidence |
|--------|----------|--------|--------|------------|
| **GLD > 79 + USDU < 25 + XLF < 70** | **90%** | +4.9% | 🔴 n=10 | Low |
| GLD > 79 + USDU < 25 + TLT > 60 | **100%** | +4.2% | 🔴 n=4 | Low |
| XLF > 79 + TLT > 60 | 89% | +9.0% | 🔴 n=9 | Low |
| HYG > 70 + GLD > 79 | **91%** | +5.6% | 🔴 n=11 | Low |

**⚠️ DANGER SIGNAL:** XLF > 70 + USDU < 25 → 11% win, -11.5% avg (n=18) - EXIT NAIL

---

# CURE (3x Healthcare) Signals

## Overbought/Oversold
| Signal | Win Rate | 5d Avg | Sample | Confidence |
|--------|----------|--------|--------|------------|
| **RSI < 21** | **85%** | +7.3% | 🟡 n=33 | Moderate |
| RSI < 25 | **81%** | +5.4% | 🟡 n=70 | Moderate |
| RSI < 30 | 64% | +3.3% | 🟢 n=167 | High |
| RSI > 79 | 40% | -0.8% | 🟢 n=95 | **Exit signal** |
| RSI > 85 | 33% | -0.9% | 🟡 n=15 | **Strong exit** |

**Pattern:** Classic mean-reversion. Buy extreme oversold, sell overbought.

---

# FAS (3x Financials) Signals

## Buy Signals
| Signal | Win Rate | Avg Return | Sample | Confidence |
|--------|----------|------------|--------|------------|
| **GLD > 79 + USDU < 25** | 71%/92% (5d/10d) | +5.8% (10d) | 🔴 n=13 | Low |
| RSI < 30 | 63% | +3.3% | 🟢 n=195 | High |
| SPY < 21 | 78% | +7.0% | 🟡 n=18 | Moderate |

## Exit Signals
| Signal | Win Rate | 5d Avg | Sample | Action |
|--------|----------|--------|--------|--------|
| RSI > 82 | 38% | -1.3% | 🟡 n=40 | Exit |
| **RSI > 85** | **8%** | -3.6% | 🔴 n=12 | **Strong exit** |

**Note:** FAS responds well to GLD/USDU signal. RSI > 85 is extremely bearish.

---

# LABU (3x Biotech) Signals

## Buy Signals
| Signal | Win Rate | 5d Avg | Sample | Confidence |
|--------|----------|--------|--------|------------|
| **RSI < 21** | **73%** | +11.2% | 🔴 n=11 | Low |
| RSI < 25 | **66%** | +5.7% | 🟡 n=59 | Moderate |
| SPY < 21 | **89%** | +12.2% | 🟡 n=18 | Moderate |

## Caution Signals
| Signal | Win Rate | Note |
|--------|----------|------|
| RSI > 70 | 42% | Slight negative edge |
| RSI > 85 | 67% | Momentum continues (n=6) |

**Pattern:** Buy extreme oversold. GLD/USDU signal does NOT work for LABU (54% win only).

---

# Quick Reference Card

## 🟢 BUY Triggers

| Signal | Asset | Win Rate | Sample | Confidence |
|--------|-------|----------|--------|------------|
| GLD > 79 + USDU < 25 + XLP > 65 | TQQQ | **100%** | n=2 episodes | 🔴 Low |
| GLD > 79 + USDU < 25 | TQQQ/UPRO/AMD/NVDA | **85-88%** | n=7 episodes | 🔴 Low |
| GLD > 79 + USDU < 25 + XLF < 70 | **NAIL** | **90%** | n=10 | 🔴 Low |
| GLD > 79 + USDU < 25 | **FAS** (10d) | **92%** | n=13 | 🔴 Low |
| SPY RSI < 21 | UPRO | **94%** | n=18 | 🟡 Moderate |
| SMH 100d below SMA200 + RSI50 < 45 | SOXL | **97%** | n=33 | 🟢 High |
| BTC < 30 + UVXY < 40 | BTC | **77%** | n=30 | 🟢 High |
| **CURE RSI < 21** | CURE | **85%** | n=33 | 🟡 Moderate |
| **CURE RSI < 25** | CURE | **81%** | n=70 | 🟡 Moderate |
| **LABU RSI < 21** | LABU | **73%** | n=11 | 🔴 Low |
| **FAS RSI < 30** | FAS | 63% | n=195 | 🟢 High |

## 🔴 EXIT/SHORT Triggers

| Signal | Asset | Win Rate | Sample | Confidence |
|--------|-------|----------|--------|------------|
| SMH > 79 + USDU > 70 | SOXS | **100%** | n=8 | 🔴 Low |
| SMH > 79 + IWM < 50 | SOXS | **86%** | n=7 | 🔴 Low |
| SPY RSI > 85 | Exit UPRO | 36% (exit) | n=11 | 🔴 Low |
| SMH 40% above SMA200 | Exit SOXL | Historical top | n=5 | 🔴 Low |
| **CURE RSI > 79** | Exit CURE | 40% | n=95 | 🟢 High |
| **FAS RSI > 85** | Exit FAS | **8%** | n=12 | 🔴 Low |
| **XLF > 70 + USDU < 25** | Exit NAIL | **11%** | n=18 | 🟡 Moderate |

## 🟡 HEDGE Triggers

| Signal | Asset | Win Rate | Sample | Confidence |
|--------|-------|----------|--------|------------|
| QQQ > 79 + SPY < 79 | UVXY | **72%** | n=41 | 🟢 High |
| SPY > 79 + QQQ > 79 | UVXY | **76%** | n=29 | 🟡 Moderate |

---

*Document auto-generated from backtested signals. All signals use Wilder RSI(10) unless otherwise noted. Past performance does not guarantee future results. Always consider sample size when sizing positions.*
