# OVX Oil Vol Spike — Strategy Record

> Monitor SIGNAL GROUP 34 (spec'd as "Group 31" — renumbered; the dashboard
> already uses Group 31 for BofA Signposts). TIER 3, monitor-only, manual
> execution at Fidelity. `_strategy_template.md` was not found in the repo, so
> this follows the sections requested in the build spec.

## Identity

- **Signal:** OVX (CBOE Crude Oil Volatility Index) RSI(10) > 79.
- **Primary vehicle:** ERX (2x Energy Bull), **1-day hold**.
- **Secondary leg:** UCO, **gated on USO > SMA50** (see caveats).
- **Cadence:** ~2.4 firings/year. Manual — `^OVX` is a CBOE index, not a
  security, so it **cannot** be referenced in a Composer symphony.
- **Status:** TIER 3. NOT "validated" — that label requires explicit sign-off.
- **Home:** monitor alert (Group 34) + dashboard display card. No automation.

## Thesis

An implied-oil-vol spike (OVX RSI>79) marks a dislocation in energy markets that
energy *equities* bounce from over the next day regardless of oil's trend
direction. ERX (2x energy equity) captures that bounce with a 1-day hold:
66.7% WR (n=42), +2.24% avg, and a robust ~40× edge-over-cost margin. Implied
vol carries information realized vol does not — a Composer-native realized-vol
proxy (USO stddev20 / SPY stddev20) correlates 0.882 in *levels* but only 0.278
in *daily changes*, and its edges collapse to ~1/3. There is no substitute for
the ^OVX read.

The hold is **1 day, not 5/10/20** on purpose: longer holds show higher average
returns in the 2x era but their permutation p_WR collapses (0.17 at 5d) — those
averages are a few outliers, not a repeatable edge. House rule: 1d WR first.

## Known Caveats

- **Leverage-era split.** ERX went 3x → 2x on 2020-03-24. Pooled n mixes two
  instruments. The **2x era (n=20)** is what's tradeable today: 1d 75.0% WR
  +1.79%. Treat the pooled 66.7% as the conservative anchor.
- **Tail risk.** 3x-era ERX 1d ranged −24.5%/+40.3% (March 2020); 5d worst
  −71.4%. The 1-day hold is what caps the damage — 2x-era-only worst 1d is
  −7.44%. Do not extend the hold to chase the 5d average.
- **UCO leg fails without a trend.** UCO (leveraged oil *futures*) carries roll
  decay that only an uptrend overcomes: USO>SMA50 → 10d 70.6% +4.05%; USO<SMA50
  → 20d 34.6% −8.78%. **Gate the UCO leg on USO > SMA50. Do not gate ERX** (ERX
  works both ways: 64.7% up-trend / 61.5% down-trend).
- **Portfolio concentration.** The Hormuz manual basket is already ~$206K of
  energy-adjacent names (PSCE, SU, GLNG, STNG, DOW, LIN, APD, RTX). ERX on this
  trigger is a **2x-leveraged ADD to an existing large energy sleeve**, not a new
  uncorrelated stream. The alert says so.
- **Brier is meaningless for years.** At ~2.4 firings/year a calibration score
  needs many years of live fires. Always read `n` beside the Brier.
- **Context divergence is display-only.** OVX >15% over SMA50 with VIX < SMA50 is
  a short-vol / long-equity *regime color*, permutation p=0.22 — NOT a trade. The
  "VIX plays catch-up to OVX" thesis is backwards. Never render it as an alert.
- **Rejected, do not re-test:** SPY short on the trigger (noise, p=0.47); OVX
  >1yr-90th-pctile (no edge); OVX +40%/20d without the RSI gate (weaker); OVX
  RSI<25 (84% pre-2020). See the provenance block for details.

## Holy Grail Role

**None.** This is a tactical overlay on an already-large energy sleeve, not a
diversifying stream. It adds leveraged energy beta on top of energy the book
already holds — the opposite of what the 7-stream Holy Grail framework is for.
Size it as a small tactical add, not a portfolio pillar.

---

## Provenance — backtest record (do not edit)

```
SIGNAL GROUP 34 (spec'd as 31) — OVX OIL VOL SPIKE
Tested 2026-07-24. Status: TIER 3 (monitor-only, manual execution).
NOT "validated" — that label requires explicit sign-off.

Data:      Yahoo ^OVX vs ^VIX, 2007-05-10 to 2026-07-23, n=4831 trading days
Method:    T+1 aligned (signal at close t, return measured close t -> close t+1),
           Wilder RSI(10), 5-day episode deduplication, Wilson 95% CIs,
           10K-iteration permutation tests vs unconditional distribution.

TRIGGER:   OVX RSI(10) > 79       [46 episodes, ~2.4/year]

PRIMARY VEHICLE — ERX (2x Energy Bull), n=42 episodes
  Hold   WR      Wilson 95% CI     avg      median   edge vs base
  1d     66.7%   (51.6, 79.0)      +2.24%   +1.95%   +14.5pp   <- TRADE THIS
  5d     59.5%   (44.5, 73.0)      +3.23%   +4.50%   +5.8pp
  10d    59.5%   (44.5, 73.0)      +2.37%   +6.62%   +6.6pp
  20d    59.5%   (44.5, 73.0)      +4.26%   +3.36%   +5.8pp
  Permutation: 1d p_mean=0.0012, p_WR=0.0388

LEVERAGE-ERA SPLIT (ERX went 3x -> 2x on 2020-03-24; pooled n mixes
two different instruments. Realized median |ERX|/|XLE| 1d = 2.87 in the
3x era, 2.04 in the 2x era.)
  2x era (>=2020-03-24), n=20:  1d 75.0% CI(53,89) +1.79% | 5d 65.0% +4.04%
                                10d 60.0% +4.75%          | 20d 65.0% +6.25%
  3x era (<2020-03-24),  n=22:  1d 59.1% CI(39,77) +2.64% | 5d 54.5% +2.50%
  The 2x era is what is tradeable today. n=20 -> Tier 3.

SLIPPAGE STRESS (house standard 5 bps leveraged, then 2x at 10 bps)
  ERX 1d gross +2.24% -> net +2.19% -> 2x-stress +2.14%
  WR 66.7% -> 64.3% at 2x stress. Edge is ~40x transaction cost. Robust.

TAIL RISK
  ERX 1d: worst -24.5%, best +40.3%  (both 3x era, March 2020)
  ERX 5d: worst -71.4%               (3x era)
  2x era only: worst 1d -7.44%, worst 5d -12.42%
  -> 1d hold caps damage in a way the 5d hold does not. Hold 1 day.

WHY 1d AND NOT 5d/10d
  5d/10d show higher average returns in the 2x era (+4.60%, +6.31%) but
  permutation p_WR collapses to 0.17 and 0.28 — those averages are carried
  by a handful of outliers, not a repeatable edge. House rule: 1d WR first.

SECONDARY LEGS ON THE SAME TRIGGER
  UCO  1d  64.3% n=42  +2.97%  +12.7pp  p_mean<0.0001  (post-2020: 71% n=24 +4.44%)
  XLE  1d  63.0% n=46  +0.71%  +10.7pp  p_mean=0.0089
  GUSH 1d  63.3% n=30  +2.08%           p_mean=0.0295
  DIG  1d  60.9% n=46  +1.17%
  OIH  1d  58.7% n=46  +1.20%
  UVXY 5d  67.0% n=24  +7.96%  post-2020 ONLY  p_WR=0.0013, p_mean=0.0066
           (pre-2020 was 38% WR n=13 and negative — post-2020 structural,
            same regime family as VIXM<25->HIBL. Monitor with Brier.)

OIL-DIRECTION CONDITIONING — ERX does NOT need the filter, UCO DOES
  ERX 1d, USO > SMA50 (n=17): 64.7% +1.20%
  ERX 1d, USO < SMA50 (n=26): 61.5% +2.53%   <- works both ways
  UCO,    USO > SMA50 (n=17): 5d 64.7% +2.16% | 10d 70.6% +4.05%
  UCO,    USO < SMA50 (n=26): 5d 42.3% -5.08% | 20d 34.6% -8.78%  <- fails
  Mechanism: an oil-vol spike dislocates energy equities regardless of trend
  and they bounce; leveraged oil futures carry roll decay that only a
  trend overcomes. GATE THE UCO LEG ON USO > SMA50. Do not gate ERX.

DO NOT TRADE — tested and rejected, do not re-test
  SPY short on OVX RSI>79: 5d 45.7% n=46, -14.3pp edge, p_mean=0.47. Noise.
  OVX > 1yr 90th percentile: no edge (SPY 5d 53.8% -6.1pp, USO 5d 42.9%).
  OVX +40% in 20d without the RSI gate: weaker (UVXY 5d 26.5%).
  OVX RSI < 25: VIX 5d 72% n=25 +6.19% but 84% of sample is pre-2020. Weak.
  Composer-native realized-vol proxy (USO stddev20 / SPY stddev20):
    corr to OVX is 0.882 in LEVELS but only 0.278 in DAILY CHANGES.
    Edges drop to ~1/3 (UCO 5d 58.0%, +6.2pp). Implied vol carries
    information realized vol does not. The proxy is not a substitute.

CONTEXT SIGNAL — display only, NOT tradeable
  OVX >15% above SMA50 AND VIX < SMA50, n=62 episodes:
    SPY 1d 61.3% | 5d 69.4% (+9.5pp) | 10d 71.0% | 20d 67.7%
    UVXY 5d 33.3%    VIX level 10d 35.5%
    post-2020 n=27: SPY 5d 93%, UVXY 5d 15% (-7.50%)
    Permutation p_mean=0.22, p_WR=0.081 -> NOT significant.
  Interpretation: the "VIX plays catch-up to OVX" thesis is BACKWARDS.
  Slow-grind oil-vol divergence with a calm VIX is a short-vol / long-equity
  regime, not a catch-up trade. Render as regime color. Never as an alert.

MULTIPLE TESTING
  19 signal specs tested this session. Bonferroni threshold p<0.0026.
  Surviving: ERX 1d p_mean=0.0012; UVXY 5d post-2020 p_WR=0.0013.
  Not surviving: ERX 1d p_WR=0.0388, XLE 1d p_mean=0.0089.

COMPOSER
  ^OVX is a CBOE index, not a security. It CANNOT be referenced in a
  Composer symphony. This signal is monitor-alert + manual execution at
  Fidelity only. Do not build a symphony around it.

PORTFOLIO CAVEAT
  The Hormuz manual basket is already ~$206K of energy-adjacent names
  (PSCE, SU, GLNG, STNG, DOW, LIN, APD, RTX). ERX on this trigger is a
  2x-leveraged ADD to an existing large energy sleeve, not a new
  uncorrelated Holy Grail stream. The alert text must say so.

STATE AT TIME OF TESTING (2026-07-23 close)
  OVX 68.97 (95.1 pctile all-history) | VIX 18.70 | ratio 3.69 (98.2 pctile)
  OVX RSI(10) 72.4 -> NOT FIRING | OVX +19.7% vs SMA50 | VIX +7.7% vs SMA50
  OVX 20d change +44.9% vs VIX +0.4%
  Last trigger: 2026-03-16. 2026 OVX peak: 120.9 (Hormuz).
```

## Wiring

- **Monitor** `signal_monitor_complete.py`: `ovx_group34(data, indicators)` →
  SIGNAL GROUP 34 in `check_signals`; Tier A (ERX buy), Tier B (UCO leg, gated),
  Tier C (approach watch); readings persisted to `dashboard_state.json` under
  `ovx`; OVX line in the email body every run. Tickers added: `^OVX ^VIX ERX USO`.
- **Dashboard** `chf_dashboard_server.py`: `compute_ovx_signals(raw_data,
  indicators)` → payload key `ovx`; 🛢️ Group 34 card; Brier signal
  `ovx_gt79_erx` (target ERX, 1d, base rate 0.667, tier 3). Tickers added:
  `^OVX ERX`.
