# Signal Research — daily emerging-signal scan

Backtest-gated, **net-new** trading signals mined daily from the price store by
[`signal_research.py`](../../signal_research.py), complementing (not duplicating)
the live signal monitor's existing groups. Refreshed on the **4:05 PM ET**
post-close run, right after the price-store update.

## What it scans (four families)

| Family | Examples | Avoids overlap with |
| --- | --- | --- |
| `rotation` | 3/6/12-mo cross-sectional momentum → new top-8 leaders | — |
| `volatility_breakout` | Donchian 52-wk breakout, Bollinger squeeze breakout, low-ATR regime | — |
| `mean_reversion` | Bollinger %B<0, distance-from-50DMA z<-2, 4+ down-day streak, capitulation volume | monitor's RSI(10) bands |
| `exhaustion_fade` | **short side:** Bollinger %B>1, distance-from-50DMA z>+2.5, 4+ up-day streak, parabolic >25% over 50DMA | monitor's RSI(10) overbought |
| `macro_ratio` | detrended z-scores of HYG/TLT, XLY/XLP, SMH/SPY, GLD/SPY, CPER/GLD (both extremes) | monitor's QQQ/SPY, QQQ/RSP, QQQE/QQQ |

Short signals carry a suggested inverse instrument (e.g. `SPY → SPXU`, `QQQ → SQQQ`, `SMH → SOXS`), or `short <ticker>` when none is mapped.

## Backtest gate

Each candidate is walk-forward backtested over full store history (rising-edge
entries, forward `horizon`-day return vs. buy-and-hold baseline). A signal must
clear the gate (`n ≥ 25`, `win_rate ≥ 0.60`, `avg_return > 0`, `edge > 0`) to
appear at all, then it is classified as:

- **`results`** — the condition **fires today**.
- **`watchlist`** — it passed the gate and is **just inside its threshold today**
  (primed / approaching), but has not fired yet. Forward-looking context.

Each entry is also tagged:

- **`composer_ready`** — a pure daily-close rule you can port into a Composer symphony.
- **`manual_swing`** — a multi-day discretionary hold.

## Files (the dashboard data contract)

| File | Purpose |
| --- | --- |
| `latest.json` | Today's full result set (same as the newest `history/` file). |
| `history/YYYY-MM-DD.json` | Per-day archive, self-contained. Idempotent (re-running a day overwrites it). |
| `index.json` | `{"latest": "YYYY-MM-DD", "dates": [...newest first]}` — enumerate the archive. |
| `latest.md` | Human-readable digest (also used in commit messages). |

### `latest.json` / `history/*.json` schema

```jsonc
{
  "generated_at_utc": "2026-06-19T20:10:00+00:00",
  "data_through": "2026-06-19",          // latest bar in the store
  "universe_size": 89,
  "params": { "min_sample": 25, "win_rate_min": 0.60 },
  "summary": {
    "n_results": 3,
    "n_watchlist": 6,
    "by_family": { "mean_reversion": 1, "volatility_breakout": 2 },
    "by_mode":   { "composer_ready": 3 }
  },
  "results": [                            // firing today; sorted by score (desc)
    {
      "family": "mean_reversion",
      "name": "4+ down-day streak",
      "ticker": "OILU",                  // ticker the condition fired on
      "suggested_instrument": "OILU",    // leveraged (long) / inverse (short) proxy
      "direction": "long",               // "long" or "short"
      "horizon_days": 5,
      "mode": "composer_ready",
      "today_value": "down streak 4",    // human-readable current reading
      "backtest": {
        "n": 31, "win_rate": 0.71, "avg_return": 0.0522,
        "median_return": 0.0661, "baseline_return": 0.0087,
        "edge": 0.0434, "fired_today": true
      },
      "score": 0.2416                    // edge × sqrt(n); ranking key
    }
  ],
  "watchlist": [ /* same shape; passed gate, primed but fired_today=false */ ]
}
```

## Dashboard tab

A self-contained, dependency-free tab ships here: **[`dashboard.html`](dashboard.html)**.
It fetches the published JSON and renders the newest day on top (🔥 Firing today
+ ⏳ Approaching/primed), with prior days in a collapsible archive below.

- **Use it standalone:** open the file in a browser, or host it (e.g. GitHub
  Pages). By default it reads the `main` branch via raw.githubusercontent.com;
  override with `?branch=<branch>` or `?base=<url-or-path>`, or the **Source**
  box in the header. To preview before this branch is merged, set the branch to
  `claude/price-store-daily-updates-evpzb8`.
- **Embed in chf-dashboard:** everything is namespaced under
  `#signal-research-tab` (scoped CSS) and a single IIFE — paste the `<style>`,
  the `<div id="signal-research-tab">`, and the `<script>` into a Jinja template
  or tab pane. No globals leak, no external dependencies.

### Raw data URLs (what the tab consumes)

To wire it into anything else, fetch these and render newest-first with prior
days archived below:

```
https://raw.githubusercontent.com/mjo156-ship-it/Market-Signals/main/data/signal_research/index.json
https://raw.githubusercontent.com/mjo156-ship-it/Market-Signals/main/data/signal_research/latest.json
https://raw.githubusercontent.com/mjo156-ship-it/Market-Signals/main/data/signal_research/history/<date>.json
```

Suggested layout: show `latest.json`'s `results` (firing today) on top, then the
`watchlist` (approaching/primed) below it — within each, group by `mode` so
Composer-ready vs manual-swing are distinct, and use `direction` to badge
long vs short fades. Below that, an accordion of older dates from
`index.json["dates"]`, each lazy-loading its `history/<date>.json`.

## Run it manually

```bash
python signal_research.py --run        # scan + backtest + write outputs
python signal_research.py --report     # scan + print, no write
python signal_research.py --self-test  # offline sanity checks
```

Reads only the local price store (no network), so it runs anywhere the store is
present.
