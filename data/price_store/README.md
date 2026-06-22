# Price Store — self-updating historical OHLCV

A **continuously updated, reproducible** price store for backtesting. Stores
**raw, unadjusted** OHLCV plus a separate corporate-actions log, and applies
split/dividend adjustment **at read time**. Refreshed automatically by the
post-close (4:05 PM ET) signal-monitor run. Built and read via
[`price_store.py`](../../price_store.py) at the repo root.

> This is **additive** and independent of the older `data/ohlcv/` CSV store
> (`scripts/update_ohlcv.py` + `ohlcv_daily.yml`), which keeps running
> unchanged. Use whichever you prefer; new backtests should prefer this one.

## Why raw + read-time adjustment?

yfinance's `Adj Close` is rewritten retroactively on **every** new
dividend/split, so appending only adjusted rows silently puts new rows on a
different basis than old ones. **Unadjusted prints never change**, which makes
a daily append/upsert safe. We keep raw prices and reconstruct the adjusted
series on demand from the actions log.

## Files

| File | Schema |
| --- | --- |
| `prices.parquet` | `date, ticker, open, high, low, close, volume, source, ingested_at` |
| `actions.parquet` | `date, ticker, type` (`dividend`/`split`), `value` |
| `quarantine.parquet` | rejected rows + `reason`, `flagged_at` (audit log; never read for backtests) |
| `_last_run.json` | last update manifest (ok/skipped/failed, actions detected, quarantined) |

`source` is `seed_csv` (historical CSV base), `stooq`, or `yahoo` (live updates).

## How it stays correct

- **Trailing-window upsert.** Each update re-pulls ~5 trading days and
  overwrites overlapping dates. This self-heals preliminary closes Yahoo
  revises after hours, gaps from a failed/rate-limited run, and late prints.
- **Corporate-action detection.** Each run checks `yf.Ticker(t).actions`. A
  newly-seen split or dividend triggers a **full re-pull** of that ticker
  (raw prices shift on a split — a 10-for-1 would otherwise show a phantom
  −90% day).
- **Validation gate.** Before any write, rows with NaNs, zero volume on a real
  NYSE session (calendar via `pandas_market_calendars`), or absurd daily
  returns (unless a split is logged that day) are quarantined, not committed.
- **Idempotent & crash-safe.** Upserts dedup on `(ticker, date)`; parquet is
  written atomically (`tmp` + `os.replace`). Running the updater multiple times
  a day is a no-op-or-correct, never a duplicate.

## Reading prices (drop-in for `yf.download`)

```python
from price_store import get_prices

# adjusted (default): splits+dividends applied at read time
df = get_prices(['SPY', 'QQQ'], start='2020-01-01', adjusted=True)   # MultiIndex cols
spy = get_prices('SPY', start='2020-01-01')                          # single-level cols
raw = get_prices('SPY', adjusted=False)                             # raw stored prints
tidy = get_prices(['SPY', 'QQQ'], layout='long')                    # date,ticker,ohlcv
```

## Operating it

```bash
python price_store.py --seed                 # one-time: ingest data/ohlcv CSVs + log actions
python price_store.py --seed --no-seed-actions   # offline seed (skips action fetch)
python price_store.py --update               # trailing-window refresh (full universe)
python price_store.py --update --tickers SPY,QQQ
python price_store.py --report               # per-ticker first/last date, rows, gaps
python price_store.py --self-test            # offline acceptance tests (no network)
```

The universe is read from `data/ohlcv/tickers.txt` (full ~91-ticker backtest
set), not just the tickers the monitor scores.

## Seeding notes

The committed seed was ingested **offline** from `data/ohlcv/*.csv` (raw OHLCV;
`Adj Close` discarded), so `actions.parquet` starts empty. On the **first live
post-close run**, action detection sees every ticker's history as new and
performs a one-time full re-pull per ticker, populating `actions.parquet` and
rewriting raw history with proper split alignment. After that, only newly
detected actions trigger a re-pull. Numbered duplicate files (`SPY2`, `TQQQ2`,
…) and Stooq-format index files are handled by the seeder (longest clean series
wins, logged) if present — none exist in this repo today.
