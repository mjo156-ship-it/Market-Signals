# OHLCV Historical Data

One CSV per ticker, kept fresh by `.github/workflows/ohlcv_daily.yml`.

## Schema

```
Date,Open,High,Low,Close,Adj Close,Volume
2024-01-02,477.42,478.40,475.20,476.71,469.18,89870900
...
```

- `Date` is `YYYY-MM-DD` (UTC trading day).
- `Close` is the raw daily close at that point in time.
- **`Adj Close` is canonical for backtesting** — it includes retroactive
  adjustments for splits and dividends. Use it instead of `Close` for
  return calculations.

## Fetch a ticker

```
https://raw.githubusercontent.com/mjo156-ship-it/Market-Signals/refs/heads/main/data/ohlcv/{TICKER}.csv
```

For example: `data/ohlcv/SPY.csv`, `data/ohlcv/BTC-USD.csv` (hyphen preserved).

## Update schedule

| When (UTC)     | When (ET, DST/STD)    | Mode      | Purpose                                  |
| -------------- | --------------------- | --------- | ---------------------------------------- |
| 21:00 weekdays | 5 PM ET / 4 PM ET     | `append`  | Add today's bar                          |
| 22:30 Fridays  | 6:30 PM ET / 5:30 PM  | `rewrite` | Refresh full history (catches Adj Close splits/dividends) |

Daily `append` is enough for the latest `Close`. The weekly `rewrite` is
required because `Adj Close` for older rows can change retroactively when
yfinance applies a new split or dividend — append-only would leave those
rows stale.

## Ops

- `data/ohlcv/tickers.txt` — the ticker universe (one per line).
- `data/ohlcv/_last_run.json` — per-run manifest with succeeded /
  skipped / failed tickers and sanity-check warnings (CSV close vs.
  live snapshot price).
- Per-ticker errors are isolated; the workflow exits 0 on partial
  failure so good tickers still commit. Exit 1 only if every ticker
  fails (network outage).

## Manual update

```bash
python scripts/update_ohlcv.py --mode=append    # default; daily delta
python scripts/update_ohlcv.py --mode=rewrite   # full reseed
python scripts/update_ohlcv.py --mode=backfill  # catch up missed days
```
