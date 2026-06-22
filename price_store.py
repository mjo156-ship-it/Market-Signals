#!/usr/bin/env python3
"""
Self-updating historical price store (raw / unadjusted OHLCV + corporate actions).

Design (see data/price_store/README.md for the long version):

  * Store **raw, UNADJUSTED** OHLCV in one tidy Parquet table. Unadjusted
    prints never change, so daily upserts are always safe. Adjustment for
    splits/dividends is computed at READ time from a separate actions log.
  * The daily update re-pulls a **trailing window** (~5 trading days) and
    UPSERTS it (overwrites overlapping dates). This self-heals preliminary
    closes that Yahoo revises after hours, gaps from a failed run, and late
    prints -- no single appended bar.
  * Corporate actions are detected each run via ``yf.Ticker(t).actions``.
    A newly-seen split or dividend triggers a **full re-pull** of that
    ticker (raw prices shift on a split, so the whole series is rewritten).
  * A **validation gate** quarantines NaNs, zero-volume rows on real NYSE
    sessions, and absurd daily returns (unless a split is logged that day)
    before anything is written. Quarantined rows are logged, never committed
    into the live table.
  * Everything is wrapped so a failure here can NEVER kill the caller
    (the signal monitor).

Storage layout (Parquet, queried via DuckDB):

  data/price_store/prices.parquet
      date, ticker, open, high, low, close, volume, source, ingested_at
  data/price_store/actions.parquet
      date, ticker, type ('dividend'|'split'), value
  data/price_store/quarantine.parquet   (append-only audit log)
      date, ticker, open, high, low, close, volume, source, reason, flagged_at
  data/price_store/_last_run.json        (per-run manifest)

CLI:

  python price_store.py --seed [--csv-dir data/ohlcv]   # one-time ingest
  python price_store.py --update [--tickers SPY,QQQ]     # trailing-window refresh
  python price_store.py --report                         # coverage report
  python price_store.py --self-test                      # offline acceptance tests

Read API (drop-in for yf.download in backtests):

  from price_store import get_prices
  df = get_prices(['SPY', 'QQQ'], start='2020-01-01', adjusted=True)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
STORE_DIR = REPO_ROOT / "data" / "price_store"
PRICES_PARQUET = STORE_DIR / "prices.parquet"
ACTIONS_PARQUET = STORE_DIR / "actions.parquet"
QUARANTINE_PARQUET = STORE_DIR / "quarantine.parquet"
MANIFEST_FILE = STORE_DIR / "_last_run.json"
SEED_TICKERS_FILE = REPO_ROOT / "data" / "ohlcv" / "tickers.txt"
DEFAULT_SEED_CSV_DIR = REPO_ROOT / "data" / "ohlcv"

SEED_START = "1990-01-01"
TRAILING_DAYS = 5            # trading days re-pulled on each update
MAX_DAILY_RETURN = 0.60      # |return| above this is quarantined unless a split is logged
PRICE_COLS = ["open", "high", "low", "close", "volume"]

# Canonical dtypes for the prices table.
_PRICE_SCHEMA = {
    "date": "datetime64[ns]",
    "ticker": "string",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",   # float so missing/partial volume survives a round-trip
    "source": "string",
    "ingested_at": "datetime64[ns]",
}
_ACTION_SCHEMA = {
    "date": "datetime64[ns]",
    "ticker": "string",
    "type": "string",
    "value": "float64",
}


def _now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# NYSE trading calendar (cached). Falls back to a Mon-Fri heuristic if
# pandas_market_calendars is unavailable so the module still imports.
# ---------------------------------------------------------------------------

_CAL = None
_CAL_OK = True


def _calendar():
    global _CAL, _CAL_OK
    if _CAL is None and _CAL_OK:
        try:
            import pandas_market_calendars as mcal
            _CAL = mcal.get_calendar("NYSE")
        except Exception as e:  # pragma: no cover - exercised only without the dep
            _CAL_OK = False
            _log(f"[calendar] WARN pandas_market_calendars unavailable ({e}); "
                 f"falling back to Mon-Fri heuristic")
    return _CAL


def trading_days(start, end) -> pd.DatetimeIndex:
    """NYSE sessions in [start, end] inclusive (normalized to midnight)."""
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if end < start:
        return pd.DatetimeIndex([])
    cal = _calendar()
    if cal is not None:
        sched = cal.schedule(start_date=start, end_date=end)
        return pd.DatetimeIndex(sched.index).normalize()
    # Fallback: business days (overcounts holidays, but never undercounts).
    return pd.bdate_range(start, end)


def is_trading_day(day) -> bool:
    day = pd.Timestamp(day).normalize()
    return len(trading_days(day, day)) > 0


# ---------------------------------------------------------------------------
# Fetcher abstraction -- lets tests inject synthetic data (Yahoo egress is
# not required to run the acceptance suite).
# ---------------------------------------------------------------------------

class YFinanceFetcher:
    """Real data source. Always pulls UNADJUSTED prices (auto_adjust=False)."""

    def history(self, ticker: str, start: Optional[str] = None,
                end: Optional[str] = None, period: Optional[str] = None) -> pd.DataFrame:
        import yfinance as yf
        kwargs = {"auto_adjust": False, "progress": False, "actions": False}
        if period is not None:
            kwargs["period"] = period
        else:
            kwargs["start"] = start or SEED_START
            if end is not None:
                kwargs["end"] = end
        df = yf.download(ticker, **kwargs)
        if df is None or len(df) == 0:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        keep = [c for c in PRICE_COLS if c in df.columns]
        df = df[keep].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "date"
        return df

    def actions(self, ticker: str) -> pd.DataFrame:
        """Return tidy actions: columns date, type, value."""
        import yfinance as yf
        try:
            raw = yf.Ticker(ticker).actions
        except Exception as e:
            _log(f"  [actions] {ticker}: fetch failed ({e})")
            return _empty_actions()
        return _normalize_actions(raw, ticker)


def _empty_actions() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "ticker", "type", "value"]).astype(_ACTION_SCHEMA)


def _normalize_actions(raw: Optional[pd.DataFrame], ticker: str) -> pd.DataFrame:
    """Convert a yfinance .actions frame (Dividends / Stock Splits) to tidy rows."""
    if raw is None or len(raw) == 0:
        return _empty_actions()
    raw = raw.copy()
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    rows = []
    for dt, r in raw.iterrows():
        div = float(r.get("Dividends", 0) or 0)
        spl = float(r.get("Stock Splits", 0) or 0)
        if div and not np.isnan(div) and div != 0:
            rows.append((dt, ticker, "dividend", div))
        if spl and not np.isnan(spl) and spl != 0:
            rows.append((dt, ticker, "split", spl))
    if not rows:
        return _empty_actions()
    out = pd.DataFrame(rows, columns=["date", "ticker", "type", "value"])
    return out.astype(_ACTION_SCHEMA)


# ---------------------------------------------------------------------------
# Low-level parquet IO (atomic + crash-safe)
# ---------------------------------------------------------------------------

def _read_parquet(path: Path, schema: dict) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in schema.items()})
    df = pd.read_parquet(path)
    for c, t in schema.items():
        if c not in df.columns:
            df[c] = pd.Series(dtype=t)
    return df[list(schema.keys())].astype(schema)


def _atomic_write_parquet(df: pd.DataFrame, path: Path, schema: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy().astype(schema)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)  # atomic on POSIX


def read_prices_raw() -> pd.DataFrame:
    return _read_parquet(PRICES_PARQUET, _PRICE_SCHEMA)


def read_actions() -> pd.DataFrame:
    return _read_parquet(ACTIONS_PARQUET, _ACTION_SCHEMA)


def _upsert(existing: pd.DataFrame, new: pd.DataFrame, keys: Sequence[str],
            sort_by: Sequence[str]) -> pd.DataFrame:
    """Combine, then keep the LAST row per key (newest wins). Idempotent."""
    if new is None or len(new) == 0:
        return existing
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=list(keys), keep="last")
    return combined.sort_values(list(sort_by)).reset_index(drop=True)


def upsert_prices(new_rows: pd.DataFrame) -> int:
    """Upsert price rows on (ticker, date). Returns rows written/overwritten."""
    if new_rows is None or len(new_rows) == 0:
        return 0
    new_rows = new_rows.astype(_PRICE_SCHEMA)
    existing = read_prices_raw()
    merged = _upsert(existing, new_rows, keys=["ticker", "date"],
                     sort_by=["ticker", "date"])
    _atomic_write_parquet(merged, PRICES_PARQUET, _PRICE_SCHEMA)
    return len(new_rows)


def upsert_actions(new_actions: pd.DataFrame) -> int:
    if new_actions is None or len(new_actions) == 0:
        return 0
    new_actions = new_actions.astype(_ACTION_SCHEMA)
    existing = read_actions()
    merged = _upsert(existing, new_actions, keys=["ticker", "date", "type"],
                     sort_by=["ticker", "date", "type"])
    _atomic_write_parquet(merged, ACTIONS_PARQUET, _ACTION_SCHEMA)
    # number of genuinely-new actions
    return len(merged) - len(existing)


def quarantine_rows(rows: pd.DataFrame) -> None:
    if rows is None or len(rows) == 0:
        return
    schema = {**{c: _PRICE_SCHEMA[c] for c in ["date", "ticker", *PRICE_COLS, "source"]},
              "reason": "string", "flagged_at": "datetime64[ns]"}
    rows = rows.copy()
    rows["flagged_at"] = _now()
    rows = rows[list(schema.keys())].astype(schema)
    existing = _read_parquet(QUARANTINE_PARQUET, schema)
    out = pd.concat([existing, rows], ignore_index=True)
    _atomic_write_parquet(out, QUARANTINE_PARQUET, schema)


# ---------------------------------------------------------------------------
# Validation gate
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame, ticker: str, actions_df: pd.DataFrame,
             prev_close: Optional[float] = None,
             check_returns: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a candidate price frame into (clean, quarantined).

    Rejects, with a logged reason:
      * any NaN in O/H/L/C/V
      * zero volume on a real NYSE session
      * |daily return| > MAX_DAILY_RETURN unless a split is logged that day

    `prev_close` seeds the first row's return when df is a trailing slice.
    `check_returns=False` skips the return rule -- used during seeding, where
    no actions are logged yet so raw split-day prints would falsely trip it
    (they're legitimate raw data; read-time adjustment handles them).
    """
    if df is None or len(df) == 0:
        empty = df.copy() if df is not None else pd.DataFrame()
        return empty, empty

    df = df.sort_values("date").reset_index(drop=True)
    reasons = pd.Series([""] * len(df), index=df.index)

    # 1) NaNs
    nan_mask = df[PRICE_COLS].isna().any(axis=1)
    reasons[nan_mask] = "nan_in_ohlcv"

    # 2) Zero volume on a true trading day
    sessions = set(trading_days(df["date"].min(), df["date"].max()))
    zero_vol = (df["volume"].fillna(0) == 0) & df["date"].isin(sessions)
    zero_vol &= ~nan_mask
    reasons[zero_vol] = "zero_volume_on_session"

    # 3) Absurd daily return (unless a split is logged on that date)
    split_dates = set(
        actions_df.loc[actions_df["type"] == "split", "date"]
        if actions_df is not None and len(actions_df) else []
    )
    if check_returns:
        closes = df["close"].astype(float)
        prev = closes.shift(1)
        if prev_close is not None and len(prev) > 0:
            prev.iloc[0] = prev_close
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = (closes - prev) / prev
        absurd = ret.abs() > MAX_DAILY_RETURN
        absurd &= ~df["date"].isin(split_dates)
        absurd = absurd.fillna(False) & (reasons == "")
        reasons[absurd] = "extreme_return"

    bad = reasons != ""
    clean = df[~bad].copy()
    quarantined = df[bad].copy()
    if len(quarantined):
        quarantined["reason"] = reasons[bad].values
    return clean, quarantined


# ---------------------------------------------------------------------------
# Seeding from the existing CSV snapshots
# ---------------------------------------------------------------------------

def _read_seed_csv(path: Path) -> tuple[pd.DataFrame, str]:
    """
    Parse a seed CSV into (frame[date,open,high,low,close,volume], source).

    Handles Yahoo format (has 'Adj Close', which we DISCARD -- we keep the
    raw OHLC) and Stooq format (no 'Adj Close', often CRLF / YYYYMMDD dates
    or <TICKER>,<DATE> style headers).
    """
    raw = pd.read_csv(path)
    cols = {c.lower().strip().strip("<>"): c for c in raw.columns}
    source = "yahoo" if "adj close" in cols else (
        "stooq" if "ticker" in cols or "per" in cols else "seed_csv")

    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    date_c = col("date")
    o, h, l, c, v = (col("open"), col("high"), col("low"),
                     col("close"), col("vol", "volume"))
    if not all([date_c, o, h, l, c]):
        raise ValueError(f"{path.name}: missing OHLC columns ({list(raw.columns)})")

    df = pd.DataFrame({
        "date": pd.to_datetime(raw[date_c].astype(str), errors="coerce"),
        "open": pd.to_numeric(raw[o], errors="coerce"),
        "high": pd.to_numeric(raw[h], errors="coerce"),
        "low": pd.to_numeric(raw[l], errors="coerce"),
        "close": pd.to_numeric(raw[c], errors="coerce"),
        "volume": pd.to_numeric(raw[v], errors="coerce") if v else 0.0,
    })
    df = df.dropna(subset=["date"]).copy()
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize()
    return df, ("seed_csv" if source == "yahoo" else source)


def _base_ticker(stem: str) -> str:
    """Strip a trailing duplicate-marker digit: SPY2->SPY, UVXY3->UVXY, _d suffix off."""
    s = stem.upper()
    if s.endswith("_D"):
        s = s[:-2]
    while s and s[-1].isdigit():
        s = s[:-1]
    return s


def seed_from_csv(csv_dir: Path = DEFAULT_SEED_CSV_DIR,
                  fetcher: Optional[YFinanceFetcher] = None,
                  fetch_actions: bool = True) -> dict:
    """
    Ingest the historical CSV base. For numbered duplicates (SPY/SPY2/SPY3),
    pick the LONGEST clean series per ticker and log which file won.

    `fetch_actions=True` also seeds the actions log from yfinance so read-time
    adjustment of the historical base is correct. Set False for offline runs.
    """
    csv_dir = Path(csv_dir)
    files = sorted(p for p in csv_dir.glob("*.csv")
                   if not p.name.startswith("_"))
    _log(f"[seed] scanning {len(files)} CSV files in {csv_dir}")

    # Group candidate files by base ticker.
    groups: dict[str, list[Path]] = {}
    for p in files:
        groups.setdefault(_base_ticker(p.stem), []).append(p)

    all_clean, all_quar = [], []
    winners, ingested_tickers = [], []
    for ticker, candidates in sorted(groups.items()):
        best = None  # (clean_df, source, path, n_clean, n_quar)
        for p in candidates:
            try:
                df, source = _read_seed_csv(p)
            except Exception as e:
                _log(f"  [seed] skip {p.name}: {e}")
                continue
            clean, quar = validate(df, ticker, _empty_actions(), check_returns=False)
            if best is None or len(clean) > best[3] or (
                    len(clean) == best[3] and len(quar) < best[4]):
                best = (clean, source, p, len(clean), len(quar))
        if best is None or best[3] == 0:
            _log(f"  [seed] {ticker}: no usable rows")
            continue
        clean, source, p, n_clean, n_quar = best
        clean = clean.assign(ticker=ticker, source=source, ingested_at=_now())
        all_clean.append(clean[list(_PRICE_SCHEMA.keys())])
        if n_quar:
            # re-derive the quarantined rows from the winning file for the audit log
            _df, _src = _read_seed_csv(p)
            _, quar = validate(_df, ticker, _empty_actions(), check_returns=False)
            if len(quar):
                quarantine_rows(quar.assign(ticker=ticker, source=source))
        reason = (f"chosen from {len(candidates)} file(s): "
                  f"{p.name} ({n_clean} clean rows"
                  + (f", beat {[c.name for c in candidates if c != p]}" if len(candidates) > 1 else "")
                  + ")")
        winners.append({"ticker": ticker, "file": p.name, "rows": n_clean,
                        "source": source, "note": reason})
        ingested_tickers.append(ticker)
        if len(candidates) > 1:
            _log(f"  [seed] {ticker}: {reason}")

    if not all_clean:
        _log("[seed] nothing to ingest")
        return {"tickers": 0, "rows": 0}

    prices = pd.concat(all_clean, ignore_index=True)
    written = upsert_prices(prices)
    _log(f"[seed] ingested {written} rows across {len(ingested_tickers)} tickers")

    # Seed the actions log so the historical base adjusts correctly.
    n_actions = 0
    if fetch_actions:
        fetcher = fetcher or YFinanceFetcher()
        for ticker in ingested_tickers:
            try:
                acts = fetcher.actions(ticker)
                n_actions += upsert_actions(acts)
            except Exception as e:
                _log(f"  [seed] {ticker}: actions fetch failed ({e})")
        _log(f"[seed] logged {n_actions} corporate actions")

    return {"tickers": len(ingested_tickers), "rows": written,
            "actions": n_actions, "winners": winners}


# ---------------------------------------------------------------------------
# Daily update
# ---------------------------------------------------------------------------

def _stored_action_keys(ticker: str) -> set:
    acts = read_actions()
    acts = acts[acts["ticker"] == ticker]
    return set(zip(acts["date"], acts["type"], acts["value"].round(6)))


def _full_repull(ticker: str, fetcher) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist = fetcher.history(ticker, start=SEED_START)
    if hist is None or len(hist) == 0:
        return pd.DataFrame(), pd.DataFrame()
    df = hist.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df, fetcher.actions(ticker)


def update_ticker(ticker: str, fetcher, trailing_days: int = TRAILING_DAYS) -> dict:
    """
    Update one ticker:
      1. Detect corporate actions; a newly-seen split/dividend forces a FULL
         re-pull + rewrite (raw prices shift on a split).
      2. Otherwise re-pull a trailing window and UPSERT (overwrite overlaps).
         The window auto-extends back to the last stored bar, so an arbitrarily
         large gap (stale CSV seed, multi-day outage) is backfilled with no
         permanent hole -- even for tickers that have no corporate actions.
    Validates before writing; quarantines suspect rows.
    """
    info = {"ticker": ticker, "mode": None, "rows": 0, "quarantined": 0,
            "new_actions": 0}

    # --- corporate-action detection -------------------------------------
    fetched_actions = fetcher.actions(ticker)
    new_action_keys = set(
        zip(fetched_actions["date"], fetched_actions["type"],
            fetched_actions["value"].round(6))
    ) - _stored_action_keys(ticker)
    info["new_actions"] = len(new_action_keys)

    if new_action_keys:
        upsert_actions(fetched_actions)
        df, acts = _full_repull(ticker, fetcher)
        info["mode"] = "full_repull"
        if len(df) == 0:
            info["error"] = "repull returned no data"
            return info
        # Rewrite the ticker's whole series: drop old rows, insert fresh.
        clean, quar = validate(df, ticker, acts)
        existing = read_prices_raw()
        existing = existing[existing["ticker"] != ticker]
        clean = clean.assign(ticker=ticker, source="yahoo", ingested_at=_now())
        merged = _upsert(existing, clean[list(_PRICE_SCHEMA.keys())],
                         keys=["ticker", "date"], sort_by=["ticker", "date"])
        _atomic_write_parquet(merged, PRICES_PARQUET, _PRICE_SCHEMA)
        if len(quar):
            quar = quar.assign(ticker=ticker, source="yahoo")
            quarantine_rows(quar)
        info["rows"] = len(clean)
        info["quarantined"] = len(quar)
        return info

    # --- trailing-window upsert (auto-extends to backfill any gap) -------
    info["mode"] = "trailing"
    today = pd.Timestamp(_now().date())
    sessions = trading_days(today - pd.Timedelta(days=trailing_days * 3), today)
    trailing_start = (sessions[-trailing_days] if len(sessions) >= trailing_days
                      else (sessions[0] if len(sessions) else today))

    existing = read_prices_raw()
    tk_rows = existing[existing["ticker"] == ticker]
    last_stored = tk_rows["date"].max() if len(tk_rows) else None
    if last_stored is None:
        # Brand-new ticker -> pull full history.
        start_ts = pd.Timestamp(SEED_START)
        info["mode"] = "full_history"
    else:
        # Pull from the EARLIER of the trailing window and the last stored bar.
        # This backfills any gap (e.g. a stale CSV seed, or a multi-day failed
        # run) with no permanent hole, regardless of whether the ticker has
        # corporate actions, while still re-pulling recent bars for self-heal.
        start_ts = min(trailing_start, pd.Timestamp(last_stored))
        if start_ts < trailing_start:
            info["mode"] = "backfill"
    start = start_ts.strftime("%Y-%m-%d")

    hist = fetcher.history(ticker, start=start)
    if hist is None or len(hist) == 0:
        info["note"] = "no rows returned (holiday/halt?)"
        return info
    df = hist.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

    # prev_close (last stored bar before the window) for the first row's return
    prior = existing[(existing["ticker"] == ticker)
                     & (existing["date"] < df["date"].min())]
    prev_close = float(prior.sort_values("date")["close"].iloc[-1]) if len(prior) else None

    clean, quar = validate(df, ticker, read_actions()[read_actions()["ticker"] == ticker],
                           prev_close=prev_close)
    clean = clean.assign(ticker=ticker, source="yahoo", ingested_at=_now())
    info["rows"] = upsert_prices(clean[list(_PRICE_SCHEMA.keys())])
    if len(quar):
        quar = quar.assign(ticker=ticker, source="yahoo")
        quarantine_rows(quar)
        info["quarantined"] = len(quar)
    return info


def _resolve_universe(tickers: Optional[Iterable[str]]) -> list[str]:
    if tickers:
        return list(tickers)
    if SEED_TICKERS_FILE.exists():
        with open(SEED_TICKERS_FILE) as f:
            return [ln.strip() for ln in f if ln.strip()]
    # else: everything already in the store
    return sorted(read_prices_raw()["ticker"].dropna().unique().tolist())


def update_price_store(tickers: Optional[Iterable[str]] = None,
                       fetcher=None,
                       trailing_days: int = TRAILING_DAYS,
                       write_manifest: bool = True) -> dict:
    """
    Refresh the whole universe (trailing-window upsert + action handling).

    Per-ticker failures are isolated; the run as a whole is also wrapped so a
    failure can never propagate to the caller (the signal monitor). Safe to
    run multiple times a day -- upserts make it idempotent.
    """
    summary = {"timestamp": _now().isoformat(), "ok": [], "failed": [],
               "skipped": [], "actions_detected": [], "quarantined": 0,
               "rows_written": 0}
    try:
        fetcher = fetcher or YFinanceFetcher()
        universe = _resolve_universe(tickers)
        _log(f"[update] universe = {len(universe)} tickers, "
             f"trailing_days={trailing_days}")
        for t in universe:
            try:
                info = update_ticker(t, fetcher, trailing_days)
                summary["rows_written"] += info.get("rows", 0)
                summary["quarantined"] += info.get("quarantined", 0)
                if info.get("new_actions"):
                    summary["actions_detected"].append(t)
                if info.get("error") or (info.get("rows", 0) == 0 and "note" in info):
                    summary["skipped"].append({"ticker": t,
                                               "reason": info.get("error") or info.get("note")})
                else:
                    summary["ok"].append(t)
                tag = info.get("mode", "?")
                _log(f"  {'✓' if t in summary['ok'] else '-'} {t:<10} "
                     f"{tag:<12} rows={info.get('rows',0)} "
                     f"q={info.get('quarantined',0)} "
                     f"acts={info.get('new_actions',0)}")
            except Exception as e:
                summary["failed"].append({"ticker": t, "error": f"{type(e).__name__}: {e}"})
                _log(f"  ✗ {t:<10} {type(e).__name__}: {e}")

        cov = coverage_report(as_dict=True)
        summary["coverage"] = {"tickers": cov["n_tickers"],
                               "latest_date": cov["latest_date"],
                               "total_rows": cov["total_rows"]}
        if write_manifest:
            STORE_DIR.mkdir(parents=True, exist_ok=True)
            with open(MANIFEST_FILE, "w") as f:
                json.dump(summary, f, indent=2, default=str)
        _log(f"[update] done: {len(summary['ok'])} ok, "
             f"{len(summary['skipped'])} skipped, {len(summary['failed'])} failed, "
             f"{len(summary['actions_detected'])} with new actions, "
             f"{summary['quarantined']} quarantined")
    except Exception as e:  # the outer guard -- must never raise to the caller
        summary["fatal"] = f"{type(e).__name__}: {e}"
        _log(f"[update] FATAL (suppressed): {summary['fatal']}")
    return summary


# ---------------------------------------------------------------------------
# Read API (with read-time adjustment)
# ---------------------------------------------------------------------------

def _adjust_one(df: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """
    Back-adjust a single ticker's raw OHLCV for splits + dividends.

      split_factor[d] = product of split ratios with ex-date > d  (old prices
                        divided by it; volume multiplied by it)
      div_factor[d]   = product of (1 - amt/prev_close) for dividends with
                        ex-date > d
      adjusted price  = raw * div_factor / split_factor
    """
    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"]
    n = len(df)
    split_factor = pd.Series(1.0, index=df.index)
    div_factor = pd.Series(1.0, index=df.index)

    if actions is not None and len(actions):
        splits = actions[actions["type"] == "split"].sort_values("date")
        for _, a in splits.iterrows():
            ex, ratio = pd.Timestamp(a["date"]), float(a["value"])
            if ratio > 0:
                split_factor[dates < ex] *= ratio
        divs = actions[actions["type"] == "dividend"].sort_values("date")
        closes = df["close"].astype(float)
        for _, a in divs.iterrows():
            ex, amt = pd.Timestamp(a["date"]), float(a["value"])
            prior = closes[dates < ex]
            if amt <= 0 or len(prior) == 0:
                continue
            pc = prior.iloc[-1]
            if pc <= 0:
                continue
            f = 1.0 - amt / pc
            if 0 < f <= 1:
                div_factor[dates < ex] *= f

    price_adj = (div_factor / split_factor)
    out = df.copy()
    for c in ["open", "high", "low", "close"]:
        out[c] = df[c].astype(float) * price_adj
    out["volume"] = df["volume"].astype(float) * split_factor
    return out


def _query_raw(tickers: Sequence[str], start: Optional[str],
               end: Optional[str]) -> pd.DataFrame:
    """Filtered read of the prices table via DuckDB (predicate pushdown)."""
    if not PRICES_PARQUET.exists():
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in _PRICE_SCHEMA.items()})
    import duckdb
    con = duckdb.connect()
    clauses, params = [], [str(PRICES_PARQUET)]
    if tickers:
        clauses.append(f"ticker IN ({','.join(['?'] * len(tickers))})")
        params += list(tickers)
    if start:
        clauses.append("date >= ?")
        params.append(pd.Timestamp(start))
    if end:
        clauses.append("date <= ?")
        params.append(pd.Timestamp(end))
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    q = f"SELECT * FROM read_parquet(?){where} ORDER BY ticker, date"
    try:
        return con.execute(q, params).df()
    finally:
        con.close()


def get_prices(tickers, start: Optional[str] = None, end: Optional[str] = None,
               adjusted: bool = True, layout: str = "wide") -> pd.DataFrame:
    """
    Read prices for one or more tickers -- a drop-in replacement for
    ``yf.download(...)`` in backtests.

    adjusted=True  -> O/H/L/C back-adjusted for splits+dividends, volume
                      split-adjusted (computed at read time from the actions
                      log). adjusted=False returns the raw stored prints.
    layout="wide"  -> single ticker: columns Open/High/Low/Close/Volume,
                      DatetimeIndex. Multiple tickers: MultiIndex columns
                      (Field, Ticker) just like yfinance.
    layout="long"  -> tidy frame: date, ticker, open, high, low, close, volume.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = list(tickers)
    raw = _query_raw(tickers, start, end)
    if len(raw) == 0:
        return pd.DataFrame()
    raw["date"] = pd.to_datetime(raw["date"])

    acts = read_actions()
    frames = []
    for t in tickers:
        sub = raw[raw["ticker"] == t]
        if len(sub) == 0:
            continue
        sub = _adjust_one(sub, acts[acts["ticker"] == t]) if adjusted else \
            sub.sort_values("date").reset_index(drop=True)
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    tidy = pd.concat(frames, ignore_index=True)

    if layout == "long":
        return tidy[["date", "ticker", *PRICE_COLS]].reset_index(drop=True)

    # wide
    rename = {"open": "Open", "high": "High", "low": "Low",
              "close": "Close", "volume": "Volume"}
    tidy = tidy.rename(columns=rename)
    fields = ["Open", "High", "Low", "Close", "Volume"]
    if len(tickers) == 1:
        out = tidy.set_index("date")[fields].sort_index()
        out.index.name = "Date"
        return out
    wide = tidy.pivot(index="date", columns="ticker", values=fields)
    wide.index.name = "Date"
    return wide.sort_index()


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def coverage_report(as_dict: bool = False):
    prices = read_prices_raw()
    if len(prices) == 0:
        rep = {"n_tickers": 0, "total_rows": 0, "latest_date": None, "per_ticker": []}
        return rep if as_dict else _print_coverage(rep)
    prices["date"] = pd.to_datetime(prices["date"])
    per = []
    for t, g in prices.groupby("ticker", observed=True):
        g = g.sort_values("date")
        first, last = g["date"].min(), g["date"].max()
        expected = len(trading_days(first, last))
        gaps = max(expected - len(g), 0)
        per.append({"ticker": str(t), "rows": len(g),
                    "first": first.strftime("%Y-%m-%d"),
                    "last": last.strftime("%Y-%m-%d"),
                    "missing_sessions": int(gaps)})
    per.sort(key=lambda r: r["ticker"])
    rep = {"n_tickers": len(per), "total_rows": int(len(prices)),
           "latest_date": prices["date"].max().strftime("%Y-%m-%d"),
           "per_ticker": per}
    return rep if as_dict else _print_coverage(rep)


def _print_coverage(rep: dict) -> dict:
    _log(f"\nCoverage: {rep['n_tickers']} tickers, {rep['total_rows']} rows, "
         f"latest={rep['latest_date']}")
    _log(f"{'ticker':<10} {'rows':>7}  {'first':<11} {'last':<11} {'gaps':>5}")
    _log("-" * 50)
    for r in rep["per_ticker"]:
        flag = "  <-- gaps" if r["missing_sessions"] else ""
        _log(f"{r['ticker']:<10} {r['rows']:>7}  {r['first']:<11} "
             f"{r['last']:<11} {r['missing_sessions']:>5}{flag}")
    return rep


# ---------------------------------------------------------------------------
# Offline acceptance tests (no network required)
# ---------------------------------------------------------------------------

def _self_test() -> int:
    """
    Prove the acceptance criteria with a synthetic, deterministic fetcher:
      1. seed from CSV          2. update                3. update twice = no dup
      4. simulated gap self-heals    5. simulated split detected + no phantom return
    Runs in an isolated temp store so it never touches real data.
    """
    import tempfile, shutil
    global STORE_DIR, PRICES_PARQUET, ACTIONS_PARQUET, QUARANTINE_PARQUET, MANIFEST_FILE
    saved = (STORE_DIR, PRICES_PARQUET, ACTIONS_PARQUET, QUARANTINE_PARQUET, MANIFEST_FILE)
    tmp = Path(tempfile.mkdtemp(prefix="pricestore_test_"))
    STORE_DIR = tmp
    PRICES_PARQUET = tmp / "prices.parquet"
    ACTIONS_PARQUET = tmp / "actions.parquet"
    QUARANTINE_PARQUET = tmp / "quarantine.parquet"
    MANIFEST_FILE = tmp / "_last_run.json"

    failures = []

    def check(name, cond):
        _log(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    class SyntheticFetcher:
        """Deterministic price generator with a controllable trailing window,
        a fakeable gap, and an injectable split."""
        def __init__(self):
            self.split_on = {}        # ticker -> (date, ratio)
            self.missing = set()      # (ticker, date-string) to omit (simulated gap)
            self.last = "2026-06-18"  # latest available session (06-19 is Juneteenth)

        def _series(self, ticker, start, end):
            days = trading_days(start, end)
            rows = []
            base = 100.0 + (hash(ticker) % 50)
            for i, d in enumerate(days):
                if (ticker, d.strftime("%Y-%m-%d")) in self.missing:
                    continue
                price = base + i  # gently rising, 1.0/day -> small returns
                sp = self.split_on.get(ticker)
                if sp and d >= pd.Timestamp(sp[0]):
                    price = price / sp[1]          # raw price drops post-split
                rows.append((d, price, price + 1, price - 1, price, 1_000_000.0))
            df = pd.DataFrame(rows, columns=["date", "open", "high", "low",
                                             "close", "volume"]).set_index("date")
            return df

        def history(self, ticker, start=None, end=None, period=None):
            start = start or "2026-06-01"
            return self._series(ticker, start, self.last)

        def actions(self, ticker):
            sp = self.split_on.get(ticker)
            if not sp:
                return _empty_actions()
            return pd.DataFrame([(pd.Timestamp(sp[0]), ticker, "split", float(sp[1]))],
                                columns=["date", "ticker", "type", "value"]).astype(_ACTION_SCHEMA)

    try:
        fx = SyntheticFetcher()

        # 1) seed
        res = seed_from_csv(DEFAULT_SEED_CSV_DIR, fetch_actions=False)
        check("seed ingested >0 tickers", res.get("tickers", 0) > 0)
        check("seed ingested >0 rows", res.get("rows", 0) > 0)
        seeded = read_prices_raw()
        # numbered-duplicate dedup: never two files for one ticker
        check("no duplicate (ticker,date) after seed",
              not seeded.duplicated(["ticker", "date"]).any())

        test_tickers = ["SPY", "QQQ"]

        # 2) update
        update_price_store(test_tickers, fetcher=fx, write_manifest=False)
        after1 = read_prices_raw()
        n1 = len(after1[after1["ticker"] == "SPY"])
        check("update brought SPY through the latest session (2026-06-18)",
              after1[after1["ticker"] == "SPY"]["date"].max() == pd.Timestamp("2026-06-18"))

        # 3) idempotency -- run again, expect identical row count
        update_price_store(test_tickers, fetcher=fx, write_manifest=False)
        after2 = read_prices_raw()
        n2 = len(after2[after2["ticker"] == "SPY"])
        check("running update twice produces no duplicate rows", n1 == n2)
        check("no duplicate (ticker,date) after double update",
              not after2.duplicated(["ticker", "date"]).any())

        # 4) simulated failed run leaves a hole -> next trailing window heals it
        gap_day = "2026-06-17"
        # manually punch a hole in the store
        store = read_prices_raw()
        store = store[~((store["ticker"] == "SPY") &
                        (store["date"] == pd.Timestamp(gap_day)))]
        _atomic_write_parquet(store, PRICES_PARQUET, _PRICE_SCHEMA)
        check("gap exists before heal",
              len(read_prices_raw().query("ticker=='SPY' and date==@pd.Timestamp(@gap_day)")) == 0)
        update_price_store(["SPY"], fetcher=fx, write_manifest=False)
        healed = read_prices_raw()
        check("trailing-window update self-healed the gap",
              len(healed[(healed["ticker"] == "SPY") &
                         (healed["date"] == pd.Timestamp(gap_day))]) == 1)

        # 4b) LARGE staleness gap (seed older than the trailing window) on an
        # action-less ticker -> must still backfill the whole span, no hole.
        store = read_prices_raw()
        store = store[~((store["ticker"] == "SPY") &
                        (store["date"] > pd.Timestamp("2026-06-09")))]
        _atomic_write_parquet(store, PRICES_PARQUET, _PRICE_SCHEMA)
        update_price_store(["SPY"], fetcher=fx, write_manifest=False)  # SPY has no actions
        bf = read_prices_raw().query("ticker=='SPY'")
        wanted = ["2026-06-12", "2026-06-15", "2026-06-16", "2026-06-18"]
        got = {pd.Timestamp(d) for d in bf["date"]}
        check("large staleness gap backfilled (no permanent hole)",
              all(pd.Timestamp(d) in got for d in wanted))

        # 5) simulated split: detected -> full re-pull -> no phantom return
        fx.split_on["QQQ"] = ("2026-06-18", 10.0)  # 10-for-1
        update_price_store(["QQQ"], fetcher=fx, write_manifest=False)
        acts = read_actions()
        check("split detected & logged",
              len(acts[(acts["ticker"] == "QQQ") & (acts["type"] == "split")]) == 1)
        # raw store shows the -90% drop on the split date...
        rawq = read_prices_raw().query("ticker=='QQQ'").sort_values("date")
        raw_ret = rawq["close"].pct_change()
        check("raw series shows the split-day crash",
              raw_ret.min() < -0.5)
        # ...but the ADJUSTED read removes the phantom return
        adj = get_prices("QQQ", adjusted=True)
        adj_ret = adj["Close"].pct_change().dropna()
        check("adjusted read has no phantom split return (|ret|<10%)",
              adj_ret.abs().max() < 0.10)

        # validation gate: NaN + zero-volume get quarantined
        bad = pd.DataFrame({
            "date": [pd.Timestamp("2026-06-22"), pd.Timestamp("2026-06-23")],
            "open": [np.nan, 100.0], "high": [1, 100.0], "low": [1, 100.0],
            "close": [1, 100.0], "volume": [5, 0.0],
        })
        clean, quar = validate(bad, "ZZZ", _empty_actions())
        check("validation quarantines NaN + zero-volume rows", len(quar) == 2)

    finally:
        STORE_DIR, PRICES_PARQUET, ACTIONS_PARQUET, QUARANTINE_PARQUET, MANIFEST_FILE = saved
        shutil.rmtree(tmp, ignore_errors=True)

    _log("")
    if failures:
        _log(f"SELF-TEST FAILED: {len(failures)} check(s): {failures}")
        return 1
    _log("SELF-TEST PASSED: all acceptance checks green")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", action="store_true", help="ingest historical CSV base")
    ap.add_argument("--csv-dir", default=str(DEFAULT_SEED_CSV_DIR),
                    help="directory of seed CSVs (default: data/ohlcv)")
    ap.add_argument("--no-seed-actions", action="store_true",
                    help="skip fetching corporate actions during seed (offline)")
    ap.add_argument("--update", action="store_true", help="trailing-window refresh")
    ap.add_argument("--tickers", default=None,
                    help="comma-separated ticker subset (default: full universe)")
    ap.add_argument("--trailing-days", type=int, default=TRAILING_DAYS)
    ap.add_argument("--report", action="store_true", help="print coverage report")
    ap.add_argument("--self-test", action="store_true", help="run offline acceptance tests")
    args = ap.parse_args(argv)

    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else None

    if args.self_test:
        return _self_test()
    did = False
    if args.seed:
        seed_from_csv(Path(args.csv_dir), fetch_actions=not args.no_seed_actions)
        did = True
    if args.update:
        update_price_store(tickers, trailing_days=args.trailing_days)
        did = True
    if args.report or not did:
        coverage_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
