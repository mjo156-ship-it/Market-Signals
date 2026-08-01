#!/usr/bin/env python3
"""
composer_oos_discover.py — OOS-confidence report for Composer *Discover* strategies.

Evaluates public/Discover symphonies exactly the way the watchlist tab does, but
with the goal of surfacing strategies whose out-of-sample edge is BOTH strong and
well-sampled (long enough OOS window to trust). For each symphony:

    in-sample  = backtest of the frozen definition BEFORE its freeze date
    out-sample = backtest ON/AFTER the freeze date (genuine forward test)
    confidence = statistical tier from the OOS window length, because the SE of an
                 annualized Sharpe is ~ sqrt(252/n_days): short windows can't tell
                 skill from noise (<6mo directional, ~1y weak, ~2y usable, 3y+ solid).

Then ranks by a confidence-GATED score so a dazzling 3-month OOS doesn't outrank a
solid 3-year one.

SOURCE OF THE DISCOVER LIST (priority order):
    1. $COMPOSER_DISCOVER_SEED path, else data/composer_discover_seed.json
       — a committed JSON list: [{"id","name","last_semantic_update_at"}, ...].
       This is the reliable path: paste the Discover network-response JSON (same as
       the watchlist) and normalize it into that file.
    2. Auto-fetch: try candidate read-only Discover endpoints with the API keys.
       Discover is a logged-in web feature and is most likely browser-session gated
       (the watchlist was), so this usually returns nothing — it's a best-effort
       attempt that logs exactly what each endpoint returned so we know.

Read-only by construction: reuses composer_oos.backtest (the one permitted POST)
and read GETs. Writes data/composer_oos_discover.jsonl and prints a ranked report.
"""
from __future__ import annotations
import os, sys, json, time
from datetime import date

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos as oos

# P2-8: request each symphony's full available history (Composer clamps to the
# earliest its constituents allow), so the in-sample window is per-symphony, not
# a hardcoded 2015. Configurable floor.
EARLY_START = os.environ.get("COMPOSER_OOS_EARLY_START", "1990-01-01")
OUT = os.environ.get("COMPOSER_OOS_DISC_PATH", "data/composer_oos_discover.jsonl")
SEED = os.environ.get("COMPOSER_DISCOVER_SEED", "data/composer_discover_seed.json")
# Only backtest strategies whose OOS window is at least this many days — the
# confidence tier is a function of the freeze date alone, so recently-tweaked
# "Thin" strategies (no meaningful OOS) can be skipped without a backtest. This
# keeps the run tractable when the seed is the full library. 0 = evaluate all.
MIN_OOS_DAYS = int(os.environ.get("COMPOSER_DISCOVER_MIN_OOS_DAYS", "0"))

# Candidate read-only Discover endpoints, tried in order if no seed file exists.
# Siblings of the known watchlist endpoint (trading-api.composer.trade/api/v1/
# watchlist) plus a couple of public guesses. Best-effort — logged, never fatal.
_DISCOVER_CANDIDATES = [
    "https://trading-api.composer.trade/api/v1/discover",
    "https://trading-api.composer.trade/api/v1/discover/symphonies",
    "https://trading-api.composer.trade/api/v1/community/symphonies",
    "https://api.composer.trade/api/v1/discover",
    "https://api.composer.trade/api/v0.1/discover",
    "https://api.composer.trade/api/v0.1/public/symphonies",
]


# OOS confidence tier now lives in composer_oos._confidence (shared with the
# watchlist ledger so both carry oos_conf / oos_conf_rank / sharpe_se).
_confidence = oos._confidence


def _cs(c):
    return oos._curve_stats(c)


def _ext(curve):
    """Calmar / Sortino / Ulcer / Martin from an equity curve (see watchlist)."""
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
        dd = v / peak - 1
        mdd = min(mdd, dd)
        sq += (dd * 100.0) ** 2
    ulcer = (sq / len(vals)) ** 0.5
    calmar = round(cagr / abs(mdd), 2) if (cagr is not None and mdd < 0) else None
    martin = round((cagr * 100) / ulcer, 2) if (cagr is not None and ulcer > 0) else None
    return {"calmar": calmar, "sortino": sortino,
            "ulcer": round(ulcer, 2), "martin": martin}


def _backtest_resilient(sid, start, end, tries=5):
    """oos.backtest with extra backoff on the backtest endpoint's burst rate limit."""
    for k in range(tries):
        try:
            return oos.backtest(sid, start, end)
        except RuntimeError as e:
            if "rate limited" in str(e).lower() and k < tries - 1:
                time.sleep(6 * (k + 1))
                continue
            raise


# ── Discover list sourcing ───────────────────────────────────────────────────
def _freeze_iso(frz):
    """Freeze date -> 'YYYY-MM-DD'. Accepts ISO strings or epoch (ms/s/days),
    the way Composer's Algolia index returns last_semantic_update_at (epoch ms)."""
    if isinstance(frz, (int, float)):
        return oos._to_date(frz)
    s = str(frz)
    if s[:4].isdigit() and "-" in s:      # already ISO-ish
        return s[:10]
    if s.isdigit():                        # epoch as string
        return oos._to_date(int(s))
    return s[:10]


def _norm(items):
    """Normalize a list of symphony dicts to (id, freeze_date, name) tuples.
    Accepts the field spellings Composer uses across endpoints."""
    out = []
    for s in items or []:
        if not isinstance(s, dict):
            continue
        sid = s.get("id") or s.get("symphony_id") or s.get("symphonyId")
        frz = (s.get("last_semantic_update_at") or s.get("lastSemanticUpdateAt")
               or s.get("last_semantic_update") or s.get("frozen_at"))
        name = s.get("name") or s.get("title") or sid
        if not sid or frz is None:
            continue
        out.append((sid, _freeze_iso(frz), name))
    return out


def _find_list(obj):
    """Dig a symphony list out of an arbitrary JSON response shape."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("symphonies", "discover", "results", "items", "data", "rows"):
            v = obj.get(key)
            if isinstance(v, list) and v:
                return v
        # single nested container
        for v in obj.values():
            got = _find_list(v)
            if got:
                return got
    return []


def _load_seed():
    if os.path.exists(SEED):
        try:
            with open(SEED) as f:
                raw = json.load(f)
        except Exception as e:
            print(f"[discover] seed {SEED} unreadable: {e}", file=sys.stderr)
            return []
        got = _norm(_find_list(raw) if not isinstance(raw, list) else raw)
        print(f"[discover] seed {SEED}: {len(got)} symphonies with freeze dates")
        return got
    return []


def _auto_fetch():
    """Best-effort read-only probe of candidate Discover endpoints. Logs each so we
    learn whether Discover is reachable with the API keys or (as expected) gated."""
    try:
        headers = oos._headers()
    except Exception as e:
        print(f"[discover] no API keys for auto-fetch: {e}", file=sys.stderr)
        return []
    for url in _DISCOVER_CANDIDATES:
        try:
            r = requests.get(url, headers=headers, timeout=30)
        except Exception as e:
            print(f"[discover] {url} -> ERROR {type(e).__name__}: {str(e)[:80]}")
            continue
        ct = r.headers.get("content-type", "")
        print(f"[discover] {url} -> HTTP {r.status_code} ({ct.split(';')[0]}, "
              f"{len(r.content)} bytes)")
        if r.status_code != 200 or "json" not in ct:
            continue
        try:
            got = _norm(_find_list(r.json()))
        except Exception:
            continue
        if got:
            print(f"[discover] auto-fetch OK via {url}: {len(got)} symphonies")
            # Cache what we found so it becomes the committed seed.
            try:
                os.makedirs(os.path.dirname(SEED) or ".", exist_ok=True)
                with open(SEED, "w") as f:
                    json.dump([{"id": i, "last_semantic_update_at": d, "name": n}
                               for i, d, n in got], f, indent=2)
                print(f"[discover] cached discovered list -> {SEED}")
            except Exception as e:
                print(f"[discover] could not cache seed: {e}", file=sys.stderr)
            return got
    print("[discover] auto-fetch found nothing (Discover likely browser-session "
          "gated, as the watchlist was). Provide data/composer_discover_seed.json.")
    return []


def discover_list():
    return _load_seed() or _auto_fetch()


# ── Main ─────────────────────────────────────────────────────────────────────
def main(rewrite=False):
    asof = date.today().isoformat()
    history = oos.load_jsonl(OUT)          # prior runs, for run-over-run deltas
    watch = discover_list()
    if not watch:
        print("[discover] no Discover symphonies to evaluate — leaving the "
              "existing ledger untouched (preserving history).", file=sys.stderr)
        # Do NOT truncate: a transient empty seed must not wipe accumulated
        # history. Only seed an empty file if none exists yet.
        os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
        if not os.path.exists(OUT):
            open(OUT, "w").close()
        return

    # Freeze-date prefilter: skip strategies whose OOS window is shorter than
    # MIN_OOS_DAYS *before* backtesting. Confidence is a function of the freeze
    # date alone, so this drops the Thin (recently-tweaked) tail for free and
    # keeps the run tractable when the seed is the full library.
    if MIN_OOS_DAYS > 0:
        today = date.today()
        kept = []
        for sid, oosdate, name in watch:
            try:
                y, m, d = (int(x) for x in oosdate[:10].split("-"))
                age = (today - date(y, m, d)).days
            except Exception:
                age = 0
            if age >= MIN_OOS_DAYS:
                kept.append((sid, oosdate, name))
        print(f"[discover] freeze-date prefilter: {len(kept)}/{len(watch)} have "
              f">= {MIN_OOS_DAYS}d OOS; skipping {len(watch) - len(kept)} thin ones")
        watch = kept

    rows = []
    for i, (sid, oosdate, name) in enumerate(watch, 1):
        time.sleep(0.5)
        try:
            curve, _ = _backtest_resilient(sid, EARLY_START, asof)
        except Exception as e:
            print(f"[{i}/{len(watch)}] {name}: FAIL {type(e).__name__}: {str(e)[:80]}",
                  file=sys.stderr)
            continue
        if not curve:
            print(f"[{i}/{len(watch)}] {name}: empty curve", file=sys.stderr)
            continue
        oos_curve = {d: v for d, v in curve.items() if d >= oosdate}
        ins = _cs({d: v for d, v in curve.items() if d < oosdate})
        out = _cs(oos_curve)
        oute = _ext(oos_curve)
        conf, conf_rank, sharpe_se = _confidence(out.get("n_days"))
        if ins.get("cagr_pct") is not None and out.get("cagr_pct") is not None:
            gap = round(out["cagr_pct"] - ins["cagr_pct"], 2)
        else:
            gap = None
        extra = oos.oos_extra_fields(oos_curve, out, ins.get("cagr_pct"), gap,
                                     full_curve=curve)
        is_days = ins.get("n_days")
        row = {
            "date": asof, "scope": "oos_split", "sym_id": sid, "name": name,
            "oos_date": oosdate,
            "bt_start": min(curve), "bt_end": max(curve),
            "is_days": is_days, "is_years": round(is_days / 252, 1) if is_days else None,
            "oos_days": out.get("n_days"),
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
        print(f"[{i}/{len(watch)}] {name}: OOS {out.get('n_days')}d "
              f"cagr {out.get('cagr_pct')} sharpe {out.get('oos_sharpe')} conf {conf}")

    # P1-4: stamp run-over-run Sharpe deltas, then append (keeping prior runs)
    # with dedup on (sym_id, date). --rewrite restores the old truncate behaviour.
    oos.stamp_sharpe_deltas(rows, history, asof)
    kept, added = oos.append_history(OUT, rows, asof, rewrite=rewrite)
    print(f"[discover] {'rewrote' if rewrite else 'appended'}: "
          f"{added} new rows + {kept} historical -> {OUT}")

    # Confidence-gated ranking: only strategies with a usable OOS window (>=1y,
    # rank>=2) compete on the sample-size-shrunk Sharpe (P2-7): sharpe_lcb =
    # OOS Sharpe − 1 SE, so a longer-OOS strategy outranks a flashier short one.
    def score(r):
        lcb = r.get("sharpe_lcb")
        return (r.get("oos_conf_rank") or 0, lcb if lcb is not None else -9)
    hi = sorted([r for r in rows if (r.get("oos_conf_rank") or 0) >= 2
                 and (r.get("oos_sharpe") or -9) > 0], key=score, reverse=True)
    hdr = (f"{'conf':>4} {'oosDays':>7} {'OOScagr':>7} {'OOSsh':>5} {'shLCB':>6} {'gap':>7} "
           f"{'calmar':>6}  name")
    print("\n=== HIGH-CONFIDENCE OOS (>=1y window, positive OOS Sharpe) ===")
    print(hdr); print("-" * len(hdr))
    for r in hi:
        lcb = r.get("sharpe_lcb")
        print(f"{r.get('oos_conf') or '?':>4} {r.get('oos_days') or 0:7d} "
              f"{r.get('oos_cagr_pct') or 0:7.1f} {r.get('oos_sharpe') or 0:5.2f} "
              f"{(lcb if lcb is not None else 0):6.2f} "
              f"{r.get('cagr_gap_pct') or 0:7.1f} {r.get('oos_calmar') or 0:6.2f}  "
              f"{(r.get('name') or '')[:44]}")
    print(f"\n[discover] {len(rows)}/{len(watch)} evaluated, "
          f"{len(hi)} high-confidence -> {OUT}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrite", action="store_true",
                    help="truncate the ledger instead of appending history")
    main(rewrite=ap.parse_args().rewrite)
