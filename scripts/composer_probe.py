#!/usr/bin/env python3
"""
composer_probe.py — READ-ONLY discovery for Composer's watchlist / discover API.

WHY
    composer_oos.py only captures *invested* symphonies (symphony-stats-meta).
    The Watchlist (184 symphonies, with an "Out of Sample Date" per entry) and
    Discover leaderboards are richer OOS sources, but their backing API paths are
    not documented in this repo. This script discovers them two ways:

      1. Download the app's compiled JS bundle (a public static asset, no auth)
         and extract every /api/... path + any 'watch'/'discover' string literal.
      2. With the API key, GET a set of candidate endpoints and print the status
         and JSON shape of whatever responds — so we can see the real structure.

    It writes NOTHING and changes NOTHING. GETs only, routed through
    composer_oos._request so the same read-only safety guard applies.

USAGE
    python composer_probe.py                       # bundle scan + candidate GETs
    python composer_probe.py --bundle <url>        # override the JS bundle URL
    python composer_probe.py --no-bundle           # skip the bundle download
"""
from __future__ import annotations
import os, sys, re, json, time, argparse

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos as oos  # reuse _request (guarded, read-only) + list_accounts

# The user's current dev bundle. Hash changes on each Composer deploy; override
# with --bundle if this 404s.
DEFAULT_BUNDLE = ("https://app.composer.trade/js/compiled/dev/"
                  "main.B7C472E05E6CF9BD355ECB2EFA1B18D7.js")

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# Regexes over the minified bundle.
# Catch every api/vN[...] path token, with OR without a leading slash/quote —
# the watchlist path appears as the bare literal "api/v1/watchlist".
_PATH_RE = re.compile(r"""(/?api/v\d[a-zA-Z0-9_./{}$:-]*)""")
_KW_RE = re.compile(r"""["'`]([^"'`]*(?:watchlist|watch|discover|popular|community|"""
                    r"""out-of-sample|out_of_sample|oos)[^"'`]*)["'`]""", re.I)
# Show how a path is assembled (host/base) by printing context around it.
_CTX_TERMS = ("api/v1/watchlist", "api/v1/discover", "best_of_community",
              "stable_performers", "recent_performers", "/discover")


def scan_bundle(url: str) -> None:
    print(f"\n=== JS BUNDLE SCAN: {url}", flush=True)
    try:
        r = requests.get(url, headers={"user-agent": UA}, timeout=60)
        print(f"  HTTP {r.status_code}  {len(r.content)} bytes", flush=True)
        r.raise_for_status()
    except Exception as e:
        print(f"  bundle fetch failed: {type(e).__name__}: {e}", flush=True)
        return
    js = r.text
    paths = sorted(set(_PATH_RE.findall(js)))
    print(f"\n  -- {len(paths)} distinct api/vN paths --", flush=True)
    for p in paths:
        print(f"    {p}", flush=True)
    print("\n  -- context around key terms (reveals host/base) --", flush=True)
    for term in _CTX_TERMS:
        i = js.find(term)
        if i >= 0:
            print(f"    [{term}] ...{js[max(0,i-70):i+len(term)+30]}...", flush=True)
        else:
            print(f"    [{term}] (not found)", flush=True)
    kws = sorted({s for s in _KW_RE.findall(js) if len(s) < 120})
    # keep the ones that look like paths/keys, drop prose
    kws = [s for s in kws if ("/" in s or "-" in s or "_" in s) and " " not in s.strip()]
    print(f"\n  -- {len(kws)} watch/discover/oos-ish string literals --", flush=True)
    for s in kws[:120]:
        print(f"    {s}", flush=True)


def probe(path: str) -> None:
    try:
        d = oos._get(path)
    except Exception as e:
        print(f"  [{path}] -> {type(e).__name__}: {str(e)[:120]}", flush=True)
        return
    if isinstance(d, dict):
        keys = list(d.keys())
        print(f"  [{path}] -> 200 dict keys={keys}", flush=True)
        # show one nested sample if a list of records is present
        for k, v in d.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                print(f"      {k}[0] keys={list(v[0].keys())}", flush=True)
                print(f"      {k}[0]={json.dumps(v[0])[:400]}", flush=True)
                break
    elif isinstance(d, list):
        print(f"  [{path}] -> 200 list len={len(d)}", flush=True)
        if d and isinstance(d[0], dict):
            print(f"      [0] keys={list(d[0].keys())}", flush=True)
            print(f"      [0]={json.dumps(d[0])[:400]}", flush=True)
    else:
        print(f"  [{path}] -> 200 {type(d).__name__}", flush=True)


_FORBIDDEN = ("/deploy/", "/trading/", "/dry-run", "/rebalance")


def raw_get(url: str) -> None:
    """GET an absolute URL with the API key. Read-only: refuses state-changing
    paths and only ever issues GET."""
    if any(f in url for f in _FORBIDDEN):
        print(f"  [{url}] -> refused (forbidden path)", flush=True)
        return
    try:
        r = requests.get(url, headers=oos._headers(), timeout=30)
        ct = r.headers.get("content-type", "")
        note = ""
        if r.ok and "json" in ct:
            d = r.json()
            if isinstance(d, dict):
                note = f"dict keys={list(d.keys())[:12]}"
                for k, v in d.items():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        note += f" | {k}[0] keys={list(v[0].keys())[:14]}"
                        break
            elif isinstance(d, list):
                note = f"list len={len(d)}"
                if d and isinstance(d[0], dict):
                    note += f" [0] keys={list(d[0].keys())[:14]}"
        else:
            note = (r.text or "")[:100].replace("\n", " ")
        print(f"  [{url}] -> {r.status_code} {note}", flush=True)
    except Exception as e:
        print(f"  [{url}] -> {type(e).__name__}: {str(e)[:120]}", flush=True)


def probe_v1() -> None:
    print("\n=== V1 ENDPOINT PROBES (both likely hosts) ===", flush=True)
    hosts = ["https://api.composer.trade", "https://stagehand.composer.trade"]
    paths = ["/api/v1/watchlist", "/api/v1/watchlist/",
             "/api/v1/discover", "/api/v1/discover/community",
             "/api/v1/discover/best_of_community",
             "/api/v1/discover/stable_performers",
             "/api/v1/discover/recent_performers"]
    for h in hosts:
        for p in paths:
            time.sleep(0.5)
            raw_get(h + p)


def probe_candidates() -> None:
    print("\n=== CANDIDATE ENDPOINT PROBES (authed GET) ===", flush=True)
    try:
        accts = oos.list_accounts()
    except Exception as e:
        print(f"  accounts/list failed ({type(e).__name__}: {e}) — key set? aborting probes",
              flush=True)
        return
    uuids = [a.get("account_uuid") for a in accts if a.get("account_uuid")]
    print(f"  accounts/list OK: {len(uuids)} account(s)", flush=True)

    # User/global-level candidates
    for p in [
        "/api/v0.1/watchlist",
        "/api/v0.1/watchlists",
        "/api/v0.1/portfolio/watchlist",
        "/api/v0.1/discover",
        "/api/v0.1/discover/symphonies",
        "/api/v0.1/symphonies/discover",
        "/api/v0.1/symphonies/popular",
        "/api/v0.1/discover/popular",
    ]:
        probe(p)

    # Account-scoped candidates
    for u in uuids[:1]:
        for p in [
            f"/api/v0.1/portfolio/accounts/{u}/watchlist",
            f"/api/v0.1/accounts/{u}/watchlist",
        ]:
            probe(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--no-bundle", action="store_true")
    a = ap.parse_args()
    if not a.no_bundle:
        scan_bundle(a.bundle)
    probe_v1()
    probe_candidates()
