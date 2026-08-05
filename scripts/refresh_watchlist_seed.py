#!/usr/bin/env python3
"""refresh_watchlist_seed.py — update the watched-symphony list from a capture.

The Watchlist OOS list is browser-session-gated (Composer's watchlist API), so it
can't be auto-fetched — it's the inline WATCH list in composer_oos_watchlist.py.
This lets you refresh it from a browser capture instead of hand-editing that list,
so symphonies you newly follow show up on the Watchlist OOS tab.

CAPTURE (once, in your logged-in browser):
  1. Open Composer → Watchlist. DevTools → Network → filter Fetch/XHR.
  2. Find the request whose response lists your watched symphonies (objects with
     id / name / last_semantic_update_at). Right-click → Copy → Copy response.
     Save it to a file, e.g. watchlist_capture.json.

REFRESH:
    python scripts/refresh_watchlist_seed.py watchlist_capture.json
    # or pipe:  pbpaste | python scripts/refresh_watchlist_seed.py -

It normalizes the capture (same shape-tolerant parser the producer uses), writes
data/composer_watchlist_seed.json (which the producer prefers over the inline
list), and prints an added/removed diff so you can see exactly what changed. A
guard refuses a capture < 50% of the current list (likely a partial grab) unless
you pass --force.
"""
from __future__ import annotations
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos_discover as d          # reuse the shape-tolerant _find_list/_norm
import composer_oos_watchlist as wl


def _read_raw(src):
    if src in (None, "-", "/dev/stdin"):
        if sys.stdin.isatty():
            sys.exit("no capture given. Pass a file "
                     "(refresh_watchlist_seed.py watchlist_capture.json) or pipe JSON via stdin.")
        return json.load(sys.stdin)
    with open(src) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Refresh the watchlist seed from a browser capture.")
    ap.add_argument("capture", nargs="?", default="-",
                    help="path to the pasted watchlist response JSON, or '-' for stdin")
    ap.add_argument("--force", action="store_true",
                    help="write even if the capture is much smaller than the current list")
    args = ap.parse_args()

    try:
        raw = _read_raw(args.capture)
    except SystemExit:
        raise
    except Exception as e:
        sys.exit(f"could not read capture: {e}")

    items = d._norm(raw if isinstance(raw, list) else d._find_list(raw))
    if not items:
        sys.exit("no symphonies with freeze dates found in the capture — is it the right "
                 "Network response? (each entry needs an id + last_semantic_update_at)")
    new_ids = {sid for sid, _, _ in items}

    # baseline = current seed if present, else the inline WATCH list
    cur = wl._load_wl_seed() or wl.WATCH
    cur_ids = {sid for sid, _, _ in cur}
    added, removed = new_ids - cur_ids, cur_ids - new_ids
    print(f"capture: {len(items)} watched · current: {len(cur)}")
    print(f"  + {len(added)} new · − {len(removed)} unfollowed · = {len(new_ids & cur_ids)} unchanged")

    if cur and len(new_ids) < 0.5 * len(cur_ids) and not args.force:
        sys.exit(f"REFUSING: capture ({len(new_ids)}) is < 50% of the current list "
                 f"({len(cur_ids)}) — this looks partial. Re-capture the full watchlist, "
                 f"or pass --force if the shrink is real.")

    seed = [{"id": sid, "name": name, "last_semantic_update_at": frz}
            for sid, frz, name in sorted(items, key=lambda t: (t[2] or "").lower())]
    os.makedirs(os.path.dirname(wl.WL_SEED) or ".", exist_ok=True)
    with open(wl.WL_SEED, "w") as f:
        json.dump(seed, f, indent=1)
    print(f"wrote {wl.WL_SEED}: {len(seed)} symphonies (producer will use this over the inline list)")

    if added:
        names = sorted(name for sid, frz, name in items if sid in added)
        shown = ", ".join(names[:12]) + (" …" if len(names) > 12 else "")
        print(f"  newly tracked: {shown}")


if __name__ == "__main__":
    main()
