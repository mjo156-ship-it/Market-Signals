#!/usr/bin/env python3
"""refresh_discover_seed.py — update the Discover seed from a fresh capture.

Composer's Discover feed is browser-session-gated, so the seed can't be
auto-fetched (the producer's auto-fetch always returns nothing). This turns
"hand-edit a 1000-entry JSON" into one command, and — crucially — feeds the
survival panel so attrition finally becomes measurable.

CAPTURE (once, in your logged-in browser):
  1. Open Composer Discover. DevTools → Network → filter Fetch/XHR.
  2. Find the request whose response is the list of symphonies (an array of
     objects with id / name / last_semantic_update_at). Right-click → Copy →
     Copy response. Save it to a file, e.g. discover_capture.json.

REFRESH:
    python scripts/refresh_discover_seed.py discover_capture.json
    # or pipe it:  pbpaste | python scripts/refresh_discover_seed.py -

It normalizes the capture (same shape-tolerant parser the producer uses), writes
data/composer_discover_seed.json, prints an added/removed diff, and stamps the
survival panel — so any strategies that dropped out of Discover are recorded as
delisted right away instead of waiting for the weekly run. A guard refuses a
capture that's suspiciously smaller than the current seed (likely a partial or
paginated grab) unless you pass --force.
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composer_oos_discover as d


def _read_raw(src):
    if src in (None, "-", "/dev/stdin"):
        if sys.stdin.isatty():
            sys.exit("no capture given. Pass a file "
                     "(refresh_discover_seed.py discover_capture.json) or pipe JSON via stdin.")
        return json.load(sys.stdin)
    with open(src) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Refresh the Discover seed from a browser capture.")
    ap.add_argument("capture", nargs="?", default="-",
                    help="path to the pasted Discover response JSON, or '-' for stdin")
    ap.add_argument("--force", action="store_true",
                    help="write even if the capture is much smaller than the current seed")
    ap.add_argument("--asof", default=date.today().isoformat(),
                    help="date to stamp the survival panel (default: today)")
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

    old = d._load_seed()                       # list of (sid, freeze, name)
    old_ids = {sid for sid, _, _ in old}
    added, removed = new_ids - old_ids, old_ids - new_ids
    print(f"capture: {len(items)} strategies · current seed: {len(old)}")
    print(f"  + {len(added)} new · − {len(removed)} gone · = {len(new_ids & old_ids)} unchanged")

    if old and len(new_ids) < 0.5 * len(old_ids) and not args.force:
        sys.exit(f"REFUSING: capture ({len(new_ids)}) is < 50% of the current seed "
                 f"({len(old_ids)}) — this looks partial (e.g. one page of Discover). "
                 f"Re-capture the full feed, or pass --force if the shrink is real.")

    seed_out = [{"id": sid, "name": name, "last_semantic_update_at": frz}
                for sid, frz, name in sorted(items, key=lambda t: (t[2] or "").lower())]
    os.makedirs(os.path.dirname(d.SEED) or ".", exist_ok=True)
    with open(d.SEED, "w") as f:
        json.dump(seed_out, f, indent=1)
    print(f"wrote {d.SEED}: {len(seed_out)} strategies")

    # Stamp the survival panel now so delistings register immediately (the weekly
    # producer run would otherwise be the first to notice).
    d._update_survival(items, args.asof)

    if removed:
        names = sorted(name for sid, frz, name in old if sid in removed)
        shown = ", ".join(names[:8]) + (" …" if len(names) > 8 else "")
        print(f"  delisted since last seed (now frozen in the survival panel): {shown}")


if __name__ == "__main__":
    main()
