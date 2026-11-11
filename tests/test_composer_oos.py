"""Offline checks for the OOS split helpers (no network). Run: python -m pytest tests/"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import composer_oos as c  # noqa: E402

VERS = [{"version_id": "v4", "created_at": "2026-09-04T10:00"}, {"version_id": "v3", "created_at": "2026-09-01T10:00"},
        {"version_id": "v2", "created_at": "2026-03-01T10:00"}, {"version_id": "v1", "created_at": "2025-06-10T10:00"}]
LOGIC = {"v4": {"id": "a", "name": "X renamed", "children": [{"ticker": "SPY", "weight": .5}]},
         "v3": {"id": "b", "name": "X", "children": [{"ticker": "SPY", "weight": .5}]},
         "v2": {"id": "c", "name": "X", "children": [{"ticker": "SPY", "weight": .4}]},
         "v1": {"id": "d", "name": "X", "children": [{"ticker": "QQQ", "weight": .4}]}}


def _api(vers, logic):
    def get(path, params=None):
        return vers if path.endswith("/versions") else logic[path.split("/versions/")[1].split("/")[0]]
    return get


def test_last_logic_change_ignores_renames_copies_and_resaves():
    assert c.last_logic_change_date("s", get=_api(VERS, LOGIC)) == "2026-09-01"      # the 09-04 save was a rename
    assert c.last_logic_change_date("s", get=_api(VERS[:1], LOGIC)) is None           # a copy has one version
    same = {k: LOGIC["v4"] for k in LOGIC}
    assert c.last_logic_change_date("s", get=_api(VERS, same)) is None                # re-saves only
    assert c.last_logic_change_date("s", get=_api(VERS, LOGIC), max_walk=2) is None   # change beyond the look-back


def test_effective_oos_date_moves_only_forward():
    assert c.effective_oos_date("s", "2025-06-10", lookup=lambda s: "2026-09-01") == ("2026-09-01", "version")
    assert c.effective_oos_date("s", "2026-10-01", lookup=lambda s: "2026-09-01") == ("2026-10-01", "seed")
    assert c.effective_oos_date("s", "2025-06-10", lookup=lambda s: None) == ("2025-06-10", "seed")


def test_volatility_is_the_spread_of_daily_returns_and_never_negative():
    falling = {f"2026-01-{d:02d}": 100 * (0.995 ** d) * (1.01 if d % 2 else 0.99) for d in range(1, 29)}
    v = c.annualized_vol_pct(falling)
    assert v is not None and v > 0                      # CAGR / Sharpe gave a negative "volatility" here
    assert c.annualized_vol_pct({"2026-01-01": 1.0}) is None


def test_new_own_symphonies_are_those_created_after_the_capture():
    own = [{"id": "old", "name": "In capture", "created_at": "2026-08-03T00:00", "is_draft": True},
           {"id": "skip", "name": "Older, never watched", "created_at": "2026-05-01T00:00", "is_draft": True},
           {"id": "new1", "name": "Copy of something", "created_at": "2026-08-20T00:00", "is_draft": True},
           {"id": "new2", "name": "Saved one", "created_at": "2026-10-12T00:00", "is_draft": False},
           {"id": "copy", "name": "Saved one (Buy Copy)", "created_at": "2026-11-03T00:00", "is_draft": False}]
    got = c.new_own_symphonies(["old", "someone-elses"], own=own)
    assert [r["id"] for r in got] == ["new1", "new2"]
    assert c.new_own_symphonies(["someone-elses"], own=own) == []     # no own symphony in the capture: no cutoff


def test_funding_gate_blocks_only_when_behind_the_market():
    days = [f"2026-{m:02d}-{d:02d}" for m in (5, 6, 7) for d in range(1, 29)]

    def curve(daily):
        out, v = {}, 100.0
        for i, d in enumerate(days):
            v *= 1 + daily(i)
            out[d] = v
        return out
    mkt = lambda i: -0.001 + 0.01 * (-1) ** i                    # market down ~8% with daily swings
    spy = curve(mkt)
    with_mkt = curve(lambda i: 0.5 * mkt(i) + 0.0003)             # beta 0.5, a little ahead of it, still down
    behind = curve(lambda i: 0.5 * mkt(i) - 0.002)                # beta 0.5, bleeding much more
    g = c.funding_gate(c._curve_stats(with_mkt)["cum_pct"], with_mkt, spy, fallback_beta=1.0)
    assert g["funding_block"] is False and g["funding_caution"] is True
    g = c.funding_gate(c._curve_stats(behind)["cum_pct"], behind, spy, fallback_beta=1.0)
    assert g["funding_block"] is True and g["funding_caution"] is False
    assert c.funding_gate(-3.0, with_mkt, {})["funding_block"] is True              # no SPY data: old rule
    assert c.funding_gate(2.0, with_mkt, spy)["funding_block"] is False
