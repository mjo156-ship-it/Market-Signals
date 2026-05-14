"""
Composer Dry-Run Trade Preview
==============================

Calls Composer's dry-run API to get the EXACT trades each symphony will execute
at next rebalance. Replaces the heuristic forecaster — this is Composer's own
evaluation, not a guess.

API endpoints used:
  POST /api/v0.1/dry-run                          (all symphonies, all accounts)
  POST /api/v0.1/dry-run/trade-preview/{symph_id} (single symphony)

Response shape (per symphony, per account):
    {
        "rebalanced": true,
        "next_rebalance_after": "...",
        "next_rebalance_date": "...",
        "symphony_value": 250954.0,
        "symphony_name": "Top of Tech Simplified (20% UVXY)",
        "recommended_trades": [
            {
                "ticker": "UVXY",
                "notional": 50190.8,    # dollar amount to trade (signed)
                "quantity": 1400.0,     # shares
                "prev_value": 45000.0,
                "prev_weight": 0.18,
                "next_weight": 0.20,
            },
            ...
        ],
        "queued_cash_change": 0.0,
    }

The forecaster's job is just:
  1. Call /api/v0.1/dry-run with the user's account UUIDs
  2. Pretty-format the output for the email
  3. Aggregate across symphonies for portfolio-level lean

Rate limit: 1 req/sec (Composer default). The single dry-run call covers all
symphonies in all accounts — one request total per refresh.

USAGE
-----
    from composer_dry_run import fetch_dry_run_preview, format_dry_run_for_email

    preview = fetch_dry_run_preview(account_uuids=[...])
    text = format_dry_run_for_email(preview)
"""

import os
import time
from collections import defaultdict, Counter
import requests as req_lib


COMPOSER_BASE = "https://api.composer.trade/api/v0.1"
COMPOSER_KEY_ID = os.environ.get('COMPOSER_KEY_ID', '')
COMPOSER_KEY_SECRET = os.environ.get('COMPOSER_KEY_SECRET', '')


# ════════════════════════════════════════════════════════════════════
# AUTHENTICATED API CALLS
# ════════════════════════════════════════════════════════════════════
def _composer_post(path, body, timeout=30):
    """Authenticated POST to Composer API."""
    if not COMPOSER_KEY_ID or not COMPOSER_KEY_SECRET:
        return None
    try:
        r = req_lib.post(
            f"{COMPOSER_BASE}{path}",
            json=body,
            headers={
                "x-api-key-id": COMPOSER_KEY_ID,
                "authorization": f"Bearer {COMPOSER_KEY_SECRET}",
                "content-type": "application/json",
                "accept": "application/json",
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except req_lib.HTTPError as e:
        body_txt = ''
        try:
            body_txt = e.response.text[:300]
        except Exception:
            pass
        print(f"  Composer POST error ({path}): {e} | body: {body_txt}")
        return None
    except Exception as e:
        print(f"  Composer POST error ({path}): {e}")
        return None


_DEBUG = os.environ.get('COMPOSER_DEBUG') == '1'


def _composer_get(path, timeout=15):
    """Authenticated GET to Composer API."""
    if not COMPOSER_KEY_ID or not COMPOSER_KEY_SECRET:
        return None
    try:
        r = req_lib.get(
            f"{COMPOSER_BASE}{path}",
            headers={
                "x-api-key-id": COMPOSER_KEY_ID,
                "authorization": f"Bearer {COMPOSER_KEY_SECRET}",
            },
            timeout=timeout,
        )
        if _DEBUG:
            print(f"[Composer DEBUG] GET {path} → status={r.status_code} "
                  f"body[:300]={r.text[:300]!r}")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  Composer GET error ({path}): {e}")
        return None


def list_account_uuids():
    """Return all broker_account_uuids associated with the API key, or None on failure.

    Composer's /accounts/list returns {accounts: [{account_uuid, account_type, ...}]}.
    """
    resp = _composer_get('/accounts/list')
    if _DEBUG:
        if resp is None:
            print("[Composer DEBUG] /accounts/list returned None")
        else:
            keys = sorted(resp.keys()) if isinstance(resp, dict) else f'(non-dict: {type(resp).__name__})'
            n_accts = len(resp.get('accounts', [])) if isinstance(resp, dict) else 0
            sample_keys = sorted(resp['accounts'][0].keys()) if (isinstance(resp, dict) and resp.get('accounts')) else None
            print(f"[Composer DEBUG] /accounts/list resp keys={keys} accounts_len={n_accts} sample_acct_keys={sample_keys}")
    if not resp or 'accounts' not in resp:
        return None
    return [a.get('account_uuid') for a in resp['accounts'] if a.get('account_uuid')]


# ════════════════════════════════════════════════════════════════════
# DRY-RUN PREVIEW
# ════════════════════════════════════════════════════════════════════
def fetch_dry_run_preview(account_uuids=None):
    """
    Fetch dry-run rebalance preview for all symphonies in given accounts.

    Args:
        account_uuids: list of broker account UUIDs to evaluate. If None, the
                       accounts list is auto-discovered via /accounts/list.

    Returns:
        list of dicts, one per account, each with shape:
            {
                'broker_account_uuid': str,
                'account_type': str,
                'account_name': str,
                'broker': str,
                'dry_run_result': {
                    symphony_id: {
                        'rebalanced': bool,
                        'symphony_name': str,
                        'symphony_value': float,
                        'recommended_trades': [...],
                        'queued_cash_change': float,
                        'next_rebalance_date': str,
                    },
                    ...
                },
                'dry_run_missing_symphonies': {symphony_id: error_msg},
                'dry_run_total_symphonies': int,
            }
        Returns None on auth failure or API error, or [] if no accounts found.
    """
    # Composer's /dry-run does NOT auto-evaluate "all accounts" when the body
    # omits account_uuids — it returns []. So when caller passes None, discover
    # them explicitly via /accounts/list first.
    if not account_uuids:
        discovered = list_account_uuids()
        if discovered is None:
            return None
        if not discovered:
            return []
        account_uuids = discovered
    body = {
        'send_segment_event': False,
        'account_uuids': list(account_uuids),
    }
    return _composer_post('/dry-run', body)


def fetch_single_symphony_preview(symphony_id, account_uuid, amount=None):
    """
    Fetch dry-run preview for a single symphony.
    Useful for spot-checking forecaster behavior or fresh-deploy analysis.

    Args:
        symphony_id: symphony UUID
        account_uuid: broker account UUID
        amount: optional fresh-deploy amount in USD (for what-if sizing)

    Returns:
        dict with shape from API docs (see module header), or None on failure.
    """
    body = {'broker_account_uuid': account_uuid}
    if amount is not None:
        body['amount'] = float(amount)
    return _composer_post(f'/dry-run/trade-preview/{symphony_id}', body)


# ════════════════════════════════════════════════════════════════════
# PARSING / SUMMARIZATION
# ════════════════════════════════════════════════════════════════════
def parse_dry_run_response(dry_run_result):
    """
    Flatten the per-account, per-symphony dry-run response into a normalized list:

        [
            {
                'account_name': 'Roth IRA',
                'account_uuid': '...',
                'symphony_id': '...',
                'symphony_name': 'Opus 4.6 67/14/.55',
                'symphony_value': 60920.0,
                'will_rebalance': True,
                'next_rebalance_date': '2026-05-08',
                'trades': [
                    {'ticker': 'UPRO', 'notional': +12184.0, 'quantity': +135,
                     'prev_weight': 0.50, 'next_weight': 0.70},
                    ...
                ],
                'cash_change': 0.0,
            },
            ...
        ]
    """
    out = []
    if not dry_run_result:
        return out

    for acct_block in dry_run_result:
        acct_name = acct_block.get('account_name') or acct_block.get('account_type') or 'Account'
        acct_uuid = acct_block.get('broker_account_uuid', '')
        for sym_id, sym in (acct_block.get('dry_run_result') or {}).items():
            trades = sym.get('recommended_trades') or []
            out.append({
                'account_name': acct_name,
                'account_uuid': acct_uuid,
                'symphony_id': sym_id,
                'symphony_name': sym.get('symphony_name', sym_id),
                'symphony_value': sym.get('symphony_value', 0),
                'will_rebalance': sym.get('rebalanced', False),
                'next_rebalance_date': sym.get('next_rebalance_date', ''),
                'trades': trades,
                'cash_change': sym.get('queued_cash_change', 0),
            })
    return out


def aggregate_target_holdings(parsed_rotations):
    """
    Sum the next_weight × symphony_value across symphonies to get the target
    dollar exposure per ticker — i.e. what your portfolio is rotating INTO.

    Returns:
        list of (ticker, target_dollars, symphony_count) tuples,
        sorted by target_dollars descending.
    """
    target = defaultdict(float)
    counts = Counter()

    for r in parsed_rotations:
        sv = r.get('symphony_value', 0)
        for t in r.get('trades', []):
            ticker = t.get('ticker', '')
            next_w = t.get('next_weight', 0)
            target[ticker] += sv * next_w
            counts[ticker] += 1

    rows = [(ticker, dollars, counts[ticker])
            for ticker, dollars in target.items()
            if dollars > 0.01]
    rows.sort(key=lambda x: -x[1])
    return rows


def aggregate_net_trade_flow(parsed_rotations):
    """
    Sum signed notional changes across all symphonies → net dollar flow per
    ticker. Positive = net buy across portfolio. Negative = net sell.

    This is the most important number for "what is my portfolio trading into?"

    Returns:
        list of (ticker, net_notional, n_symphonies_buying, n_symphonies_selling)
        sorted by absolute net_notional descending.
    """
    flow = defaultdict(float)
    buys = Counter()
    sells = Counter()

    for r in parsed_rotations:
        for t in r.get('trades', []):
            ticker = t.get('ticker', '')
            n = t.get('notional', 0) or 0
            flow[ticker] += n
            if n > 0:
                buys[ticker] += 1
            elif n < 0:
                sells[ticker] += 1

    rows = [(ticker, flow[ticker], buys[ticker], sells[ticker])
            for ticker in flow]
    rows.sort(key=lambda x: -abs(x[1]))
    return rows


# ════════════════════════════════════════════════════════════════════
# RISK CLASSIFICATION & CONCENTRATION WARNINGS
# ════════════════════════════════════════════════════════════════════
# Tickers grouped by risk class. Each class has its own threshold rules.
# Tickers can appear in multiple classes (e.g. UVXY is both leveraged AND vol).

VOL_PRODUCTS = {
    # Long volatility (decay-prone)
    'UVXY', 'UVIX', 'VXX', 'VIXY', 'TVIX',
    # Short volatility (catastrophic-tail-risk)
    'SVXY', 'SVIX',
    # Mid-curve vol
    'VIXM',
}

LEV_3X = {
    # Equity index 3x
    'TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'SPXL', 'SPXS',
    'TNA', 'TZA', 'UDOW', 'SDOW',
    # Sector 3x
    'SOXL', 'SOXS', 'TECL', 'TECS', 'FAS', 'FAZ',
    'CURE', 'NAIL', 'LABU', 'LABD', 'DRN', 'DRV',
    'DPST', 'DUSL', 'TPOR', 'WANT', 'WEBL', 'PILL',
    'DFEN', 'BNKU', 'RETL',
    # Country 3x
    'EDC', 'EDZ', 'YINN', 'YANG', 'KORU', 'EURL', 'EURZ',
    'INDL', 'BRZU', 'MEXX',
    # Commodity / theme 3x
    'BOIL', 'KOLD', 'UCO', 'SCO', 'GUSH', 'DRIP',
    'JNUG', 'JDST', 'NUGT', 'DUST', 'ERX', 'ERY',
    'TMF', 'TMV', 'TYO', 'TYD',
    # Single-stock 3x
    'TSLL', 'TSLT', 'NVDL', 'NVD',
    # Misc 3x
    'HIBL', 'HIBS', 'FNGU', 'FNGD', 'CWEB',
}

LEV_2X = {
    # Equity 2x
    'QLD', 'QID', 'SSO', 'SDS', 'DDM', 'DXD',
    'UWM', 'TWM',
    # Sector 2x
    'USD', 'SSG', 'AGQ', 'ZSL', 'BIB', 'BIS',
    'UYG', 'SKF', 'UCC', 'SCC',
    # Misc 2x
    'BITX', 'ETHU',
}

# Map ticker -> set of risk classes it belongs to
def _risk_classes(ticker):
    classes = set()
    if ticker in VOL_PRODUCTS:
        classes.add('vol')
    if ticker in LEV_3X:
        classes.add('lev3x')
    if ticker in LEV_2X:
        classes.add('lev2x')
    return classes


# Thresholds — % of total portfolio value
# (warn_pct, alert_pct) where alert is the louder bolded warning
THRESHOLDS = {
    'vol':   (5.0, 10.0),   # vol products: warn at 5%, alert at 10%
    'lev3x': (20.0, 30.0),  # 3x leveraged: warn at 20%, alert at 30%
    'lev2x': (35.0, 50.0),  # 2x leveraged: warn at 35%, alert at 50%
    'concentration': (25.0, 35.0),  # any single ticker: warn at 25%, alert at 35%
}

# Within-class aggregate thresholds — sum across all tickers in class
# Catches "death by a thousand 3x positions" when no single one is huge
CLASS_AGGREGATE_THRESHOLDS = {
    'vol':   (8.0, 15.0),
    'lev3x': (40.0, 60.0),
    'lev2x': (50.0, 70.0),
}


def compute_target_exposure(parsed_rotations):
    """
    Compute the post-rebalance target dollar exposure per ticker.

    For each symphony rotating: sum (next_weight * symphony_value) per ticker.
    For each symphony NOT rotating: we don't have prev_weights for non-trade
    holdings in the dry-run response, so this estimates ROTATIONAL exposure
    only — i.e. what the rebalancing symphonies will own after their trades.

    Returns dict: {ticker: target_dollars}
    """
    target = defaultdict(float)
    for r in parsed_rotations:
        sv = r.get('symphony_value', 0)
        for t in r.get('trades', []):
            ticker = t.get('ticker', '')
            next_w = t.get('next_weight', 0) or 0
            target[ticker] += sv * next_w
    return dict(target)


def compute_concentration_warnings(parsed_rotations, total_portfolio_value=None):
    """
    Detect concentration exceeding risk thresholds.

    Args:
        parsed_rotations: output of parse_dry_run_response()
        total_portfolio_value: total $ across all accounts (used as denominator).
            If None, falls back to sum of symphony values from the dry-run
            (which under-counts cash and non-rebalancing holdings).

    Returns:
        dict with shape:
            {
                'denominator': float,             # used for % calculations
                'denominator_source': str,        # 'portfolio' or 'rotation_sum'
                'per_ticker': [
                    {'ticker': str, 'target_dollars': float, 'pct': float,
                     'class': str, 'severity': 'warn'|'alert',
                     'threshold_pct': float, 'symphonies': [str, ...]},
                    ...
                ],
                'per_class': [
                    {'class': str, 'total_dollars': float, 'pct': float,
                     'severity': 'warn'|'alert', 'threshold_pct': float,
                     'tickers': [(ticker, dollars), ...]},
                    ...
                ],
            }
    """
    target = compute_target_exposure(parsed_rotations)

    # Denominator
    if total_portfolio_value and total_portfolio_value > 0:
        denom = float(total_portfolio_value)
        denom_src = 'portfolio'
    else:
        denom = sum(r.get('symphony_value', 0) for r in parsed_rotations)
        denom_src = 'rotation_sum'

    if denom <= 0:
        return {'denominator': 0, 'denominator_source': denom_src,
                'per_ticker': [], 'per_class': []}

    # Track which symphonies hold each ticker
    symphonies_per_ticker = defaultdict(set)
    for r in parsed_rotations:
        for t in r.get('trades', []):
            ticker = t.get('ticker', '')
            next_w = t.get('next_weight', 0) or 0
            if next_w > 0:
                symphonies_per_ticker[ticker].add(r['symphony_name'])

    # Per-ticker warnings — consolidate so each ticker appears ONCE with all
    # classes it breaches listed
    per_ticker_raw = []
    for ticker, dollars in target.items():
        if dollars <= 0:
            continue
        pct = dollars / denom * 100

        breaches = []  # list of (class, severity, threshold)
        for cls in _risk_classes(ticker):
            warn, alert = THRESHOLDS[cls]
            if pct >= alert:
                breaches.append((cls, 'alert', alert))
            elif pct >= warn:
                breaches.append((cls, 'warn', warn))

        # Single-ticker concentration
        warn, alert = THRESHOLDS['concentration']
        if pct >= alert:
            breaches.append(('concentration', 'alert', alert))
        elif pct >= warn and not _risk_classes(ticker):
            # Only flag concentration if no risk-class warning already fires
            # (avoid double-flagging UVXY at 27% as both vol-alert AND
            # concentration-warn)
            breaches.append(('concentration', 'warn', warn))

        if breaches:
            # Most severe wins for sort key
            severity = 'alert' if any(b[1] == 'alert' for b in breaches) else 'warn'
            per_ticker_raw.append({
                'ticker': ticker,
                'target_dollars': dollars,
                'pct': pct,
                'breaches': breaches,
                'severity': severity,
                'symphonies': sorted(symphonies_per_ticker[ticker]),
            })

    # Sort: alerts first, then by % descending
    per_ticker_raw.sort(key=lambda x: (0 if x['severity'] == 'alert' else 1, -x['pct']))

    # Flatten breaches into the legacy per-ticker format for back-compat,
    # but with each ticker appearing once (highest-severity breach reported,
    # other breaches in 'also_breaches' list)
    per_ticker = []
    for w in per_ticker_raw:
        # Pick the most severe breach as primary, alert > warn
        breaches_sorted = sorted(w['breaches'],
                                  key=lambda b: (0 if b[1] == 'alert' else 1,
                                                  -b[2]))
        primary = breaches_sorted[0]
        per_ticker.append({
            'ticker': w['ticker'],
            'target_dollars': w['target_dollars'],
            'pct': w['pct'],
            'class': primary[0],
            'severity': primary[1],
            'threshold_pct': primary[2],
            'symphonies': w['symphonies'],
            'all_breaches': [{'class': b[0], 'severity': b[1],
                              'threshold_pct': b[2]} for b in breaches_sorted],
        })

    # Per-class aggregates
    class_totals = defaultdict(lambda: {'dollars': 0, 'tickers': []})
    for ticker, dollars in target.items():
        for cls in _risk_classes(ticker):
            class_totals[cls]['dollars'] += dollars
            class_totals[cls]['tickers'].append((ticker, dollars))

    per_class = []
    for cls, info in class_totals.items():
        pct = info['dollars'] / denom * 100
        warn, alert = CLASS_AGGREGATE_THRESHOLDS[cls]
        sev = 'alert' if pct >= alert else ('warn' if pct >= warn else None)
        if sev:
            per_class.append({
                'class': cls,
                'total_dollars': info['dollars'],
                'pct': pct,
                'severity': sev,
                'threshold_pct': alert if sev == 'alert' else warn,
                'tickers': sorted(info['tickers'], key=lambda x: -x[1]),
            })

    per_class.sort(key=lambda x: (0 if x['severity'] == 'alert' else 1, -x['pct']))

    return {
        'denominator': denom,
        'denominator_source': denom_src,
        'per_ticker': per_ticker,
        'per_class': per_class,
    }


def format_concentration_warnings(warnings):
    """
    Format concentration warnings as plain text. Returns empty string if no
    warnings — only renders a section when something actually fires.

    Renders ALERTS first (in bold via uppercase + asterisks for plain-text email),
    then WARNs.
    """
    per_ticker = warnings.get('per_ticker', [])
    per_class = warnings.get('per_class', [])

    if not per_ticker and not per_class:
        return ''

    denom = warnings.get('denominator', 0)
    denom_src = warnings.get('denominator_source', 'unknown')
    denom_note = (' (vs total portfolio)' if denom_src == 'portfolio'
                  else ' (vs rebalancing-symphony sum — actual % may be lower)')

    out = "\n" + "=" * 70 + "\n"
    out += "*** RISK CONCENTRATION WARNINGS ***\n"
    out += "=" * 70 + "\n"
    out += f"Denominator: ${denom:,.0f}{denom_note}\n\n"

    # Class names for display
    CLASS_LABELS = {
        'vol': 'Volatility products',
        'lev3x': '3x leveraged ETFs',
        'lev2x': '2x leveraged ETFs',
        'concentration': 'Single-ticker concentration',
    }

    # ALERTS — single-ticker
    alerts_t = [w for w in per_ticker if w['severity'] == 'alert']
    if alerts_t:
        out += "*** ALERT — TICKER EXPOSURE ABOVE HARD THRESHOLD ***\n"
        out += "-" * 70 + "\n"
        for w in alerts_t:
            # Build "[reason1, reason2]" string for all breaches
            breach_strs = []
            for b in w.get('all_breaches', [{'class': w['class'],
                                              'severity': w['severity'],
                                              'threshold_pct': w['threshold_pct']}]):
                lab = CLASS_LABELS.get(b['class'], b['class'])
                breach_strs.append(f"{lab} >{b['threshold_pct']:.0f}%")
            reasons = ' AND '.join(breach_strs)

            sym_list = ', '.join(w['symphonies'][:3])
            if len(w['symphonies']) > 3:
                sym_list += f" +{len(w['symphonies'])-3} more"
            out += (f"  {w['ticker']:<6}  ${w['target_dollars']:>11,.0f}  "
                    f"{w['pct']:>5.1f}% of portfolio\n")
            out += f"          [{reasons}]\n"
            out += f"          via: {sym_list}\n"
        out += "\n"

    # ALERTS — class aggregate
    alerts_c = [w for w in per_class if w['severity'] == 'alert']
    if alerts_c:
        out += "*** ALERT — CLASS AGGREGATE ABOVE HARD THRESHOLD ***\n"
        out += "-" * 70 + "\n"
        for w in alerts_c:
            label = CLASS_LABELS.get(w['class'], w['class'])
            ticker_summary = ', '.join(f"{t} ${d:,.0f}"
                                        for t, d in w['tickers'][:5])
            if len(w['tickers']) > 5:
                ticker_summary += f" +{len(w['tickers'])-5} more"
            out += (f"  {label}: ${w['total_dollars']:,.0f} = {w['pct']:.1f}% "
                    f"[threshold {w['threshold_pct']:.0f}%]\n")
            out += f"          tickers: {ticker_summary}\n"
        out += "\n"

    # WARNs — single-ticker
    warns_t = [w for w in per_ticker if w['severity'] == 'warn']
    if warns_t:
        out += "WARN — ticker exposure approaching threshold:\n"
        for w in warns_t:
            breach_strs = []
            for b in w.get('all_breaches', [{'class': w['class'],
                                              'severity': w['severity'],
                                              'threshold_pct': w['threshold_pct']}]):
                lab = CLASS_LABELS.get(b['class'], b['class'])
                breach_strs.append(f"{lab} >{b['threshold_pct']:.0f}%")
            reasons = ' AND '.join(breach_strs)
            out += (f"  {w['ticker']:<6}  ${w['target_dollars']:>11,.0f}  "
                    f"{w['pct']:>5.1f}%  [{reasons}]\n")
        out += "\n"

    # WARNs — class aggregate
    warns_c = [w for w in per_class if w['severity'] == 'warn']
    if warns_c:
        out += "WARN — class aggregate approaching threshold:\n"
        for w in warns_c:
            label = CLASS_LABELS.get(w['class'], w['class'])
            out += (f"  {label}: ${w['total_dollars']:,.0f} = {w['pct']:.1f}% "
                    f"[threshold {w['threshold_pct']:.0f}%]\n")
        out += "\n"

    return out


# ════════════════════════════════════════════════════════════════════
# EMAIL FORMATTING
# ════════════════════════════════════════════════════════════════════
def format_dry_run_for_email(parsed_rotations, max_trades_shown=4,
                              total_portfolio_value=None):
    """
    Format the dry-run preview as plain-text for inclusion in the signal email.

    Args:
        parsed_rotations: output of parse_dry_run_response()
        max_trades_shown: cap trade rows per symphony (excess shown as "+N more")
        total_portfolio_value: total $ across all accounts. If provided, used as
            denominator for concentration % calculations. If None, falls back to
            sum of rebalancing symphony values (under-counts cash).

    Returns a string.
    """
    if not parsed_rotations:
        return ("\n" + "=" * 70 + "\n"
                "COMPOSER NEXT-REBALANCE PREVIEW\n"
                "=" * 70 + "\n"
                "Dry-run unavailable (no API key, no rebalances pending, "
                "or API error).\n")

    out = "\n" + "=" * 70 + "\n"
    out += "🧭 COMPOSER NEXT-REBALANCE PREVIEW (dry-run from API)\n"
    out += "=" * 70 + "\n"

    # Group by account
    by_acct = defaultdict(list)
    for r in parsed_rotations:
        by_acct[r['account_name']].append(r)

    will_rebalance = [r for r in parsed_rotations if r['will_rebalance']]
    no_rebalance = [r for r in parsed_rotations if not r['will_rebalance']]

    out += (f"Symphonies evaluated: {len(parsed_rotations)} | "
            f"Will rebalance: {len(will_rebalance)} | "
            f"Hold: {len(no_rebalance)}\n\n")

    for acct_name in sorted(by_acct.keys()):
        rotations = by_acct[acct_name]
        rebalancing_in_acct = [r for r in rotations if r['will_rebalance']]
        if not rebalancing_in_acct:
            continue
        out += f"\n── {acct_name} ──\n"

        for r in sorted(rebalancing_in_acct, key=lambda x: -x['symphony_value']):
            sname = r['symphony_name'][:40]
            sv = r['symphony_value']
            out += f"\n  {sname}  (${sv:,.0f})\n"
            trades = sorted(r['trades'], key=lambda t: -abs(t.get('notional', 0)))
            shown = trades[:max_trades_shown]
            for t in shown:
                ticker = t.get('ticker', '?')
                notional = t.get('notional', 0)
                prev_w = (t.get('prev_weight') or 0) * 100
                next_w = (t.get('next_weight') or 0) * 100
                arrow = '→'
                if notional >= 0:
                    notional_str = f"+${notional:,.0f}"
                else:
                    notional_str = f"-${abs(notional):,.0f}"
                out += (f"    {ticker:<6} {prev_w:>5.1f}% {arrow} {next_w:>5.1f}%  "
                        f"({notional_str})\n")
            if len(trades) > max_trades_shown:
                out += f"    ... +{len(trades) - max_trades_shown} more trades\n"

    # Holds — just list, no detail
    if no_rebalance:
        out += "\n── Symphonies holding (no rebalance) ──\n"
        for r in sorted(no_rebalance, key=lambda x: -x['symphony_value'])[:10]:
            out += f"  {r['symphony_name'][:40]:<40} ${r['symphony_value']:>10,.0f}\n"
        if len(no_rebalance) > 10:
            out += f"  ... +{len(no_rebalance) - 10} more\n"

    # Concentration warnings — show prominently BEFORE the net flow table
    # so they catch the eye before the user reads the summary numbers
    warnings = compute_concentration_warnings(parsed_rotations, total_portfolio_value)
    out += format_concentration_warnings(warnings)

    # Net portfolio flow
    flow = aggregate_net_trade_flow(parsed_rotations)
    if flow:
        out += "\n── PORTFOLIO NET TRADE FLOW (across all symphonies) ──\n"
        out += f"  {'Ticker':<8} {'Net Notional':>14} {'Buys':>6} {'Sells':>6}\n"
        for ticker, net, nb, ns in flow[:12]:
            if net >= 0:
                net_str = f"+${net:,.0f}"
            else:
                net_str = f"-${abs(net):,.0f}"
            out += f"  {ticker:<8} {net_str:>15} {nb:>6} {ns:>6}\n"

    out += "\n  Net positive = portfolio rotating INTO. Net negative = rotating OUT OF.\n"
    return out


# ════════════════════════════════════════════════════════════════════
# SELF-TEST
# ════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    if not COMPOSER_KEY_ID:
        print("Set COMPOSER_KEY_ID and COMPOSER_KEY_SECRET to test.")
        raise SystemExit(0)

    print("Fetching dry-run preview for all accounts...")
    raw = fetch_dry_run_preview()
    if raw is None:
        print("API returned None — check credentials and connectivity.")
        raise SystemExit(1)

    print(f"Got response for {len(raw)} account(s).")
    parsed = parse_dry_run_response(raw)
    print(f"Parsed {len(parsed)} symphonies total.\n")
    print(format_dry_run_for_email(parsed))
