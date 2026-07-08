#!/usr/bin/env python3
"""
Ocean Grove STR Opportunity Report — v1.0
=========================================
Weekly report of properties FOR SALE in Ocean Grove, NJ with an assessment of
their short-term-rental (STR) potential: gross rental earnings, operating
income, and operating income NET OF TAX after applying a 100% bonus-depreciation
benefit in the purchase year (basis split 40% land / 60% building).

The report flags every property that can BREAK EVEN on an annual operating
basis, and grades each one on whether that break-even is *sustainable* (recurs
every year) or *purchase-year only* (carried by the one-time depreciation
shield).

--------------------------------------------------------------------------------
WHY A REAL-ESTATE REPORT LIVES IN A MARKET-SIGNALS REPO
--------------------------------------------------------------------------------
It is a self-contained analytics generator that follows the same house style as
`snapshot_generator.py` / `signal_research.py`: pure-Python, config at the top,
writes a JSON data contract + a rendered Markdown report into `data/`, archives
a dated copy into `history/`, and is refreshed on a schedule by a GitHub Action
(`.github/workflows/ocean_grove_str.yml`, weekly on Mondays).

--------------------------------------------------------------------------------
DATA SOURCE (pluggable)
--------------------------------------------------------------------------------
`fetch_listings()` resolves the for-sale set in this priority order:

  1. $OG_LISTINGS_JSON  — path to a JSON file of real listings you drop in
                          (e.g. exported from an MLS/portal). Same schema as
                          SAMPLE_LISTINGS below.
  2. $RENTCAST_API_KEY  — pull live Ocean Grove for-sale listings from RentCast.
                          (Wired as a stub; enable by filling in _fetch_rentcast.)
  3. SAMPLE_LISTINGS    — a curated, clearly-labelled *representative* set of
                          Ocean Grove listings used when no live source is
                          configured, so the report runs out-of-the-box. These
                          are illustrative, NOT live MLS rows.

The financial engine is identical regardless of source, so wiring a real feed
later changes nothing downstream.

--------------------------------------------------------------------------------
STR REVENUE (AirDNA-calibratable)
--------------------------------------------------------------------------------
Each listing's gross STR revenue is resolved most-specific-first:
  1. listing `airdna_projected_revenue`  — address-level AirDNA Rentalizer.
  2. AirDNA market data by bedroom count  — from data/ocean_grove/airdna_market.json
     ($OG_AIRDNA_JSON), populated from an AirDNA MarketMinder subscription. See
     airdna_market.example.json for the template.
  3. Built-in seasonal model              — used when no AirDNA figure applies.
The chosen basis is recorded per listing (`str_revenue.basis`) and summarized in
`meta.revenue_source`.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  python ocean_grove_str.py                    # write JSON + MD into data/ocean_grove
  python ocean_grove_str.py --output-dir path  # custom output dir
  python ocean_grove_str.py --asof 2026-07-06  # override the report "week of" date
  python ocean_grove_str.py --no-archive       # skip writing history/<date>.json

All modelling assumptions below are overridable via environment variables (see
CONFIG); the resolved values are echoed into the report's `assumptions` block so
every number is reproducible and auditable.
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ==============================================================================
# CONFIG — every assumption is overridable via an environment variable so the
# weekly workflow (or a local run) can re-price the whole market without a code
# change. Defaults reflect a high-earner buyer using the STR "material
# participation" treatment (losses non-passive, usable against ordinary income).
# ==============================================================================

def _envf(name, default):
    """Read a float from the environment, falling back to `default`."""
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default

def _envi(name, default):
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default

# --- Depreciation basis split (per the report spec: 40% land / 60% building) ---
LAND_ALLOCATION      = _envf("OG_LAND_ALLOCATION", 0.40)   # non-depreciable
BUILDING_ALLOCATION  = _envf("OG_BUILDING_ALLOCATION", 0.60)  # depreciable
BONUS_DEPR_RATE      = _envf("OG_BONUS_DEPR_RATE", 1.00)   # 100% bonus, year 1

# --- Tax ---
# Blended federal + NJ marginal rate for a high earner. STR losses are assumed
# NON-PASSIVE (avg-stay <=7 days + material participation, i.e. the "STR
# loophole"), so the year-1 depreciation loss offsets ordinary income.
MARGINAL_TAX_RATE    = _envf("OG_MARGINAL_TAX_RATE", 0.37)

# --- Financing (drives the primary "operating break-even" definition) ---
DOWN_PAYMENT_PCT     = _envf("OG_DOWN_PAYMENT_PCT", 0.25)
MORTGAGE_RATE        = _envf("OG_MORTGAGE_RATE", 0.065)    # 30-yr fixed APR
MORTGAGE_TERM_YEARS  = _envi("OG_MORTGAGE_TERM_YEARS", 30)
CLOSING_COST_PCT     = _envf("OG_CLOSING_COST_PCT", 0.03)  # % of price, cash to close

# --- STR operating cost assumptions ---
MGMT_FEE_PCT         = _envf("OG_MGMT_FEE_PCT", 0.15)      # PM / co-host, % of gross
PLATFORM_FEE_PCT     = _envf("OG_PLATFORM_FEE_PCT", 0.03)  # Airbnb/Vrbo host fee
CLEANING_NET_PCT     = _envf("OG_CLEANING_NET_PCT", 0.04)  # cleaning not recovered from guest
SUPPLIES_PCT         = _envf("OG_SUPPLIES_PCT", 0.03)      # consumables / restock
CAPEX_RESERVE_PCT    = _envf("OG_CAPEX_RESERVE_PCT", 0.05) # reserve for turnover wear
INSURANCE_PCT        = _envf("OG_INSURANCE_PCT", 0.006)    # coastal STR policy, % of price
MAINT_PCT            = _envf("OG_MAINT_PCT", 0.010)        # repairs & maintenance, % of price
UTILITIES_ANNUAL     = _envf("OG_UTILITIES_ANNUAL", 4800)  # elec/gas/water/internet/streaming
LICENSE_ANNUAL       = _envf("OG_LICENSE_ANNUAL", 750)     # Neptune Twp STR license + bond amort.

# --- "Near break-even" band: how close (in $/yr) counts as almost-there ---
NEAR_BREAKEVEN_BAND  = _envf("OG_NEAR_BREAKEVEN_BAND", 7500)

# --- Seasonal STR revenue model (Ocean Grove is an intensely seasonal shore
# market: Sat-Sat weekly summer bookings, quiet winters). Peak weekly rate is
# looked up by bedroom count, then scaled per season. A listing may override
# with an explicit `str_peak_weekly` field. ---
PEAK_WEEKLY_BY_BED = {   # $/week at peak-summer pricing, by bedroom count
    0: 2400,  # studio
    1: 2800,
    2: 3800,
    3: 5000,
    4: 6800,
    5: 8800,  # 5+ bedrooms
}

# weeks sum to 52; rate_mult scales the peak weekly rate; occ is effective
# booked-share of that season.
SEASONS = {
    "peak":     {"weeks": 11, "rate_mult": 1.00, "occ": 0.90},  # ~mid-Jun..Labor Day
    "shoulder": {"weeks": 8,  "rate_mult": 0.55, "occ": 0.60},  # late-May, Sep, early-Oct
    "off":      {"weeks": 33, "rate_mult": 0.32, "occ": 0.22},  # Oct..May
}

# ==============================================================================
# REPRESENTATIVE LISTING SET
# ------------------------------------------------------------------------------
# Illustrative Ocean Grove (Neptune Twp, Monmouth County) listings used when no
# live data source is configured. Prices/taxes reflect the mid-2026 market
# (median list ~$785K, ~$625/sqft, ~1.79% effective tax, CMA 99-year ground
# lease on most lots). These are NOT live MLS rows — replace via $OG_LISTINGS_JSON
# or a live API for production use.
#
# Fields:
#   id, address, property_type, beds, baths, sqft, list_price,
#   annual_property_tax, monthly_condo_fee, annual_ground_lease, url,
#   [str_peak_weekly]  (optional per-listing peak weekly override)
# ==============================================================================
SAMPLE_LISTINGS = [
    {"id": "OG-001", "address": "29 Embury Ave", "property_type": "Single-family (Victorian)",
     "beds": 3, "baths": 2.0, "sqft": 1500, "list_price": 785000,
     "annual_property_tax": 11150, "monthly_condo_fee": 0, "annual_ground_lease": 1000,
     "url": "https://example.com/listing/OG-001"},
    {"id": "OG-002", "address": "108 Heck Ave", "property_type": "Cottage",
     "beds": 2, "baths": 1.0, "sqft": 980, "list_price": 649000,
     "annual_property_tax": 9400, "monthly_condo_fee": 0, "annual_ground_lease": 900,
     "url": "https://example.com/listing/OG-002"},
    {"id": "OG-003", "address": "71 Cookman Ave, Unit 3", "property_type": "Condo",
     "beds": 1, "baths": 1.0, "sqft": 620, "list_price": 415000,
     "annual_property_tax": 5200, "monthly_condo_fee": 385, "annual_ground_lease": 0,
     "url": "https://example.com/listing/OG-003"},
    {"id": "OG-004", "address": "14 Ocean Pathway", "property_type": "Single-family (Grand Victorian)",
     "beds": 5, "baths": 3.0, "sqft": 3000, "list_price": 1795000,
     "annual_property_tax": 24500, "monthly_condo_fee": 0, "annual_ground_lease": 1200,
     "url": "https://example.com/listing/OG-004"},
    {"id": "OG-005", "address": "45 Webb Ave", "property_type": "Single-family",
     "beds": 4, "baths": 2.5, "sqft": 2200, "list_price": 1150000,
     "annual_property_tax": 16800, "monthly_condo_fee": 0, "annual_ground_lease": 1100,
     "url": "https://example.com/listing/OG-005"},
    {"id": "OG-006", "address": "88 Main Ave, Unit 2", "property_type": "Condo (studio)",
     "beds": 0, "baths": 1.0, "sqft": 480, "list_price": 329000,
     "annual_property_tax": 4300, "monthly_condo_fee": 340, "annual_ground_lease": 0,
     "url": "https://example.com/listing/OG-006"},
    {"id": "OG-007", "address": "6 Surf Ave", "property_type": "Single-family (near beach)",
     "beds": 4, "baths": 3.0, "sqft": 2400, "list_price": 1495000,
     "annual_property_tax": 21000, "monthly_condo_fee": 0, "annual_ground_lease": 1150,
     "url": "https://example.com/listing/OG-007"},
    {"id": "OG-008", "address": "122 Central Ave", "property_type": "Condo",
     "beds": 2, "baths": 2.0, "sqft": 1100, "list_price": 560000,
     "annual_property_tax": 7800, "monthly_condo_fee": 520, "annual_ground_lease": 0,
     "url": "https://example.com/listing/OG-008"},
    {"id": "OG-009", "address": "53 Clark Ave", "property_type": "Single-family",
     "beds": 3, "baths": 1.5, "sqft": 1400, "list_price": 739000,
     "annual_property_tax": 10600, "monthly_condo_fee": 0, "annual_ground_lease": 950,
     "url": "https://example.com/listing/OG-009"},
    {"id": "OG-010", "address": "17 Pilgrim Pathway", "property_type": "Single-family (large)",
     "beds": 6, "baths": 4.0, "sqft": 3400, "list_price": 2150000,
     "annual_property_tax": 29000, "monthly_condo_fee": 0, "annual_ground_lease": 1300,
     "url": "https://example.com/listing/OG-010"},
    {"id": "OG-011", "address": "40 Abbott Ave", "property_type": "Single-family",
     "beds": 3, "baths": 2.0, "sqft": 1600, "list_price": 829000,
     "annual_property_tax": 12000, "monthly_condo_fee": 0, "annual_ground_lease": 1000,
     "url": "https://example.com/listing/OG-011"},
    {"id": "OG-012", "address": "62 Mt Hermon Way", "property_type": "Single-family (Victorian)",
     "beds": 3, "baths": 2.0, "sqft": 1650, "list_price": 895000,
     "annual_property_tax": 13200, "monthly_condo_fee": 0, "annual_ground_lease": 1050,
     "url": "https://example.com/listing/OG-012"},
]


# ==============================================================================
# DATA SOURCE RESOLUTION
# ==============================================================================
def _fetch_rentcast(api_key):
    """Live Ocean Grove for-sale pull from RentCast (stub).

    Left intentionally unimplemented: fill in the /listings/sale request and map
    the response into the SAMPLE_LISTINGS schema. Returning None makes the caller
    fall back to the representative set, so enabling this is a purely additive
    change.
    """
    return None


def fetch_listings():
    """Resolve the for-sale set. Returns (listings, source_label)."""
    path = os.environ.get("OG_LISTINGS_JSON")
    if path and Path(path).exists():
        with open(path) as f:
            data = json.load(f)
        listings = data["listings"] if isinstance(data, dict) and "listings" in data else data
        return listings, f"file:{path}"

    api_key = os.environ.get("RENTCAST_API_KEY")
    if api_key:
        live = _fetch_rentcast(api_key)
        if live:
            return live, "rentcast"

    return SAMPLE_LISTINGS, "representative-sample"


# ==============================================================================
# FINANCIAL MODEL
# ==============================================================================
def estimate_str_revenue(listing, airdna=None):
    """Annual gross STR revenue for a listing, with basis.

    Revenue is resolved in priority order (most specific first):
      1. listing `airdna_projected_revenue`  — address-level AirDNA Rentalizer.
      2. AirDNA market data for the listing's bedroom count — annual_revenue, or
         adr x occupancy x 365 if annual_revenue is absent.
      3. The seasonal model (per-listing `str_peak_weekly` override or the
         bedroom-count peak-weekly lookup, scaled across peak/shoulder/off).

    Returns (gross_annual, breakdown_by_season, peak_weekly, basis). When AirDNA
    supplies the annual total, the seasonal breakdown is the model's shape scaled
    to that total (illustrative distribution — AirDNA's own monthly seasonality
    is not applied unless present in a future field).
    """
    beds = int(listing.get("beds", 2) or 0)
    bkey = min(beds, 5)
    peak_weekly = listing.get("str_peak_weekly")
    if peak_weekly is None:
        peak_weekly = PEAK_WEEKLY_BY_BED.get(bkey, PEAK_WEEKLY_BY_BED[2])

    # Unscaled model seasonal shape — the pure estimate, and the distribution used
    # to display an AirDNA annual total across seasons.
    shape = {}
    model_total = 0.0
    for name, s in SEASONS.items():
        rev = peak_weekly * s["rate_mult"] * s["occ"] * s["weeks"]
        shape[name] = rev
        model_total += rev

    gross = None
    basis = "seasonal-model"
    by_bed = (airdna or {}).get("by_bedroom", {}) if airdna else {}
    ad = by_bed.get(str(bkey)) or by_bed.get(bkey)  # tolerate str/int keys
    if listing.get("airdna_projected_revenue"):
        gross = float(listing["airdna_projected_revenue"]); basis = "airdna-rentalizer"
    elif ad and ad.get("annual_revenue"):
        gross = float(ad["annual_revenue"]); basis = "airdna-market-bedroom"
    elif ad and ad.get("adr") and ad.get("occupancy"):
        gross = float(ad["adr"]) * float(ad["occupancy"]) * 365; basis = "airdna-adr-occ"
    if gross is None:
        gross = model_total

    scale = (gross / model_total) if model_total else 0
    breakdown = {}
    for name, s in SEASONS.items():
        breakdown[name] = {
            "weeks": s["weeks"],
            "weekly_rate": round(peak_weekly * s["rate_mult"] * scale),
            "occupancy": s["occ"],
            "revenue": round(shape[name] * scale),
        }
    return round(gross), breakdown, round(peak_weekly * scale), basis


def load_airdna():
    """Load AirDNA market calibration if present. Returns (data, path) or (None, None).

    Path resolves from $OG_AIRDNA_JSON (default data/ocean_grove/airdna_market.json).
    Populate it from your AirDNA MarketMinder dashboard — see airdna_market.example.json.
    """
    path = os.environ.get("OG_AIRDNA_JSON", "data/ocean_grove/airdna_market.json")
    if path and Path(path).exists():
        try:
            return json.loads(Path(path).read_text()), path
        except (ValueError, OSError):
            return None, None
    return None, None


def _stabilized_aftertax_financed(noi, price, down_pct):
    """Stabilized (year 2+, no depreciation) after-tax operating cash flow at a
    given down-payment fraction, financed. Used to solve for the break-even
    down payment."""
    loan = price * (1 - down_pct)
    ds, y1int = annual_debt_service_and_year1_interest(loan)
    pretax = noi - ds
    tax_stab = (noi - y1int) * MARGINAL_TAX_RATE   # signed; negative => benefit
    return pretax - tax_stab


def break_even_down_payment(noi, price):
    """Down-payment fraction at which stabilized after-tax financed cash flow
    reaches $0. Returns 0.0 if it self-funds even fully financed, or None if it
    can never break even (NOI too low to cover costs even all-cash). Otherwise
    the fraction in (0, 1) found by bisection — a lower value is more attractive
    (less equity needed to reach annual operating break-even)."""
    if _stabilized_aftertax_financed(noi, price, 0.0) >= 0:
        return 0.0
    if _stabilized_aftertax_financed(noi, price, 1.0) < 0:
        return None  # unprofitable even unlevered
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if _stabilized_aftertax_financed(noi, price, mid) >= 0:
            hi = mid
        else:
            lo = mid
    return round((lo + hi) / 2, 4)


def annual_debt_service_and_year1_interest(loan_amount):
    """Standard amortized mortgage. Returns (annual_payment, year1_interest)."""
    if loan_amount <= 0:
        return 0.0, 0.0
    r = MORTGAGE_RATE / 12.0
    n = MORTGAGE_TERM_YEARS * 12
    if r == 0:
        monthly = loan_amount / n
    else:
        monthly = loan_amount * r / (1 - (1 + r) ** (-n))
    annual_payment = monthly * 12
    # Year-1 interest = sum of the monthly interest components over 12 months.
    balance = loan_amount
    year1_interest = 0.0
    for _ in range(12):
        interest = balance * r
        principal = monthly - interest
        year1_interest += interest
        balance -= principal
    return annual_payment, year1_interest


def operating_expenses(listing, gross_revenue):
    """Annual STR operating expenses. Excludes debt service and depreciation."""
    price = listing["list_price"]
    exp = {
        "management":       round(MGMT_FEE_PCT * gross_revenue),
        "platform_fees":    round(PLATFORM_FEE_PCT * gross_revenue),
        "cleaning_net":     round(CLEANING_NET_PCT * gross_revenue),
        "supplies":         round(SUPPLIES_PCT * gross_revenue),
        "capex_reserve":    round(CAPEX_RESERVE_PCT * gross_revenue),
        "property_tax":     round(listing.get("annual_property_tax", price * 0.0179)),
        "insurance":        round(INSURANCE_PCT * price),
        "maintenance":      round(MAINT_PCT * price),
        "utilities":        round(UTILITIES_ANNUAL),
        "condo_fee":        round(listing.get("monthly_condo_fee", 0) * 12),
        "ground_lease":     round(listing.get("annual_ground_lease", 0)),
        "str_license":      round(LICENSE_ANNUAL),
    }
    exp["total"] = sum(exp.values())
    return exp


def analyze(listing, airdna=None):
    """Full STR economics for one listing."""
    price = listing["list_price"]

    gross, season_breakdown, peak_weekly, revenue_basis = estimate_str_revenue(listing, airdna)
    exp = operating_expenses(listing, gross)
    noi = gross - exp["total"]                       # Net Operating Income (pre-debt, pre-tax)

    # --- Financing ---
    loan = price * (1 - DOWN_PAYMENT_PCT)
    annual_debt_service, year1_interest = annual_debt_service_and_year1_interest(loan)
    cash_invested = price * DOWN_PAYMENT_PCT + price * CLOSING_COST_PCT

    pretax_cf_financed = noi - annual_debt_service   # operating cash flow, financed
    pretax_cf_cash     = noi                         # operating cash flow, all-cash

    # --- Bonus depreciation (100% of the 60% building basis, in the purchase year) ---
    building_basis = price * BUILDING_ALLOCATION
    bonus_depreciation = building_basis * BONUS_DEPR_RATE

    # --- Year-1 tax (with the depreciation shield) ---
    # Taxable income = NOI - mortgage interest - depreciation. A negative value is
    # a loss; under STR material participation it offsets ordinary income, so the
    # cash benefit = -taxable_income * marginal_rate. Depreciation is non-cash, so
    # after-tax CASH flow = pre-tax cash flow + tax_benefit.
    taxable_y1_financed = noi - year1_interest - bonus_depreciation
    taxable_y1_cash     = noi - bonus_depreciation
    tax_y1_financed = taxable_y1_financed * MARGINAL_TAX_RATE   # negative => benefit
    tax_y1_cash     = taxable_y1_cash * MARGINAL_TAX_RATE
    aftertax_cf_y1_financed = pretax_cf_financed - tax_y1_financed
    aftertax_cf_y1_cash     = pretax_cf_cash - tax_y1_cash

    # --- Stabilized (year 2+): building basis fully expensed, so NO further
    # depreciation. Interest stays roughly year-1 level early in the amortization,
    # which we use as a representative stabilized figure. ---
    taxable_stab_financed = noi - year1_interest
    taxable_stab_cash     = noi
    tax_stab_financed = taxable_stab_financed * MARGINAL_TAX_RATE
    tax_stab_cash     = taxable_stab_cash * MARGINAL_TAX_RATE
    aftertax_cf_stab_financed = pretax_cf_financed - tax_stab_financed
    aftertax_cf_stab_cash     = pretax_cf_cash - tax_stab_cash

    # --- Yield metrics ---
    cap_rate      = noi / price if price else 0
    gross_yield   = gross / price if price else 0
    cash_on_cash  = pretax_cf_financed / cash_invested if cash_invested else 0

    # --- Break-even flags -----------------------------------------------------
    # The report is asked to (a) flag properties that can break even on an annual
    # operating basis, and (b) assume the 100% bonus-depreciation benefit in the
    # purchase year. Those two interact, so we surface three distinct tests:
    #
    #   * purchase_year  — year-1 after-tax operating cash flow >= 0, financed.
    #     With 100% bonus depreciation this is ~always true (the depreciation
    #     shield swamps the operating loss) — it is the direct, literal answer to
    #     "break even in the purchase year with bonus depreciation."
    #   * sustained      — stabilized (yr 2+, no depreciation) after-tax cash flow
    #     >= 0, financed. The strict "self-funds every year" test.
    #   * operating_allcash — stabilized after-tax cash flow >= 0 UNLEVERED, i.e.
    #     the asset covers its operating costs net of tax without financing drag.
    breaks_even_purchase_year = aftertax_cf_y1_financed >= 0
    breaks_even_sustained = aftertax_cf_stab_financed >= 0
    breaks_even_operating_allcash = aftertax_cf_stab_cash >= 0

    # Discriminating metrics that tie the two together:
    #   * break_even_down_pct — equity fraction needed to self-fund annually.
    #   * break_even_horizon_years — how many years the one-time year-1 tax
    #     windfall keeps the CUMULATIVE after-tax position non-negative before
    #     stabilized operating losses draw it down (None => sustained, no draw-down).
    be_down = break_even_down_payment(noi, price)
    if aftertax_cf_stab_financed >= 0:
        horizon = None  # sustained — cumulative position never turns negative
    elif aftertax_cf_y1_financed < 0:
        horizon = 0
    else:
        horizon = int(1 + aftertax_cf_y1_financed // (-aftertax_cf_stab_financed))

    if breaks_even_sustained:
        rating = "SUSTAINABLE_BREAKEVEN"
        rating_label = "✅ Sustainable break-even"
    elif horizon is not None and horizon >= 5:
        rating = "DEPR_FUNDED_BREAKEVEN"
        rating_label = "🟢 Break-even over hold (depreciation-funded)"
    elif aftertax_cf_stab_financed >= -NEAR_BREAKEVEN_BAND:
        rating = "NEAR_BREAKEVEN"
        rating_label = "🟡 Near break-even"
    else:
        rating = "CASHFLOW_NEGATIVE"
        rating_label = "🔴 Cash-flow negative"

    return {
        **{k: listing.get(k) for k in (
            "id", "address", "property_type", "beds", "baths", "sqft",
            "list_price", "url")},
        "str_revenue": {
            "gross_annual": gross,
            "peak_weekly": peak_weekly,
            "basis": revenue_basis,
            "by_season": season_breakdown,
        },
        "operating_expenses": exp,
        "noi": round(noi),
        "financing": {
            "loan_amount": round(loan),
            "annual_debt_service": round(annual_debt_service),
            "year1_interest": round(year1_interest),
            "cash_invested": round(cash_invested),
            "down_payment_pct": DOWN_PAYMENT_PCT,
            "rate": MORTGAGE_RATE,
        },
        "depreciation": {
            "building_basis": round(building_basis),
            "land_basis": round(price * LAND_ALLOCATION),
            "bonus_depreciation_year1": round(bonus_depreciation),
        },
        "operating_income": {
            "financed": {
                "pretax_cash_flow": round(pretax_cf_financed),
                "year1_tax_benefit": round(-tax_y1_financed),
                "aftertax_cash_flow_year1": round(aftertax_cf_y1_financed),
                "aftertax_cash_flow_stabilized": round(aftertax_cf_stab_financed),
            },
            "all_cash": {
                "pretax_cash_flow": round(pretax_cf_cash),
                "year1_tax_benefit": round(-tax_y1_cash),
                "aftertax_cash_flow_year1": round(aftertax_cf_y1_cash),
                "aftertax_cash_flow_stabilized": round(aftertax_cf_stab_cash),
            },
        },
        "yields": {
            "cap_rate": round(cap_rate, 4),
            "gross_yield": round(gross_yield, 4),
            "cash_on_cash": round(cash_on_cash, 4),
        },
        "breakeven": {
            "down_payment_pct": be_down,      # equity needed to self-fund annually
            "horizon_years": horizon,          # yrs the yr-1 tax windfall funds losses
        },
        "flags": {
            "breaks_even_purchase_year": breaks_even_purchase_year,
            "breaks_even_sustained": breaks_even_sustained,
            "breaks_even_operating_allcash": breaks_even_operating_allcash,
        },
        "rating": rating,
        "rating_label": rating_label,
    }


# ==============================================================================
# RENDERING
# ==============================================================================
def _money(x):
    if x is None:
        return "—"
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def _pct(x):
    return f"{x*100:.1f}%" if x is not None else "—"


def build_report(asof_date, archive=True):
    listings, source = fetch_listings()
    airdna, airdna_path = load_airdna()
    analyzed = [analyze(l, airdna) for l in listings]

    # Revenue provenance: how each listing's gross STR revenue was derived.
    basis_counts = {}
    for a in analyzed:
        b = a["str_revenue"]["basis"]
        basis_counts[b] = basis_counts.get(b, 0) + 1

    # Rank by sustainable after-tax cash flow (financed), best first.
    analyzed.sort(
        key=lambda a: a["operating_income"]["financed"]["aftertax_cash_flow_stabilized"],
        reverse=True,
    )

    purchase_year = [a for a in analyzed if a["flags"]["breaks_even_purchase_year"]]
    sustained = [a for a in analyzed if a["flags"]["breaks_even_sustained"]]
    allcash = [a for a in analyzed if a["flags"]["breaks_even_operating_allcash"]]

    def _median(vals):
        vals = sorted(v for v in vals if v is not None)
        if not vals:
            return None
        n = len(vals)
        return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2

    now_utc = datetime.now(timezone.utc)
    week_monday = asof_date - timedelta(days=asof_date.weekday())

    report = {
        "meta": {
            "report": "Ocean Grove STR Opportunity Report",
            "version": "1.0",
            "generated_utc": now_utc.isoformat(),
            "week_of": week_monday.strftime("%Y-%m-%d"),
            "asof": asof_date.strftime("%Y-%m-%d"),
            "market": "Ocean Grove, NJ (Neptune Township, Monmouth County)",
            "data_source": source,
            "data_source_note": (
                "Representative sample listings — illustrative, not live MLS. "
                "Wire a live feed via $OG_LISTINGS_JSON or $RENTCAST_API_KEY."
                if source == "representative-sample" else
                "Listings resolved from a configured live/file source."
            ),
            "revenue_source": {
                "primary": ("airdna" if airdna_path and any(k.startswith("airdna") for k in basis_counts)
                            else "seasonal-model"),
                "by_basis": basis_counts,
                "airdna_file": airdna_path,
                "airdna_as_of": (airdna or {}).get("as_of") if airdna else None,
                "note": (
                    "STR revenue calibrated to AirDNA MarketMinder figures."
                    if airdna_path and any(k.startswith("airdna") for k in basis_counts) else
                    "STR revenue from the built-in seasonal model. Drop AirDNA MarketMinder "
                    "figures into data/ocean_grove/airdna_market.json (see .example.json) to calibrate."
                ),
            },
            "listing_count": len(analyzed),
        },
        "assumptions": {
            "basis_split": {"land": LAND_ALLOCATION, "building": BUILDING_ALLOCATION},
            "bonus_depreciation_rate": BONUS_DEPR_RATE,
            "marginal_tax_rate": MARGINAL_TAX_RATE,
            "tax_treatment": "STR losses assumed non-passive (avg stay <=7d + material participation)",
            "financing": {
                "down_payment_pct": DOWN_PAYMENT_PCT,
                "mortgage_rate": MORTGAGE_RATE,
                "term_years": MORTGAGE_TERM_YEARS,
                "closing_cost_pct": CLOSING_COST_PCT,
            },
            "operating_cost_pcts": {
                "management": MGMT_FEE_PCT, "platform_fees": PLATFORM_FEE_PCT,
                "cleaning_net": CLEANING_NET_PCT, "supplies": SUPPLIES_PCT,
                "capex_reserve": CAPEX_RESERVE_PCT, "insurance_of_price": INSURANCE_PCT,
                "maintenance_of_price": MAINT_PCT,
            },
            "operating_cost_fixed": {
                "utilities_annual": UTILITIES_ANNUAL, "str_license_annual": LICENSE_ANNUAL,
            },
            "seasonal_model": SEASONS,
            "peak_weekly_by_bed": PEAK_WEEKLY_BY_BED,
            "near_breakeven_band": NEAR_BREAKEVEN_BAND,
            "breakeven_definition": (
                "Three tests are reported. purchase_year = year-1 after-tax operating "
                "cash flow (financed) >= $0, i.e. break-even net of tax WITH the 100% "
                "bonus-depreciation benefit (nearly always true — the shield swamps the "
                "operating loss). sustained = stabilized (yr 2+, no depreciation) after-tax "
                "cash flow (financed) >= $0 (the strict self-funding test). "
                "operating_allcash = stabilized after-tax cash flow >= $0 unlevered. "
                "break_even_down_pct = equity fraction needed to self-fund annually; "
                "horizon_years = years the one-time yr-1 tax windfall keeps the cumulative "
                "position non-negative."
            ),
        },
        "summary": {
            "listing_count": len(analyzed),
            "breakeven_purchase_year": len(purchase_year),
            "breakeven_sustained": len(sustained),
            "breakeven_operating_allcash": len(allcash),
            "median_list_price": _median([a["list_price"] for a in analyzed]),
            "median_gross_str_revenue": _median([a["str_revenue"]["gross_annual"] for a in analyzed]),
            "median_noi": _median([a["noi"] for a in analyzed]),
            "median_breakeven_down_pct": _median(
                [a["breakeven"]["down_payment_pct"] for a in analyzed
                 if a["breakeven"]["down_payment_pct"] is not None]),
        },
        "listings": analyzed,
    }

    markdown = render_markdown(report)
    return report, markdown


def render_markdown(report):
    m = report["meta"]
    a = report["assumptions"]
    s = report["summary"]
    lines = []

    lines.append(f"# 🏖️ Ocean Grove STR Opportunity Report")
    lines.append("")
    lines.append(f"**Week of {m['week_of']}**  ·  {m['market']}  ·  generated {m['generated_utc'][:16].replace('T', ' ')} UTC")
    lines.append("")
    if m["data_source"] == "representative-sample":
        lines.append(f"> ⚠️ **Data source: representative sample** — {m['data_source_note']}")
        lines.append("")
    rs = m.get("revenue_source", {})
    if rs:
        if rs.get("primary") == "airdna":
            lines.append(f"> 📊 **STR revenue: AirDNA MarketMinder**"
                         + (f" (as of {rs['airdna_as_of']})" if rs.get("airdna_as_of") else "")
                         + f" — {rs['note']}")
        else:
            lines.append(f"> 📈 **STR revenue: built-in seasonal model** — {rs['note']}")
        lines.append("")

    # --- Executive summary ---
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **{s['listing_count']}** listings assessed")
    lines.append(f"- **{s['breakeven_operating_allcash']}** can break even on an annual operating basis net of tax when **unlevered** (the asset covers operating costs after tax)")
    lines.append(f"- **{s['breakeven_sustained']}** self-fund every year as **financed** at {int(report['assumptions']['financing']['down_payment_pct']*100)}% down / {_pct(report['assumptions']['financing']['mortgage_rate'])} (strict sustained break-even)")
    lines.append(f"- **{s['breakeven_purchase_year']}** break even in the **purchase year** net of tax once the 100% bonus-depreciation benefit is applied")
    lines.append(f"- Median list price **{_money(s['median_list_price'])}** · median gross STR revenue **{_money(s['median_gross_str_revenue'])}** · median NOI **{_money(s['median_noi'])}** · median break-even down payment **{_pct(s['median_breakeven_down_pct'])}**")
    lines.append("")
    lines.append("> **Read this first.** In this market, 100% bonus depreciation makes essentially every "
                 "property cash-positive *in the purchase year* (a one-time tax windfall of 60% of price "
                 "× your rate). The real question is what happens *after* that. The two columns that "
                 "matter are **stabilized after-tax cash flow** (does it self-fund every year?) and the "
                 "**break-even down payment** (how much equity it takes to get there). Lower list price / "
                 "higher-yield units come closest.")
    lines.append("")

    # --- Break-even flags ---
    lines.append("## 🚩 Break-even flags")
    lines.append("")
    sustained = [x for x in report["listings"] if x["flags"]["breaks_even_sustained"]]
    if sustained:
        lines.append("**Self-funding as financed** — break even on an annual operating basis every year, "
                     "net of tax, at the assumed financing (strongest signal):")
        lines.append("")
        lines.append("| Property | Type | Bd | List | Stabilized after-tax CF | Break-even down |")
        lines.append("|---|---|--:|--:|--:|--:|")
        for x in sustained:
            fin = x["operating_income"]["financed"]
            lines.append(
                f"| {x['address']} | {x['property_type']} | {x['beds']} | {_money(x['list_price'])} "
                f"| {_money(fin['aftertax_cash_flow_stabilized'])} | {_pct(x['breakeven']['down_payment_pct'])} |"
            )
        lines.append("")
    else:
        lines.append("_No listing self-funds every year at the assumed financing "
                     f"({int(report['assumptions']['financing']['down_payment_pct']*100)}% down / "
                     f"{_pct(report['assumptions']['financing']['mortgage_rate'])})._ "
                     "Below are the closest to annual operating break-even — ranked by stabilized "
                     "after-tax cash flow and the equity needed to self-fund. Every one still breaks "
                     "even in the **purchase year** on the bonus-depreciation shield.")
        lines.append("")
        lines.append("| Property | Type | Bd | List | Stabilized after-tax CF | Break-even down | Yrs funded by yr-1 shield |")
        lines.append("|---|---|--:|--:|--:|--:|--:|")
        for x in report["listings"][:5]:
            fin = x["operating_income"]["financed"]
            hz = x["breakeven"]["horizon_years"]
            lines.append(
                f"| {x['address']} | {x['property_type']} | {x['beds']} | {_money(x['list_price'])} "
                f"| {_money(fin['aftertax_cash_flow_stabilized'])} | {_pct(x['breakeven']['down_payment_pct'])} "
                f"| {'sustained' if hz is None else hz} |"
            )
        lines.append("")

    # --- Full ranking ---
    lines.append("## All listings — ranked by stabilized after-tax cash flow (financed)")
    lines.append("")
    lines.append("| # | Property | Bd/Ba | List | Gross STR | NOI | Cap | Pre-tax CF | Yr-1 after-tax | Stabilized after-tax | BE down | Rating |")
    lines.append("|--:|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|")
    for i, x in enumerate(report["listings"], 1):
        fin = x["operating_income"]["financed"]
        lines.append(
            f"| {i} | {x['address']} | {x['beds']}/{x['baths']} | {_money(x['list_price'])} "
            f"| {_money(x['str_revenue']['gross_annual'])} | {_money(x['noi'])} | {_pct(x['yields']['cap_rate'])} "
            f"| {_money(fin['pretax_cash_flow'])} | {_money(fin['aftertax_cash_flow_year1'])} "
            f"| {_money(fin['aftertax_cash_flow_stabilized'])} | {_pct(x['breakeven']['down_payment_pct'])} "
            f"| {x['rating_label']} |"
        )
    lines.append("")

    # --- Per-property detail ---
    lines.append("## Property detail")
    lines.append("")
    for x in report["listings"]:
        fin = x["operating_income"]["financed"]
        cash = x["operating_income"]["all_cash"]
        dep = x["depreciation"]
        exp = x["operating_expenses"]
        lines.append(f"### {x['address']} — {x['rating_label']}")
        lines.append("")
        lines.append(f"{x['property_type']} · {x['beds']} bd / {x['baths']} ba · {x['sqft']:,} sqft · list **{_money(x['list_price'])}**  ")
        lines.append(f"[listing]({x['url']})")
        lines.append("")
        _basis_lbl = {
            "seasonal-model": "seasonal model",
            "airdna-rentalizer": "AirDNA Rentalizer (address-level)",
            "airdna-market-bedroom": "AirDNA market (by bedroom)",
            "airdna-adr-occ": "AirDNA ADR×occupancy",
        }.get(x["str_revenue"].get("basis"), x["str_revenue"].get("basis"))
        lines.append(f"- **Gross STR revenue:** {_money(x['str_revenue']['gross_annual'])}/yr "
                     f"(peak {_money(x['str_revenue']['peak_weekly'])}/wk · basis: {_basis_lbl})")
        lines.append(f"- **Operating expenses:** {_money(exp['total'])}/yr "
                     f"(mgmt {_money(exp['management'])}, tax {_money(exp['property_tax'])}, "
                     f"ins {_money(exp['insurance'])}, maint {_money(exp['maintenance'])}, "
                     f"utils {_money(exp['utilities'])}"
                     + (f", condo {_money(exp['condo_fee'])}" if exp['condo_fee'] else "")
                     + (f", ground lease {_money(exp['ground_lease'])}" if exp['ground_lease'] else "")
                     + ")")
        lines.append(f"- **NOI:** {_money(x['noi'])}/yr · cap rate {_pct(x['yields']['cap_rate'])} · gross yield {_pct(x['yields']['gross_yield'])}")
        lines.append(f"- **Debt service:** {_money(x['financing']['annual_debt_service'])}/yr "
                     f"(loan {_money(x['financing']['loan_amount'])} @ {_pct(x['financing']['rate'])}, "
                     f"yr-1 interest {_money(x['financing']['year1_interest'])})")
        lines.append(f"- **Bonus depreciation (yr 1):** {_money(dep['bonus_depreciation_year1'])} "
                     f"(building basis {_money(dep['building_basis'])} @ {int(BONUS_DEPR_RATE*100)}%; "
                     f"land {_money(dep['land_basis'])} not depreciated)")
        lines.append(f"- **Financed operating income:** pre-tax {_money(fin['pretax_cash_flow'])}/yr → "
                     f"year-1 net of tax **{_money(fin['aftertax_cash_flow_year1'])}** "
                     f"(tax benefit {_money(fin['year1_tax_benefit'])}) → "
                     f"stabilized net of tax **{_money(fin['aftertax_cash_flow_stabilized'])}**")
        lines.append(f"- **All-cash operating income:** pre-tax {_money(cash['pretax_cash_flow'])}/yr → "
                     f"year-1 net of tax {_money(cash['aftertax_cash_flow_year1'])} → "
                     f"stabilized net of tax {_money(cash['aftertax_cash_flow_stabilized'])}")
        lines.append(f"- **Cash-on-cash (pre-tax, financed):** {_pct(x['yields']['cash_on_cash'])} "
                     f"on {_money(x['financing']['cash_invested'])} invested")
        be = x["breakeven"]
        hz = be["horizon_years"]
        lines.append(f"- **Break-even:** self-funds annually at **{_pct(be['down_payment_pct'])} down**; "
                     + ("sustained (never draws down)." if hz is None
                        else f"as financed, the year-1 tax windfall funds ~**{hz} years** of operating losses before the cumulative position turns negative."))
        lines.append("")

    # --- Assumptions ---
    lines.append("## Assumptions & methodology")
    lines.append("")
    lines.append(f"- **Depreciation basis split:** {int(a['basis_split']['land']*100)}% land / "
                 f"{int(a['basis_split']['building']*100)}% building · "
                 f"**{int(a['bonus_depreciation_rate']*100)}% bonus depreciation** taken in the purchase year "
                 f"on the building basis. Land is not depreciable.")
    lines.append(f"- **Marginal tax rate:** {_pct(a['marginal_tax_rate'])} — {a['tax_treatment']}. "
                 f"The year-1 depreciation loss is assumed usable against ordinary income; the resulting "
                 f"tax reduction is treated as a cash benefit.")
    lines.append(f"- **Financing:** {int(a['financing']['down_payment_pct']*100)}% down, "
                 f"{a['financing']['term_years']}-yr fixed @ {_pct(a['financing']['mortgage_rate'])}, "
                 f"{int(a['financing']['closing_cost_pct']*100)}% closing costs.")
    lines.append(f"- **Operating costs:** management {_pct(a['operating_cost_pcts']['management'])}, "
                 f"platform {_pct(a['operating_cost_pcts']['platform_fees'])}, "
                 f"cleaning-net {_pct(a['operating_cost_pcts']['cleaning_net'])}, "
                 f"supplies {_pct(a['operating_cost_pcts']['supplies'])}, "
                 f"capex reserve {_pct(a['operating_cost_pcts']['capex_reserve'])} (all % of gross); "
                 f"insurance {_pct(a['operating_cost_pcts']['insurance_of_price'])} & "
                 f"maintenance {_pct(a['operating_cost_pcts']['maintenance_of_price'])} of price; "
                 f"utilities {_money(a['operating_cost_fixed']['utilities_annual'])}/yr; "
                 f"STR license {_money(a['operating_cost_fixed']['str_license_annual'])}/yr.")
    lines.append(f"- **Seasonal revenue model:** peak {SEASONS['peak']['weeks']}wk @ {int(SEASONS['peak']['occ']*100)}% occ, "
                 f"shoulder {SEASONS['shoulder']['weeks']}wk @ {int(SEASONS['shoulder']['occ']*100)}% ("
                 f"{int(SEASONS['shoulder']['rate_mult']*100)}% of peak rate), "
                 f"off {SEASONS['off']['weeks']}wk @ {int(SEASONS['off']['occ']*100)}% "
                 f"({int(SEASONS['off']['rate_mult']*100)}% of peak rate). "
                 f"Peak weekly rate by bedroom count.")
    lines.append(f"- **Break-even definition:** {a['breakeven_definition']}")
    lines.append("")
    lines.append("### Ocean Grove notes")
    lines.append("")
    lines.append("- Most Ocean Grove lots sit on a **99-year ground lease from the Ocean Grove Camp "
                 "Meeting Association (CMA)** — the buyer typically owns the *building* on leased land. "
                 "An annual ground-lease fee is included in operating costs where applicable. Because the "
                 "land is not owned, a buyer could arguably justify a higher building allocation than the "
                 "40/60 split assumed here (which would *increase* the year-1 depreciation shield).")
    lines.append("- **Neptune Township has been tightening STR rules** (minimum-stay and bonding "
                 "requirements). Confirm current licensing/minimum-stay before underwriting; these can "
                 "materially change occupancy and cost assumptions.")
    lines.append("- Summer bookings are typically **Saturday-to-Saturday weekly**, reflected in the "
                 "weekly-rate seasonal model above.")
    lines.append("")
    lines.append("> **Not investment, tax, or legal advice.** A simplified underwriting model with "
                 "estimated inputs. Depreciation recapture on sale, passive-activity limits, financing "
                 "eligibility on leased land, and actual booking performance are not modeled. Verify every "
                 "figure and consult a CPA/attorney before transacting.")
    lines.append("")

    return "\n".join(lines)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Ocean Grove STR Opportunity Report")
    parser.add_argument("--output-dir", default="data/ocean_grove")
    parser.add_argument("--asof", default=None, help="Report as-of date YYYY-MM-DD (default: today UTC)")
    parser.add_argument("--no-archive", action="store_true", help="Do not write history/<date>.json")
    args = parser.parse_args()

    if args.asof:
        asof = datetime.strptime(args.asof, "%Y-%m-%d").date()
    else:
        asof = datetime.now(timezone.utc).date()

    report, markdown = build_report(asof, archive=not args.no_archive)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "latest.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "latest.md").write_text(markdown)

    if not args.no_archive:
        hist = out / "history"
        hist.mkdir(parents=True, exist_ok=True)
        (hist / f"{report['meta']['week_of']}.json").write_text(json.dumps(report, indent=2, default=str))

    # index.json — dashboard contract: list of archived weeks (newest first) +
    # the latest week. Mirrors data/signal_research/index.json so the dashboard
    # tab can enumerate the weekly archive.
    hist = out / "history"
    weeks = sorted((p.stem for p in hist.glob("*.json")), reverse=True) if hist.exists() else []
    latest_week = report["meta"]["week_of"]
    if latest_week not in weeks:
        weeks = [latest_week] + weeks
    (out / "index.json").write_text(json.dumps({"latest": latest_week, "weeks": weeks}, indent=2))

    s = report["summary"]
    print(f"Ocean Grove STR Report — week of {report['meta']['week_of']} ({report['meta']['data_source']})")
    rs = report["meta"].get("revenue_source", {})
    print(f"  Revenue source: {rs.get('primary')} {dict(rs.get('by_basis', {}))}")
    print(f"  {s['listing_count']} listings · {s['breakeven_operating_allcash']} unlevered break-even · "
          f"{s['breakeven_sustained']} financed-sustained · {s['breakeven_purchase_year']} purchase-year")
    print(f"  Wrote {out/'latest.json'}, {out/'latest.md'}")
    for x in report["listings"]:
        fin = x["operating_income"]["financed"]
        print(f"    {x['rating_label'][:2]} {x['address']:<26} "
              f"list {_money(x['list_price']):>11} · yr1 {_money(fin['aftertax_cash_flow_year1']):>10} · "
              f"stab {_money(fin['aftertax_cash_flow_stabilized']):>10}")


if __name__ == "__main__":
    main()
