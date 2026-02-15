import streamlit as st
import streamlit.components.v1 as components

# Prefer secrets, fallback to your provided URL
DEFAULT_CITYMAP_URL = "https://citymaps.capetown.gov.za/EGISViewer/"
CITYMAP_VIEWER_URL = st.secrets.get("CITYMAP_VIEWER_URL", DEFAULT_CITYMAP_URL)

import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
from copy import deepcopy

# =========================
# CONFIG
# =========================
ZONING_PRESETS = {
    "GR2 (Suburban)": {"ff": 1.0, "height": 15, "coverage": 0.6},
    "GR4 (Flats)": {"ff": 1.5, "height": 24, "coverage": 0.6},
    "MU1 (Mixed Use)": {"ff": 1.5, "height": 15, "coverage": 0.75},
    "MU2 (High Density)": {"ff": 4.0, "height": 25, "coverage": 1.0},
    "GB7 (CBD/High Rise)": {"ff": 12.0, "height": 60, "coverage": 1.0},
}

DC_BASE_RATE = 514.10
ROADS_TRANSPORT_PORTION = 285.35

# 2026 hard cost tiers
COST_TIERS = {
    "Economic (R10,000/m²)": 10000.0,
    "Mid-Tier (R18,000/m²)": 18000.0,
    "Luxury (R25,000+/m²)": 25000.0,
}

# 2026 sectional title exit price ranges (per sellable m²)
# Note: Clifton/Bantry Bay upper assumed 170,000 due to truncated image.
DEFAULT_EXIT_PRICES = [
    {"suburb": "Clifton / Bantry Bay", "min_price_per_m2": 120000, "max_price_per_m2": 170000},
    {"suburb": "Sea Point / Green Point", "min_price_per_m2": 65000, "max_price_per_m2": 85000},
    {"suburb": "City Bowl (CBD / Gardens)", "min_price_per_m2": 45000, "max_price_per_m2": 60000},
    {"suburb": "Claremont / Rondebosch", "min_price_per_m2": 40000, "max_price_per_m2": 52000},
    {"suburb": "Woodstock / Salt River", "min_price_per_m2": 32000, "max_price_per_m2": 42000},
    {"suburb": "Durbanville / Sunningdale", "min_price_per_m2": 25000, "max_price_per_m2": 35000},
    {"suburb": "Khayelitsha / Mitchells Plain", "min_price_per_m2": 10000, "max_price_per_m2": 15000},
]

IH_PRICE_PER_SELLABLE_M2 = 15000

# Professional fee ranges (from your image)
PROF_FEE_RANGES = {
    "Architect": (0.05, 0.07),
    "Quantity Surveyor (QS)": (0.02, 0.03),
    "Structural Engineer": (0.015, 0.02),
    "Civil Engineer": (0.01, 0.015),
    "Electrical/Mech Engineer": (0.01, 0.02),
    "Project Manager": (0.02, 0.03),
}
PROF_FEE_TARGET_TOTAL = 0.135  # midpoint of ~12–15%

# =========================
# PAGE
# =========================
st.set_page_config(page_title="ResiDuo — Cape Town Feasibility", layout="wide")
st.title("ResiDuo — Cape Town Feasibility")


# =========================
# OVERLAY MODEL
# =========================
@dataclass(frozen=True)
class HeritageOverlay:
    enabled: bool
    bonus_suppression_pct: float  # suppresses density bonus
    cost_uplift_pct: float        # uplifts construction input
    fees_uplift_pct: float        # uplifts total professional fee rate
    profit_uplift_pct: float      # uplifts profit % of GDV


def apply_heritage_overlay(
    density_bonus_pct: float,
    base_cost_value: float,
    base_fees_rate: float,
    base_profit_rate: float,
    overlay: HeritageOverlay,
) -> tuple[float, float, float, float]:
    if not overlay.enabled:
        return density_bonus_pct, base_cost_value, base_fees_rate, base_profit_rate

    adj_bonus = max(0.0, density_bonus_pct * (1.0 - overlay.bonus_suppression_pct / 100.0))
    adj_cost = base_cost_value * (1.0 + overlay.cost_uplift_pct / 100.0)
    adj_fees = base_fees_rate * (1.0 + overlay.fees_uplift_pct / 100.0)
    adj_profit = base_profit_rate * (1.0 + overlay.profit_uplift_pct / 100.0)
    return adj_bonus, adj_cost, adj_fees, adj_profit


# =========================
# HELPERS
# =========================
def pt_discount(pt_zone_value: str) -> float:
    return {"PT1": 0.8, "PT2": 0.5}.get(pt_zone_value, 1.0)


def normalize_exit_price_db(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns: suburb, min_price_per_m2, max_price_per_m2
    Migrate: suburb, exit_price_per_m2 -> min=max
    """
    if df is None or df.empty:
        return pd.DataFrame(DEFAULT_EXIT_PRICES)

    df = df.copy()
    col_map = {c.lower().strip(): c for c in df.columns}
    if "suburb" not in col_map:
        return pd.DataFrame(DEFAULT_EXIT_PRICES)

    suburb_col = col_map["suburb"]
    min_col = col_map.get("min_price_per_m2")
    max_col = col_map.get("max_price_per_m2")
    single_col = col_map.get("exit_price_per_m2")

    if min_col is None or max_col is None:
        if single_col is not None:
            df["min_price_per_m2"] = df[single_col]
            df["max_price_per_m2"] = df[single_col]
        else:
            return pd.DataFrame(DEFAULT_EXIT_PRICES)
    else:
        df["min_price_per_m2"] = df[min_col]
        df["max_price_per_m2"] = df[max_col]

    df["suburb"] = df[suburb_col].astype(str).str.strip()
    df["min_price_per_m2"] = pd.to_numeric(df["min_price_per_m2"], errors="coerce")
    df["max_price_per_m2"] = pd.to_numeric(df["max_price_per_m2"], errors="coerce")
    df = df.dropna(subset=["suburb", "min_price_per_m2", "max_price_per_m2"])
    df = df.sort_values("suburb").drop_duplicates(subset=["suburb"], keep="last")

    swap = df["min_price_per_m2"] > df["max_price_per_m2"]
    if swap.any():
        df.loc[swap, ["min_price_per_m2", "max_price_per_m2"]] = df.loc[
            swap, ["max_price_per_m2", "min_price_per_m2"]
        ].values

    return df[["suburb", "min_price_per_m2", "max_price_per_m2"]].reset_index(drop=True)


def load_exit_price_db() -> pd.DataFrame:
    if "exit_price_db" not in st.session_state:
        st.session_state.exit_price_db = pd.DataFrame(DEFAULT_EXIT_PRICES)
    st.session_state.exit_price_db = normalize_exit_price_db(st.session_state.exit_price_db)
    return st.session_state.exit_price_db


def set_exit_price_db_from_upload(uploaded_file) -> tuple[bool, str]:
    try:
        df = pd.read_csv(uploaded_file)
        df2 = normalize_exit_price_db(df)
        if df2.empty:
            return False, "CSV loaded but no valid rows found."
        st.session_state.exit_price_db = df2
        return True, f"Loaded {len(df2)} suburb exit price ranges."
    except Exception as e:
        return False, f"Failed to load CSV: {e}"


def default_prof_fee_components_scaled_to_target() -> dict[str, float]:
    mids = {k: (v[0] + v[1]) / 2.0 for k, v in PROF_FEE_RANGES.items()}
    s = sum(mids.values())
    if s <= 0:
        return {k: 0.0 for k in mids}
    scale = PROF_FEE_TARGET_TOTAL / s
    return {k: mids[k] * scale for k in mids}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_market_price_from_db(db: pd.DataFrame, suburb: str, price_point: str) -> tuple[float | None, float | None, float | None]:
    if not suburb or db.empty:
        return None, None, None
    row = db.loc[db["suburb"] == suburb]
    if row.empty:
        return None, None, None
    mn = float(row["min_price_per_m2"].iloc[0])
    mx = float(row["max_price_per_m2"].iloc[0])
    if price_point == "Low":
        p = mn
    elif price_point == "High":
        p = mx
    else:
        p = (mn + mx) / 2.0
    return p, mn, mx


def compute_model(inputs: dict) -> dict:
    zoning_key = inputs["zoning_key"]
    ff = ZONING_PRESETS[zoning_key]["ff"]

    land_area = float(inputs["land_area"])
    existing_bulk = float(inputs["existing_gba"])
    density_bonus = float(inputs["density_bonus"])
    ih_percent = float(inputs["ih_percent"])
    pt_zone = inputs["pt_zone"]

    efficiency = float(inputs["efficiency_ratio"])
    profit_rate = float(inputs["profit_margin"])
    base_fee_rate = float(inputs["prof_fee_total"])

    overlay = HeritageOverlay(
        enabled=bool(inputs["heritage_enabled"]),
        bonus_suppression_pct=float(inputs["heritage_bonus_suppression"]),
        cost_uplift_pct=float(inputs["heritage_cost_uplift"]),
        fees_uplift_pct=float(inputs["heritage_fees_uplift"]),
        profit_uplift_pct=float(inputs["heritage_profit_uplift"]),
    )

    cost_mode = inputs["cost_mode"]  # "R / m²" or "% of GDV"
    pct_gdv_scope = inputs["pct_gdv_scope"]
    base_cost_sqm = float(inputs["const_cost_sqm"])
    base_cost_pct_gdv = float(inputs["const_cost_pct_gdv"])

    cost_input = base_cost_sqm if cost_mode == "R / m²" else base_cost_pct_gdv
    adj_bonus, adj_cost_input, adj_fee_rate, adj_profit_rate = apply_heritage_overlay(
        density_bonus_pct=density_bonus,
        base_cost_value=cost_input,
        base_fees_rate=base_fee_rate,
        base_profit_rate=profit_rate,
        overlay=overlay,
    )

    base_bulk = land_area * ff
    proposed_bulk = base_bulk * (1.0 + adj_bonus / 100.0)
    proposed_sellable = proposed_bulk * efficiency

    net_increase_bulk = max(0.0, proposed_bulk - existing_bulk)

    ih_increase_bulk = net_increase_bulk * (ih_percent / 100.0)
    market_increase_bulk = net_increase_bulk - ih_increase_bulk

    ih_sellable = ih_increase_bulk * efficiency
    market_sellable = max(0.0, proposed_sellable - ih_sellable)

    disc = pt_discount(pt_zone)
    roads_dc = market_increase_bulk * ROADS_TRANSPORT_PORTION * disc
    other_dc = market_increase_bulk * (DC_BASE_RATE - ROADS_TRANSPORT_PORTION)
    total_dc = roads_dc + other_dc
    total_potential_dc = net_increase_bulk * DC_BASE_RATE
    dc_savings = total_potential_dc - total_dc

    market_price = float(inputs["market_price"])
    ih_price = float(inputs.get("ih_price", IH_PRICE_PER_SELLABLE_M2))

    gdv = (market_sellable * market_price) + (ih_sellable * ih_price)

    if cost_mode == "R / m²":
        adj_cost_sqm = float(adj_cost_input)
        construction_costs = proposed_bulk * adj_cost_sqm
        hard_plus_dc = construction_costs + total_dc
        prof_fees = hard_plus_dc * adj_fee_rate
        adj_cost_pct_gdv = None
    else:
        adj_cost_pct_gdv = float(adj_cost_input)
        if pct_gdv_scope == "Hard cost only":
            construction_costs = gdv * adj_cost_pct_gdv
            hard_plus_dc = construction_costs + total_dc
            prof_fees = hard_plus_dc * adj_fee_rate
        else:
            target_all_in = gdv * adj_cost_pct_gdv
            construction_costs = (target_all_in - (adj_fee_rate * total_dc)) / (1.0 + adj_fee_rate)
            construction_costs = max(0.0, construction_costs)
            hard_plus_dc = construction_costs + total_dc
            prof_fees = hard_plus_dc * adj_fee_rate
        adj_cost_sqm = None

    profit = gdv * adj_profit_rate
    rlv = gdv - (construction_costs + total_dc + prof_fees + profit)

    implied_cost_sqm = (construction_costs / proposed_bulk) if proposed_bulk > 0 else 0.0

    return {
        "base_bulk": base_bulk,
        "proposed_bulk": proposed_bulk,
        "proposed_sellable": proposed_sellable,
        "net_increase_bulk": net_increase_bulk,
        "ih_sellable": ih_sellable,
        "market_sellable": market_sellable,
        "market_price": market_price,
        "ih_price": ih_price,
        "gdv": gdv,
        "construction_costs": construction_costs,
        "total_dc": total_dc,
        "dc_savings": dc_savings,
        "prof_fees": prof_fees,
        "profit": profit,
        "rlv": rlv,
        "brownfield_credit": existing_bulk > proposed_bulk,
        "adj_bonus": adj_bonus,
        "adj_fee_rate": adj_fee_rate,
        "adj_profit_rate": adj_profit_rate,
        "cost_mode": cost_mode,
        "pct_gdv_scope": pct_gdv_scope,
        "adj_cost_sqm": adj_cost_sqm,
        "adj_cost_pct_gdv": adj_cost_pct_gdv,
        "implied_cost_sqm": implied_cost_sqm,
        "efficiency": efficiency,
    }


def fmt_money(x: float) -> str:
    return f"R {x:,.0f}"


def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def build_audit_table(inputs: dict, res: dict) -> pd.DataFrame:
    rows = [
        ("Zoning preset", inputs["zoning_key"]),
        ("Land area (m²)", f"{inputs['land_area']:,.0f}"),
        ("Existing GBA (bulk m²)", f"{inputs['existing_gba']:,.0f}"),
        ("Density bonus (input)", f"{inputs['density_bonus']:.1f}%"),
        ("Efficiency (sellable/bulk)", f"{inputs['efficiency_ratio']*100:.0f}%"),
        ("IH % (net increase)", f"{inputs['ih_percent']:.0f}%"),
        ("PT zone", inputs["pt_zone"]),
        ("Market price (sellable m²)", fmt_money(inputs["market_price"])),
        ("IH price (sellable m²)", fmt_money(inputs.get("ih_price", IH_PRICE_PER_SELLABLE_M2))),
        ("Hard cost mode", inputs["cost_mode"]),
        ("Hard cost (R/m² bulk)", fmt_money(inputs["const_cost_sqm"]) if inputs["cost_mode"] == "R / m²" else "—"),
        ("Hard cost (%GDV)", f"{inputs['const_cost_pct_gdv']*100:.1f}%" if inputs["cost_mode"] == "% of GDV" else "—"),
        ("%GDV scope", inputs["pct_gdv_scope"] if inputs["cost_mode"] == "% of GDV" else "—"),
        ("Prof fees (total)", fmt_pct(inputs["prof_fee_total"])),
        ("Profit (target)", fmt_pct(inputs["profit_margin"])),
        ("Overlay: heritage enabled", "Yes" if inputs["heritage_enabled"] else "No"),
        ("—", "—"),
        ("Proposed bulk (m²)", f"{res['proposed_bulk']:,.0f}"),
        ("Proposed sellable (m²)", f"{res['proposed_sellable']:,.0f}"),
        ("GDV", fmt_money(res["gdv"])),
        ("Construction costs", fmt_money(res["construction_costs"])),
        ("Development charges (DC)", fmt_money(res["total_dc"])),
        ("Professional fees", fmt_money(res["prof_fees"])),
        ("Developer profit", fmt_money(res["profit"])),
        ("Residual land value (RLV)", fmt_money(res["rlv"])),
        ("DC savings vs full DC", fmt_money(res["dc_savings"])),
    ]
    return pd.DataFrame(rows, columns=["Item", "Value"])


def diff_inputs(current: dict, compare: dict) -> pd.DataFrame:
    keys = [
        "land_area", "existing_gba", "zoning_key", "pt_zone",
        "density_bonus", "ih_percent", "efficiency_ratio",
        "exit_price_source", "selected_suburb", "price_point",
        "market_price", "ih_price",
        "cost_mode", "const_cost_sqm", "const_cost_pct_gdv", "pct_gdv_scope",
        "prof_fee_total", "profit_margin",
        "heritage_enabled", "heritage_bonus_suppression", "heritage_cost_uplift", "heritage_fees_uplift", "heritage_profit_uplift",
    ]
    rows = []
    for k in keys:
        a = current.get(k)
        b = compare.get(k)
        if a == b:
            continue
        def norm(x):
            if isinstance(x, float):
                return round(x, 6)
            return x
        rows.append([k, norm(b), norm(a)])
    return pd.DataFrame(rows, columns=["Parameter", "Compare scenario", "Current"])


# =========================
# STATE INIT
# =========================
def default_inputs() -> dict:
    db = load_exit_price_db()
    suburbs = sorted(db["suburb"].tolist())
    default_suburb = suburbs[0] if suburbs else ""
    price_point = "Mid"
    auto_price, _, _ = compute_market_price_from_db(db, default_suburb, price_point)
    if auto_price is None:
        auto_price = 35000.0

    fee_defaults = default_prof_fee_components_scaled_to_target()
    prof_total = sum(fee_defaults.values())

    return {
        # Site
        "land_area": 1000.0,
        "existing_gba": 200.0,
        "zoning_key": "GR2 (Suburban)",
        "pt_zone": "Standard",

        # Policy
        "ih_percent": 20.0,
        "density_bonus": 20.0,

        # Benchmarks
        "efficiency_ratio": 0.85,
        "profit_margin": 0.20,

        # Prices
        "exit_price_source": "Suburb database",
        "selected_suburb": default_suburb,
        "price_point": price_point,
        "market_price": float(auto_price),
        "ih_price": float(IH_PRICE_PER_SELLABLE_M2),
        "override_market_price": False,

        # Fees
        "prof_fee_components": fee_defaults,
        "prof_fee_total": float(prof_total),

        # Cost input
        "build_tier": "Mid-Tier (R18,000/m²)",
        "cost_mode": "R / m²",
        "const_cost_sqm": float(COST_TIERS["Mid-Tier (R18,000/m²)"]),
        "const_cost_pct_gdv": 0.50,
        "pct_gdv_scope": "Hard cost only",

        # Heritage overlay
        "heritage_enabled": False,
        "heritage_bonus_suppression": 50.0,
        "heritage_cost_uplift": 8.0,
        "heritage_fees_uplift": 5.0,
        "heritage_profit_uplift": 5.0,
    }


if "scenarios" not in st.session_state:
    base = default_inputs()
    st.session_state.scenarios = {
        "Base": deepcopy(base),
        "Scenario A": deepcopy(base),
        "Scenario B": deepcopy(base),
    }

if "active_inputs" not in st.session_state:
    st.session_state.active_inputs = deepcopy(st.session_state.scenarios["Base"])

if "active_scenario" not in st.session_state:
    st.session_state.active_scenario = "Base"

if "pending_load" not in st.session_state:
    st.session_state.pending_load = None

if st.session_state.pending_load:
    nm = st.session_state.pending_load
    if nm in st.session_state.scenarios:
        st.session_state.active_inputs = deepcopy(st.session_state.scenarios[nm])
        st.session_state.active_scenario = nm
    st.session_state.pending_load = None
    st.rerun()

db = load_exit_price_db()

# =========================
# OPTION C LAYOUT
# =========================
left, mid, right = st.columns([0.95, 1.45, 0.95])

# -------------------------
# LEFT: INPUTS (FORM)
# -------------------------
with left:
    st.subheader("Inputs")

    with st.form("inputs_form", border=True):
        inputs = deepcopy(st.session_state.active_inputs)

        with st.expander("1) Site", expanded=True):
            inputs["land_area"] = st.number_input("Land Area (m²)", min_value=0.0, value=float(inputs["land_area"]), step=50.0)
            inputs["existing_gba"] = st.number_input("Existing GBA (m² bulk)", min_value=0.0, value=float(inputs["existing_gba"]), step=25.0)
            inputs["zoning_key"] = st.selectbox("Zoning preset", list(ZONING_PRESETS.keys()), index=list(ZONING_PRESETS.keys()).index(inputs["zoning_key"]))
            inputs["pt_zone"] = st.selectbox("PT zone (roads DC discount)", ["Standard", "PT1", "PT2"], index=["Standard", "PT1", "PT2"].index(inputs["pt_zone"]))

        with st.expander("2) Policy", expanded=True):
            inputs["ih_percent"] = float(st.slider("Inclusionary Housing (% of net increase)", 0, 30, int(inputs["ih_percent"])))
            inputs["density_bonus"] = float(st.slider("Density bonus (%)", 0, 50, int(inputs["density_bonus"])))

        with st.expander("3) 2026 Benchmarks", expanded=True):
            inputs["efficiency_ratio"] = float(st.slider("Efficiency ratio (sellable ÷ bulk)", 0.60, 0.95, float(inputs["efficiency_ratio"]), step=0.01))
            inputs["profit_margin"] = st.slider("Developer profit (% of GDV)", 10, 30, int(round(inputs["profit_margin"] * 100))) / 100.0

        with st.expander("4) Exit prices (2026)", expanded=True):
            inputs["exit_price_source"] = st.radio("Exit price source", ["Suburb database", "Manual entry"], index=0 if inputs["exit_price_source"] == "Suburb database" else 1)
            suburbs = sorted(db["suburb"].dropna().astype(str).unique().tolist())

            if inputs["exit_price_source"] == "Suburb database":
                if inputs.get("selected_suburb") not in suburbs and suburbs:
                    inputs["selected_suburb"] = suburbs[0]
                inputs["selected_suburb"] = st.selectbox("Suburb group", suburbs, index=suburbs.index(inputs["selected_suburb"]) if inputs.get("selected_suburb") in suburbs else 0)
                inputs["price_point"] = st.radio("Point in range", ["Low", "Mid", "High"], index=["Low", "Mid", "High"].index(inputs.get("price_point", "Mid")), horizontal=True)

                auto_price, mn, mx = compute_market_price_from_db(db, inputs["selected_suburb"], inputs["price_point"])
                if auto_price is None:
                    auto_price, mn, mx = float(inputs.get("market_price", 35000.0)), None, None

                inputs["override_market_price"] = st.checkbox("Override suburb price", value=bool(inputs.get("override_market_price", False)))

                if inputs["override_market_price"]:
                    inputs["market_price"] = st.number_input("Market exit price (R/sellable m²)", min_value=0.0, value=float(inputs.get("market_price", auto_price)), step=500.0)
                else:
                    inputs["market_price"] = float(auto_price)
                    if mn is not None and mx is not None:
                        st.caption(f"Auto: R {auto_price:,.0f}/m² ({inputs['price_point']}) from R {mn:,.0f}–R {mx:,.0f}")
            else:
                inputs["market_price"] = st.number_input("Market exit price (R/sellable m²)", min_value=0.0, value=float(inputs.get("market_price", 35000.0)), step=500.0)

            inputs["ih_price"] = st.number_input("IH exit price (R/sellable m²)", min_value=0.0, value=float(inputs.get("ih_price", IH_PRICE_PER_SELLABLE_M2)), step=500.0)

        with st.expander("5) Construction costs", expanded=True):
            inputs["build_tier"] = st.selectbox("Hard cost tier (benchmark)", list(COST_TIERS.keys()), index=list(COST_TIERS.keys()).index(inputs["build_tier"]))
            tier_default = float(COST_TIERS[inputs["build_tier"]])

            inputs["cost_mode"] = st.radio("Cost input mode", ["R / m²", "% of GDV"], index=0 if inputs["cost_mode"] == "R / m²" else 1)

            if inputs["cost_mode"] == "R / m²":
                inputs["const_cost_sqm"] = st.number_input("Hard construction cost (R/bulk m²)", min_value=0.0, value=float(inputs.get("const_cost_sqm", tier_default)), step=250.0)
                snap = st.checkbox("Snap to tier default", value=False)
                if snap:
                    inputs["const_cost_sqm"] = tier_default
                inputs["const_cost_pct_gdv"] = float(inputs.get("const_cost_pct_gdv", 0.50))
                inputs["pct_gdv_scope"] = inputs.get("pct_gdv_scope", "Hard cost only")
            else:
                inputs["const_cost_pct_gdv"] = st.slider("Construction cost (%GDV)", 10, 90, int(round(inputs.get("const_cost_pct_gdv", 0.50) * 100))) / 100.0
                inputs["pct_gdv_scope"] = st.radio("%GDV applies to…", ["Hard cost only", "Hard + soft (includes prof fees)"], index=0 if inputs.get("pct_gdv_scope", "Hard cost only") == "Hard cost only" else 1)
                inputs["const_cost_sqm"] = float(inputs.get("const_cost_sqm", tier_default))

        with st.expander("6) Professional fees (itemised)", expanded=False):
            fee_components = deepcopy(inputs.get("prof_fee_components", default_prof_fee_components_scaled_to_target()))
            total = 0.0
            for name, (lo, hi) in PROF_FEE_RANGES.items():
                dv = clamp(float(fee_components.get(name, (lo + hi) / 2.0)), lo, hi)
                val_pct = st.slider(f"{name} (%)", float(lo * 100), float(hi * 100), float(dv * 100), step=0.1)
                fee_components[name] = val_pct / 100.0
                total += fee_components[name]

            inputs["prof_fee_components"] = fee_components
            inputs["prof_fee_total"] = float(total)
            st.caption(f"Total professional fees: {total*100:.2f}% (guide: ~12–15%)")

        with st.expander("7) Overlays — Built heritage", expanded=False):
            inputs["heritage_enabled"] = st.checkbox("Enable heritage overlay", value=bool(inputs.get("heritage_enabled", False)))
            dis = not inputs["heritage_enabled"]
            inputs["heritage_bonus_suppression"] = float(st.slider("Bonus suppression (%)", 0, 100, int(inputs.get("heritage_bonus_suppression", 50)), disabled=dis))
            inputs["heritage_cost_uplift"] = float(st.slider("Construction uplift (%)", 0, 40, int(inputs.get("heritage_cost_uplift", 8)), disabled=dis))
            inputs["heritage_fees_uplift"] = float(st.slider("Fees uplift (%)", 0, 40, int(inputs.get("heritage_fees_uplift", 5)), disabled=dis))
            inputs["heritage_profit_uplift"] = float(st.slider("Profit uplift (%)", 0, 40, int(inputs.get("heritage_profit_uplift", 5)), disabled=dis))

        apply_clicked = st.form_submit_button("✅ Apply inputs", use_container_width=True)

    if apply_clicked:
        st.session_state.active_inputs = deepcopy(inputs)
        st.toast("Inputs applied", icon="✅")
        st.rerun()

    # ---- Exit DB management (OUTSIDE the form) ----
    with st.expander("📦 Exit price database (upload / reset)", expanded=False):
        up = st.file_uploader("Upload CSV (suburb + min/max or exit_price_per_m2)", type=["csv"], key="exit_db_uploader")
        if up is not None:
            ok, msg = set_exit_price_db_from_upload(up)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

        if st.button("Reset exit DB to defaults", use_container_width=True):
            st.session_state.exit_price_db = pd.DataFrame(DEFAULT_EXIT_PRICES)
            st.success("Reset exit DB.")
            st.rerun()

        st.dataframe(load_exit_price_db(), use_container_width=True, hide_index=True)


# -------------------------
# MID: OUTPUTS (dashboard)
# -------------------------
with mid:
    active_inputs = deepcopy(st.session_state.active_inputs)
    res = compute_model(active_inputs)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Residual Land Value", fmt_money(res["rlv"]))
    k2.metric("GDV", fmt_money(res["gdv"]))
    k3.metric("Construction", fmt_money(res["construction_costs"]))
    k4.metric("DCs", fmt_money(res["total_dc"]))
    k5.metric("DC savings", fmt_money(res["dc_savings"]))

    flags = []
    if res["brownfield_credit"]:
        flags.append("Existing bulk exceeds proposed bulk (brownfield credit / net increase = 0).")
    if res["adj_profit_rate"] < 0.20:
        flags.append("Profit < 20% of GDV (may be unbankable).")
    if res["efficiency"] < 0.80:
        flags.append("Efficiency < 80% (design risk / sellable loss).")
    if flags:
        st.warning(" • " + "\n • ".join(flags))

    chips = [
        f"Zoning: {active_inputs['zoning_key']}",
        f"Efficiency: {res['efficiency']*100:.0f}%",
        f"IH: {active_inputs['ih_percent']:.0f}%",
        f"Bonus (input→effective): {active_inputs['density_bonus']:.0f}% → {res['adj_bonus']:.1f}%",
        f"Profit: {res['adj_profit_rate']*100:.1f}%",
        f"Fees: {res['adj_fee_rate']*100:.2f}%",
        f"PT: {active_inputs['pt_zone']}",
    ]
    st.caption(" | ".join(chips))

    wf = go.Figure(go.Waterfall(
        name="Residual breakdown",
        orientation="v",
        measure=["relative", "relative", "relative", "relative", "relative", "total"],
        x=["GDV", "Construction", "DCs", "Prof Fees", "Profit", "Residual Land"],
        y=[
            res["gdv"],
            -res["construction_costs"],
            -res["total_dc"],
            -res["prof_fees"],
            -res["profit"],
            res["rlv"],
        ],
        connector={"line": {"color": "rgb(80,80,80)"}},
    ))
    wf.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420)
    st.plotly_chart(wf, use_container_width=True)

    st.subheader("Sensitivity (IH % vs Density Bonus)")
    ih_levels = [0, 10, 20, 30]
    bonus_levels = [0, 20, 40]

    z = []
    text = []
    for ih in ih_levels:
        row = []
        row_text = []
        for b in bonus_levels:
            tmp_in = deepcopy(active_inputs)
            tmp_in["ih_percent"] = float(ih)
            tmp_in["density_bonus"] = float(b)
            tmp = compute_model(tmp_in)
            row.append(tmp["rlv"] / 1_000_000)
            row_text.append(f"R {tmp['rlv']/1_000_000:.1f}M")
        z.append(row)
        text.append(row_text)

    hm = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"{b}% Bonus" for b in bonus_levels],
        y=[f"{ih}% IH" for ih in ih_levels],
        text=text,
        hovertemplate="%{y} / %{x}<br>%{text}<extra></extra>",
        showscale=True,
    ))
    hm.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(hm, use_container_width=True)


# -------------------------
# RIGHT: SCENARIOS + AUDIT + MAP
# -------------------------
with right:
    st.subheader("Scenarios & Audit")

    scenarios = st.session_state.scenarios
    scenario_names = list(scenarios.keys())

    sel = st.selectbox("Scenario", scenario_names, index=scenario_names.index(st.session_state.active_scenario))

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Load", use_container_width=True):
            st.session_state.pending_load = sel
            st.rerun()
    with b2:
        if st.button("Save", use_container_width=True):
            scenarios[sel] = deepcopy(st.session_state.active_inputs)
            st.session_state.scenarios = scenarios
            st.toast(f"Saved current inputs → {sel}", icon="💾")
    with b3:
        if st.button("Reset", use_container_width=True):
            scenarios[sel] = default_inputs()
            st.session_state.scenarios = scenarios
            st.toast(f"Reset {sel} to defaults", icon="↩️")

    st.markdown("**Compare**")
    compare_to = st.selectbox("Compare current vs", scenario_names, index=scenario_names.index(sel))
    compare_inputs = scenarios[compare_to]
    current_inputs = st.session_state.active_inputs

    df_diff = diff_inputs(current_inputs, compare_inputs)
    if df_diff.empty:
        st.caption("No differences vs selected compare scenario.")
    else:
        st.dataframe(df_diff, use_container_width=True, hide_index=True)

    st.markdown("**Audit trail**")
    active_res = compute_model(current_inputs)
    audit_df = build_audit_table(current_inputs, active_res)
    st.dataframe(audit_df, use_container_width=True, hide_index=True)

    with st.expander("🗺️ City Map Viewer", expanded=False):
        st.link_button("Open in new tab", CITYMAP_VIEWER_URL)
        components.iframe(CITYMAP_VIEWER_URL, height=420, scrolling=True)
