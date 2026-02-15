import os
import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass

# =========================
# CITY MAP VIEWER
# =========================
DEFAULT_CITYMAP_URL = "https://citymaps.capetown.gov.za/EGISViewer/"
CITYMAP_VIEWER_URL = st.secrets.get("CITYMAP_VIEWER_URL", DEFAULT_CITYMAP_URL)

# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="Residuō | Cape Town Feasibility", layout="wide")

# =========================
# ASSETS / LOGO
# =========================
ASSETS_DIR = st.secrets.get("ASSETS_DIR", "assets")
LOGO_CANDIDATES = [
    os.path.join(ASSETS_DIR, "residuo_brand.png"),
    os.path.join(ASSETS_DIR, "residuo_D_wordmark_transparent_darktext_1200w_clean.png"),
    os.path.join(ASSETS_DIR, "residuo_D_wordmark_transparent_lighttext_1200w_clean.png"),
    os.path.join(ASSETS_DIR, "residuo_icon.png"),
    os.path.join(ASSETS_DIR, "residuo_D_icon_256.png"),
]
def pick_logo_path() -> str | None:
    for p in LOGO_CANDIDATES:
        if os.path.exists(p):
            return p
    return None

LOGO_PATH = pick_logo_path()

# =========================
# CONFIG DATA
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

# 2026 benchmarks
IH_PRICE_PER_M2 = 15000  # capped IH exit price (assumption)
COST_TIERS = {
    "Economic (R10,000/m²)": 10000.0,
    "Mid-Tier (R18,000/m²)": 18000.0,
    "Luxury (R25,000+/m²)": 25000.0,
}

# Exit prices: 2026 sectional title new apartments (R/m² sellable) — ranges
DEFAULT_EXIT_PRICES = [
    {"suburb": "Clifton / Bantry Bay", "min_price_per_m2": 120000, "max_price_per_m2": 170000},  # assumed max
    {"suburb": "Sea Point / Green Point", "min_price_per_m2": 65000, "max_price_per_m2": 85000},
    {"suburb": "City Bowl (CBD / Gardens)", "min_price_per_m2": 45000, "max_price_per_m2": 60000},
    {"suburb": "Claremont / Rondebosch", "min_price_per_m2": 40000, "max_price_per_m2": 52000},
    {"suburb": "Woodstock / Salt River", "min_price_per_m2": 32000, "max_price_per_m2": 42000},
    {"suburb": "Durbanville / Sunningdale", "min_price_per_m2": 25000, "max_price_per_m2": 35000},
    {"suburb": "Khayelitsha / Mitchells Plain", "min_price_per_m2": 10000, "max_price_per_m2": 15000},
]

# Professional fee ranges (from your 2026 image) — as fractions of 1.0
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
# OVERLAYS
# =========================
@dataclass(frozen=True)
class HeritageOverlay:
    enabled: bool
    bonus_suppression_pct: float   # suppresses density bonus
    cost_uplift_pct: float         # uplifts construction input (R/m² or %GDV)
    fees_uplift_pct: float         # uplifts total professional fee rate
    profit_uplift_pct: float       # uplifts profit % of GDV


def apply_heritage_overlay(
    density_bonus_pct: float,
    base_cost_value: float,
    base_fees_rate: float,
    base_profit_rate: float,
    overlay: HeritageOverlay,
) -> tuple[float, float, float, float]:
    """Returns (adj_bonus_pct, adj_cost_value, adj_fees_rate, adj_profit_rate)."""
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
def fmt_money(x: float) -> str:
    return f"R {x:,.0f}"

def fmt_money2(x: float) -> str:
    return f"R {x:,.2f}"

def fmt_pct(x: float, dp: int = 1) -> str:
    return f"{x*100:.{dp}f}%"

def pt_discount(pt_zone_value: str) -> float:
    return {"PT1": 0.8, "PT2": 0.5}.get(pt_zone_value, 1.0)

def normalize_exit_price_db(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures DB has: suburb, min_price_per_m2, max_price_per_m2
    Migrates old schema: suburb, exit_price_per_m2 -> min=max.
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

    swap_mask = df["min_price_per_m2"] > df["max_price_per_m2"]
    if swap_mask.any():
        df.loc[swap_mask, ["min_price_per_m2", "max_price_per_m2"]] = df.loc[
            swap_mask, ["max_price_per_m2", "min_price_per_m2"]
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

def compute_model(
    land_area_m2: float,
    existing_bulk_m2: float,
    ff: float,
    density_bonus_pct: float,
    efficiency_ratio: float,
    ih_pct: float,
    pt_zone_value: str,
    market_price_sellable_m2: float,
    ih_price_sellable_m2: float,
    profit_pct_gdv: float,
    prof_fee_rate_total: float,
    overlay: HeritageOverlay,
    cost_mode: str,            # "R / m²" or "% of GDV"
    base_cost_sqm: float,      # if R / m², applies to BULK
    base_cost_pct_gdv: float,  # if %GDV
    pct_gdv_scope: str,        # "Hard cost only" or "Hard + soft"
):
    base_bulk = land_area_m2 * ff

    cost_input = base_cost_sqm if cost_mode == "R / m²" else base_cost_pct_gdv
    adj_bonus_pct, adj_cost_input, adj_fee_rate, adj_profit_rate = apply_heritage_overlay(
        density_bonus_pct=density_bonus_pct,
        base_cost_value=cost_input,
        base_fees_rate=prof_fee_rate_total,
        base_profit_rate=profit_pct_gdv,
        overlay=overlay,
    )

    proposed_bulk = base_bulk * (1.0 + adj_bonus_pct / 100.0)
    proposed_sellable = proposed_bulk * efficiency_ratio

    # Net increase (bulk) for DC logic
    net_increase_bulk = max(0.0, proposed_bulk - existing_bulk_m2)

    # IH requirement applied to net increase bulk; revenue uses sellable
    ih_increase_bulk = net_increase_bulk * (ih_pct / 100.0)
    market_increase_bulk = net_increase_bulk - ih_increase_bulk

    ih_sellable = ih_increase_bulk * efficiency_ratio
    market_sellable = max(0.0, proposed_sellable - ih_sellable)

    # DCs on market share of net increase bulk; PT discount on roads
    disc = pt_discount(pt_zone_value)
    roads_dc = market_increase_bulk * ROADS_TRANSPORT_PORTION * disc
    other_dc = market_increase_bulk * (DC_BASE_RATE - ROADS_TRANSPORT_PORTION)
    total_dc = roads_dc + other_dc

    total_potential_dc = net_increase_bulk * DC_BASE_RATE
    dc_savings = total_potential_dc - total_dc

    # GDV on sellable
    gdv = (market_sellable * market_price_sellable_m2) + (ih_sellable * ih_price_sellable_m2)

    # Construction + fees
    adj_cost_sqm = None
    adj_cost_pct = None

    if cost_mode == "R / m²":
        adj_cost_sqm = adj_cost_input
        construction_costs = proposed_bulk * adj_cost_sqm
        hard_plus_dc = construction_costs + total_dc
        prof_fees = hard_plus_dc * adj_fee_rate
    else:
        adj_cost_pct = adj_cost_input
        if pct_gdv_scope == "Hard cost only":
            construction_costs = gdv * adj_cost_pct
            hard_plus_dc = construction_costs + total_dc
            prof_fees = hard_plus_dc * adj_fee_rate
        else:
            # all-in (construction + prof fees) = p*GDV, where fees = fee_rate*(construction + DC)
            target_all_in = gdv * adj_cost_pct
            construction_costs = (target_all_in - (adj_fee_rate * total_dc)) / (1.0 + adj_fee_rate)
            construction_costs = max(0.0, construction_costs)
            hard_plus_dc = construction_costs + total_dc
            prof_fees = hard_plus_dc * adj_fee_rate

    profit = gdv * adj_profit_rate
    rlv = gdv - (construction_costs + total_dc + prof_fees + profit)

    implied_cost_sqm = (construction_costs / proposed_bulk) if proposed_bulk > 0 else 0.0

    return {
        "adj_bonus_pct": float(adj_bonus_pct),
        "proposed_bulk": float(proposed_bulk),
        "proposed_sellable": float(proposed_sellable),
        "net_increase_bulk": float(net_increase_bulk),
        "market_sellable": float(market_sellable),
        "ih_sellable": float(ih_sellable),
        "gdv": float(gdv),
        "construction_costs": float(construction_costs),
        "total_dc": float(total_dc),
        "dc_savings": float(dc_savings),
        "prof_fees": float(prof_fees),
        "profit": float(profit),
        "rlv": float(rlv),
        "fee_rate": float(adj_fee_rate),
        "profit_rate": float(adj_profit_rate),
        "cost_mode": cost_mode,
        "pct_gdv_scope": pct_gdv_scope,
        "adj_cost_sqm": None if adj_cost_sqm is None else float(adj_cost_sqm),
        "adj_cost_pct": None if adj_cost_pct is None else float(adj_cost_pct),
        "implied_cost_sqm": float(implied_cost_sqm),
        "efficiency_ratio": float(efficiency_ratio),
        "brownfield_credit": bool(existing_bulk_m2 > proposed_bulk),
    }

def build_audit(res: dict, inputs: dict) -> pd.DataFrame:
    rows = []

    def add(section: str, item: str, value: str):
        rows.append({"Section": section, "Item": item, "Value": value})

    # Areas
    add("Areas", "Land area", f"{inputs['land_area']:,.0f} m²")
    add("Areas", "Existing bulk (GBA)", f"{inputs['existing_bulk']:,.0f} m²")
    add("Areas", "Zoning preset", inputs["zoning_key"])
    add("Areas", "Floor factor (FF)", f"{inputs['ff']:.2f}")
    add("Areas", "Density bonus (input)", f"{inputs['density_bonus']:.1f}%")
    add("Areas", "Density bonus (effective)", f"{res['adj_bonus_pct']:.1f}%")
    add("Areas", "Proposed bulk", f"{res['proposed_bulk']:,.0f} m²")
    add("Areas", "Efficiency ratio", f"{res['efficiency_ratio']*100:.0f}%")
    add("Areas", "Sellable area", f"{res['proposed_sellable']:,.0f} m²")

    # Revenue
    add("Revenue", "Market sellable", f"{res['market_sellable']:,.0f} m²")
    add("Revenue", "IH sellable", f"{res['ih_sellable']:,.0f} m²")
    add("Revenue", "Market price (sellable)", fmt_money(inputs["market_price"]))
    add("Revenue", "IH price (sellable)", fmt_money(inputs["ih_price"]))
    add("Revenue", "GDV", fmt_money(res["gdv"]))

    # DCs
    add("DCs", "PT zone", inputs["pt_zone"])
    add("DCs", "Net increase (bulk)", f"{res['net_increase_bulk']:,.0f} m²")
    add("DCs", "Total DC payable", fmt_money(res["total_dc"]))
    add("DCs", "DC savings vs full", fmt_money(res["dc_savings"]))

    # Costs
    add("Costs", "Construction mode", inputs["cost_mode"])
    if res["cost_mode"] == "R / m²":
        add("Costs", "Hard cost input", f"{fmt_money(res['adj_cost_sqm'])} / bulk m²")
    else:
        add("Costs", "%GDV scope", inputs["pct_gdv_scope"])
        add("Costs", "Construction input", f"{res['adj_cost_pct']*100:.1f}% of GDV")
        add("Costs", "Implied hard cost", f"{fmt_money(res['implied_cost_sqm'])} / bulk m²")
    add("Costs", "Construction costs", fmt_money(res["construction_costs"]))

    # Fees & Profit
    add("Profit", "Professional fees rate", fmt_pct(res["fee_rate"], 2))
    add("Profit", "Professional fees (value)", fmt_money(res["prof_fees"]))
    add("Profit", "Developer profit rate", fmt_pct(res["profit_rate"], 1))
    add("Profit", "Developer profit (value)", fmt_money(res["profit"]))

    # RLV
    add("Land", "Residual land value", fmt_money(res["rlv"]))

    return pd.DataFrame(rows)

def risk_flags(res: dict, inputs: dict) -> list[str]:
    flags = []
    if res["brownfield_credit"]:
        flags.append("Brownfield credit: existing bulk exceeds proposed bulk (DCs may be zero).")
    if inputs["profit_margin"] < 0.20:
        flags.append("Profit < 20%: SA lenders often reject lower profit targets.")
    if inputs["efficiency_ratio"] < 0.80:
        flags.append("Efficiency < 80%: design/layout risk; GDV sensitivity increases.")
    if inputs["heritage_enabled"]:
        flags.append("Heritage overlay active: bonus suppressed and risk uplifts applied.")
    if inputs["ih_percent"] >= 25:
        flags.append("High IH%: check market cross-subsidy and absorption.")
    return flags

# =========================
# OPTION A — DASHBOARD FIRST
# =========================

# Header + logo
hcol1, hcol2 = st.columns([0.25, 0.75], vertical_alignment="center")
with hcol1:
    if LOGO_PATH:
        st.image(LOGO_PATH, use_container_width=True)
with hcol2:
    st.markdown("## Residuō — Cape Town Feasibility")
    st.caption("Dashboard-first layout • Quick profile → KPIs → Waterfall → Tabs (Inputs / Sensitivity / Audit / Data)")

st.divider()

# Quick profile row (top)
db = load_exit_price_db()
suburb_list = sorted(db["suburb"].dropna().astype(str).unique().tolist())
if not suburb_list:
    suburb_list = ["(no suburb data loaded)"]

q1, q2, q3, q4, q5 = st.columns([1.3, 1.0, 1.0, 0.9, 0.9], vertical_alignment="bottom")

with q1:
    erf_ref = st.text_input("Property / Erf / Reference", value=st.session_state.get("erf_ref", ""))
    st.session_state["erf_ref"] = erf_ref

with q2:
    selected_suburb = st.selectbox("Suburb group (exit prices)", suburb_list, index=0)

with q3:
    price_point = st.radio("Exit price point", ["Low", "Mid", "High"], horizontal=True, index=1)

with q4:
    zoning_key = st.selectbox("Zoning preset", list(ZONING_PRESETS.keys()), index=0)

with q5:
    pt_zone = st.selectbox("PT zone", ["Standard", "PT1", "PT2"], index=0)

# Resolve exit price from DB
row = db.loc[db["suburb"] == selected_suburb] if "suburb" in db.columns else pd.DataFrame()
db_min = db_max = db_price = None
if not row.empty:
    db_min = float(row["min_price_per_m2"].iloc[0])
    db_max = float(row["max_price_per_m2"].iloc[0])
    if price_point == "Low":
        db_price = db_min
    elif price_point == "High":
        db_price = db_max
    else:
        db_price = (db_min + db_max) / 2.0

# Tabs (inputs live there, but we need default values now for model run)
# We'll keep defaults in session_state so the dashboard can run immediately.
def ss_get(k, default):
    if k not in st.session_state:
        st.session_state[k] = default
    return st.session_state[k]

# Core defaults
land_area = ss_get("land_area", 1000.0)
existing_bulk = ss_get("existing_bulk", 200.0)
ih_percent = ss_get("ih_percent", 20)
density_bonus = ss_get("density_bonus", 20)
efficiency_ratio = ss_get("efficiency_ratio", 0.85)
profit_margin = ss_get("profit_margin", 0.20)

# Construction defaults (2026 tier)
build_tier = ss_get("build_tier", "Mid-Tier (R18,000/m²)")
if build_tier not in COST_TIERS:
    build_tier = "Mid-Tier (R18,000/m²)"
tier_cost_default = COST_TIERS[build_tier]

cost_mode = ss_get("cost_mode", "R / m²")
pct_gdv_scope = ss_get("pct_gdv_scope", "Hard cost only")
const_cost_sqm = ss_get("const_cost_sqm", float(tier_cost_default))
const_cost_pct_gdv = ss_get("const_cost_pct_gdv", 0.50)

# Market price (override allowed in Inputs tab)
market_price = ss_get("market_price", float(db_price) if db_price is not None else 35000.0)
override_price = ss_get("override_price", False)
if (db_price is not None) and (not override_price):
    market_price = float(db_price)

# Professional fees (defaults scaled to target total)
default_fee_components = default_prof_fee_components_scaled_to_target()
fee_components = ss_get("fee_components", default_fee_components)
if not isinstance(fee_components, dict) or not fee_components:
    fee_components = default_fee_components
base_prof_fee_rate = sum(float(v) for v in fee_components.values())

# Heritage overlay defaults
heritage_enabled = ss_get("heritage_enabled", False)
heritage_bonus_suppression = ss_get("heritage_bonus_suppression", 50.0)
heritage_cost_uplift = ss_get("heritage_cost_uplift", 8.0)
heritage_fees_uplift = ss_get("heritage_fees_uplift", 5.0)
heritage_profit_uplift = ss_get("heritage_profit_uplift", 5.0)

heritage_overlay = HeritageOverlay(
    enabled=bool(heritage_enabled),
    bonus_suppression_pct=float(heritage_bonus_suppression),
    cost_uplift_pct=float(heritage_cost_uplift),
    fees_uplift_pct=float(heritage_fees_uplift),
    profit_uplift_pct=float(heritage_profit_uplift),
)

# Model run
ff = ZONING_PRESETS[zoning_key]["ff"]
res = compute_model(
    land_area_m2=float(land_area),
    existing_bulk_m2=float(existing_bulk),
    ff=float(ff),
    density_bonus_pct=float(density_bonus),
    efficiency_ratio=float(efficiency_ratio),
    ih_pct=float(ih_percent),
    pt_zone_value=pt_zone,
    market_price_sellable_m2=float(market_price),
    ih_price_sellable_m2=float(IH_PRICE_PER_M2),
    profit_pct_gdv=float(profit_margin),
    prof_fee_rate_total=float(base_prof_fee_rate),
    overlay=heritage_overlay,
    cost_mode=cost_mode,
    base_cost_sqm=float(const_cost_sqm),
    base_cost_pct_gdv=float(const_cost_pct_gdv),
    pct_gdv_scope=pct_gdv_scope,
)

# KPI strip (Row 1)
k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
k1.metric("Residual Land Value", fmt_money2(res["rlv"]))
k2.metric("RLV / m² land", fmt_money2(res["rlv"] / land_area) if land_area > 0 else "—")
k3.metric("GDV", fmt_money(res["gdv"]))
k4.metric("Profit", fmt_money(res["profit"]))
k5.metric("Total DC", fmt_money(res["total_dc"]))
k6.metric("DC Savings", fmt_money(res["dc_savings"]))
k7.metric("Buildable Bulk", f"{res['proposed_bulk']:,.0f} m²")

st.divider()

# Row 2: Waterfall + Assumptions & flags
left, right = st.columns([0.62, 0.38], vertical_alignment="top")

with left:
    st.subheader("Residual breakdown")
    fig = go.Figure(go.Waterfall(
        name="RLV Breakdown",
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
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=430)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Assumptions snapshot")

    # compact “chips”
    chip_css = """
    <style>
      .chipwrap{display:flex;flex-wrap:wrap;gap:8px;margin-top:6px}
      .chip{padding:6px 10px;border-radius:16px;border:1px solid rgba(0,0,0,0.15);font-size:12px}
      .muted{opacity:0.75}
    </style>
    """
    st.markdown(chip_css, unsafe_allow_html=True)

    chips = [
        f"Zoning: <b>{zoning_key}</b>",
        f"FF: <b>{ff:.2f}</b>",
        f"Efficiency: <b>{efficiency_ratio*100:.0f}%</b>",
        f"IH: <b>{ih_percent:.0f}%</b>",
        f"Bonus: <b>{density_bonus:.0f}%</b> → <b>{res['adj_bonus_pct']:.1f}%</b>",
        f"Exit price: <b>{fmt_money(market_price)}/m²</b> <span class='muted'>({price_point})</span>",
        f"Hard cost: <b>{build_tier}</b>",
        f"Profit: <b>{profit_margin*100:.0f}%</b>",
        f"Fees: <b>{base_prof_fee_rate*100:.2f}%</b>",
        f"PT: <b>{pt_zone}</b>",
    ]
    st.markdown(
        "<div class='chipwrap'>" + "".join([f"<div class='chip'>{c}</div>" for c in chips]) + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Flags")
    flags = risk_flags(
        res,
        {
            "profit_margin": profit_margin,
            "efficiency_ratio": efficiency_ratio,
            "heritage_enabled": heritage_enabled,
            "ih_percent": ih_percent,
        },
    )
    if not flags:
        st.success("No red flags triggered by current assumptions.")
    else:
        for f in flags:
            st.warning(f)

# =========================
# Row 3: Tabs
# =========================
tab_inputs, tab_sens, tab_audit, tab_data, tab_notes = st.tabs(
    ["Inputs", "Sensitivity", "Audit trail", "Exit prices", "Notes"]
)

with tab_inputs:
    st.subheader("Inputs")

    c1, c2, c3 = st.columns([1.2, 1.2, 1.0], vertical_alignment="top")

    with c1:
        with st.expander("Site", expanded=True):
            st.session_state["land_area"] = st.number_input("Land area (m²)", value=float(land_area), min_value=0.0, step=50.0)
            st.session_state["existing_bulk"] = st.number_input("Existing GBA (m² bulk)", value=float(existing_bulk), min_value=0.0, step=25.0)

        with st.expander("Policy", expanded=True):
            st.session_state["ih_percent"] = st.slider("Inclusionary Housing (%)", 0, 30, int(ih_percent))
            st.session_state["density_bonus"] = st.slider("Density Bonus (%)", 0, 50, int(density_bonus))
            st.session_state["efficiency_ratio"] = st.slider("Efficiency ratio (sellable ÷ bulk)", 0.60, 0.95, float(efficiency_ratio), step=0.01)

    with c2:
        with st.expander("Exit price (suburb database)", expanded=True):
            st.session_state["override_price"] = st.checkbox("Override suburb exit price", value=bool(override_price))
            if st.session_state["override_price"] or db_price is None:
                st.session_state["market_price"] = st.number_input(
                    "Market sales price (R per sellable m²)", value=float(market_price), min_value=0.0, step=500.0
                )
            else:
                st.session_state["market_price"] = float(db_price)
                st.caption(f"Auto from DB: {fmt_money(db_price)} / m² ({price_point})")
            st.caption(f"IH exit price (fixed): {fmt_money(IH_PRICE_PER_M2)} / m²")

        with st.expander("Profit & fees", expanded=True):
            st.session_state["profit_margin"] = st.slider("Developer profit (% of GDV)", 10, 30, int(round(profit_margin * 100))) / 100.0

            st.markdown("**Professional fee components (sum used in model)**")
            new_components = {}
            for name, (lo, hi) in PROF_FEE_RANGES.items():
                default_v = float(fee_components.get(name, (lo + hi) / 2.0))
                new_components[name] = st.slider(
                    f"{name} (%)",
                    min_value=float(lo * 100),
                    max_value=float(hi * 100),
                    value=float(default_v * 100),
                    step=0.1,
                ) / 100.0
            st.session_state["fee_components"] = new_components
            st.caption(f"Total prof fees: {sum(new_components.values())*100:.2f}% (image ~12–15%)")

    with c3:
        with st.expander("Construction costs", expanded=True):
            st.session_state["build_tier"] = st.selectbox("2026 hard cost tier", list(COST_TIERS.keys()), index=list(COST_TIERS.keys()).index(build_tier))
            tier_cost_default_now = COST_TIERS[st.session_state["build_tier"]]

            st.session_state["cost_mode"] = st.radio("Cost input mode", ["R / m²", "% of GDV"], index=0 if cost_mode == "R / m²" else 1)

            if st.session_state["cost_mode"] == "R / m²":
                st.session_state["const_cost_sqm"] = st.number_input(
                    "Hard construction cost (R per bulk m²)",
                    value=float(st.session_state.get("const_cost_sqm", tier_cost_default_now)),
                    min_value=0.0,
                    step=250.0,
                )
                st.session_state["const_cost_pct_gdv"] = float(st.session_state.get("const_cost_pct_gdv", 0.50))
                st.session_state["pct_gdv_scope"] = "Hard cost only"
            else:
                st.session_state["pct_gdv_scope"] = st.radio("%GDV applies to…", ["Hard cost only", "Hard + soft (includes prof fees)"], index=0 if pct_gdv_scope == "Hard cost only" else 1)
                st.session_state["const_cost_pct_gdv"] = st.slider("Construction cost (% of GDV)", 10, 90, int(round(const_cost_pct_gdv * 100))) / 100.0
                st.session_state["const_cost_sqm"] = float(st.session_state.get("const_cost_sqm", tier_cost_default_now))

        with st.expander("Overlays", expanded=False):
            st.session_state["heritage_enabled"] = st.checkbox("Enable Built Heritage Overlay", value=bool(heritage_enabled))
            st.session_state["heritage_bonus_suppression"] = st.slider("Bonus suppression (%)", 0, 100, int(round(heritage_bonus_suppression)))
            st.session_state["heritage_cost_uplift"] = st.slider("Construction uplift (%)", 0, 40, int(round(heritage_cost_uplift)))
            st.session_state["heritage_fees_uplift"] = st.slider("Fees uplift (%)", 0, 40, int(round(heritage_fees_uplift)))
            st.session_state["heritage_profit_uplift"] = st.slider("Profit uplift (%)", 0, 40, int(round(heritage_profit_uplift)))

        with st.expander("City Map Viewer", expanded=False):
            st.caption("If the embed doesn’t load (browser blocking), use the button.")
            st.link_button("Open City Map Viewer", CITYMAP_VIEWER_URL)
            components.iframe(CITYMAP_VIEWER_URL, height=420, scrolling=True)

    st.info("Changes apply immediately. If you want an explicit 'Apply' button workflow, say so and I’ll convert it.")

with tab_sens:
    st.subheader("Sensitivity Analysis — IH % vs Density Bonus")

    ih_levels = [0, 10, 20, 30]
    bonus_levels = [0, 20, 40]

    z = []
    text = []
    for ih in ih_levels:
        zr = []
        tr = []
        for bonus in bonus_levels:
            tmp = compute_model(
                land_area_m2=float(land_area),
                existing_bulk_m2=float(existing_bulk),
                ff=float(ff),
                density_bonus_pct=float(bonus),
                efficiency_ratio=float(efficiency_ratio),
                ih_pct=float(ih),
                pt_zone_value=pt_zone,
                market_price_sellable_m2=float(market_price),
                ih_price_sellable_m2=float(IH_PRICE_PER_M2),
                profit_pct_gdv=float(profit_margin),
                prof_fee_rate_total=float(sum(st.session_state.get("fee_components", fee_components).values())),
                overlay=heritage_overlay,
                cost_mode=cost_mode,
                base_cost_sqm=float(const_cost_sqm),
                base_cost_pct_gdv=float(const_cost_pct_gdv),
                pct_gdv_scope=pct_gdv_scope,
            )
            rlv_m = tmp["rlv"] / 1_000_000
            zr.append(rlv_m)
            tr.append(f"{rlv_m:.1f}M")
        z.append(zr)
        text.append(tr)

    heat = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[f"{b}% Bonus" for b in bonus_levels],
            y=[f"{i}% IH" for i in ih_levels],
            text=text,
            texttemplate="%{text}",
            hovertemplate="IH: %{y}<br>Bonus: %{x}<br>RLV: R %{z:.2f}M<extra></extra>",
        )
    )
    heat.update_layout(height=450, margin=dict(l=30, r=30, t=30, b=30))
    st.plotly_chart(heat, use_container_width=True)

    # Also show table version (quick copy)
    df_matrix = pd.DataFrame(
        [[f"R {val:.1f}M" for val in row] for row in z],
        index=[f"{i}% IH" for i in ih_levels],
        columns=[f"{b}% Bonus" for b in bonus_levels],
    )
    st.table(df_matrix)

with tab_audit:
    st.subheader("Audit trail (grouped)")

    inputs_for_audit = {
        "land_area": float(land_area),
        "existing_bulk": float(existing_bulk),
        "zoning_key": zoning_key,
        "ff": float(ff),
        "density_bonus": float(density_bonus),
        "efficiency_ratio": float(efficiency_ratio),
        "ih_percent": float(ih_percent),
        "market_price": float(market_price),
        "ih_price": float(IH_PRICE_PER_M2),
        "pt_zone": pt_zone,
        "cost_mode": cost_mode,
        "pct_gdv_scope": pct_gdv_scope,
        "profit_margin": float(profit_margin),
        "heritage_enabled": bool(heritage_enabled),
    }

    audit_df = build_audit(res, inputs_for_audit)

    # Show as sectioned tables
    for section in ["Areas", "Revenue", "DCs", "Costs", "Profit", "Land"]:
        part = audit_df[audit_df["Section"] == section][["Item", "Value"]].reset_index(drop=True)
        st.markdown(f"#### {section}")
        st.dataframe(part, use_container_width=True, hide_index=True)

with tab_data:
    st.subheader("Exit price database")
    st.caption("Upload a CSV if you want your own suburb groups and ranges.")
    uploaded = st.file_uploader("Upload exit price CSV", type=["csv"])
    if uploaded is not None:
        ok, msg = set_exit_price_db_from_upload(uploaded)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    cA, cB = st.columns([1, 1])
    with cA:
        if st.button("Reset to 2026 defaults"):
            st.session_state.exit_price_db = pd.DataFrame(DEFAULT_EXIT_PRICES)
            st.success("Exit prices reset.")

    with cB:
        st.caption("Supported columns: suburb + (min_price_per_m2,max_price_per_m2) OR suburb + exit_price_per_m2")

    st.dataframe(load_exit_price_db(), use_container_width=True)

with tab_notes:
    st.subheader("Notes")
    st.write(
        """
- **Efficiency ratio** applies to *revenue* (sellable area), while **construction and DCs remain bulk-based**.
- **Profit** defaults to **20% of GDV** (typical SA lender threshold).
- **Professional fees** are the **sum of your 2026 line items** (Architect/QS/Engineers/PM).
- If the City Map Viewer embed is blank, the site is likely blocking iframes; use the open button instead.
        """
    )
