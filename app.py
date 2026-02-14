# app.py — Residuo (IH + RLV Calculator) — CLEANED SINGLE FILE
# - Adds brand header + sidebar logo/tagline
# - Safely uses favicon if assets/residuo_icon.png exists
# - Keeps City Map Viewer embed + link fallback
# - Removes accidental duplication and keeps your existing logic intact

import os
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# =========================
# BRANDING (Residuo)
# =========================
TAGLINE = "Unlock Land's True Value"
LOGO_PATH = "assets/residuo_D_wordmark_transparent_darktext_1200w_clean.png"
ICON_PATH = "assets/residuo_D_glyph_transparent_256_clean.png"


def _file_exists(path: str) -> bool:
    try:
        return os.path.isfile(path)
    except Exception:
        return False


# Page config MUST run before any other st.* UI calls
page_icon = ICON_PATH if _file_exists(ICON_PATH) else "🏗️"
st.set_page_config(page_title="Residuo", page_icon=page_icon, layout="wide")

# Optional tighter spacing
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Top brand header
brand_cols = st.columns([3, 7], vertical_alignment="center")
with brand_cols[0]:
    if _file_exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.error(f"Logo not found at: {LOGO_PATH}")
with brand_cols[1]:
    st.markdown(
        f"""
        <div style="padding-top:8px;">
          <h1 style="margin:0; line-height:1.0;">Residuo</h1>
          <div style="font-size:1.05rem; opacity:0.75; margin-top:6px;">{TAGLINE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# Sidebar branding
if _file_exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_container_width=True)
    st.sidebar.caption(TAGLINE)
    st.sidebar.divider()

# =========================
# CITY MAP VIEWER
# =========================
DEFAULT_CITYMAP_URL = "https://citymaps.capetown.gov.za/EGISViewer/"
CITYMAP_VIEWER_URL = st.secrets.get("CITYMAP_VIEWER_URL", DEFAULT_CITYMAP_URL)

with st.expander("🗺️ City of Cape Town Map Viewer", expanded=False):
    st.caption(
        "Embedded viewer (if allowed by the site). If it doesn’t load, use the button below to open it in a new tab."
    )
    st.link_button("Open City Map Viewer", CITYMAP_VIEWER_URL)
    components.iframe(CITYMAP_VIEWER_URL, height=520, scrolling=True)

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

IH_PRICE_PER_M2 = 15000  # IH capped price (assumption)

# 2026 hard cost tiers (benchmarks)
COST_TIERS = {
    "Economic (R10,000/m²)": 10000.0,
    "Mid-Tier (R18,000/m²)": 18000.0,
    "Luxury (R25,000+/m²)": 25000.0,
}

# Exit price DB (2026 sectional title new apartments) — ranges per m²
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

# Professional fee ranges (as %)
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
    bulk_reduction_pct: float  # interpreted as "BONUS suppression %"
    cost_uplift_pct: float  # uplifts construction input (R/m² or %GDV)
    fees_uplift_pct: float  # uplifts total professional fee rate
    profit_uplift_pct: float  # uplifts profit % of GDV


def apply_heritage_overlay(
    density_bonus_pct: float,
    base_cost_value: float,
    base_fees_rate: float,
    base_profit_rate: float,
    overlay: HeritageOverlay,
) -> tuple[float, float, float, float]:
    if not overlay.enabled:
        return density_bonus_pct, base_cost_value, base_fees_rate, base_profit_rate

    # Override density bonus via suppression factor (per your requirement)
    adj_bonus = max(0.0, density_bonus_pct * (1.0 - overlay.bulk_reduction_pct / 100.0))
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
    Ensures DB has: suburb, min_price_per_m2, max_price_per_m2
    Migrates old schema (exit_price_per_m2) -> min=max.
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

    # swap if min > max
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
    """
    Uses midpoints of each range then scales all components so sum equals PROF_FEE_TARGET_TOTAL.
    """
    mids = {k: (v[0] + v[1]) / 2.0 for k, v in PROF_FEE_RANGES.items()}
    s = sum(mids.values())
    if s <= 0:
        return {k: 0.0 for k in mids}
    scale = PROF_FEE_TARGET_TOTAL / s
    return {k: mids[k] * scale for k in mids}


def compute_model(
    land_area_m2: float,
    existing_gba_bulk_m2: float,
    ff: float,
    density_bonus_pct: float,
    efficiency_ratio: float,
    ih_pct: float,
    pt_zone_value: str,
    market_price_per_sellable_m2: float,
    ih_price_per_sellable_m2: float,
    profit_pct_gdv: float,
    base_prof_fee_rate: float,
    overlay: HeritageOverlay,
    cost_mode: str,
    base_cost_sqm: float,
    base_cost_pct_gdv: float,
    pct_gdv_scope: str,
) -> dict:
    base_bulk = land_area_m2 * ff

    cost_input = base_cost_sqm if cost_mode == "R / m²" else base_cost_pct_gdv
    adj_bonus_pct, adj_cost_input, adj_fees_rate, adj_profit_rate = apply_heritage_overlay(
        density_bonus_pct=density_bonus_pct,
        base_cost_value=cost_input,
        base_fees_rate=base_prof_fee_rate,
        base_profit_rate=profit_pct_gdv,
        overlay=overlay,
    )

    proposed_bulk = base_bulk * (1.0 + adj_bonus_pct / 100.0)
    proposed_sellable = proposed_bulk * efficiency_ratio

    net_increase_bulk = max(0.0, proposed_bulk - existing_gba_bulk_m2)

    ih_increase_bulk = net_increase_bulk * (ih_pct / 100.0)
    market_increase_bulk = net_increase_bulk - ih_increase_bulk

    ih_sellable = ih_increase_bulk * efficiency_ratio
    market_sellable = max(0.0, proposed_sellable - ih_sellable)

    disc = pt_discount(pt_zone_value)
    roads_dc = market_increase_bulk * ROADS_TRANSPORT_PORTION * disc
    other_dc = market_increase_bulk * (DC_BASE_RATE - ROADS_TRANSPORT_PORTION)
    total_dc = roads_dc + other_dc

    total_potential_dc = net_increase_bulk * DC_BASE_RATE
    dc_savings = total_potential_dc - total_dc

    gdv = (market_sellable * market_price_per_sellable_m2) + (ih_sellable * ih_price_per_sellable_m2)

    adj_cost_sqm = None
    adj_cost_pct_gdv = None

    if cost_mode == "R / m²":
        adj_cost_sqm = adj_cost_input
        construction_costs = proposed_bulk * adj_cost_sqm
        hard_plus_dc = construction_costs + total_dc
        prof_fees = hard_plus_dc * adj_fees_rate
    else:
        adj_cost_pct_gdv = adj_cost_input

        if pct_gdv_scope == "Hard cost only":
            construction_costs = gdv * adj_cost_pct_gdv
            hard_plus_dc = construction_costs + total_dc
            prof_fees = hard_plus_dc * adj_fees_rate
        else:
            # Solve: construction + fees*(construction + dc) = target_all_in
            target_all_in = gdv * adj_cost_pct_gdv
            construction_costs = (target_all_in - (adj_fees_rate * total_dc)) / (1.0 + adj_fees_rate)
            construction_costs = max(0.0, construction_costs)
            hard_plus_dc = construction_costs + total_dc
            prof_fees = hard_plus_dc * adj_fees_rate

    profit = gdv * adj_profit_rate
    rlv = gdv - (construction_costs + total_dc + prof_fees + profit)

    implied_cost_sqm = (construction_costs / proposed_bulk) if proposed_bulk > 0 else 0.0

    return {
        "proposed_bulk": proposed_bulk,
        "proposed_sellable": proposed_sellable,
        "market_sellable": market_sellable,
        "ih_sellable": ih_sellable,
        "total_dc": total_dc,
        "dc_savings": dc_savings,
        "gdv": gdv,
        "construction_costs": construction_costs,
        "prof_fees": prof_fees,
        "profit": profit,
        "rlv": rlv,
        "brownfield_credit": existing_gba_bulk_m2 > proposed_bulk,
        "adj_bonus_pct": float(adj_bonus_pct),
        "adj_fees_rate": float(adj_fees_rate),
        "adj_profit_rate": float(adj_profit_rate),
        "cost_mode": cost_mode,
        "pct_gdv_scope": pct_gdv_scope,
        "adj_cost_sqm": adj_cost_sqm,
        "adj_cost_pct_gdv": adj_cost_pct_gdv,
        "implied_cost_sqm": implied_cost_sqm,
        "efficiency_ratio": float(efficiency_ratio),
    }


# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("1. Site Parameters")
land_area = st.sidebar.number_input("Land Area (m²)", value=1000.0, min_value=0.0, step=50.0)
existing_gba = st.sidebar.number_input("Existing GBA on Site (m² bulk)", value=200.0, min_value=0.0, step=25.0)
zoning_key = st.sidebar.selectbox("Zoning Preset", list(ZONING_PRESETS.keys()))
pt_zone = st.sidebar.selectbox("PT Zone (Parking/DC Discount)", ["Standard", "PT1", "PT2"])

st.sidebar.header("2. Policy Toggles")
ih_percent = st.sidebar.slider("Inclusionary Housing (%)", 0, 30, 20)
density_bonus = st.sidebar.slider("Density Bonus (%)", 0, 50, 20)

st.sidebar.header("3. 2026 Benchmarks")
efficiency_ratio = st.sidebar.slider("Efficiency Ratio (sellable ÷ bulk)", 0.60, 0.95, 0.85, step=0.01)
profit_margin = st.sidebar.slider("Developer Profit (as % of GDV)", 10, 30, 20) / 100.0
build_tier = st.sidebar.selectbox("Hard Cost Tier (2026)", list(COST_TIERS.keys()), index=1)
tier_cost_default = COST_TIERS[build_tier]

st.sidebar.header("4. Exit Prices (Sectional Title 2026 est.)")
exit_price_source = st.sidebar.radio("Market exit price source", ["Suburb database", "Manual entry"], index=0)
db = load_exit_price_db()

with st.sidebar.expander("Upload exit price CSV (optional)"):
    uploaded = st.file_uploader("CSV with suburb + price columns", type=["csv"])
    if uploaded is not None:
        ok, msg = set_exit_price_db_from_upload(uploaded)
        if ok:
            st.success(msg)
            db = load_exit_price_db()
        else:
            st.error(msg)

    if st.button("Reset to 2026 defaults"):
        st.session_state.exit_price_db = pd.DataFrame(DEFAULT_EXIT_PRICES)
        db = load_exit_price_db()
        st.success("Exit price database reset.")

selected_suburb = None
db_min = db_max = db_price = None
price_point = "Mid"

if exit_price_source == "Suburb database":
    suburbs = sorted(db["suburb"].dropna().astype(str).unique().tolist())
    selected_suburb = st.sidebar.selectbox("Select suburb group", suburbs, index=0 if suburbs else None)
    price_point = st.sidebar.radio("Apply which point in range?", ["Low", "Mid", "High"], index=1, horizontal=True)

    if selected_suburb:
        row = db.loc[db["suburb"] == selected_suburb]
        if not row.empty:
            db_min = float(row["min_price_per_m2"].iloc[0])
            db_max = float(row["max_price_per_m2"].iloc[0])
            db_price = db_min if price_point == "Low" else db_max if price_point == "High" else (db_min + db_max) / 2.0

    override_price = st.sidebar.checkbox("Override suburb exit price", value=False)

    if (db_price is None) or override_price:
        market_price = st.sidebar.number_input(
            "Market Sales Price (R per sellable m²)",
            value=float(db_price) if db_price else 35000.0,
            min_value=0.0,
            step=500.0,
        )
    else:
        market_price = db_price
        st.sidebar.caption(f"Auto exit price: **R {market_price:,.0f}/m²** ({price_point})")
else:
    market_price = st.sidebar.number_input("Market Sales Price (R per sellable m²)", value=35000.0, min_value=0.0, step=500.0)

st.sidebar.header("5. Professional Fees (2026 ranges)")
default_components = default_prof_fee_components_scaled_to_target()
with st.sidebar.expander("Set professional fee components (uses sum of items)"):
    fee_components = {}
    for name, (lo, hi) in PROF_FEE_RANGES.items():
        fee_components[name] = (
            st.sidebar.slider(
                f"{name} (%)",
                float(lo * 100),
                float(hi * 100),
                float(default_components[name] * 100),
                step=0.1,
            )
            / 100.0
        )
base_prof_fee_rate = sum(fee_components.values())
st.sidebar.caption(f"Total Professional Fees (sum): **{base_prof_fee_rate*100:.2f}%** (~12–15%)")

st.sidebar.header("6. Construction Cost Input")
cost_mode = st.sidebar.radio("Construction cost input mode", ["R / m²", "% of GDV"], index=0)
pct_gdv_scope = "Hard cost only"
if cost_mode == "% of GDV":
    pct_gdv_scope = st.sidebar.radio("%GDV applies to…", ["Hard cost only", "Hard + soft (includes prof fees)"], index=0)

if cost_mode == "R / m²":
    const_cost_sqm = st.sidebar.number_input(
        "Hard Construction Cost (R per m² bulk)",
        value=float(tier_cost_default),
        min_value=0.0,
        step=250.0,
    )
    const_cost_pct_gdv = 0.0
else:
    const_cost_pct_gdv = st.sidebar.slider("Construction Cost (% of GDV)", 10, 90, 50) / 100.0
    const_cost_sqm = 0.0

st.sidebar.header("7. Overlays")
st.sidebar.subheader("🏛️ Built Heritage Overlay")
heritage_enabled = st.sidebar.checkbox("Enable Built Heritage Overlay", value=False)
heritage_bonus_suppression = st.sidebar.slider("Bonus suppression (%)", 0, 100, 50, disabled=not heritage_enabled)
heritage_cost_uplift = st.sidebar.slider("Construction cost uplift (%)", 0, 40, 8, disabled=not heritage_enabled)
heritage_fees_uplift = st.sidebar.slider("Professional fees uplift (%)", 0, 40, 5, disabled=not heritage_enabled)
heritage_profit_uplift = st.sidebar.slider("Profit requirement uplift (%)", 0, 40, 5, disabled=not heritage_enabled)

heritage_overlay = HeritageOverlay(
    enabled=heritage_enabled,
    bulk_reduction_pct=float(heritage_bonus_suppression),
    cost_uplift_pct=float(heritage_cost_uplift),
    fees_uplift_pct=float(heritage_fees_uplift),
    profit_uplift_pct=float(heritage_profit_uplift),
)

# =========================
# ENGINE + DISPLAY
# =========================
ff = ZONING_PRESETS[zoning_key]["ff"]
res = compute_model(
    land_area_m2=land_area,
    existing_gba_bulk_m2=existing_gba,
    ff=ff,
    density_bonus_pct=density_bonus,
    efficiency_ratio=efficiency_ratio,
    ih_pct=ih_percent,
    pt_zone_value=pt_zone,
    market_price_per_sellable_m2=market_price,
    ih_price_per_sellable_m2=IH_PRICE_PER_M2,
    profit_pct_gdv=profit_margin,
    base_prof_fee_rate=base_prof_fee_rate,
    overlay=heritage_overlay,
    cost_mode=cost_mode,
    base_cost_sqm=const_cost_sqm,
    base_cost_pct_gdv=const_cost_pct_gdv,
    pct_gdv_scope=pct_gdv_scope,
)

col1, col2 = st.columns([1, 1], vertical_alignment="top")

with col1:
    st.subheader("Results")
    st.metric("Residual Land Value (RLV)", f"R {res['rlv']:,.2f}")

    st.success(
        f"**🌟 Incentive Effect**  \n"
        f"With **{ih_percent}% IH** and **{pt_zone}**, you save:  \n"
        f"### R {res['dc_savings']:,.2f}  \n"
        f"*in Development Charges.*"
    )

    if res["brownfield_credit"]:
        st.warning("⚠️ Existing bulk exceeds proposed bulk. Net increase is zero; no DCs payable (Brownfield Credit).")

    if heritage_overlay.enabled:
        st.info(
            f"**🏛️ Heritage overlay active**  \n"
            f"- Bonus suppression: **{heritage_overlay.bulk_reduction_pct:.0f}%** → effective bonus **{res['adj_bonus_pct']:.1f}%**  \n"
            f"- Cost uplift: **{heritage_overlay.cost_uplift_pct:.0f}%**  \n"
            f"- Fees uplift: **{heritage_overlay.fees_uplift_pct:.0f}%**  \n"
            f"- Profit uplift: **{heritage_overlay.profit_uplift_pct:.0f}%**"
        )

with col2:
    st.subheader("RLV Waterfall")
    fig = go.Figure(
        go.Waterfall(
            name="RLV Breakdown",
            orientation="v",
            measure=["relative", "relative", "relative", "relative", "relative", "total"],
            x=["GDV", "Construction", "DCs", "Professional Fees", "Profit", "Residual Land"],
            y=[
                res["gdv"],
                -res["construction_costs"],
                -res["total_dc"],
                -res["prof_fees"],
                -res["profit"],
                res["rlv"],
            ],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

# =========================
# SENSITIVITY MATRIX
# =========================
st.subheader("Sensitivity Analysis: IH % vs Density Bonus (2026-aware)")

ih_levels = [0, 10, 20, 30]
bonus_levels = [0, 20, 40]

matrix_data = []
for ih in ih_levels:
    row = []
    for bonus in bonus_levels:
        tmp = compute_model(
            land_area_m2=land_area,
            existing_gba_bulk_m2=existing_gba,
            ff=ff,
            density_bonus_pct=bonus,
            efficiency_ratio=efficiency_ratio,
            ih_pct=ih,
            pt_zone_value=pt_zone,
            market_price_per_sellable_m2=market_price,
            ih_price_per_sellable_m2=IH_PRICE_PER_M2,
            profit_pct_gdv=profit_margin,
            base_prof_fee_rate=base_prof_fee_rate,
            overlay=heritage_overlay,
            cost_mode=cost_mode,
            base_cost_sqm=const_cost_sqm,
            base_cost_pct_gdv=const_cost_pct_gdv,
            pct_gdv_scope=pct_gdv_scope,
        )
        row.append(f"R {tmp['rlv']/1_000_000:.1f}M")
    matrix_data.append(row)

df_matrix = pd.DataFrame(
    matrix_data,
    index=[f"{x}% IH" for x in ih_levels],
    columns=[f"{x}% Bonus" for x in bonus_levels],
)
st.table(df_matrix)

with st.expander("View exit price database (2026 estimates)"):
    st.dataframe(load_exit_price_db(), use_container_width=True)
