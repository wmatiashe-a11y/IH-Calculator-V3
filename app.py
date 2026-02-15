import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass

# =========================================================
# City of Cape Town Map Viewer
# =========================================================
DEFAULT_CITYMAP_URL = "https://citymaps.capetown.gov.za/EGISViewer/"
CITYMAP_VIEWER_URL = st.secrets.get("CITYMAP_VIEWER_URL", DEFAULT_CITYMAP_URL)

# =========================================================
# CONFIG (Cape Town 2026-style defaults)
# =========================================================
ZONING_PRESETS = {
    "GR2 (Suburban)": {"ff": 1.0, "height": 15, "coverage": 0.6},
    "GR4 (Flats)": {"ff": 1.5, "height": 24, "coverage": 0.6},
    "MU1 (Mixed Use)": {"ff": 1.5, "height": 15, "coverage": 0.75},
    "MU2 (High Density)": {"ff": 4.0, "height": 25, "coverage": 1.0},
    "GB7 (CBD/High Rise)": {"ff": 12.0, "height": 60, "coverage": 1.0},
}

DC_BASE_RATE = 514.10
ROADS_TRANSPORT_PORTION = 285.35

DEFAULT_EFFICIENCY = 0.85
IH_PRICE_PER_M2 = 15000  # IH capped price (sellable m² basis)

COST_TIERS = {
    "Economic (R10,000/m²)": 10000.0,
    "Mid-Tier (R18,000/m²)": 18000.0,
    "Luxury (R25,000+/m²)": 25000.0,
}

# Exit price DB (2026 sectional title new apartments) — sellable m² basis
DEFAULT_EXIT_PRICES = [
    {"suburb": "Clifton / Bantry Bay", "min_price_per_m2": 120000, "max_price_per_m2": 170000},  # assumed upper
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
DEFAULT_PROFIT = 0.20

st.set_page_config(page_title="Resíduo • Cape Town Feasibility", layout="wide")
st.title("Resíduo — Cape Town Feasibility Lens")

# =========================================================
# Canonical state keys (the model reads ONLY these)
# =========================================================
STATE_KEYS = {
    "qp_address": "",
    "selected_suburb": "",
    "zoning_key": "GR2 (Suburban)",
    "pt_zone": "Standard",
    "density_bonus": 20,
    "land_area": 1000.0,
    "existing_gba": 200.0,
    "ih_percent": 20,
    "efficiency_ratio": DEFAULT_EFFICIENCY,
    "profit_margin_pct": 20,  # stored as int percent
    "build_tier": "Mid-Tier (R18,000/m²)",
    "cost_mode": "R / m²",
    "pct_gdv_scope": "Hard cost only",
    "const_cost_sqm": 18000.0,
    "const_cost_pct_gdv": 0.50,
    "exit_price_source": "Suburb database",
    "price_point": "Mid",
    "override_exit_price": False,
    "market_price_manual": 35000.0,
    "ih_exit_price": IH_PRICE_PER_M2,
    "heritage_enabled": False,
    "heritage_bonus_suppression": 50,
    "heritage_cost_uplift": 8,
    "heritage_fees_uplift": 5,
    "heritage_profit_uplift": 5,
    "project_notes": "",
}


def init_state():
    for k, v in STATE_KEYS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if "prof_fee_components" not in st.session_state:
        st.session_state.prof_fee_components = default_prof_fee_components_scaled_to_target()

    load_exit_price_db()

    # set initial suburb if empty
    db = load_exit_price_db()
    suburbs = sorted(db["suburb"].dropna().astype(str).unique().tolist())
    if not st.session_state.selected_suburb and suburbs:
        st.session_state.selected_suburb = suburbs[0]


def sync_state(from_key: str, to_key: str):
    """Copy widget value into canonical model state (single direction)."""
    if from_key in st.session_state:
        st.session_state[to_key] = st.session_state[from_key]


# =========================================================
# Models / Overlays
# =========================================================
@dataclass(frozen=True)
class HeritageOverlay:
    enabled: bool
    bulk_reduction_pct: float  # interpreted as BONUS suppression %
    cost_uplift_pct: float
    fees_uplift_pct: float
    profit_uplift_pct: float


def apply_heritage_overlay(
    density_bonus_pct: float,
    base_cost_value: float,
    base_fees_rate: float,
    base_profit_rate: float,
    overlay: HeritageOverlay,
) -> tuple[float, float, float, float]:
    if not overlay.enabled:
        return density_bonus_pct, base_cost_value, base_fees_rate, base_profit_rate

    adj_bonus = max(0.0, density_bonus_pct * (1.0 - overlay.bulk_reduction_pct / 100.0))
    adj_cost = base_cost_value * (1.0 + overlay.cost_uplift_pct / 100.0)
    adj_fees = base_fees_rate * (1.0 + overlay.fees_uplift_pct / 100.0)
    adj_profit = base_profit_rate * (1.0 + overlay.profit_uplift_pct / 100.0)
    return adj_bonus, adj_cost, adj_fees, adj_profit


# =========================================================
# Helpers
# =========================================================
def pt_discount(pt_zone_value: str) -> float:
    return {"PT1": 0.8, "PT2": 0.5}.get(pt_zone_value, 1.0)


def normalize_exit_price_db(df: pd.DataFrame) -> pd.DataFrame:
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


def fmt_r(x: float) -> str:
    return f"R {x:,.0f}"


def fmt_r2(x: float) -> str:
    return f"R {x:,.2f}"


def fmt_pct(x: float, dp: int = 1) -> str:
    return f"{x*100:.{dp}f}%"


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
):
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
            target_all_in = gdv * adj_cost_pct_gdv
            construction_costs = (target_all_in - (adj_fees_rate * total_dc)) / (1.0 + adj_fees_rate)
            construction_costs = max(0.0, construction_costs)
            hard_plus_dc = construction_costs + total_dc
            prof_fees = hard_plus_dc * adj_fees_rate

    profit = gdv * adj_profit_rate
    rlv = gdv - (construction_costs + total_dc + prof_fees + profit)

    implied_cost_sqm = (construction_costs / proposed_bulk) if proposed_bulk > 0 else 0.0

    return {
        "base_bulk": base_bulk,
        "proposed_bulk": proposed_bulk,
        "proposed_sellable": proposed_sellable,
        "net_increase_bulk": net_increase_bulk,
        "market_sellable": market_sellable,
        "ih_sellable": ih_sellable,
        "market_increase_bulk": market_increase_bulk,
        "ih_increase_bulk": ih_increase_bulk,
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


# =========================================================
# Init
# =========================================================
init_state()

# =========================================================
# HEADER — Quick Profile (unique hdr_* keys)
# =========================================================
db = load_exit_price_db()
suburbs = sorted(db["suburb"].dropna().astype(str).unique().tolist()) or [""]

# Seed header widgets from canonical state (only if first time)
if "hdr_seeded" not in st.session_state:
    st.session_state.hdr_qp_address = st.session_state.qp_address
    st.session_state.hdr_selected_suburb = st.session_state.selected_suburb
    st.session_state.hdr_zoning_key = st.session_state.zoning_key
    st.session_state.hdr_pt_zone = st.session_state.pt_zone
    st.session_state.hdr_density_bonus = st.session_state.density_bonus
    st.session_state.hdr_seeded = True

hdr = st.container()
with hdr:
    c1, c2, c3, c4, c5 = st.columns([2.3, 1.4, 1.4, 1.1, 1.2])
    with c1:
        st.text_input(
            "Property Quick-Profile (Erf / Address / Label)",
            key="hdr_qp_address",
            placeholder="e.g. Erf 12345, Sea Point",
            on_change=sync_state,
            args=("hdr_qp_address", "qp_address"),
        )
    with c2:
        st.selectbox(
            "Suburb group (exit price)",
            suburbs,
            key="hdr_selected_suburb",
            on_change=sync_state,
            args=("hdr_selected_suburb", "selected_suburb"),
        )
    with c3:
        st.selectbox(
            "Zoning preset",
            list(ZONING_PRESETS.keys()),
            key="hdr_zoning_key",
            on_change=sync_state,
            args=("hdr_zoning_key", "zoning_key"),
        )
    with c4:
        st.selectbox(
            "PT Zone",
            ["Standard", "PT1", "PT2"],
            key="hdr_pt_zone",
            on_change=sync_state,
            args=("hdr_pt_zone", "pt_zone"),
        )
    with c5:
        st.selectbox(
            "Density bonus",
            [0, 10, 20, 30, 40, 50],
            key="hdr_density_bonus",
            on_change=sync_state,
            args=("hdr_density_bonus", "density_bonus"),
        )

    with st.expander("🗺️ City of Cape Town Map Viewer", expanded=False):
        st.link_button("Open City Map Viewer", CITYMAP_VIEWER_URL)
        components.iframe(CITYMAP_VIEWER_URL, height=520, scrolling=True)

# =========================================================
# Resolve exit price
# =========================================================
selected_suburb = st.session_state.selected_suburb
price_point = st.session_state.price_point

db_row = db.loc[db["suburb"] == selected_suburb] if selected_suburb else pd.DataFrame()
db_min = db_max = db_price = None
if not db_row.empty:
    db_min = float(db_row["min_price_per_m2"].iloc[0])
    db_max = float(db_row["max_price_per_m2"].iloc[0])
    if price_point == "Low":
        db_price = db_min
    elif price_point == "High":
        db_price = db_max
    else:
        db_price = (db_min + db_max) / 2.0

if st.session_state.exit_price_source == "Suburb database" and (db_price is not None) and (not st.session_state.override_exit_price):
    market_price = db_price
else:
    market_price = float(st.session_state.market_price_manual)

# =========================================================
# Professional fees total
# =========================================================
base_prof_fee_rate = float(sum(st.session_state.prof_fee_components.values()))

# =========================================================
# Cost inputs
# =========================================================
tier_cost_default = COST_TIERS.get(st.session_state.build_tier, 18000.0)
if st.session_state.cost_mode == "R / m²":
    const_cost_sqm = float(st.session_state.const_cost_sqm) if st.session_state.const_cost_sqm else tier_cost_default
    const_cost_pct_gdv = 0.0
else:
    const_cost_sqm = 0.0
    const_cost_pct_gdv = float(st.session_state.const_cost_pct_gdv)

# =========================================================
# Overlay object
# =========================================================
heritage_overlay = HeritageOverlay(
    enabled=bool(st.session_state.heritage_enabled),
    bulk_reduction_pct=float(st.session_state.heritage_bonus_suppression),
    cost_uplift_pct=float(st.session_state.heritage_cost_uplift),
    fees_uplift_pct=float(st.session_state.heritage_fees_uplift),
    profit_uplift_pct=float(st.session_state.heritage_profit_uplift),
)

# =========================================================
# Compute
# =========================================================
ff = ZONING_PRESETS[st.session_state.zoning_key]["ff"]

res = compute_model(
    land_area_m2=float(st.session_state.land_area),
    existing_gba_bulk_m2=float(st.session_state.existing_gba),
    ff=float(ff),
    density_bonus_pct=float(st.session_state.density_bonus),
    efficiency_ratio=float(st.session_state.efficiency_ratio),
    ih_pct=float(st.session_state.ih_percent),
    pt_zone_value=str(st.session_state.pt_zone),
    market_price_per_sellable_m2=float(market_price),
    ih_price_per_sellable_m2=float(st.session_state.ih_exit_price),
    profit_pct_gdv=float(st.session_state.profit_margin_pct) / 100.0,
    base_prof_fee_rate=base_prof_fee_rate,
    overlay=heritage_overlay,
    cost_mode=str(st.session_state.cost_mode),
    base_cost_sqm=float(const_cost_sqm),
    base_cost_pct_gdv=float(const_cost_pct_gdv),
    pct_gdv_scope=str(st.session_state.pct_gdv_scope),
)

# =========================================================
# KPI strip
# =========================================================
k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
k1.metric("Residual Land Value", fmt_r2(res["rlv"]))
k2.metric("RLV / m² land", fmt_r2(res["rlv"] / max(1.0, float(st.session_state.land_area))))
k3.metric("GDV", fmt_r2(res["gdv"]))
k4.metric("Profit (R)", fmt_r2(res["profit"]))
k5.metric("Total DCs", fmt_r2(res["total_dc"]))
k6.metric("DC Savings", fmt_r2(res["dc_savings"]))
k7.metric("Proposed bulk", f"{res['proposed_bulk']:,.0f} m²")

# =========================================================
# Main row: Waterfall + Snapshot
# =========================================================
left, right = st.columns([1.55, 1.0], gap="large")
with left:
    st.subheader("Residual breakdown")
    fig = go.Figure(go.Waterfall(
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
    ))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=460)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Assumptions snapshot")
    st.write(
        f"**Zoning:** {st.session_state.zoning_key}  \n"
        f"**FF:** {ff:.2f}  \n"
        f"**Efficiency:** {fmt_pct(res['efficiency_ratio'], 0)}  \n"
        f"**IH % (net increase):** {st.session_state.ih_percent}%  \n"
        f"**Market exit price:** {fmt_r(market_price)}/m² sellable  \n"
        f"**IH exit price:** {fmt_r(float(st.session_state.ih_exit_price))}/m² sellable  \n"
        f"**Profit target:** {fmt_pct(res['adj_profit_rate'])} of GDV  \n"
        f"**Prof fees (total):** {fmt_pct(res['adj_fees_rate'], 2)}  \n"
        f"**PT zone:** {st.session_state.pt_zone}  \n"
        f"**Density bonus (effective):** {res['adj_bonus_pct']:.1f}%"
    )

    st.divider()
    st.subheader("Risk flags")

    flags = []
    if res["brownfield_credit"]:
        flags.append("Existing bulk exceeds proposed bulk → **Net increase is 0** (brownfield credit; DCs may be 0).")
    if (float(st.session_state.profit_margin_pct) / 100.0) < 0.20:
        flags.append("Profit < **20%** → may be rejected by banks (typical SA funding threshold).")
    if res["efficiency_ratio"] < 0.80:
        flags.append("Efficiency < **80%** → design/parking/service risk (check layouts).")
    if heritage_overlay.enabled:
        flags.append("Built Heritage overlay enabled → bonus suppressed + uplifted costs/fees/profit.")

    if not flags:
        st.success("No major flags triggered by current inputs.")
    else:
        for f in flags:
            st.warning(f)

# =========================================================
# Tabs
# =========================================================
tabs = st.tabs(["Inputs", "Sensitivity", "Audit trail", "Exit prices DB", "Notes"])

# -------------------------
# Inputs tab (unique in_* keys, synced to canonical state)
# -------------------------
with tabs[0]:
    st.subheader("Inputs")

    # seed input widgets once
    if "in_seeded" not in st.session_state:
        # mirror canonical into in_*
        for k in STATE_KEYS.keys():
            st.session_state[f"in_{k}"] = st.session_state[k]
        st.session_state.in_seeded = True

    a, b = st.columns([1, 1], gap="large")

    with a:
        with st.expander("1) Site", expanded=True):
            st.number_input("Land Area (m²)", min_value=0.0, step=50.0, key="in_land_area", on_change=sync_state, args=("in_land_area", "land_area"))
            st.number_input("Existing GBA on Site (m² bulk)", min_value=0.0, step=25.0, key="in_existing_gba", on_change=sync_state, args=("in_existing_gba", "existing_gba"))

        with st.expander("2) Policy", expanded=True):
            st.slider("Inclusionary Housing (%) on net increase", 0, 30, key="in_ih_percent", on_change=sync_state, args=("in_ih_percent", "ih_percent"))
            st.slider("Efficiency ratio (sellable ÷ bulk)", 0.60, 0.95, step=0.01, key="in_efficiency_ratio", on_change=sync_state, args=("in_efficiency_ratio", "efficiency_ratio"))
            st.slider("Developer Profit (% of GDV)", 10, 30, key="in_profit_margin_pct", on_change=sync_state, args=("in_profit_margin_pct", "profit_margin_pct"))

        with st.expander("3) Exit prices", expanded=False):
            st.radio("Market exit price source", ["Suburb database", "Manual entry"], key="in_exit_price_source", horizontal=True, on_change=sync_state, args=("in_exit_price_source", "exit_price_source"))
            st.radio("Range point", ["Low", "Mid", "High"], key="in_price_point", horizontal=True, on_change=sync_state, args=("in_price_point", "price_point"))
            st.checkbox("Override suburb exit price", key="in_override_exit_price", on_change=sync_state, args=("in_override_exit_price", "override_exit_price"))
            st.number_input("Market Sales Price (manual) — R per sellable m²", min_value=0.0, step=500.0, key="in_market_price_manual", on_change=sync_state, args=("in_market_price_manual", "market_price_manual"))
            st.number_input("IH Exit Price — R per sellable m²", min_value=0.0, step=500.0, key="in_ih_exit_price", on_change=sync_state, args=("in_ih_exit_price", "ih_exit_price"))

    with b:
        with st.expander("4) Construction cost", expanded=True):
            st.selectbox("Hard cost tier (2026)", list(COST_TIERS.keys()), key="in_build_tier", on_change=sync_state, args=("in_build_tier", "build_tier"))
            st.radio("Construction cost input mode", ["R / m²", "% of GDV"], key="in_cost_mode", horizontal=True, on_change=sync_state, args=("in_cost_mode", "cost_mode"))

            if st.session_state.cost_mode == "% of GDV":
                st.radio("%GDV applies to…", ["Hard cost only", "Hard + soft (includes prof fees)"], key="in_pct_gdv_scope", on_change=sync_state, args=("in_pct_gdv_scope", "pct_gdv_scope"))
                st.slider("Construction Cost (% of GDV)", 0.10, 0.90, step=0.01, key="in_const_cost_pct_gdv", on_change=sync_state, args=("in_const_cost_pct_gdv", "const_cost_pct_gdv"))
            else:
                st.number_input("Hard Construction Cost (R per m² bulk)", min_value=0.0, step=250.0, key="in_const_cost_sqm", on_change=sync_state, args=("in_const_cost_sqm", "const_cost_sqm"))

        with st.expander("5) Professional fees (2026 ranges)", expanded=False):
            st.caption("Applied as % of (construction + DCs). Total = sum of items.")
            new_components = {}
            for name, (lo, hi) in PROF_FEE_RANGES.items():
                cur = float(st.session_state.prof_fee_components.get(name, (lo + hi) / 2.0))
                new_components[name] = st.slider(
                    f"{name} (%)",
                    min_value=float(lo * 100),
                    max_value=float(hi * 100),
                    value=float(cur * 100),
                    step=0.1,
                    key=f"in_fee_{name}",
                ) / 100.0
            st.session_state.prof_fee_components = new_components
            st.info(f"Total professional fees: **{sum(new_components.values())*100:.2f}%**")

        with st.expander("6) Built Heritage overlay", expanded=False):
            st.checkbox("Enable Built Heritage Overlay", key="in_heritage_enabled", on_change=sync_state, args=("in_heritage_enabled", "heritage_enabled"))
            st.slider("Bonus suppression (%)", 0, 100, key="in_heritage_bonus_suppression", disabled=not st.session_state.heritage_enabled, on_change=sync_state, args=("in_heritage_bonus_suppression", "heritage_bonus_suppression"))
            st.slider("Construction cost uplift (%)", 0, 40, key="in_heritage_cost_uplift", disabled=not st.session_state.heritage_enabled, on_change=sync_state, args=("in_heritage_cost_uplift", "heritage_cost_uplift"))
            st.slider("Professional fees uplift (%)", 0, 40, key="in_heritage_fees_uplift", disabled=not st.session_state.heritage_enabled, on_change=sync_state, args=("in_heritage_fees_uplift", "heritage_fees_uplift"))
            st.slider("Profit uplift (%)", 0, 40, key="in_heritage_profit_uplift", disabled=not st.session_state.heritage_enabled, on_change=sync_state, args=("in_heritage_profit_uplift", "heritage_profit_uplift"))

    st.caption("All Inputs use unique keys (no duplicates) and sync into the model state used by the dashboard.")

# -------------------------
# Sensitivity
# -------------------------
with tabs[1]:
    st.subheader("Sensitivity — IH % vs Density Bonus")

    ih_levels = [0, 10, 20, 30]
    bonus_levels = [0, 20, 40]

    matrix_num = []
    matrix_lbl = []
    for ih in ih_levels:
        row_num = []
        row_lbl = []
        for bonus in bonus_levels:
            tmp = compute_model(
                land_area_m2=float(st.session_state.land_area),
                existing_gba_bulk_m2=float(st.session_state.existing_gba),
                ff=float(ff),
                density_bonus_pct=float(bonus),
                efficiency_ratio=float(st.session_state.efficiency_ratio),
                ih_pct=float(ih),
                pt_zone_value=str(st.session_state.pt_zone),
                market_price_per_sellable_m2=float(market_price),
                ih_price_per_sellable_m2=float(st.session_state.ih_exit_price),
                profit_pct_gdv=float(st.session_state.profit_margin_pct) / 100.0,
                base_prof_fee_rate=float(sum(st.session_state.prof_fee_components.values())),
                overlay=heritage_overlay,
                cost_mode=str(st.session_state.cost_mode),
                base_cost_sqm=float(const_cost_sqm),
                base_cost_pct_gdv=float(const_cost_pct_gdv),
                pct_gdv_scope=str(st.session_state.pct_gdv_scope),
            )
            row_num.append(tmp["rlv"] / 1_000_000)
            row_lbl.append(f"R {tmp['rlv']/1_000_000:.1f}M")
        matrix_num.append(row_num)
        matrix_lbl.append(row_lbl)

    df_matrix = pd.DataFrame(
        matrix_lbl,
        index=[f"{x}% IH" for x in ih_levels],
        columns=[f"{x}% Bonus" for x in bonus_levels],
    )
    st.table(df_matrix)

    fig_hm = go.Figure(
        data=go.Heatmap(
            z=matrix_num,
            x=[f"{x}% Bonus" for x in bonus_levels],
            y=[f"{x}% IH" for x in ih_levels],
            hovertemplate="Bonus=%{x}<br>IH=%{y}<br>RLV=%{z:.1f}M<extra></extra>",
        )
    )
    fig_hm.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_hm, use_container_width=True)

# -------------------------
# Audit trail
# -------------------------
with tabs[2]:
    st.subheader("Audit trail (grouped)")
    rows = []

    def sep(title: str):
        rows.append({"Section": title, "Item": "", "Value": ""})

    def row(section: str, item: str, value: str):
        rows.append({"Section": section, "Item": item, "Value": value})

    sep("Areas")
    row("Areas", "Land area", f"{float(st.session_state.land_area):,.0f} m²")
    row("Areas", "Existing bulk (GBA)", f"{float(st.session_state.existing_gba):,.0f} m²")
    row("Areas", "Base bulk (land × FF)", f"{res['base_bulk']:,.0f} m²")
    row("Areas", "Proposed bulk", f"{res['proposed_bulk']:,.0f} m²")
    row("Areas", "Efficiency", fmt_pct(res["efficiency_ratio"], 0))
    row("Areas", "Proposed sellable", f"{res['proposed_sellable']:,.0f} m²")
    row("Areas", "Market sellable", f"{res['market_sellable']:,.0f} m²")
    row("Areas", "IH sellable", f"{res['ih_sellable']:,.0f} m²")

    sep("Revenue")
    row("Revenue", "Market exit price (sellable)", f"{fmt_r(market_price)}/m²")
    row("Revenue", "IH exit price (sellable)", f"{fmt_r(float(st.session_state.ih_exit_price))}/m²")
    row("Revenue", "GDV", fmt_r2(res["gdv"]))

    sep("Costs")
    if res["cost_mode"] == "R / m²":
        row("Costs", "Hard cost input", f"{fmt_r(res['adj_cost_sqm'])}/m² bulk")
    else:
        row("Costs", "Construction input", f"{fmt_pct(res['adj_cost_pct_gdv'], 1)} of GDV ({res['pct_gdv_scope']})")
    row("Costs", "Construction costs", fmt_r2(res["construction_costs"]))
    row("Costs", "Development Charges (DCs)", fmt_r2(res["total_dc"]))
    row("Costs", "Professional fees (total)", f"{fmt_pct(res['adj_fees_rate'], 2)} of (hard+DC)")
    row("Costs", "Professional fees (R)", fmt_r2(res["prof_fees"]))

    sep("Profit")
    row("Profit", "Profit rate", fmt_pct(res["adj_profit_rate"], 1))
    row("Profit", "Profit (R)", fmt_r2(res["profit"]))

    sep("Residual")
    row("Residual", "Residual land value (RLV)", fmt_r2(res["rlv"]))
    row("Residual", "RLV per m² land", fmt_r2(res["rlv"] / max(1.0, float(st.session_state.land_area))))

    sep("DC Incentive Effect")
    row("DC Incentive Effect", "Potential DCs on all net increase", fmt_r2(max(0.0, res["net_increase_bulk"]) * DC_BASE_RATE))
    row("DC Incentive Effect", "Actual DCs payable", fmt_r2(res["total_dc"]))
    row("DC Incentive Effect", "DC savings", fmt_r2(res["dc_savings"]))

    sep("Overlays")
    row("Overlays", "Built heritage enabled", "Yes" if heritage_overlay.enabled else "No")
    if heritage_overlay.enabled:
        row("Overlays", "Bonus suppression", f"{heritage_overlay.bulk_reduction_pct:.0f}%")
        row("Overlays", "Cost uplift", f"{heritage_overlay.cost_uplift_pct:.0f}%")
        row("Overlays", "Fees uplift", f"{heritage_overlay.fees_uplift_pct:.0f}%")
        row("Overlays", "Profit uplift", f"{heritage_overlay.profit_uplift_pct:.0f}%")

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# -------------------------
# Exit prices DB
# -------------------------
with tabs[3]:
    st.subheader("Exit price database (sellable m², 2026 estimates)")
    c1, c2 = st.columns([1.2, 1.0])

    with c1:
        uploaded = st.file_uploader("Upload CSV (suburb + min/max OR suburb + exit_price_per_m2)", type=["csv"])
        if uploaded is not None:
            ok, msg = set_exit_price_db_from_upload(uploaded)
            st.success(msg) if ok else st.error(msg)

        if st.button("Reset DB to 2026 defaults"):
            st.session_state.exit_price_db = pd.DataFrame(DEFAULT_EXIT_PRICES)
            load_exit_price_db()
            st.success("Reset complete.")

    with c2:
        st.caption("Current selection")
        st.write(f"**Suburb group:** {selected_suburb}")
        st.write(f"**Range point:** {st.session_state.price_point}")
        if db_min is not None and db_max is not None:
            st.write(f"**Range:** {fmt_r(db_min)} – {fmt_r(db_max)} / m²")
        st.write(f"**Applied market price:** {fmt_r(market_price)} / m²")

    st.dataframe(load_exit_price_db(), use_container_width=True)

# -------------------------
# Notes
# -------------------------
with tabs[4]:
    st.subheader("Notes")
    st.text_area("Project notes (saved in session)", key="project_notes", height=220)
