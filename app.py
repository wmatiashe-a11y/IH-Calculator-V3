import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass

# --- CONFIGURATION & ZONING PRESETS ---
ZONING_PRESETS = {
    "GR2 (Suburban)": {"ff": 1.0, "height": 15, "coverage": 0.6},
    "GR4 (Flats)": {"ff": 1.5, "height": 24, "coverage": 0.6},
    "MU1 (Mixed Use)": {"ff": 1.5, "height": 15, "coverage": 0.75},
    "MU2 (High Density)": {"ff": 4.0, "height": 25, "coverage": 1.0},
    "GB7 (CBD/High Rise)": {"ff": 12.0, "height": 60, "coverage": 1.0},
}

DC_BASE_RATE = 514.10  # Total DC per m2
ROADS_TRANSPORT_PORTION = 285.35

IH_PRICE_PER_M2 = 15000            # capped IH price (assumption)
DEFAULT_PROF_FEE_RATE = 0.12       # % of (construction + DCs)

# --- EXIT PRICE DATABASE (DEFAULT / PLACEHOLDER) ---
# Replace these with your real suburb-specific exit prices when ready.
DEFAULT_EXIT_PRICES = [
    {"suburb": "Sea Point", "exit_price_per_m2": 65000},
    {"suburb": "Green Point", "exit_price_per_m2": 60000},
    {"suburb": "CBD", "exit_price_per_m2": 52000},
    {"suburb": "Woodstock", "exit_price_per_m2": 42000},
    {"suburb": "Salt River", "exit_price_per_m2": 38000},
    {"suburb": "Observatory", "exit_price_per_m2": 36000},
    {"suburb": "Rondebosch", "exit_price_per_m2": 45000},
    {"suburb": "Claremont", "exit_price_per_m2": 48000},
    {"suburb": "Kenilworth", "exit_price_per_m2": 43000},
    {"suburb": "Newlands", "exit_price_per_m2": 52000},
    {"suburb": "Century City", "exit_price_per_m2": 40000},
    {"suburb": "Bellville", "exit_price_per_m2": 28000},
]

st.set_page_config(page_title="CPT RLV Calculator", layout="wide")
st.title("🏗️ Cape Town Redevelopment: RLV & IH Sensitivity")

# -----------------------
# Heritage overlay model
# -----------------------
@dataclass(frozen=True)
class HeritageOverlay:
    enabled: bool
    bulk_reduction_pct: float      # interpreted as "BONUS suppression %"
    cost_uplift_pct: float         # increases construction costs input (R/m² or %GDV)
    fees_uplift_pct: float         # increases professional fees rate
    profit_uplift_pct: float       # increases required profit % of GDV


def apply_heritage_overlay(
    density_bonus_pct: float,
    base_cost_value: float,
    base_fees_rate: float,
    base_profit_rate: float,
    overlay: HeritageOverlay,
) -> tuple[float, float, float, float]:
    """
    Returns:
      (adj_density_bonus_pct, adj_cost_value, adj_fees_rate, adj_profit_rate)

    Heritage behavior:
    - Suppresses density bonus (override)
    - Uplifts apply to the chosen construction input (R/m² OR % of GDV)
    """
    if not overlay.enabled:
        return density_bonus_pct, base_cost_value, base_fees_rate, base_profit_rate

    adj_bonus = density_bonus_pct * (1.0 - overlay.bulk_reduction_pct / 100.0)
    adj_bonus = max(0.0, adj_bonus)

    adj_cost = base_cost_value * (1.0 + overlay.cost_uplift_pct / 100.0)
    adj_fees = base_fees_rate * (1.0 + overlay.fees_uplift_pct / 100.0)
    adj_profit = base_profit_rate * (1.0 + overlay.profit_uplift_pct / 100.0)
    return adj_bonus, adj_cost, adj_fees, adj_profit


# --- HELPERS ---
def pt_discount(pt_zone_value: str) -> float:
    if pt_zone_value == "PT1":
        return 0.8
    if pt_zone_value == "PT2":
        return 0.5
    return 1.0


def load_exit_price_db() -> pd.DataFrame:
    """
    Session-backed exit price database:
    - starts with DEFAULT_EXIT_PRICES
    - can be replaced by a user-uploaded CSV
    """
    if "exit_price_db" not in st.session_state:
        st.session_state.exit_price_db = pd.DataFrame(DEFAULT_EXIT_PRICES)
    return st.session_state.exit_price_db


def set_exit_price_db_from_upload(uploaded_file) -> tuple[bool, str]:
    """
    Expected CSV columns:
      suburb, exit_price_per_m2
    """
    try:
        df = pd.read_csv(uploaded_file)
        cols = {c.lower().strip(): c for c in df.columns}
        if "suburb" not in cols or "exit_price_per_m2" not in cols:
            return False, "CSV must include columns: suburb, exit_price_per_m2"

        df2 = df[[cols["suburb"], cols["exit_price_per_m2"]]].copy()
        df2.columns = ["suburb", "exit_price_per_m2"]
        df2["suburb"] = df2["suburb"].astype(str).str.strip()
        df2["exit_price_per_m2"] = pd.to_numeric(df2["exit_price_per_m2"], errors="coerce")
        df2 = df2.dropna(subset=["suburb", "exit_price_per_m2"])
        df2 = df2.sort_values("suburb").drop_duplicates(subset=["suburb"], keep="last")

        if df2.empty:
            return False, "CSV loaded but no valid rows found."

        st.session_state.exit_price_db = df2
        return True, f"Loaded {len(df2)} suburb exit prices."
    except Exception as e:
        return False, f"Failed to load CSV: {e}"


def compute_model(
    land_area_m2: float,
    existing_gba_m2: float,
    ff: float,
    density_bonus_pct: float,
    ih_pct: float,
    pt_zone_value: str,
    market_price_per_m2: float,
    ih_price_per_m2: float,
    profit_pct_gdv: float,
    base_prof_fee_rate: float,
    overlay: HeritageOverlay,
    cost_mode: str,
    base_cost_sqm: float,
    base_cost_pct_gdv: float,
    pct_gdv_scope: str,
):
    base_bulk = land_area_m2 * ff

    # Apply overlay (bonus override + uplifts)
    cost_input = base_cost_sqm if cost_mode == "R / m²" else base_cost_pct_gdv
    adj_bonus_pct, adj_cost_input, adj_fees_rate, adj_profit_rate = apply_heritage_overlay(
        density_bonus_pct=density_bonus_pct,
        base_cost_value=cost_input,
        base_fees_rate=base_prof_fee_rate,
        base_profit_rate=profit_pct_gdv,
        overlay=overlay,
    )

    proposed_gba = base_bulk * (1.0 + adj_bonus_pct / 100.0)

    # Brownfield credit / net increase
    net_increase = max(0.0, proposed_gba - existing_gba_m2)

    # IH applied to net increase (consistent with DC basis)
    ih_gba = min(net_increase * (ih_pct / 100.0), proposed_gba)
    market_gba = proposed_gba - ih_gba
    market_gba_increase = net_increase - ih_gba

    # DCs (market share only), PT discount on roads portion
    disc = pt_discount(pt_zone_value)
    roads_dc = market_gba_increase * ROADS_TRANSPORT_PORTION * disc
    other_dc = market_gba_increase * (DC_BASE_RATE - ROADS_TRANSPORT_PORTION)
    total_dc = roads_dc + other_dc

    total_potential_dc = net_increase * DC_BASE_RATE
    dc_savings = total_potential_dc - total_dc

    # Revenue
    gdv = (market_gba * market_price_per_m2) + (ih_gba * ih_price_per_m2)

    # Construction + fees logic (mode + scope dependent)
    adj_cost_sqm = None
    adj_cost_pct_gdv = None

    if cost_mode == "R / m²":
        adj_cost_sqm = adj_cost_input
        construction_costs = proposed_gba * adj_cost_sqm
        hard_plus_dc = construction_costs + total_dc
        prof_fees = hard_plus_dc * adj_fees_rate
    else:
        # % of GDV
        adj_cost_pct_gdv = adj_cost_input

        if pct_gdv_scope == "Hard cost only":
            construction_costs = gdv * adj_cost_pct_gdv
            hard_plus_dc = construction_costs + total_dc
            prof_fees = hard_plus_dc * adj_fees_rate
        else:
            # Hard + soft (construction + prof fees) totals to %GDV without double counting fees
            target_all_in = gdv * adj_cost_pct_gdv
            construction_costs = (target_all_in - (adj_fees_rate * total_dc)) / (1.0 + adj_fees_rate)
            construction_costs = max(0.0, construction_costs)

            hard_plus_dc = construction_costs + total_dc
            prof_fees = hard_plus_dc * adj_fees_rate

    # Profit + RLV
    profit = gdv * adj_profit_rate
    rlv = gdv - (construction_costs + total_dc + prof_fees + profit)

    implied_cost_sqm = (construction_costs / proposed_gba) if proposed_gba > 0 else 0.0

    return {
        "input_bonus_pct": float(density_bonus_pct),
        "adj_bonus_pct": float(adj_bonus_pct),
        "proposed_gba": proposed_gba,
        "net_increase": net_increase,
        "ih_gba": ih_gba,
        "market_gba": market_gba,
        "market_gba_increase": market_gba_increase,
        "total_dc": total_dc,
        "dc_savings": dc_savings,
        "gdv": gdv,
        "construction_costs": construction_costs,
        "prof_fees": prof_fees,
        "profit": profit,
        "rlv": rlv,
        "brownfield_credit": existing_gba_m2 > proposed_gba,
        "adj_fees_rate": adj_fees_rate,
        "adj_profit_rate": adj_profit_rate,
        "cost_mode": cost_mode,
        "pct_gdv_scope": pct_gdv_scope,
        "adj_cost_sqm": adj_cost_sqm,
        "adj_cost_pct_gdv": adj_cost_pct_gdv,
        "implied_cost_sqm": implied_cost_sqm,
    }


# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("1. Site Parameters")
land_area = st.sidebar.number_input("Land Area (m2)", value=1000.0, min_value=0.0, step=50.0)
existing_gba = st.sidebar.number_input("Existing GBA on Site (m2)", value=200.0, min_value=0.0, step=25.0)
zoning_key = st.sidebar.selectbox("Zoning Preset", list(ZONING_PRESETS.keys()))
pt_zone = st.sidebar.selectbox("PT Zone (Parking/DC Discount)", ["Standard", "PT1", "PT2"])

st.sidebar.header("2. Policy Toggles")
ih_percent = st.sidebar.slider("Inclusionary Housing (%)", 0, 30, 20)
density_bonus = st.sidebar.slider("Density Bonus (%)", 0, 50, 20)

# -------------------------
# Exit price (suburb DB)
# -------------------------
st.sidebar.header("3. Exit Prices")
exit_price_source = st.sidebar.radio(
    "Market exit price source",
    options=["Suburb database", "Manual entry"],
    index=0,
)

db = load_exit_price_db()

# Optional CSV upload to replace the database
with st.sidebar.expander("Upload exit price CSV (optional)"):
    uploaded = st.file_uploader("CSV with columns: suburb, exit_price_per_m2", type=["csv"])
    if uploaded is not None:
        ok, msg = set_exit_price_db_from_upload(uploaded)
        if ok:
            st.success(msg)
        else:
            st.error(msg)
    st.caption("Tip: this is how you plug in your 2026 dataset without editing code.")

# If using database: select suburb and auto-fill market price
selected_suburb = None
db_price = None

if exit_price_source == "Suburb database":
    suburbs = sorted(db["suburb"].dropna().astype(str).unique().tolist())
    selected_suburb = st.sidebar.selectbox("Select suburb", suburbs, index=0 if suburbs else None)
    if selected_suburb:
        row = db.loc[db["suburb"] == selected_suburb, "exit_price_per_m2"]
        db_price = float(row.iloc[0]) if len(row) else None

    override_price = st.sidebar.checkbox("Override suburb exit price", value=False)
    if (db_price is None) or override_price:
        market_price = st.sidebar.number_input(
            "Market Sales Price (per m2)",
            value=float(db_price) if db_price is not None else 35000.0,
            min_value=0.0,
            step=500.0,
        )
    else:
        market_price = db_price
        st.sidebar.caption(f"Auto: **R {market_price:,.0f}/m²** (from {selected_suburb})")

else:
    market_price = st.sidebar.number_input("Market Sales Price (per m2)", value=35000.0, min_value=0.0, step=500.0)

# -------------------------
# Financial inputs
# -------------------------
st.sidebar.header("4. Costs & Returns")
cost_mode = st.sidebar.radio(
    "Construction cost input mode",
    options=["R / m²", "% of GDV"],
    index=0,
)

pct_gdv_scope = "Hard cost only"
if cost_mode == "% of GDV":
    pct_gdv_scope = st.sidebar.radio(
        "%GDV applies to…",
        options=["Hard cost only", "Hard + soft (includes prof fees)"],
        index=0,
        help=(
            "Hard cost only: %GDV sets construction (hard) costs, then prof fees are added on top.\n"
            "Hard + soft: %GDV represents construction + professional fees (all-in), "
            "so the model avoids double counting fees."
        ),
    )

if cost_mode == "R / m²":
    const_cost_sqm = st.sidebar.number_input("Construction Cost (per m2)", value=17500.0, min_value=0.0, step=250.0)
    const_cost_pct_gdv = 0.0
else:
    const_cost_pct_gdv = st.sidebar.slider(
        "Construction Cost (% of GDV)",
        min_value=10,
        max_value=90,
        value=50,
        help="Construction costs as a % of GDV (scope depends on the toggle above).",
    ) / 100.0
    const_cost_sqm = 0.0

profit_margin = st.sidebar.slider("Target Profit (as % of GDV)", 10, 25, 20) / 100

# --- BUILT HERITAGE OVERLAY INPUTS ---
st.sidebar.header("5. Overlays")
st.sidebar.subheader("🏛️ Built Heritage Overlay")

heritage_enabled = st.sidebar.checkbox("Enable Built Heritage Overlay", value=False)

heritage_bonus_suppression = st.sidebar.slider(
    "Bonus suppression (%)",
    0, 100, 50,
    disabled=not heritage_enabled,
    help="0% = full bonus achievable; 100% = bonus fully blocked."
)
heritage_cost_uplift = st.sidebar.slider(
    "Construction cost uplift (%)",
    0, 40, 8,
    disabled=not heritage_enabled,
    help="Uplifts construction input (R/m² or % of GDV)."
)
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
# ENGINE RUN
# =========================
ff = ZONING_PRESETS[zoning_key]["ff"]

res = compute_model(
    land_area_m2=land_area,
    existing_gba_m2=existing_gba,
    ff=ff,
    density_bonus_pct=density_bonus,
    ih_pct=ih_percent,
    pt_zone_value=pt_zone,
    market_price_per_m2=market_price,
    ih_price_per_m2=IH_PRICE_PER_M2,
    profit_pct_gdv=profit_margin,
    base_prof_fee_rate=DEFAULT_PROF_FEE_RATE,
    overlay=heritage_overlay,
    cost_mode=cost_mode,
    base_cost_sqm=const_cost_sqm,
    base_cost_pct_gdv=const_cost_pct_gdv,
    pct_gdv_scope=pct_gdv_scope,
)

# =========================
# UI DISPLAY
# =========================
col1, col2 = st.columns([1, 1])

with col1:
    st.metric("Residual Land Value (RLV)", f"R {res['rlv']:,.2f}")

    st.success(f"""
**🌟 Incentive Effect**  
With **{ih_percent}% IH** and **{pt_zone}**, you save:  
### R {res['dc_savings']:,.2f}  
*in Development Charges (vs charging full DCs on all net increase).*
""")

    if res["brownfield_credit"]:
        st.warning("⚠️ Existing GBA exceeds proposed GBA. Net increase is zero; no DCs payable (Brownfield Credit).")

    # Show suburb + market price source
    if exit_price_source == "Suburb database" and selected_suburb:
        st.caption(f"Exit price suburb: **{selected_suburb}** • Market price used: **R {market_price:,.0f}/m²**")
    else:
        st.caption(f"Market price used: **R {market_price:,.0f}/m²**")

    # Construction cost readout (mode-aware)
    if res["cost_mode"] == "R / m²":
        st.caption(f"Construction input: **R {res['adj_cost_sqm']:,.0f}/m²**")
    else:
        scope_label = "hard only" if res["pct_gdv_scope"] == "Hard cost only" else "hard+soft"
        st.caption(
            f"Construction input: **{(res['adj_cost_pct_gdv']*100):.1f}% of GDV** ({scope_label}) "
            f"(implied **R {res['implied_cost_sqm']:,.0f}/m²**)"
        )

    if heritage_overlay.enabled:
        st.info(f"""
**🏛️ Built Heritage Overlay Active**
- Bonus suppression: **{heritage_overlay.bulk_reduction_pct:.0f}%** → effective bonus **{res['adj_bonus_pct']:.1f}%**
- Construction uplift: **{heritage_overlay.cost_uplift_pct:.0f}%**
- Fees rate uplift: **{heritage_overlay.fees_uplift_pct:.0f}%** → **{res['adj_fees_rate']*100:.2f}%**
- Profit uplift: **{heritage_overlay.profit_uplift_pct:.0f}%** → **{res['adj_profit_rate']*100:.2f}%**
""")

    st.caption(
        f"Proposed GBA: {res['proposed_gba']:,.0f} m²  •  Net increase: {res['net_increase']:,.0f} m²  •  "
        f"IH: {res['ih_gba']:,.0f} m²  •  Market: {res['market_gba']:,.0f} m²"
    )

with col2:
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
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

# Sensitivity Matrix (uses same engine + overlay + cost mode + scope)
st.subheader("Sensitivity Analysis: IH % vs Density Bonus (overlay-consistent)")

ih_levels = [0, 10, 20, 30]
bonus_levels = [0, 20, 40]

matrix_data = []
for ih in ih_levels:
    row = []
    for bonus in bonus_levels:
        tmp = compute_model(
            land_area_m2=land_area,
            existing_gba_m2=existing_gba,
            ff=ff,
            density_bonus_pct=bonus,
            ih_pct=ih,
            pt_zone_value=pt_zone,
            market_price_per_m2=market_price,
            ih_price_per_m2=IH_PRICE_PER_M2,
            profit_pct_gdv=profit_margin,
            base_prof_fee_rate=DEFAULT_PROF_FEE_RATE,
            overlay=heritage_overlay,
            cost_mode=cost_mode,
            base_cost_sqm=const_cost_sqm,
            base_cost_pct_gdv=const_cost_pct_gdv,
            pct_gdv_scope=pct_gdv_scope,
        )
        row.append(f"R {tmp['rlv'] / 1_000_000:.1f}M")
    matrix_data.append(row)

df_matrix = pd.DataFrame(
    matrix_data,
    index=[f"{x}% IH" for x in ih_levels],
    columns=[f"{x}% Bonus" for x in bonus_levels],
)
st.table(df_matrix)

# Optional: show the current DB (so you can sanity check what’s loaded)
with st.expander("View exit price database"):
    st.dataframe(load_exit_price_db(), use_container_width=True)
