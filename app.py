import streamlit as st
import streamlit.components.v1 as components

# City of Cape Town Map Viewer
DEFAULT_CITYMAP_URL = "https://citymaps.capetown.gov.za/EGISViewer/"
CITYMAP_VIEWER_URL = st.secrets.get("CITYMAP_VIEWER_URL", DEFAULT_CITYMAP_URL)

import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass

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

IH_PRICE_PER_M2 = 15000  # IH capped exit price (sellable m²)

# 2026 hard cost tiers (benchmarks)
COST_TIERS = {
    "Economic (R10,000/m²)": 10000.0,
    "Mid-Tier (R18,000/m²)": 18000.0,
    "Luxury (R25,000+/m²)": 25000.0,
}

# Exit price DB (2026 sectional title new apartments) — ranges per m² sellable
# Note: Clifton/Bantry Bay upper assumed 170,000 due to truncated source.
DEFAULT_EXIT_PRICES = [
    {"suburb": "Clifton / Bantry Bay", "min_price_per_m2": 120000, "max_price_per_m2": 170000},
    {"suburb": "Sea Point / Green Point", "min_price_per_m2": 65000, "max_price_per_m2": 85000},
    {"suburb": "City Bowl (CBD / Gardens)", "min_price_per_m2": 45000, "max_price_per_m2": 60000},
    {"suburb": "Claremont / Rondebosch", "min_price_per_m2": 40000, "max_price_per_m2": 52000},
    {"suburb": "Woodstock / Salt River", "min_price_per_m2": 32000, "max_price_per_m2": 42000},
    {"suburb": "Durbanville / Sunningdale", "min_price_per_m2": 25000, "max_price_per_m2": 35000},
    {"suburb": "Khayelitsha / Mitchells Plain", "min_price_per_m2": 10000, "max_price_per_m2": 15000},
]

# Professional fee ranges from your 2026 image (as %)
PROF_FEE_RANGES = {
    "Architect": (0.05, 0.07),
    "Quantity Surveyor (QS)": (0.02, 0.03),
    "Structural Engineer": (0.015, 0.02),
    "Civil Engineer": (0.01, 0.015),
    "Electrical/Mech Engineer": (0.01, 0.02),
    "Project Manager": (0.02, 0.03),
}
# Target “2026 average total” per image (~12–15%) => midpoint 13.5%
PROF_FEE_TARGET_TOTAL = 0.135

st.set_page_config(page_title="Residuo — Cape Town Feasibility", layout="wide")
st.title("🏗️ Residuo — Cape Town Feasibility (Wizard + Live Dashboard)")

# =========================
# OVERLAYS
# =========================
@dataclass(frozen=True)
class HeritageOverlay:
    enabled: bool
    bulk_reduction_pct: float      # interpreted as "BONUS suppression %"
    cost_uplift_pct: float         # uplifts construction input (R/m² or %GDV)
    fees_uplift_pct: float         # uplifts total prof fee rate
    profit_uplift_pct: float       # uplifts profit % of GDV


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
    Uses midpoints of each range, then scales all components so the sum equals 13.5% (target).
    """
    mids = {k: (v[0] + v[1]) / 2.0 for k, v in PROF_FEE_RANGES.items()}
    s = sum(mids.values())
    if s <= 0:
        return {k: 0.0 for k in mids}
    scale = PROF_FEE_TARGET_TOTAL / s
    return {k: mids[k] * scale for k in mids}


def compute_market_price(db: pd.DataFrame) -> tuple[float, dict]:
    """
    Returns (market_price, meta)
    """
    src = st.session_state.get("exit_price_source", "Suburb database")

    if src == "Manual entry":
        return float(st.session_state.get("market_price_manual", 35000.0)), {
            "mode": "manual",
            "suburb": None,
            "min": None,
            "max": None,
            "point": None,
        }

    # Suburb DB mode
    suburbs = sorted(db["suburb"].dropna().astype(str).unique().tolist())
    suburb = st.session_state.get("selected_suburb", suburbs[0] if suburbs else None)
    point = st.session_state.get("price_point", "Mid")
    override = bool(st.session_state.get("override_suburb_price", False))

    db_min = db_max = db_price = None
    if suburb:
        row = db.loc[db["suburb"] == suburb]
        if not row.empty:
            db_min = float(row["min_price_per_m2"].iloc[0])
            db_max = float(row["max_price_per_m2"].iloc[0])
            if point == "Low":
                db_price = db_min
            elif point == "High":
                db_price = db_max
            else:
                db_price = (db_min + db_max) / 2.0

    if override or (db_price is None):
        return float(st.session_state.get("market_price_override", db_price if db_price else 35000.0)), {
            "mode": "db_override" if override else "db_missing",
            "suburb": suburb,
            "min": db_min,
            "max": db_max,
            "point": point,
        }

    return float(db_price), {"mode": "db", "suburb": suburb, "min": db_min, "max": db_max, "point": point}


def compute_model(
    land_area_m2: float,
    existing_gba_bulk_m2: float,
    ff: float,
    density_bonus_pct: float,
    efficiency_ratio: float,  # sellable/bulk
    ih_pct: float,
    pt_zone_value: str,
    market_price_per_sellable_m2: float,
    ih_price_per_sellable_m2: float,
    profit_pct_gdv: float,
    base_prof_fee_rate: float,  # total prof fee rate
    overlay: HeritageOverlay,
    cost_mode: str,             # "R / m²" or "% of GDV"
    base_cost_sqm: float,       # if "R / m²"
    base_cost_pct_gdv: float,   # if "% of GDV"
    pct_gdv_scope: str,         # "Hard cost only" or "Hard + soft"
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

    # Brownfield credit / net increase (bulk)
    net_increase_bulk = max(0.0, proposed_bulk - existing_gba_bulk_m2)

    # IH split on net increase bulk (for DC logic); revenue uses sellable
    ih_increase_bulk = net_increase_bulk * (ih_pct / 100.0)
    market_increase_bulk = net_increase_bulk - ih_increase_bulk

    ih_sellable = ih_increase_bulk * efficiency_ratio
    market_sellable = max(0.0, proposed_sellable - ih_sellable)

    # DCs on market share of net increase (bulk)
    disc = pt_discount(pt_zone_value)
    roads_dc = market_increase_bulk * ROADS_TRANSPORT_PORTION * disc
    other_dc = market_increase_bulk * (DC_BASE_RATE - ROADS_TRANSPORT_PORTION)
    total_dc = roads_dc + other_dc

    total_potential_dc = net_increase_bulk * DC_BASE_RATE
    dc_savings = total_potential_dc - total_dc

    # Revenue on sellable
    gdv = (market_sellable * market_price_per_sellable_m2) + (ih_sellable * ih_price_per_sellable_m2)

    # Costs
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
            # If %GDV represents (construction + prof fees) all-in, solve without double counting:
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
        "net_increase_bulk": net_increase_bulk,
        "market_increase_bulk": market_increase_bulk,
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


def fmt_money(x: float) -> str:
    return f"R {x:,.0f}"


def fmt_money2(x: float) -> str:
    return f"R {x:,.2f}"


def init_defaults():
    # Wizard state
    st.session_state.setdefault("wizard_step", "1) Site")

    # Site
    st.session_state.setdefault("land_area", 1000.0)
    st.session_state.setdefault("existing_gba", 200.0)
    st.session_state.setdefault("zoning_key", "GR2 (Suburban)")
    st.session_state.setdefault("pt_zone", "Standard")

    # Policy
    st.session_state.setdefault("ih_percent", 20)
    st.session_state.setdefault("density_bonus", 20)

    # Benchmarks
    st.session_state.setdefault("efficiency_ratio", 0.85)
    st.session_state.setdefault("profit_margin", 0.20)

    # Exit prices
    st.session_state.setdefault("exit_price_source", "Suburb database")
    st.session_state.setdefault("price_point", "Mid")
    st.session_state.setdefault("override_suburb_price", False)
    st.session_state.setdefault("market_price_override", 35000.0)
    st.session_state.setdefault("market_price_manual", 35000.0)

    # Costs
    st.session_state.setdefault("cost_mode", "R / m²")
    st.session_state.setdefault("pct_gdv_scope", "Hard cost only")
    st.session_state.setdefault("build_tier", "Mid-Tier (R18,000/m²)")
    st.session_state.setdefault("const_cost_sqm", COST_TIERS["Mid-Tier (R18,000/m²)"])
    st.session_state.setdefault("const_cost_pct_gdv", 0.50)

    # Fees (components default scaled)
    defaults = default_prof_fee_components_scaled_to_target()
    for k in PROF_FEE_RANGES.keys():
        st.session_state.setdefault(f"fee_{k}", float(defaults[k]))

    # Overlay
    st.session_state.setdefault("heritage_enabled", False)
    st.session_state.setdefault("heritage_bonus_suppression", 50)
    st.session_state.setdefault("heritage_cost_uplift", 8)
    st.session_state.setdefault("heritage_fees_uplift", 5)
    st.session_state.setdefault("heritage_profit_uplift", 5)


init_defaults()

# =========================
# OPTION B LAYOUT
# =========================
left, right = st.columns([0.38, 0.62], gap="large")

# =========================
# LEFT: Wizard inputs
# =========================
with left:
    st.subheader("🧭 Inputs Wizard")

    steps = [
        "1) Site",
        "2) Policy",
        "3) Benchmarks",
        "4) Exit Prices",
        "5) Professional Fees",
        "6) Construction Costs",
        "7) Overlays",
        "8) Review",
    ]
    st.radio("Step", steps, key="wizard_step")

    step = st.session_state["wizard_step"]
    db = load_exit_price_db()

    if step == "1) Site":
        st.markdown("**Site parameters**")
        st.number_input("Land Area (m²)", min_value=0.0, step=50.0, key="land_area")
        st.number_input("Existing GBA on Site (m² bulk)", min_value=0.0, step=25.0, key="existing_gba")
        st.selectbox("Zoning Preset", list(ZONING_PRESETS.keys()), key="zoning_key")
        st.selectbox("PT Zone (Parking/DC Discount)", ["Standard", "PT1", "PT2"], key="pt_zone")

        with st.expander("🗺️ City of Cape Town Map Viewer", expanded=False):
            st.caption("If the embedded map doesn’t load, use the button to open it in a new tab.")
            st.link_button("Open City Map Viewer", CITYMAP_VIEWER_URL)
            components.iframe(CITYMAP_VIEWER_URL, height=460, scrolling=True)

    elif step == "2) Policy":
        st.markdown("**Policy toggles**")
        st.slider("Inclusionary Housing (%)", 0, 30, key="ih_percent")
        st.slider("Density Bonus (%)", 0, 50, key="density_bonus")

    elif step == "3) Benchmarks":
        st.markdown("**2026 benchmark defaults**")
        st.slider("Efficiency Ratio (sellable ÷ bulk)", 0.60, 0.95, step=0.01, key="efficiency_ratio")
        st.slider("Developer Profit (as % of GDV)", 0.10, 0.30, step=0.01, key="profit_margin")

        st.caption("Profit default is **20%** (often required by SA development lenders).")

    elif step == "4) Exit Prices":
        st.markdown("**Market exit price (sellable m²)**")

        st.radio("Source", ["Suburb database", "Manual entry"], key="exit_price_source")

        with st.expander("Upload exit price CSV (optional)", expanded=False):
            uploaded = st.file_uploader("CSV", type=["csv"])
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
                st.success("Reset complete.")

        if st.session_state["exit_price_source"] == "Suburb database":
            suburbs = sorted(db["suburb"].dropna().astype(str).unique().tolist())
            if suburbs:
                st.selectbox("Select suburb group", suburbs, key="selected_suburb")
                st.radio("Point in range", ["Low", "Mid", "High"], horizontal=True, key="price_point")
                st.checkbox("Override suburb price", key="override_suburb_price")

                # Show current DB range
                suburb = st.session_state.get("selected_suburb")
                row = db.loc[db["suburb"] == suburb]
                if not row.empty:
                    mn = float(row["min_price_per_m2"].iloc[0])
                    mx = float(row["max_price_per_m2"].iloc[0])
                    st.caption(f"Range: **R {mn:,.0f} – R {mx:,.0f}/m²**")

                if st.session_state.get("override_suburb_price", False):
                    st.number_input("Override exit price (R/m²)", min_value=0.0, step=500.0, key="market_price_override")
            else:
                st.warning("Exit price DB is empty. Switch to Manual entry or upload a CSV.")
                st.session_state["exit_price_source"] = "Manual entry"
                st.number_input("Market exit price (R/m²)", min_value=0.0, step=500.0, key="market_price_manual")
        else:
            st.number_input("Market exit price (R/m²)", min_value=0.0, step=500.0, key="market_price_manual")

        st.caption("These prices apply to **new sectional title apartments** (sellable m²).")

    elif step == "5) Professional Fees":
        st.markdown("**Professional fees (2026 ranges)**")
        st.caption("Defaults are scaled to a **total ~13.5%** (midpoint of ~12–15%).")

        with st.expander("Fee components", expanded=True):
            for name, (lo, hi) in PROF_FEE_RANGES.items():
                st.slider(
                    f"{name} (%)",
                    min_value=float(lo * 100),
                    max_value=float(hi * 100),
                    step=0.1,
                    key=f"fee_{name}",
                )

        total_fees = sum(float(st.session_state[f"fee_{k}"]) for k in PROF_FEE_RANGES.keys())
        st.success(f"Total professional fees: **{total_fees*100:.2f}%**")

    elif step == "6) Construction Costs":
        st.markdown("**Construction cost inputs**")

        st.radio("Cost input mode", ["R / m²", "% of GDV"], key="cost_mode")

        if st.session_state["cost_mode"] == "% of GDV":
            st.radio("%GDV applies to…", ["Hard cost only", "Hard + soft (includes prof fees)"], key="pct_gdv_scope")
            st.slider("Construction Cost (% of GDV)", 0.10, 0.90, step=0.01, key="const_cost_pct_gdv")
        else:
            st.selectbox("Hard Cost Tier (2026)", list(COST_TIERS.keys()), key="build_tier")
            # If user hasn’t manually changed const_cost_sqm, keep it aligned with tier
            tier_val = COST_TIERS[st.session_state["build_tier"]]
            if "const_cost_sqm_touched" not in st.session_state:
                st.session_state["const_cost_sqm"] = float(tier_val)

            use_tier = st.checkbox("Use tier default", value=True, key="use_tier_default")
            if use_tier:
                st.session_state["const_cost_sqm"] = float(tier_val)
                st.info(f"Using tier default: **R {tier_val:,.0f}/m² bulk**")
            else:
                st.session_state["const_cost_sqm_touched"] = True
                st.number_input("Hard Construction Cost (R per m² bulk)", min_value=0.0, step=250.0, key="const_cost_sqm")

    elif step == "7) Overlays":
        st.markdown("**Overlays**")
        st.checkbox("Enable Built Heritage Overlay", key="heritage_enabled")

        disabled = not bool(st.session_state["heritage_enabled"])
        st.slider("Bonus suppression (%)", 0, 100, key="heritage_bonus_suppression", disabled=disabled)
        st.slider("Construction cost uplift (%)", 0, 40, key="heritage_cost_uplift", disabled=disabled)
        st.slider("Professional fees uplift (%)", 0, 40, key="heritage_fees_uplift", disabled=disabled)
        st.slider("Profit requirement uplift (%)", 0, 40, key="heritage_profit_uplift", disabled=disabled)

        st.caption("Overlay suppresses bonus and uplifts cost/fees/profit inputs.")

    elif step == "8) Review":
        st.markdown("**Review key inputs**")

        db = load_exit_price_db()
        market_price, price_meta = compute_market_price(db)

        total_fees = sum(float(st.session_state[f"fee_{k}"]) for k in PROF_FEE_RANGES.keys())

        st.write(
            {
                "Land area (m²)": float(st.session_state["land_area"]),
                "Existing GBA bulk (m²)": float(st.session_state["existing_gba"]),
                "Zoning": st.session_state["zoning_key"],
                "PT zone": st.session_state["pt_zone"],
                "IH %": int(st.session_state["ih_percent"]),
                "Density bonus %": int(st.session_state["density_bonus"]),
                "Efficiency %": round(float(st.session_state["efficiency_ratio"]) * 100, 0),
                "Profit %GDV": round(float(st.session_state["profit_margin"]) * 100, 1),
                "Exit price mode": price_meta["mode"],
                "Selected suburb": price_meta.get("suburb"),
                "Exit price used (R/m² sellable)": round(float(market_price), 0),
                "Cost mode": st.session_state["cost_mode"],
                "Hard cost (R/m² bulk)": float(st.session_state.get("const_cost_sqm", 0.0)),
                "Cost %GDV": float(st.session_state.get("const_cost_pct_gdv", 0.0)),
                "Cost %GDV scope": st.session_state.get("pct_gdv_scope"),
                "Prof fees total %": round(total_fees * 100, 2),
                "Heritage overlay": bool(st.session_state["heritage_enabled"]),
            }
        )

# =========================
# RIGHT: Live Dashboard
# =========================
with right:
    # Gather inputs for model
    db = load_exit_price_db()
    market_price, price_meta = compute_market_price(db)

    zoning_key = st.session_state["zoning_key"]
    ff = ZONING_PRESETS[zoning_key]["ff"]

    total_prof_fee_rate = sum(float(st.session_state[f"fee_{k}"]) for k in PROF_FEE_RANGES.keys())

    heritage_overlay = HeritageOverlay(
        enabled=bool(st.session_state["heritage_enabled"]),
        bulk_reduction_pct=float(st.session_state["heritage_bonus_suppression"]),
        cost_uplift_pct=float(st.session_state["heritage_cost_uplift"]),
        fees_uplift_pct=float(st.session_state["heritage_fees_uplift"]),
        profit_uplift_pct=float(st.session_state["heritage_profit_uplift"]),
    )

    cost_mode = st.session_state["cost_mode"]
    pct_gdv_scope = st.session_state.get("pct_gdv_scope", "Hard cost only")

    const_cost_sqm = float(st.session_state.get("const_cost_sqm", COST_TIERS["Mid-Tier (R18,000/m²)"]))
    const_cost_pct_gdv = float(st.session_state.get("const_cost_pct_gdv", 0.50))

    res = compute_model(
        land_area_m2=float(st.session_state["land_area"]),
        existing_gba_bulk_m2=float(st.session_state["existing_gba"]),
        ff=float(ff),
        density_bonus_pct=float(st.session_state["density_bonus"]),
        efficiency_ratio=float(st.session_state["efficiency_ratio"]),
        ih_pct=float(st.session_state["ih_percent"]),
        pt_zone_value=st.session_state["pt_zone"],
        market_price_per_sellable_m2=float(market_price),
        ih_price_per_sellable_m2=float(IH_PRICE_PER_M2),
        profit_pct_gdv=float(st.session_state["profit_margin"]),
        base_prof_fee_rate=float(total_prof_fee_rate),
        overlay=heritage_overlay,
        cost_mode=cost_mode,
        base_cost_sqm=const_cost_sqm,
        base_cost_pct_gdv=const_cost_pct_gdv,
        pct_gdv_scope=pct_gdv_scope,
    )

    # Mini "last change impact"
    prev_rlv = st.session_state.get("_prev_rlv")
    delta_rlv = None if prev_rlv is None else (res["rlv"] - prev_rlv)

    st.subheader("📊 Live Dashboard")

    # KPI strip
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Residual Land Value", fmt_money(res["rlv"]), delta=None if delta_rlv is None else fmt_money(delta_rlv))
    k2.metric("GDV (Market + IH)", fmt_money(res["gdv"]))
    k3.metric("Development Charges", fmt_money(res["total_dc"]))
    k4.metric("DC Savings", fmt_money(res["dc_savings"]))

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Profit (20% default)", fmt_money(res["profit"]))
    k6.metric("Prof Fees (sum)", fmt_money(res["prof_fees"]))
    k7.metric("Proposed Bulk (m²)", f"{res['proposed_bulk']:,.0f}")
    k8.metric("Sellable (m²)", f"{res['proposed_sellable']:,.0f}")

    # Assumptions chips
    chips = []
    chips.append(f"Zoning: {zoning_key}")
    chips.append(f"Efficiency: {res['efficiency_ratio']*100:.0f}%")
    chips.append(f"IH: {int(st.session_state['ih_percent'])}%")
    chips.append(f"Bonus: {int(st.session_state['density_bonus'])}% → eff {res['adj_bonus_pct']:.1f}%")
    chips.append(f"Exit: {fmt_money(market_price)}/m² (sellable)")
    if cost_mode == "R / m²":
        chips.append(f"Hard cost: {fmt_money(res['adj_cost_sqm'])}/m² (bulk)")
    else:
        chips.append(f"Cost: {res['adj_cost_pct_gdv']*100:.1f}% GDV ({pct_gdv_scope})")
    chips.append(f"Fees: {res['adj_fees_rate']*100:.2f}%")
    chips.append(f"Profit: {res['adj_profit_rate']*100:.1f}%")
    if heritage_overlay.enabled:
        chips.append("Heritage: ON")

    st.caption(" • ".join(chips))

    if res["brownfield_credit"]:
        st.warning("⚠️ Existing bulk exceeds proposed bulk. Net increase is zero; DCs should be zero (brownfield credit).")

    # Waterfall
    fig = go.Figure(
        go.Waterfall(
            name="Residual Breakdown",
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
            connector={"line": {"color": "rgb(63,63,63)"}},
        )
    )
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Sensitivity (kept compact but consistent with engine)
    with st.expander("Sensitivity: IH % vs Density Bonus (2026-aware)", expanded=False):
        ih_levels = [0, 10, 20, 30]
        bonus_levels = [0, 20, 40]
        matrix = []

        for ih in ih_levels:
            row = []
            for bonus in bonus_levels:
                tmp = compute_model(
                    land_area_m2=float(st.session_state["land_area"]),
                    existing_gba_bulk_m2=float(st.session_state["existing_gba"]),
                    ff=float(ff),
                    density_bonus_pct=float(bonus),
                    efficiency_ratio=float(st.session_state["efficiency_ratio"]),
                    ih_pct=float(ih),
                    pt_zone_value=st.session_state["pt_zone"],
                    market_price_per_sellable_m2=float(market_price),
                    ih_price_per_sellable_m2=float(IH_PRICE_PER_M2),
                    profit_pct_gdv=float(st.session_state["profit_margin"]),
                    base_prof_fee_rate=float(total_prof_fee_rate),
                    overlay=heritage_overlay,
                    cost_mode=cost_mode,
                    base_cost_sqm=const_cost_sqm,
                    base_cost_pct_gdv=const_cost_pct_gdv,
                    pct_gdv_scope=pct_gdv_scope,
                )
                row.append(f"R {tmp['rlv']/1_000_000:.1f}M")
            matrix.append(row)

        df_matrix = pd.DataFrame(
            matrix,
            index=[f"{x}% IH" for x in ih_levels],
            columns=[f"{x}% Bonus" for x in bonus_levels],
        )
        st.table(df_matrix)

    with st.expander("Exit price database (2026 estimates)", expanded=False):
        st.dataframe(load_exit_price_db(), use_container_width=True)

    # Update last value AFTER rendering
    st.session_state["_prev_rlv"] = float(res["rlv"])
