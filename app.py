# app.py — Residuo (IH + RLV Calculator) — POLISHED UI SINGLE FILE
# ✅ Professional front-end layout (brand header, KPI cards, clean sidebar sections)
# ✅ Uses your existing asset filenames (wordmark + glyph)
# ✅ City Map Viewer embed + open-in-new-tab fallback
# ✅ Clickable Plotly heatmap with metric dropdown (RLV/GDV/Profit/Costs)
# ✅ Costs metric = Hard+Soft ONLY (never includes DCs)
# ✅ Scenario breakdown includes IH exit price (R/m² sellable)
# ✅ IH exit price slider under "2) Policy" (R10k–R30k)
# ✅ Bankability gauge replaces waterfall
# ✅ Selected scenario under sensitivity recalculates LIVE
# ✅ Selected scenario now follows policy sliders by default (pins on click; slider movement unpins)
# ✅ NEW: Key outputs now includes a Built Heritage overlay BLUE popout (styled card)

import os
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# =========================
# BRANDING (Residuo)
# =========================
APP_NAME = "Residuo"
TAGLINE = "Unlock Land's True Value"

LOGO_PATH = "assets/residuo_D_wordmark_transparent_darktext_1200w_clean.png"
ICON_PATH = "assets/residuo_D_glyph_transparent_256_clean.png"

# =========================
# CITY MAP VIEWER
# =========================
DEFAULT_CITYMAP_URL = "https://citymaps.capetown.gov.za/EGISViewer/"
CITYMAP_VIEWER_URL = st.secrets.get("CITYMAP_VIEWER_URL", DEFAULT_CITYMAP_URL)

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

IH_EXIT_PRICE_DEFAULT = 15000  # default IH exit price (R/m² sellable)

COST_TIERS = {
    "Economic (R10,000/m²)": 10000.0,
    "Mid-Tier (R18,000/m²)": 18000.0,
    "Luxury (R25,000+/m²)": 25000.0,
}

DEFAULT_EXIT_PRICES = [
    {"suburb": "Clifton / Bantry Bay", "min_price_per_m2": 120000, "max_price_per_m2": 170000},
    {"suburb": "Sea Point / Green Point", "min_price_per_m2": 65000, "max_price_per_m2": 85000},
    {"suburb": "City Bowl (CBD / Gardens)", "min_price_per_m2": 45000, "max_price_per_m2": 60000},
    {"suburb": "Claremont / Rondebosch", "min_price_per_m2": 40000, "max_price_per_m2": 52000},
    {"suburb": "Woodstock / Salt River", "min_price_per_m2": 32000, "max_price_per_m2": 42000},
    {"suburb": "Durbanville / Sunningdale", "min_price_per_m2": 25000, "max_price_per_m2": 35000},
    {"suburb": "Khayelitsha / Mitchells Plain", "min_price_per_m2": 10000, "max_price_per_m2": 15000},
]

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
# UTILS
# =========================
def _file_exists(path: str) -> bool:
    try:
        return os.path.isfile(path)
    except Exception:
        return False


def fmt_money(x: float) -> str:
    try:
        return f"R {x:,.0f}"
    except Exception:
        return "—"


def fmt_pct(x: float, dp: int = 1) -> str:
    try:
        return f"{x*100:.{dp}f}%"
    except Exception:
        return "—"


# =========================
# PAGE CONFIG (must be first Streamlit call)
# =========================
page_icon = ICON_PATH if _file_exists(ICON_PATH) else "🏗️"
st.set_page_config(page_title=APP_NAME, page_icon=page_icon, layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
      [data-testid="stSidebar"] { padding-top: 1rem; }

      .kpi-card {
        border: 1px solid rgba(0,0,0,.08);
        border-radius: 16px;
        padding: 14px 16px;
        background: rgba(255,255,255,.70);
      }
      .kpi-title { font-size: 0.85rem; opacity: 0.72; margin-bottom: 4px; }
      .kpi-value { font-size: 1.35rem; font-weight: 800; line-height: 1.1; }
      .kpi-sub { font-size: 0.85rem; opacity: 0.70; margin-top: 4px; }

      .section-title { font-size: 1.05rem; font-weight: 800; margin: 0.5rem 0 0.25rem; }
      .muted { opacity: 0.75; }
      .hr { height: 1px; background: rgba(0,0,0,.08); margin: 0.75rem 0; }

      .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        background: rgba(0,0,0,.05);
        margin-left: 8px;
      }

      /* NEW: Heritage popout */
      .popout-blue {
        border: 1px solid rgba(17, 99, 255, 0.20);
        border-left: 6px solid rgba(17, 99, 255, 0.85);
        background: rgba(17, 99, 255, 0.06);
        border-radius: 16px;
        padding: 12px 14px;
      }
      .popout-title {
        font-weight: 800;
        margin-bottom: 4px;
        font-size: 0.95rem;
      }
      .popout-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px 14px;
        margin-top: 6px;
        font-size: 0.90rem;
      }
      .pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        background: rgba(17, 99, 255, 0.10);
        margin-left: 8px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# BRAND HEADER
# =========================
h1, h2 = st.columns([2.2, 7.8], vertical_alignment="center")
with h1:
    if _file_exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.markdown(f"## {APP_NAME}")
with h2:
    st.markdown(
        f"""
        <div>
          <div style="font-size:1.65rem; font-weight:900; line-height:1.05;">{APP_NAME}
            <span class="badge">Cape Town Feasibility</span>
          </div>
          <div class="muted" style="margin-top:6px;">{TAGLINE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================
# OVERLAYS
# =========================
@dataclass(frozen=True)
class HeritageOverlay:
    enabled: bool
    bulk_reduction_pct: float  # interpreted as "BONUS suppression %"
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


# =========================
# HELPERS
# =========================
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

    if cost_mode == "R / m²":
        adj_cost_sqm = adj_cost_input
        construction_costs = proposed_bulk * adj_cost_sqm
        hard_plus_dc = construction_costs + total_dc
        prof_fees = hard_plus_dc * adj_fees_rate
        adj_cost_pct_gdv = None
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
        adj_cost_sqm = None

    profit = gdv * adj_profit_rate
    rlv = gdv - (construction_costs + total_dc + prof_fees + profit)

    implied_cost_sqm = (construction_costs / proposed_bulk) if proposed_bulk > 0 else 0.0
    hard_soft_costs = construction_costs + prof_fees  # excludes DCs

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
        "hard_soft_costs": hard_soft_costs,
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
        "net_increase_bulk": float(net_increase_bulk),
    }


# =========================
# BANKABILITY GAUGE (Bullet-style)
# =========================
def make_bankability_gauge(
    res: dict,
    *,
    land_area_m2: float,
    tier_cost_sqm: float,
    cost_tier_name: str,
    profit_target: float = 0.20,
    prof_fee_band: tuple[float, float] = (0.12, 0.15),
) -> go.Figure:
    profit_pct = float(res.get("adj_profit_rate", 0.0))
    fees_pct = float(res.get("adj_fees_rate", 0.0))
    implied_cost_sqm = float(res.get("implied_cost_sqm", 0.0))
    rlv = float(res.get("rlv", 0.0))
    rlv_per_land = (rlv / land_area_m2) if land_area_m2 > 0 else 0.0

    profit_max = 0.30
    fees_max = 0.20
    fees_lo, fees_hi = prof_fee_band

    cost_max = max(tier_cost_sqm * 1.6, implied_cost_sqm * 1.25, 1.0)
    cost_min = 0.0

    rlv_land_max = max(abs(rlv_per_land) * 1.25, 1000.0)
    rlv_land_min = -rlv_land_max

    labels = [
        "Profit (as % of GDV)",
        "Professional fees (as %)",
        f"Implied hard cost (R/m² bulk) vs {cost_tier_name}",
        "RLV per m² land",
    ]

    def norm(v, vmin, vmax):
        if vmax - vmin == 0:
            return 0.0
        return (v - vmin) / (vmax - vmin)

    profit_n = max(0.0, min(1.0, norm(profit_pct, 0.0, profit_max)))
    fees_n = max(0.0, min(1.0, norm(fees_pct, 0.0, fees_max)))
    cost_n = max(0.0, min(1.0, norm(implied_cost_sqm, cost_min, cost_max)))
    rlv_n = max(0.0, min(1.0, norm(rlv_per_land, rlv_land_min, rlv_land_max)))

    profit_target_n = norm(profit_target, 0.0, profit_max)
    fees_lo_n = norm(fees_lo, 0.0, fees_max)
    fees_hi_n = norm(fees_hi, 0.0, fees_max)
    tier_n = norm(tier_cost_sqm, cost_min, cost_max)

    y = list(range(len(labels)))[::-1]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[1 - profit_target_n], y=[y[0]], base=[profit_target_n],
        orientation="h", hoverinfo="skip", marker=dict(opacity=0.18), showlegend=False
    ))
    fig.add_trace(go.Bar(
        x=[max(0.0, fees_hi_n - fees_lo_n)], y=[y[1]], base=[fees_lo_n],
        orientation="h", hoverinfo="skip", marker=dict(opacity=0.18), showlegend=False
    ))
    cost_band_lo = max(0.0, tier_cost_sqm * 0.90)
    cost_band_hi = tier_cost_sqm * 1.10
    fig.add_trace(go.Bar(
        x=[max(0.0, norm(cost_band_hi, cost_min, cost_max) - norm(cost_band_lo, cost_min, cost_max))],
        y=[y[2]], base=[norm(cost_band_lo, cost_min, cost_max)],
        orientation="h", hoverinfo="skip", marker=dict(opacity=0.18), showlegend=False
    ))

    fig.add_trace(go.Bar(
        x=[profit_n, fees_n, cost_n, rlv_n],
        y=y,
        orientation="h",
        marker=dict(line=dict(width=0)),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=[
            f"Profit: {profit_pct*100:.1f}% (target ≥ {profit_target*100:.0f}%)",
            f"Fees: {fees_pct*100:.2f}% (band {fees_lo*100:.0f}–{fees_hi*100:.0f}%)",
            f"Implied hard cost: {fmt_money(implied_cost_sqm)} / m² bulk (tier {fmt_money(tier_cost_sqm)})",
            f"RLV per land: {fmt_money(rlv_per_land)} / m² land",
        ],
        showlegend=False,
    ))

    fig.add_shape(type="line", x0=profit_target_n, x1=profit_target_n, y0=y[0]-0.35, y1=y[0]+0.35,
                  xref="x", yref="y", line=dict(width=3))
    fig.add_shape(type="line", x0=fees_lo_n, x1=fees_lo_n, y0=y[1]-0.35, y1=y[1]+0.35,
                  xref="x", yref="y", line=dict(width=2, dash="dot"))
    fig.add_shape(type="line", x0=fees_hi_n, x1=fees_hi_n, y0=y[1]-0.35, y1=y[1]+0.35,
                  xref="x", yref="y", line=dict(width=2, dash="dot"))
    fig.add_shape(type="line", x0=tier_n, x1=tier_n, y0=y[2]-0.35, y1=y[2]+0.35,
                  xref="x", yref="y", line=dict(width=2, dash="dot"))

    fig.update_layout(
        height=430,
        margin=dict(l=10, r=70, t=10, b=10),
        xaxis=dict(range=[0, 1.0], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=[
            dict(x=0.0, y=y[0], xref="x", yref="y", text=labels[0], showarrow=False, xanchor="left", yanchor="middle"),
            dict(x=0.0, y=y[1], xref="x", yref="y", text=labels[1], showarrow=False, xanchor="left", yanchor="middle"),
            dict(x=0.0, y=y[2], xref="x", yref="y", text=labels[2], showarrow=False, xanchor="left", yanchor="middle"),
            dict(x=0.0, y=y[3], xref="x", yref="y", text=labels[3], showarrow=False, xanchor="left", yanchor="middle"),
            dict(x=1.02, y=y[0], xref="x", yref="y", text=f"<b>{profit_pct*100:.1f}%</b>", showarrow=False, xanchor="left"),
            dict(x=1.02, y=y[1], xref="x", yref="y", text=f"<b>{fees_pct*100:.2f}%</b>", showarrow=False, xanchor="left"),
            dict(x=1.02, y=y[2], xref="x", yref="y", text=f"<b>{fmt_money(implied_cost_sqm)}/m²</b>", showarrow=False, xanchor="left"),
            dict(x=1.02, y=y[3], xref="x", yref="y", text=f"<b>{fmt_money(rlv_per_land)}/m²</b>", showarrow=False, xanchor="left"),
        ],
    )
    return fig


def render_heritage_popout(overlay: HeritageOverlay, res: dict, base_bonus_pct: float) -> None:
    if not overlay.enabled:
        return
    st.markdown(
        f"""
        <div class="popout-blue">
          <div class="popout-title">🏛️ Built Heritage overlay <span class="pill">Active</span></div>
          <div class="muted" style="font-size:0.9rem;">
            Overlay adjusts bonus, costs, fees and profit to reflect heritage controls / approval risk.
          </div>
          <div class="popout-grid">
            <div><b>Bonus suppression</b></div><div>{overlay.bulk_reduction_pct:.0f}%</div>
            <div><b>Base bonus</b></div><div>{base_bonus_pct:.0f}% → <b>{res["adj_bonus_pct"]:.1f}%</b></div>
            <div><b>Cost uplift</b></div><div>{overlay.cost_uplift_pct:.0f}%</div>
            <div><b>Fees uplift</b></div><div>{overlay.fees_uplift_pct:.0f}%</div>
            <div><b>Profit uplift</b></div><div>{overlay.profit_uplift_pct:.0f}%</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# SIDEBAR
# =========================
if _file_exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_container_width=True)
st.sidebar.caption(TAGLINE)
st.sidebar.markdown('<div class="hr"></div>', unsafe_allow_html=True)

st.sidebar.markdown("### 1) Site")
land_area = st.sidebar.number_input("Land Area (m²)", value=1000.0, min_value=0.0, step=50.0)
existing_gba = st.sidebar.number_input("Existing GBA (m² bulk)", value=200.0, min_value=0.0, step=25.0)
zoning_key = st.sidebar.selectbox("Zoning Preset", list(ZONING_PRESETS.keys()))
pt_zone = st.sidebar.selectbox("PT Zone (Parking/DC Discount)", ["Standard", "PT1", "PT2"])

st.sidebar.markdown("### 2) Policy")
ih_percent = st.sidebar.slider("Inclusionary Housing (%)", 0, 30, 20)
density_bonus = st.sidebar.slider("Density Bonus (%)", 0, 50, 20)
ih_exit_price = st.sidebar.slider(
    "IH Exit Price (R / sellable m²)",
    min_value=10000,
    max_value=30000,
    value=int(IH_EXIT_PRICE_DEFAULT),
    step=500,
)

st.sidebar.markdown("### 3) 2026 Benchmarks")
efficiency_ratio = st.sidebar.slider("Efficiency Ratio (sellable ÷ bulk)", 0.60, 0.95, 0.85, step=0.01)
profit_margin = st.sidebar.slider("Developer Profit (% of GDV)", 10, 30, 20) / 100.0
build_tier = st.sidebar.selectbox("Hard Cost Tier (2026)", list(COST_TIERS.keys()), index=1)
tier_cost_default = COST_TIERS[build_tier]

st.sidebar.markdown("### 4) Exit Prices")
exit_price_source = st.sidebar.radio("Exit price source", ["Suburb database", "Manual entry"], index=0)

db = load_exit_price_db()
with st.sidebar.expander("Manage suburb database", expanded=False):
    uploaded = st.file_uploader("Upload CSV (suburb + min/max)", type=["csv"])
    if uploaded is not None:
        ok, msg = set_exit_price_db_from_upload(uploaded)
        (st.success if ok else st.error)(msg)

    if st.button("Reset defaults"):
        st.session_state.exit_price_db = pd.DataFrame(DEFAULT_EXIT_PRICES)
        st.success("Reset done.")
        db = load_exit_price_db()

selected_suburb = None
db_min = db_max = db_price = None
price_point = "Mid"

if exit_price_source == "Suburb database":
    suburbs = sorted(db["suburb"].dropna().astype(str).unique().tolist())
    selected_suburb = st.sidebar.selectbox("Suburb group", suburbs, index=0 if suburbs else None)
    price_point = st.sidebar.radio("Use point in range", ["Low", "Mid", "High"], index=1, horizontal=True)

    if selected_suburb:
        row = db.loc[db["suburb"] == selected_suburb]
        if not row.empty:
            db_min = float(row["min_price_per_m2"].iloc[0])
            db_max = float(row["max_price_per_m2"].iloc[0])
            db_price = db_min if price_point == "Low" else db_max if price_point == "High" else (db_min + db_max) / 2.0

    override_price = st.sidebar.checkbox("Override suburb price", value=False)

    if (db_price is None) or override_price:
        market_price = st.sidebar.number_input(
            "Market Exit Price (R / sellable m²)",
            value=float(db_price) if db_price else 35000.0,
            min_value=0.0,
            step=500.0,
        )
    else:
        market_price = db_price
        st.sidebar.caption(
            f"Auto: **R {market_price:,.0f}/m²** ({price_point}) from **R {db_min:,.0f}–R {db_max:,.0f}/m²**"
        )
else:
    market_price = st.sidebar.number_input("Market Exit Price (R / sellable m²)", value=35000.0, min_value=0.0, step=500.0)

st.sidebar.markdown("### 5) Professional fees")
default_components = default_prof_fee_components_scaled_to_target()
with st.sidebar.expander("Fee components (sum of items)", expanded=False):
    fee_components = {}
    for name, (lo, hi) in PROF_FEE_RANGES.items():
        fee_components[name] = (
            st.sidebar.slider(
                f"{name} (%)",
                float(lo * 100),
                float(hi * 100),
                float(default_components[name] * 100),
                step=0.1,
                key=f"fee_{name}",
            )
            / 100.0
        )
base_prof_fee_rate = float(sum(fee_components.values()))
st.sidebar.caption(f"Total fees: **{base_prof_fee_rate*100:.2f}%** (~12–15%)")

st.sidebar.markdown("### 6) Construction input")
cost_mode = st.sidebar.radio("Construction cost mode", ["R / m²", "% of GDV"], index=0)
pct_gdv_scope = "Hard cost only"
if cost_mode == "% of GDV":
    pct_gdv_scope = st.sidebar.radio("%GDV applies to…", ["Hard cost only", "Hard + soft (includes prof fees)"], index=0)

if cost_mode == "R / m²":
    const_cost_sqm = st.sidebar.number_input(
        "Hard Cost (R / bulk m²)",
        value=float(tier_cost_default),
        min_value=0.0,
        step=250.0,
    )
    const_cost_pct_gdv = 0.0
else:
    const_cost_pct_gdv = st.sidebar.slider("Construction (% of GDV)", 10, 90, 50) / 100.0
    const_cost_sqm = 0.0

st.sidebar.markdown("### 7) Overlays")
with st.sidebar.expander("🏛️ Built Heritage Overlay", expanded=False):
    heritage_enabled = st.sidebar.checkbox("Enable overlay", value=False, key="herit_on")
    heritage_bonus_suppression = st.sidebar.slider("Bonus suppression (%)", 0, 100, 50, disabled=not heritage_enabled)
    heritage_cost_uplift = st.sidebar.slider("Construction uplift (%)", 0, 40, 8, disabled=not heritage_enabled)
    heritage_fees_uplift = st.sidebar.slider("Fees uplift (%)", 0, 40, 5, disabled=not heritage_enabled)
    heritage_profit_uplift = st.sidebar.slider("Profit uplift (%)", 0, 40, 5, disabled=not heritage_enabled)

heritage_overlay = HeritageOverlay(
    enabled=bool(heritage_enabled),
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
    existing_gba_bulk_m2=existing_gba,
    ff=ff,
    density_bonus_pct=density_bonus,
    efficiency_ratio=efficiency_ratio,
    ih_pct=ih_percent,
    pt_zone_value=pt_zone,
    market_price_per_sellable_m2=market_price,
    ih_price_per_sellable_m2=float(ih_exit_price),
    profit_pct_gdv=profit_margin,
    base_prof_fee_rate=base_prof_fee_rate,
    overlay=heritage_overlay,
    cost_mode=cost_mode,
    base_cost_sqm=const_cost_sqm,
    base_cost_pct_gdv=const_cost_pct_gdv,
    pct_gdv_scope=pct_gdv_scope,
)

# =========================
# MAIN LAYOUT
# =========================
left, right = st.columns([1.25, 1], vertical_alignment="top")

with left:
    st.markdown('<div class="section-title">Key outputs</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-title">Residual Land Value</div>
                <div class="kpi-value">{fmt_money(res["rlv"])}</div>
                <div class="kpi-sub">After DCs, fees, profit</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-title">GDV</div>
                <div class="kpi-value">{fmt_money(res["gdv"])}</div>
                <div class="kpi-sub">Market + IH revenue</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-title">Development Charges</div>
                <div class="kpi-value">{fmt_money(res["total_dc"])}</div>
                <div class="kpi-sub">Net increase (market)</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-title">DC Savings</div>
                <div class="kpi-value">{fmt_money(res["dc_savings"])}</div>
                <div class="kpi-sub">Vs full DC on all</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ✅ NEW: Blue popout right under KPIs (to match your desired look)
    if heritage_overlay.enabled:
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        render_heritage_popout(heritage_overlay, res, base_bonus_pct=float(density_bonus))

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Assumptions snapshot</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        st.write("**Zoning:**", zoning_key)
        st.write("**Floor factor (FF):**", ZONING_PRESETS[zoning_key]["ff"])
        st.write("**Density bonus:**", f"{density_bonus:.0f}% (adj {res['adj_bonus_pct']:.1f}%)")
    with a2:
        st.write("**Efficiency:**", f"{efficiency_ratio*100:.0f}%")
        st.write("**IH %:**", f"{ih_percent:.0f}%")
        st.write("**Exit price (Market):**", f"R {market_price:,.0f}/m²")
    with a3:
        st.write("**IH exit price:**", f"R {ih_exit_price:,.0f}/m²")
        st.write("**Fees:**", fmt_pct(res["adj_fees_rate"], 2))
        st.write("**Profit:**", fmt_pct(res["adj_profit_rate"], 1))

    if res["brownfield_credit"]:
        st.warning("⚠️ Existing bulk exceeds proposed bulk. Net increase is zero; no DCs payable (Brownfield Credit).")

with right:
    st.markdown('<div class="section-title">Bankability gauge</div>', unsafe_allow_html=True)
    st.caption("Quick lender-style checks: profit, fees, cost vs tier benchmark, and RLV per land.")
    tier_name = build_tier.split("(")[0].strip()
    gauge = make_bankability_gauge(
        res,
        land_area_m2=float(land_area),
        tier_cost_sqm=float(tier_cost_default),
        cost_tier_name=tier_name,
        profit_target=0.20,
        prof_fee_band=(0.12, 0.15),
    )
    st.plotly_chart(gauge, use_container_width=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================
# CITYMAP VIEWER
# =========================
with st.expander("🗺️ City of Cape Town Map Viewer", expanded=False):
    st.caption("If the embed is blocked, use the button to open in a new tab.")
    st.link_button("Open City Map Viewer", CITYMAP_VIEWER_URL)
    components.iframe(CITYMAP_VIEWER_URL, height=560, scrolling=True)

# =========================
# SENSITIVITY HEATMAP (clickable + metric dropdown)
# ✅ follows sliders by default; pins on click; slider movement unpins
# =========================
st.markdown('<div class="section-title">Sensitivity analysis</div>', unsafe_allow_html=True)
st.caption("Choose a metric, then click a cell to pin and inspect the scenario (IH % × Density Bonus).")

metric = st.selectbox(
    "Heatmap metric",
    ["RLV (R m)", "GDV (R m)", "Profit (R m)", "Costs (Hard+Soft) (R m)"],
    index=0,
)

with st.expander("Heatmap settings", expanded=False):
    ih_step = st.select_slider("IH step (%)", options=[5, 10], value=10)
    ih_max = st.slider("IH max (%)", min_value=10, max_value=30, value=30, step=5)
    bonus_step = st.select_slider("Bonus step (%)", options=[10, 20], value=20)
    bonus_max = st.slider("Bonus max (%)", min_value=20, max_value=50, value=40, step=10)

ih_levels = list(range(0, ih_max + 1, ih_step))
bonus_levels = list(range(0, bonus_max + 1, bonus_step))


def _nearest_level(val: float, levels: list[int]) -> int:
    if not levels:
        return int(val)
    return min(levels, key=lambda x: abs(x - float(val)))


# ---- selection state that follows policy sliders by default ----
if "sens_user_pinned" not in st.session_state:
    st.session_state.sens_user_pinned = False

if "sens_last_policy" not in st.session_state:
    st.session_state.sens_last_policy = {"ih": float(ih_percent), "bonus": float(density_bonus)}

policy_changed = (
    float(st.session_state.sens_last_policy.get("ih", ih_percent)) != float(ih_percent)
    or float(st.session_state.sens_last_policy.get("bonus", density_bonus)) != float(density_bonus)
)

if policy_changed:
    st.session_state.sens_user_pinned = False
    st.session_state.sens_last_policy = {"ih": float(ih_percent), "bonus": float(density_bonus)}

if "sens_selected" not in st.session_state:
    st.session_state.sens_selected = {
        "ih": _nearest_level(ih_percent, ih_levels),
        "bonus": _nearest_level(density_bonus, bonus_levels),
    }

if not st.session_state.sens_user_pinned:
    st.session_state.sens_selected = {
        "ih": _nearest_level(ih_percent, ih_levels),
        "bonus": _nearest_level(density_bonus, bonus_levels),
    }

sel_ih = _nearest_level(st.session_state.sens_selected.get("ih", ih_percent), ih_levels)
sel_bonus = _nearest_level(st.session_state.sens_selected.get("bonus", density_bonus), bonus_levels)
st.session_state.sens_selected = {"ih": sel_ih, "bonus": sel_bonus}


def _metric_value(res_: dict) -> float:
    if metric.startswith("RLV"):
        return res_["rlv"] / 1_000_000.0
    if metric.startswith("GDV"):
        return res_["gdv"] / 1_000_000.0
    if metric.startswith("Profit"):
        return res_["profit"] / 1_000_000.0
    return (res_["construction_costs"] + res_["prof_fees"]) / 1_000_000.0


# heatmap Z
z = []
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
            ih_price_per_sellable_m2=float(ih_exit_price),
            profit_pct_gdv=profit_margin,
            base_prof_fee_rate=base_prof_fee_rate,
            overlay=heritage_overlay,
            cost_mode=cost_mode,
            base_cost_sqm=const_cost_sqm,
            base_cost_pct_gdv=const_cost_pct_gdv,
            pct_gdv_scope=pct_gdv_scope,
        )
        row.append(_metric_value(tmp))
    z.append(row)

x_labels = [f"{b}% Bonus" for b in bonus_levels]
y_labels = [f"{i}% IH" for i in ih_levels]
zmin = min(min(r) for r in z) if z else 0.0
zmax = max(max(r) for r in z) if z else 0.0

sel_x_label = f"{sel_bonus}% Bonus"
sel_y_label = f"{sel_ih}% IH"

fig = go.Figure(
    data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        zmin=zmin,
        zmax=zmax,
        hovertemplate=(
            "<b>%{y}</b> × <b>%{x}</b><br>"
            f"{metric.split('(')[0].strip()}: <b>R %{{z:.1f}}M</b><extra></extra>"
        ),
        colorbar=dict(title=metric),
    )
)
fig.update_layout(
    height=460,
    margin=dict(l=10, r=10, t=20, b=10),
    xaxis=dict(title="Density Bonus"),
    yaxis=dict(title="Inclusionary Housing %"),
    clickmode="event+select",
)

if sel_x_label in x_labels and sel_y_label in y_labels:
    fig.add_trace(
        go.Scatter(
            x=[sel_x_label],
            y=[sel_y_label],
            mode="markers+text",
            text=["Selected"],
            textposition="top center",
            marker=dict(size=16, symbol="circle-open", line=dict(width=3)),
            hoverinfo="skip",
            showlegend=False,
        )
    )

state = st.plotly_chart(fig, use_container_width=True, key="sens_heatmap", on_select="rerun")

# click -> pin
points = []
try:
    sel = getattr(state, "selection", None)
    if isinstance(sel, dict):
        points = sel.get("points", []) or []
except Exception:
    points = []

if not points:
    try:
        if isinstance(state, dict):
            points = (state.get("selection", {}) or {}).get("points", []) or []
    except Exception:
        points = []

if points:
    p = points[0]
    sx = p.get("x")
    sy = p.get("y")
    try:
        new_bonus = int(str(sx).split("%")[0])
        new_ih = int(str(sy).split("%")[0])
        if new_ih in ih_levels and new_bonus in bonus_levels:
            st.session_state.sens_selected = {"ih": new_ih, "bonus": new_bonus}
            st.session_state.sens_user_pinned = True
            sel_ih, sel_bonus = new_ih, new_bonus
    except Exception:
        pass

# LIVE recompute selected scenario
selected_res = compute_model(
    land_area_m2=land_area,
    existing_gba_bulk_m2=existing_gba,
    ff=ff,
    density_bonus_pct=float(sel_bonus),
    efficiency_ratio=efficiency_ratio,
    ih_pct=float(sel_ih),
    pt_zone_value=pt_zone,
    market_price_per_sellable_m2=market_price,
    ih_price_per_sellable_m2=float(ih_exit_price),
    profit_pct_gdv=profit_margin,
    base_prof_fee_rate=base_prof_fee_rate,
    overlay=heritage_overlay,
    cost_mode=cost_mode,
    base_cost_sqm=const_cost_sqm,
    base_cost_pct_gdv=const_cost_pct_gdv,
    pct_gdv_scope=pct_gdv_scope,
)

st.markdown("#### Selected scenario (live)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("IH %", f"{int(sel_ih)}%")
c2.metric("Density Bonus", f"{int(sel_bonus)}%")
c3.metric(metric.split("(")[0].strip(), f"R {_metric_value(selected_res):.1f}M")
c4.metric("RLV (R m)", f"R {selected_res['rlv']/1_000_000:.1f}M")

d1, d2, d3, d4 = st.columns(4)
d1.metric("GDV (R m)", f"R {selected_res['gdv']/1_000_000:.1f}M")
d2.metric("Profit (R m)", f"R {selected_res['profit']/1_000_000:.1f}M")
d3.metric("DCs (R m)", f"R {selected_res['total_dc']/1_000_000:.1f}M")
d4.metric("Fees (R m)", f"R {selected_res['prof_fees']/1_000_000:.1f}M")

with st.expander("Scenario breakdown (live)", expanded=False):
    hard_soft = selected_res["construction_costs"] + selected_res["prof_fees"]
    st.write(
        {
            "Proposed bulk (m²)": round(selected_res["proposed_bulk"], 1),
            "Proposed sellable (m²)": round(selected_res["proposed_sellable"], 1),
            "Market sellable (m²)": round(selected_res["market_sellable"], 1),
            "IH sellable (m²)": round(selected_res["ih_sellable"], 1),
            "Market exit price (R/m² sellable)": round(market_price, 0),
            "IH exit price (R/m² sellable)": round(float(ih_exit_price), 0),
            "Construction (R)": round(selected_res["construction_costs"], 0),
            "Professional fees (R)": round(selected_res["prof_fees"], 0),
            "Costs (Hard+Soft) (R)": round(hard_soft, 0),
            "DC total (R)": round(selected_res["total_dc"], 0),
            "Profit (R)": round(selected_res["profit"], 0),
            "GDV (R)": round(selected_res["gdv"], 0),
            "RLV (R)": round(selected_res["rlv"], 0),
            "Effective bonus (%)": round(selected_res["adj_bonus_pct"], 2),
            "Fees rate (%)": round(selected_res["adj_fees_rate"] * 100, 2),
            "Profit rate (%)": round(selected_res["adj_profit_rate"] * 100, 2),
            "%GDV scope (calc logic)": pct_gdv_scope,
            "Pinned by click?": bool(st.session_state.get("sens_user_pinned", False)),
        }
    )

with st.expander("View exit price database (2026 estimates)"):
    st.dataframe(load_exit_price_db(), use_container_width=True)
