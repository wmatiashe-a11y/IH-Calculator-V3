import os
from dataclasses import dataclass
from typing import Dict, Any

import streamlit as st
import pandas as pd


# ----------------------------
# Branding (Option D)
# ----------------------------
ASSETS_DIR = "assets"

ICON_D = os.path.join(ASSETS_DIR, "residuo_D_glyph_transparent_256_clean.png")
WORDMARK_D_LIGHTTEXT = os.path.join(
    ASSETS_DIR, "residuo_D_wordmark_transparent_lighttext_1200w_clean.png"
)  # white text (best on dark theme)
WORDMARK_D_DARKTEXT = os.path.join(
    ASSETS_DIR, "residuo_D_wordmark_transparent_darktext_1200w_clean.png"
)  # dark text (best on light theme)


def safe_exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False


def current_theme_base() -> str:
    """
    Returns 'light' or 'dark' based on Streamlit theme.
    If not available, default to 'light'.
    """
    try:
        base = st.get_option("theme.base")
        if base in ("light", "dark"):
            return base
    except Exception:
        pass
    return "light"


def brand_wordmark_path() -> str:
    # If dark theme -> use white wordmark; else use dark wordmark
    return WORDMARK_D_LIGHTTEXT if current_theme_base() == "dark" else WORDMARK_D_DARKTEXT


# IMPORTANT: set_page_config must be first Streamlit call
st.set_page_config(
    page_title="Residuo — Residual Land Value Calculator",
    page_icon=ICON_D if safe_exists(ICON_D) else "🏗️",
    layout="wide",
)

# ----------------------------
# Header branding render (production)
# ----------------------------
theme = current_theme_base()
wordmark = brand_wordmark_path()

h1, h2 = st.columns([1, 8], vertical_alignment="center")
with h1:
    if safe_exists(ICON_D):
        st.image(ICON_D, width=52)
    else:
        st.markdown("🏗️")

with h2:
    if safe_exists(wordmark):
        # Constrained width so it always appears (prevents giant header)
        st.image(wordmark, width=520)
    else:
        st.title("Residuo")
        st.caption("Unlock Land’s True Value")

st.divider()

# Sidebar debug (keep ON for now; set expanded=False or remove later)
st.sidebar.header("Inputs")
with st.sidebar.expander("🧪 Branding debug", expanded=False):
    st.write("Theme base:", theme)
    st.write("cwd:", os.getcwd())
    st.write("ASSETS_DIR exists:", safe_exists(ASSETS_DIR))
    if safe_exists(ASSETS_DIR):
        try:
            st.write("assets files:", os.listdir(ASSETS_DIR))
        except Exception as e:
            st.error(f"Could not list assets/: {e}")
    st.write("ICON exists:", safe_exists(ICON_D))
    st.write("WORDMARK lighttext exists:", safe_exists(WORDMARK_D_LIGHTTEXT))
    st.write("WORDMARK darktext exists:", safe_exists(WORDMARK_D_DARKTEXT))
    st.write("Chosen wordmark:", wordmark)


# ----------------------------
# Helpers
# ----------------------------
def money(x: float) -> str:
    try:
        return f"R{float(x):,.0f}"
    except Exception:
        return "—"


def pct(x: float) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "—"


# ----------------------------
# Model
# ----------------------------
@dataclass
class Inputs:
    # Areas / planning
    plot_size_sqm: float = 500.0
    floor_factor: float = 2.0
    efficiency: float = 0.85

    # Revenue
    exit_price_per_sqm: float = 42000.0  # selling price per sellable sqm

    # Costs
    build_cost_per_sqm: float = 18000.0  # construction per GFA sqm
    prof_fees_rate: float = 0.12         # professional fees as % of construction
    contingency_rate: float = 0.05       # contingency as % of construction
    marketing_rate: float = 0.02         # marketing as % of GDV
    finance_rate: float = 0.08           # finance as % of base costs (proxy)
    other_costs: float = 0.0

    # Profit
    profit_rate: float = 0.20            # developer profit as % of GDV


def calc_rlv(i: Inputs) -> Dict[str, Any]:
    """
    Simple, transparent RLV scaffold:
    - GFA = plot_size * floor_factor
    - NSA (sellable) = GFA * efficiency
    - GDV = NSA * exit_price
    - Construction = GFA * build_cost
    - Prof fees + contingency based on construction
    - Marketing based on GDV
    - Finance proxy based on (construction + fees + contingency + other) * finance_rate
    - Profit based on GDV
    - Residual Land Value = GDV - (all costs + profit)
    """
    gfa = i.plot_size_sqm * i.floor_factor
    nsa = gfa * i.efficiency
    gdv = nsa * i.exit_price_per_sqm

    construction = gfa * i.build_cost_per_sqm
    prof_fees = construction * i.prof_fees_rate
    contingency = construction * i.contingency_rate
    marketing = gdv * i.marketing_rate

    base_costs_ex_finance = construction + prof_fees + contingency + i.other_costs
    finance = base_costs_ex_finance * i.finance_rate

    profit = gdv * i.profit_rate

    total_costs = base_costs_ex_finance + marketing + finance
    rlv = gdv - (total_costs + profit)

    audit = {
        # Areas
        "plot_size_sqm": i.plot_size_sqm,
        "floor_factor": i.floor_factor,
        "gfa_sqm": gfa,
        "efficiency": i.efficiency,
        "sellable_sqm": nsa,

        # Revenue
        "exit_price_per_sqm": i.exit_price_per_sqm,
        "gdv": gdv,

        # Costs
        "build_cost_per_sqm": i.build_cost_per_sqm,
        "construction": construction,
        "prof_fees_rate": i.prof_fees_rate,
        "prof_fees": prof_fees,
        "contingency_rate": i.contingency_rate,
        "contingency": contingency,
        "marketing_rate": i.marketing_rate,
        "marketing": marketing,
        "finance_rate": i.finance_rate,
        "finance": finance,
        "other_costs": i.other_costs,
        "total_costs": total_costs,

        # Profit + RLV
        "profit_rate": i.profit_rate,
        "profit": profit,
        "rlv": rlv,
        "rlv_per_plot_sqm": (rlv / i.plot_size_sqm) if i.plot_size_sqm else 0.0,
    }

    return {
        "gfa_sqm": gfa,
        "sellable_sqm": nsa,
        "gdv": gdv,
        "total_costs": total_costs,
        "profit": profit,
        "rlv": rlv,
        "audit": audit,
    }


# ----------------------------
# UI (Inputs)
# ----------------------------
i = Inputs(
    plot_size_sqm=st.sidebar.number_input("Plot size (m²)", min_value=0.0, value=500.0, step=10.0),
    floor_factor=st.sidebar.number_input("Floor factor (FAR)", min_value=0.0, value=2.0, step=0.1),
    efficiency=st.sidebar.slider("Efficiency (sellable/GFA)", min_value=0.50, max_value=0.95, value=0.85, step=0.01),

    exit_price_per_sqm=st.sidebar.number_input("Exit price (R/m² sellable)", min_value=0.0, value=42000.0, step=500.0),

    build_cost_per_sqm=st.sidebar.number_input("Build cost (R/m² GFA)", min_value=0.0, value=18000.0, step=500.0),
    prof_fees_rate=st.sidebar.slider("Professional fees (% of construction)", 0.00, 0.30, 0.12, 0.01),
    contingency_rate=st.sidebar.slider("Contingency (% of construction)", 0.00, 0.20, 0.05, 0.01),
    marketing_rate=st.sidebar.slider("Marketing (% of GDV)", 0.00, 0.10, 0.02, 0.005),
    finance_rate=st.sidebar.slider("Finance proxy (% of base costs)", 0.00, 0.30, 0.08, 0.01),
    other_costs=st.sidebar.number_input("Other costs (R)", min_value=0.0, value=0.0, step=5000.0),

    profit_rate=st.sidebar.slider("Developer profit (% of GDV)", 0.00, 0.40, 0.20, 0.01),
)

result = calc_rlv(i)

# ----------------------------
# Outputs
# ----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("GDV", money(result["gdv"]))
c2.metric("Total Costs", money(result["total_costs"]))
c3.metric("Profit", money(result["profit"]))
c4.metric("Residual Land Value", money(result["rlv"]))

st.caption(f"RLV per plot m²: **{money(result['audit']['rlv_per_plot_sqm'])} / m²**")

st.divider()

# ----------------------------
# Audit table
# ----------------------------
st.subheader("Audit")
audit = result["audit"]

rows = []


def add_section(title: str):
    rows.append({"Section": title, "Key": "", "Value": ""})


def add_row(section: str, key: str, value: str):
    rows.append({"Section": section, "Key": key, "Value": value})


add_section("Areas")
add_row("Areas", "Plot size (m²)", f"{audit['plot_size_sqm']:,.0f}")
add_row("Areas", "Floor factor (FAR)", f"{audit['floor_factor']:.2f}")
add_row("Areas", "GFA (m²)", f"{audit['gfa_sqm']:,.0f}")
add_row("Areas", "Efficiency", pct(audit["efficiency"]))
add_row("Areas", "Sellable (m²)", f"{audit['sellable_sqm']:,.0f}")

add_section("Revenue")
add_row("Revenue", "Exit price (R/m²)", money(audit["exit_price_per_sqm"]))
add_row("Revenue", "GDV", money(audit["gdv"]))

add_section("Costs")
add_row("Costs", "Build cost (R/m²)", money(audit["build_cost_per_sqm"]))
add_row("Costs", "Construction", money(audit["construction"]))
add_row("Costs", "Prof fees rate", pct(audit["prof_fees_rate"]))
add_row("Costs", "Prof fees", money(audit["prof_fees"]))
add_row("Costs", "Contingency rate", pct(audit["contingency_rate"]))
add_row("Costs", "Contingency", money(audit["contingency"]))
add_row("Costs", "Marketing rate", pct(audit["marketing_rate"]))
add_row("Costs", "Marketing", money(audit["marketing"]))
add_row("Costs", "Finance rate", pct(audit["finance_rate"]))
add_row("Costs", "Finance", money(audit["finance"]))
add_row("Costs", "Other costs", money(audit["other_costs"]))
add_row("Costs", "Total costs", money(audit["total_costs"]))

add_section("Profit + Land")
add_row("Profit + Land", "Profit rate", pct(audit["profit_rate"]))
add_row("Profit + Land", "Profit", money(audit["profit"]))
add_row("Profit + Land", "RLV", money(audit["rlv"]))
add_row("Profit + Land", "RLV per plot m²", money(audit["rlv_per_plot_sqm"]))

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()

# ----------------------------
# Tiny snippet: auto-switch wordmark based on theme
# ----------------------------
st.subheader("Theme-switch snippet (copy/paste)")
st.code(
    """import streamlit as st
import os

ASSETS_DIR = "assets"
ICON = os.path.join(ASSETS_DIR, "residuo_D_glyph_transparent_256_clean.png")
WORD_DARK = os.path.join(ASSETS_DIR, "residuo_D_wordmark_transparent_lighttext_1200w_clean.png")  # white text
WORD_LIGHT = os.path.join(ASSETS_DIR, "residuo_D_wordmark_transparent_darktext_1200w_clean.png")  # dark text

def exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False

st.set_page_config(
    page_title="Residuo",
    page_icon=ICON if exists(ICON) else "🏗️",
    layout="wide",
)

theme = st.get_option("theme.base") or "light"
wordmark = WORD_DARK if theme == "dark" else WORD_LIGHT

if exists(wordmark):
    st.image(wordmark, width=520)
else:
    st.title("Residuo")
""",
    language="python",
)
