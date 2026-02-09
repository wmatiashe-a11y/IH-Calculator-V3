# app.py
# IH + RLV Calculator (Cape Town-style feasibility) — FULL PATCHED SINGLE FILE
# ✅ Fixes Streamlit dataclass mutable default error (default_factory)
# ✅ Fixes log spam: disables ALL print() output under Streamlit
# ✅ Implements the 6 “real-world” tweaks:
#   1) Contingency + escalation on construction
#   2) Split marketing vs finance (finance proxy on cost outflows)
#   3) Profit basis switch (GDV or Cost)
#   4) Affordable cost multiplier (optional)
#   5) Units rounding option for DC realism
#   6) Land acquisition friction costs (optional, adds outputs; preserves existing keys)
# ✅ Preserves existing audit structure/keys and keeps legacy "finance_marketing" key
# ✅ Backwards-compatible: if split rates are 0 but legacy finance_marketing_rate > 0, uses legacy gdv*rate

import builtins
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Literal

import streamlit as st


# -----------------------------
# Streamlit-safe guard + print silencer
# -----------------------------
def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


# ✅ Guaranteed: no print() spam in Streamlit logs (even if legacy demo code exists)
if _running_in_streamlit():
    builtins.print = lambda *args, **kwargs: None  # type: ignore[assignment]


# -----------------------------
# Formatting helpers
# -----------------------------
def _money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(float(x))
    return f"{sign}R{int(round(x, 0)):,}".replace(",", " ")


def _pct(x: float) -> str:
    return f"{x*100:.1f}%"


def _safe_num(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -----------------------------
# Data models
# -----------------------------
ProfitBasis = Literal["GDV", "COST"]


@dataclass
class ZoningRules:
    floor_factor: float = 2.0           # FAR proxy (bulk = plot * floor_factor)
    efficiency: float = 0.85            # sellable = bulk * efficiency
    avg_unit_size_m2: float = 60.0      # for rough unit count
    affordable_share: float = 0.10      # inclusionary share (sellable)
    affordable_price_psm: float = 12000.0


@dataclass
class Overlays:
    include_affordable: bool = True
    affordable_cost_multiplier: float = 1.00  # tweak #4 (optional)
    heritage_cost_uplift: float = 0.00        # optional overlay
    coastal_cost_uplift: float = 0.00         # optional overlay (wind/glazing proxy)
    gov_incentive_psm: float = 0.00           # optional, reduces effective build cost (R/m²)
    bulk_cap_m2: float = 0.00                 # optional: hard cap on bulk if >0


@dataclass
class Assumptions:
    # Inputs
    plot_size_m2: float = 1000.0
    exit_price_psm: float = 45000.0      # market price (R/m² sellable)
    build_cost_psm: float = 18000.0      # hard cost (R/m² bulk)
    zoning: ZoningRules = field(default_factory=ZoningRules)     # ✅ FIX
    overlays: Overlays = field(default_factory=Overlays)         # ✅ FIX

    # Professional fees etc.
    professional_fees_rate: float = 0.10  # % of build cost
    rates_taxes_rate: float = 0.00        # placeholder

    # Tweak #1: contingency + escalation on construction
    contingency_rate: float = 0.07        # % of build cost
    escalation_rate: float = 0.06         # % of build cost (simple proxy)

    # Tweak #2: split marketing vs finance (finance proxy on cost outflows)
    marketing_rate: float = 0.03          # % of GDV
    finance_rate: float = 0.08            # % of "total_costs_before_finance" proxy

    # Legacy combined rate (kept for backwards compatibility)
    finance_marketing_rate: float = 0.00  # % of GDV (legacy)

    # Tweak #3: profit basis switch (GDV or Cost)
    profit_basis: ProfitBasis = "GDV"
    profit_rate: float = 0.18

    # Tweak #5: units rounding option for DC realism
    round_units: bool = True

    # Tweak #6: land acquisition friction costs (optional)
    include_acquisition_costs: bool = True
    acquisition_cost_rate: float = 0.08   # % of land value (transfer duty + legals + agents proxy)
    acquisition_cost_fixed: float = 0.0   # fixed add-on (R)


# -----------------------------
# Engine
# -----------------------------
def calculate_rlv(a: Assumptions) -> Dict[str, Any]:
    """
    Returns a dict with:
      - 'audit': detailed step-by-step items
      - 'outputs': headline outputs
    IMPORTANT: keeps legacy audit keys, including 'finance_marketing'.
    """
    audit: Dict[str, Any] = {}

    plot_size = max(0.0, _safe_num(a.plot_size_m2))
    floor_factor = max(0.0, _safe_num(a.zoning.floor_factor))
    eff = _clamp(_safe_num(a.zoning.efficiency, 0.85), 0.30, 0.98)

    # 1) Bulk + optional cap overlay
    gross_bulk = plot_size * floor_factor
    if _safe_num(a.overlays.bulk_cap_m2) > 0:
        gross_bulk = min(gross_bulk, _safe_num(a.overlays.bulk_cap_m2))
    audit["gross_bulk_m2"] = gross_bulk

    # 2) Sellable area
    sellable_area = gross_bulk * eff
    audit["sellable_area_m2"] = sellable_area

    # 5) Units (rough) with rounding toggle
    avg_unit = max(10.0, _safe_num(a.zoning.avg_unit_size_m2, 60.0))
    units_raw = sellable_area / avg_unit if avg_unit > 0 else 0.0
    units = int(round(units_raw)) if a.round_units else float(units_raw)
    audit["units_estimate"] = units

    # 3) Revenue (GDV) with inclusionary housing
    exit_price = max(0.0, _safe_num(a.exit_price_psm))
    affordable_price = max(0.0, _safe_num(a.zoning.affordable_price_psm, 12000.0))
    affordable_share = _clamp(_safe_num(a.zoning.affordable_share, 0.10), 0.0, 0.60)

    if a.overlays.include_affordable and affordable_share > 0:
        affordable_area = sellable_area * affordable_share
        market_area = sellable_area - affordable_area
        gdv = (market_area * exit_price) + (affordable_area * affordable_price)
        audit["affordable_area_m2"] = affordable_area
        audit["market_area_m2"] = market_area
    else:
        affordable_area = 0.0
        market_area = sellable_area
        gdv = sellable_area * exit_price
        audit["affordable_area_m2"] = 0.0
        audit["market_area_m2"] = market_area

    audit["gdv"] = gdv

    # 4) Costs (build etc.)
    base_build_cost_psm = max(0.0, _safe_num(a.build_cost_psm))

    # Optional incentive reduces cost per m² (bounded so it cannot go negative)
    incentive_psm = max(0.0, _safe_num(a.overlays.gov_incentive_psm))
    effective_build_cost_psm = max(0.0, base_build_cost_psm - incentive_psm)

    # Overlay uplifts on build cost (heritage/coastal)
    heritage_uplift = max(0.0, _safe_num(a.overlays.heritage_cost_uplift))
    coastal_uplift = max(0.0, _safe_num(a.overlays.coastal_cost_uplift))
    uplift_factor = 1.0 + heritage_uplift + coastal_uplift

    # Affordable cost multiplier (applied to the share of bulk attributable to affordable area, as proxy)
    affordable_cost_mult = max(0.5, _safe_num(a.overlays.affordable_cost_multiplier, 1.0))
    if a.overlays.include_affordable and affordable_area > 0 and sellable_area > 0:
        affordable_cost_share = affordable_area / sellable_area
    else:
        affordable_cost_share = 0.0

    # Build cost applied on bulk (common quick feasibility convention)
    build_cost_base = gross_bulk * effective_build_cost_psm * uplift_factor

    # Apply affordable multiplier to the affordable "portion" of build cost
    build_cost = build_cost_base * (
        (1.0 - affordable_cost_share) + (affordable_cost_share * affordable_cost_mult)
    )

    audit["build_cost"] = build_cost
    audit["build_cost_psm_effective"] = effective_build_cost_psm
    audit["cost_uplift_factor"] = uplift_factor
    audit["affordable_cost_multiplier"] = affordable_cost_mult

    # Tweak #1: contingency + escalation (simple % of build cost)
    contingency = build_cost * _clamp(_safe_num(a.contingency_rate, 0.07), 0.0, 0.30)
    escalation = build_cost * _clamp(_safe_num(a.escalation_rate, 0.06), 0.0, 0.30)
    audit["contingency"] = contingency
    audit["escalation"] = escalation

    professional_fees = build_cost * _clamp(_safe_num(a.professional_fees_rate, 0.10), 0.0, 0.30)
    audit["professional_fees"] = professional_fees

    rates_taxes = gdv * max(0.0, _safe_num(a.rates_taxes_rate, 0.0))
    audit["rates_taxes"] = rates_taxes

    # Base costs before marketing/finance/profit
    base_costs = build_cost + contingency + escalation + professional_fees + rates_taxes
    audit["base_costs"] = base_costs

    # Tweak #2: marketing vs finance split
    marketing = gdv * max(0.0, _safe_num(a.marketing_rate, 0.0))
    audit["marketing"] = marketing

    # Finance proxy: % of cost outflows (base_costs + marketing) — simple, stable
    finance_base = base_costs + marketing
    finance = finance_base * max(0.0, _safe_num(a.finance_rate, 0.0))
    audit["finance"] = finance

    # Legacy combined: keep key 'finance_marketing' (do not break downstream)
    legacy_rate = max(0.0, _safe_num(a.finance_marketing_rate, 0.0))
    finance_marketing = 0.0
    if (marketing == 0.0 and finance == 0.0) and legacy_rate > 0:
        # Back-compat behavior: if split rates are 0 but legacy is set, apply legacy to GDV.
        finance_marketing = gdv * legacy_rate
    audit["finance_marketing"] = finance_marketing

    # Total costs excluding profit (use split amounts + legacy if used)
    total_costs_ex_profit = base_costs + marketing + finance + finance_marketing
    audit["total_costs_ex_profit"] = total_costs_ex_profit

    # Tweak #3: profit basis switch
    profit_rate = max(0.0, _safe_num(a.profit_rate, 0.18))
    if a.profit_basis == "COST":
        profit_base = total_costs_ex_profit
        profit = profit_base * profit_rate
        audit["profit_basis"] = "COST"
    else:
        profit_base = gdv
        profit = profit_base * profit_rate
        audit["profit_basis"] = "GDV"
    audit["profit"] = profit
    audit["profit_base"] = profit_base

    # Residual Land Value (before acquisition friction)
    residual_land_value = gdv - (total_costs_ex_profit + profit)
    audit["residual_land_value"] = residual_land_value

    # Tweak #6: acquisition friction costs
    acquisition_costs = 0.0
    if a.include_acquisition_costs:
        acq_rate = max(0.0, _safe_num(a.acquisition_cost_rate, 0.08))
        acq_fixed = max(0.0, _safe_num(a.acquisition_cost_fixed, 0.0))
        land_offer_proxy = max(0.0, residual_land_value)
        acquisition_costs = (land_offer_proxy * acq_rate) + acq_fixed

    audit["acquisition_costs"] = acquisition_costs

    # Max offer outputs
    max_offer_before_acquisition = residual_land_value
    max_offer_after_acquisition = residual_land_value - acquisition_costs

    outputs = {
        "gdv": gdv,
        "residual_land_value": residual_land_value,
        "max_offer_before_acquisition": max_offer_before_acquisition,
        "max_offer_after_acquisition": max_offer_after_acquisition,
        "sellable_area_m2": sellable_area,
        "gross_bulk_m2": gross_bulk,
        "units_estimate": units,
    }

    return {"audit": audit, "outputs": outputs}


# -----------------------------
# UI helpers: "Feasibility Lens" cards
# -----------------------------
def feasibility_lens_cards(local_comps: str, bulk_eff: str, coastal_premium: str) -> None:
    st.markdown("### The Feasibility Lens")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("📍 Local Comps")
        st.info(local_comps)
    with c2:
        st.caption("🏗️ Bulk Efficiency")
        st.info(bulk_eff)
    with c3:
        st.caption("💨 Coastal Premium")
        st.info(coastal_premium)


def render_audit(audit: Dict[str, Any]) -> None:
    st.markdown("### Audit Trail")
    key_order = [
        "gross_bulk_m2",
        "sellable_area_m2",
        "units_estimate",
        "market_area_m2",
        "affordable_area_m2",
        "gdv",
        "build_cost_psm_effective",
        "cost_uplift_factor",
        "affordable_cost_multiplier",
        "build_cost",
        "contingency",
        "escalation",
        "professional_fees",
        "rates_taxes",
        "base_costs",
        "marketing",
        "finance",
        "finance_marketing",
        "total_costs_ex_profit",
        "profit_basis",
        "profit_base",
        "profit",
        "residual_land_value",
        "acquisition_costs",
    ]

    rows = []
    for k in key_order:
        if k in audit:
            rows.append((k, audit[k]))
    for k in audit.keys():
        if k not in {x for x, _ in rows}:
            rows.append((k, audit[k]))

    import pandas as pd

    df = pd.DataFrame(rows, columns=["Key", "Value"])

    # ✅ Make Arrow serialization stable (mixed types -> string for display)
    df["Value"] = df["Value"].astype(str)

    # ✅ Streamlit 1.54+: use width instead of use_container_width
    st.dataframe(df, width="stretch", hide_index=True)


# -----------------------------
# App
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="IH + RLV Calculator (Cape Town)", layout="wide")

    st.title("IH + RLV Calculator (Cape Town Feasibility)")
    st.caption("Residual Land Value with Inclusionary Housing + real-world feasibility patches.")

    st.sidebar.header("Inputs")

    plot_size = st.sidebar.number_input("Plot Size (m²)", min_value=0.0, value=1000.0, step=50.0)
    floor_factor = st.sidebar.number_input("Zoning Floor Factor (FAR proxy)", min_value=0.0, value=2.0, step=0.1)
    efficiency = st.sidebar.slider("Efficiency (Sellable % of bulk)", min_value=0.30, max_value=0.98, value=0.85, step=0.01)

    exit_price = st.sidebar.number_input("Exit Price (R/m² sellable)", min_value=0.0, value=45000.0, step=500.0)
    build_cost = st.sidebar.number_input("Build Cost (R/m² bulk)", min_value=0.0, value=18000.0, step=500.0)

    st.sidebar.divider()
    st.sidebar.subheader("Inclusionary Housing")
    include_aff = st.sidebar.toggle("Include Affordable (IH)", value=True)
    aff_share = st.sidebar.slider("Affordable Share of sellable", min_value=0.0, max_value=0.6, value=0.10, step=0.01)
    aff_price = st.sidebar.number_input("Affordable Exit Price (R/m²)", min_value=0.0, value=12000.0, step=500.0)
    aff_cost_mult = st.sidebar.slider("Affordable Cost Multiplier", min_value=0.50, max_value=2.00, value=1.00, step=0.05)

    st.sidebar.divider()
    st.sidebar.subheader("Real-world Tweaks")
    prof_fees_rate = st.sidebar.slider("Professional Fees (% of build)", 0.0, 0.30, 0.10, 0.01)
    contingency_rate = st.sidebar.slider("Contingency (% of build)", 0.0, 0.30, 0.07, 0.01)
    escalation_rate = st.sidebar.slider("Escalation (% of build)", 0.0, 0.30, 0.06, 0.01)

    marketing_rate = st.sidebar.slider("Marketing (% of GDV)", 0.0, 0.10, 0.03, 0.005)
    finance_rate = st.sidebar.slider("Finance (% of costs outflows)", 0.0, 0.20, 0.08, 0.005)

    legacy_fin_mark_rate = st.sidebar.slider("Legacy Finance+Marketing (% of GDV)", 0.0, 0.20, 0.00, 0.005)

    profit_basis = st.sidebar.selectbox("Profit Basis", ["GDV", "COST"], index=0)
    profit_rate = st.sidebar.slider("Profit (% base)", 0.0, 0.40, 0.18, 0.01)

    round_units = st.sidebar.toggle("Round Units Estimate", value=True)

    st.sidebar.divider()
    st.sidebar.subheader("Overlays (Optional)")
    heritage_uplift = st.sidebar.slider("Heritage cost uplift", 0.0, 0.20, 0.00, 0.01)
    coastal_uplift = st.sidebar.slider("Coastal/wind cost uplift", 0.0, 0.20, 0.00, 0.01)
    gov_incentive_psm = st.sidebar.number_input("Gov incentive (R/m² bulk) reduces build cost", min_value=0.0, value=0.0, step=100.0)
    bulk_cap_m2 = st.sidebar.number_input("Bulk cap (m², 0 = none)", min_value=0.0, value=0.0, step=50.0)

    st.sidebar.divider()
    st.sidebar.subheader("Acquisition Costs (Optional)")
    include_acq = st.sidebar.toggle("Include Acquisition Costs", value=True)
    acq_rate = st.sidebar.slider("Acquisition cost rate (% of land)", 0.0, 0.20, 0.08, 0.005)
    acq_fixed = st.sidebar.number_input("Acquisition fixed (R)", min_value=0.0, value=0.0, step=10000.0)

    z = ZoningRules(
        floor_factor=floor_factor,
        efficiency=efficiency,
        avg_unit_size_m2=60.0,
        affordable_share=aff_share,
        affordable_price_psm=aff_price,
    )

    ov = Overlays(
        include_affordable=include_aff,
        affordable_cost_multiplier=aff_cost_mult,
        heritage_cost_uplift=heritage_uplift,
        coastal_cost_uplift=coastal_uplift,
        gov_incentive_psm=gov_incentive_psm,
        bulk_cap_m2=bulk_cap_m2,
    )

    a = Assumptions(
        plot_size_m2=plot_size,
        exit_price_psm=exit_price,
        build_cost_psm=build_cost,
        zoning=z,
        overlays=ov,
        professional_fees_rate=prof_fees_rate,
        contingency_rate=contingency_rate,
        escalation_rate=escalation_rate,
        marketing_rate=marketing_rate,
        finance_rate=finance_rate,
        finance_marketing_rate=legacy_fin_mark_rate,
        profit_basis=profit_basis,  # type: ignore[arg-type]
        profit_rate=profit_rate,
        round_units=round_units,
        include_acquisition_costs=include_acq,
        acquisition_cost_rate=acq_rate,
        acquisition_cost_fixed=acq_fixed,
    )

    result = calculate_rlv(a)
    audit = result["audit"]
    outputs = result["outputs"]

    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        st.markdown("## Dashboard")

        feasibility_lens_cards(
            local_comps='Recent sales in this sub-zone: R42,000/m².',
            bulk_eff=f'You can build ~{int(round(outputs["gross_bulk_m2"], 0)):,}m² on this plot at FAR {floor_factor:.2f}.'.replace(",", " "),
            coastal_premium=f'High-wind uplift applied: {_pct(coastal_uplift)}.',
        )

        st.markdown("### Headline Outputs")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GDV", _money(outputs["gdv"]))
        c2.metric("Residual Land Value", _money(outputs["residual_land_value"]))
        c3.metric("Max Offer (before acquisition)", _money(outputs["max_offer_before_acquisition"]))
        c4.metric("Max Offer (after acquisition)", _money(outputs["max_offer_after_acquisition"]))

        if outputs["max_offer_after_acquisition"] < 0:
            st.error(
                "Residual is negative. This scheme is not feasible at the current assumptions "
                "(or it requires a lower land price / higher exit / lower costs / incentives)."
            )

        st.markdown("### Areas")
        a1, a2, a3 = st.columns(3)
        a1.metric("Gross Bulk (m²)", f'{int(round(outputs["gross_bulk_m2"], 0)):,}'.replace(",", " "))
        a2.metric("Sellable Area (m²)", f'{int(round(outputs["sellable_area_m2"], 0)):,}'.replace(",", " "))
        a3.metric("Units (estimate)", str(outputs["units_estimate"]))

        st.markdown("### Key Cost Lines")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Build Cost", _money(audit["build_cost"]))
        k2.metric("Contingency", _money(audit["contingency"]))
        k3.metric("Escalation", _money(audit["escalation"]))
        k4.metric("Professional Fees", _money(audit["professional_fees"]))

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Marketing", _money(audit["marketing"]))
        k6.metric("Finance", _money(audit["finance"]))
        k7.metric("Legacy Fin+Mkt", _money(audit["finance_marketing"]))
        k8.metric("Profit", _money(audit["profit"]))

    with right:
        render_audit(audit)

        with st.expander("Assumptions (debug export)"):
            st.json(asdict(a))

        st.caption(
            "Note: Finance is a simplified proxy (% of cost outflows). "
            "For bank-grade feasibility, replace with a cashflow + drawdown + interest schedule."
        )


if __name__ == "__main__":
    main()
