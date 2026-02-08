import builtins

def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

# ✅ Kill all print() output in Streamlit (stops log spam even if legacy demo blocks exist)
if _running_in_streamlit():
    builtins.print = lambda *args, **kwargs: None  # type: ignore[assignment]
# app.py
# Streamlit RLV + Inclusionary Housing (IH) Calculator (Overlay-aware) — patched + hardened
# - No top-level print() spam (Streamlit reruns safe)
# - Finance/Marketing split supported + legacy blended rate supported
# - Contingency/escalation supported
# - Profit basis switch supported
# - Optional affordable cost multiplier
# - Optional unit rounding for DC realism
# - Optional land acquisition friction costs (added outputs; existing keys preserved)
#
# Works as a single-file Streamlit app.

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Literal

import streamlit as st


# -----------------------------
# Helpers
# -----------------------------
def _running_in_streamlit() -> bool:
    """True when executing under Streamlit runtime (prevents demo prints)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _money(x: float) -> str:
    try:
        return f"R{float(x):,.0f}"
    except Exception:
        return "—"


def _pct(x: float) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "—"


# -----------------------------
# Engine
# -----------------------------
@dataclass
class Overlays:
    # Construction cost modifiers (multipliers)
    heritage_cost_multiplier: float = 1.00
    coastal_spec_multiplier: float = 1.00

    # Efficiency modifier (sellable %)
    efficiency_multiplier: float = 1.00

    # Inclusionary housing policy
    inclusionary_enabled: bool = True
    inclusionary_share: float = 0.10
    inclusionary_cap_price_sqm: float = 12000.0

    # Differential affordable build spec/cost
    affordable_cost_multiplier: float = 1.00

    # Fees
    prof_fees_rate: float = 0.15

    # New split rates (recommended)
    marketing_rate: float = 0.03
    finance_rate_on_costs: float = 0.015  # ✅ calibrated proxy default

    # Legacy blended field (older UIs may still set this)
    finance_marketing_rate: float = 0.08

    # Profit
    profit_basis: Literal["gdv", "cost"] = "gdv"
    target_profit_rate: float = 0.20

    # Risk allowances on construction
    contingency_rate: float = 0.07
    escalation_rate: float = 0.00

    # Municipal DCs
    avg_unit_size_sqm: float = 60.0
    dc_per_unit: float = 45000.0
    round_units: bool = False

    # Incentives (subtract from costs)
    municipal_incentive_amount: float = 0.0
    rates_rebate_amount: float = 0.0

    # Land acquisition friction (optional clarity)
    acquisition_cost_rate: float = 0.0
    acquisition_cost_lump_sum: float = 0.0


class RLVMachine:
    def __init__(
        self,
        plot_size: float,
        zoning_rules: Dict[str, Any],
        base_efficiency: float = 0.85,
        assumed_floors: int = 1,
        overlays: Optional[Overlays] = None,
    ):
        self.plot_size = float(plot_size)
        self.zoning = zoning_rules or {}
        self.base_efficiency = float(base_efficiency)
        self.assumed_floors = int(assumed_floors)
        self.overlays = overlays or Overlays()
        self._validate_inputs()

    def _validate_inputs(self):
        if self.plot_size <= 0:
            raise ValueError("plot_size must be > 0")

        if "floor_factor" not in self.zoning or "coverage" not in self.zoning:
            raise KeyError("zoning_rules must include 'floor_factor' and 'coverage'")

        ff = float(self.zoning["floor_factor"])
        cov = float(self.zoning["coverage"])
        if ff <= 0:
            raise ValueError("zoning_rules['floor_factor'] must be > 0")
        if not (0 < cov <= 1):
            raise ValueError("zoning_rules['coverage'] must be between 0 and 1")

        # sanity for key rates
        for name in [
            "prof_fees_rate",
            "marketing_rate",
            "finance_rate_on_costs",
            "finance_marketing_rate",
            "target_profit_rate",
            "contingency_rate",
            "escalation_rate",
            "acquisition_cost_rate",
        ]:
            v = float(getattr(self.overlays, name))
            if not (0 <= v < 1):
                raise ValueError(f"{name} must be between 0 and 1")

        if not (0 <= self.overlays.inclusionary_share < 1):
            raise ValueError("inclusionary_share must be between 0 and 1")

        if self.overlays.avg_unit_size_sqm <= 0:
            raise ValueError("avg_unit_size_sqm must be > 0")

        if self.overlays.affordable_cost_multiplier <= 0:
            raise ValueError("affordable_cost_multiplier must be > 0")

        if self.overlays.profit_basis not in ("gdv", "cost"):
            raise ValueError("profit_basis must be 'gdv' or 'cost'")

    def _round2(self, x: float) -> float:
        return round(float(x), 2)

    def calculate_rlv(
        self,
        exit_price_sqm: float,
        build_cost_sqm: float,
        include_affordable: Optional[bool] = None,
    ) -> Dict[str, Any]:
        exit_price_sqm = float(exit_price_sqm)
        build_cost_sqm = float(build_cost_sqm)
        if exit_price_sqm <= 0 or build_cost_sqm <= 0:
            raise ValueError("exit_price_sqm and build_cost_sqm must be > 0")

        inclusionary_on = (
            self.overlays.inclusionary_enabled
            if include_affordable is None
            else bool(include_affordable)
        )

        floor_factor = float(self.zoning["floor_factor"])
        coverage = float(self.zoning["coverage"])

        # 1) Bulk caps
        gfa_by_far = self.plot_size * floor_factor
        gfa_by_coverage = self.plot_size * coverage * max(self.assumed_floors, 1)

        if gfa_by_far <= gfa_by_coverage:
            total_bulk = gfa_by_far
            binding_constraint = "FAR binds"
        else:
            total_bulk = gfa_by_coverage
            binding_constraint = "Coverage binds"

        # 2) Net sellable area (clamp)
        effective_efficiency = self.base_efficiency * self.overlays.efficiency_multiplier
        effective_efficiency = max(0.0, min(1.0, effective_efficiency))
        sellable_area = total_bulk * effective_efficiency

        # 3) GDV (with inclusionary split)
        if inclusionary_on and self.overlays.inclusionary_share > 0 and sellable_area > 0:
            affordable_area = sellable_area * self.overlays.inclusionary_share
            market_area = sellable_area - affordable_area
            gdv_market = market_area * exit_price_sqm
            gdv_affordable = affordable_area * self.overlays.inclusionary_cap_price_sqm
            gdv = gdv_market + gdv_affordable
            inclusionary_hit = (
                f"{int(self.overlays.inclusionary_share * 100)}% Area Capped "
                f"@ R{self.overlays.inclusionary_cap_price_sqm:,.0f}/m²"
            )
        else:
            affordable_area = 0.0
            market_area = sellable_area
            gdv_market = sellable_area * exit_price_sqm
            gdv_affordable = 0.0
            gdv = gdv_market
            inclusionary_hit = "None"

        # 4) Hard costs + risk allowances
        build_cost_multiplier = (
            self.overlays.heritage_cost_multiplier * self.overlays.coastal_spec_multiplier
        )
        adjusted_build_cost_sqm = build_cost_sqm * build_cost_multiplier

        affordable_gfa = total_bulk * (affordable_area / sellable_area) if sellable_area > 0 else 0.0
        market_gfa = total_bulk - affordable_gfa

        base_construction_market = market_gfa * adjusted_build_cost_sqm
        base_construction_affordable = (
            affordable_gfa
            * adjusted_build_cost_sqm
            * self.overlays.affordable_cost_multiplier
        )
        base_construction = base_construction_market + base_construction_affordable

        total_construction = base_construction * (
            1.0 + self.overlays.contingency_rate + self.overlays.escalation_rate
        )

        # 5) Soft costs
        prof_fees = total_construction * self.overlays.prof_fees_rate

        # 6) DCs (optionally rounded)
        estimated_units_raw = (
            sellable_area / self.overlays.avg_unit_size_sqm
            if self.overlays.avg_unit_size_sqm > 0
            else 0.0
        )
        estimated_units = (
            float(round(estimated_units_raw))
            if self.overlays.round_units
            else float(estimated_units_raw)
        )
        muni_dcs = estimated_units * self.overlays.dc_per_unit

        # 7) Finance + marketing (backwards-compatible)
        marketing_rate = float(getattr(self.overlays, "marketing_rate", 0.0))
        finance_rate_on_costs = float(getattr(self.overlays, "finance_rate_on_costs", 0.0))
        legacy_blended = float(getattr(self.overlays, "finance_marketing_rate", 0.0))

        # If split rates are both zero but legacy exists, use legacy behavior (gdv * legacy rate)
        if marketing_rate == 0.0 and finance_rate_on_costs == 0.0 and legacy_blended > 0:
            finance_marketing = gdv * legacy_blended
            marketing = finance_marketing
            finance = 0.0
            finance_base = 0.0
            finance_mode = "legacy_gdv_blended"
        else:
            marketing = gdv_market * marketing_rate
            finance_base = total_construction + prof_fees + muni_dcs
            finance = finance_base * finance_rate_on_costs
            finance_marketing = marketing + finance
            finance_mode = "split_marketing_plus_finance_on_costs"

        # 8) Incentives
        incentives_total = float(
            self.overlays.municipal_incentive_amount + self.overlays.rates_rebate_amount
        )

        total_costs_ex_profit = (
            total_construction + prof_fees + muni_dcs + finance_marketing - incentives_total
        )

        # 9) Profit basis
        if self.overlays.profit_basis == "cost":
            target_profit = total_costs_ex_profit * self.overlays.target_profit_rate
        else:
            target_profit = gdv * self.overlays.target_profit_rate

        # 10) Residual (before acquisition)
        residual_land_value = gdv - (total_costs_ex_profit + target_profit)

        # 11) Acquisition friction costs (optional)
        acquisition_costs = (
            max(0.0, residual_land_value) * self.overlays.acquisition_cost_rate
            + self.overlays.acquisition_cost_lump_sum
        )
        residual_land_value_after_acq = residual_land_value - acquisition_costs

        audit = {
            "inputs": {
                "plot_size_sqm": self.plot_size,
                "zoning_floor_factor": floor_factor,
                "zoning_coverage": coverage,
                "assumed_floors": self.assumed_floors,
                "exit_price_sqm_market": exit_price_sqm,
                "build_cost_sqm_base": build_cost_sqm,
                "base_efficiency": self.base_efficiency,
                "efficiency_multiplier": self.overlays.efficiency_multiplier,
                "effective_efficiency": effective_efficiency,
                "overlays": asdict(self.overlays),
            },
            "bulk_calcs": {
                "gfa_by_far": self._round2(gfa_by_far),
                "gfa_by_coverage": self._round2(gfa_by_coverage),
                "total_bulk_gfa": self._round2(total_bulk),
                "binding_constraint": binding_constraint,
            },
            "area_split": {
                "sellable_area": self._round2(sellable_area),
                "market_area": self._round2(market_area),
                "affordable_area": self._round2(affordable_area),
                "inclusionary": inclusionary_hit,
            },
            "gdv": {
                "gdv_market": self._round2(gdv_market),
                "gdv_affordable": self._round2(gdv_affordable),
                "gross_development_value": self._round2(gdv),
            },
            "costs": {
                "build_cost_multiplier": self._round2(build_cost_multiplier),
                "adjusted_build_cost_sqm": self._round2(adjusted_build_cost_sqm),
                "base_construction_market": self._round2(base_construction_market),
                "base_construction_affordable": self._round2(base_construction_affordable),
                "contingency_rate": self._round2(self.overlays.contingency_rate),
                "escalation_rate": self._round2(self.overlays.escalation_rate),
                "total_construction_cost": self._round2(total_construction),
                "professional_fees": self._round2(prof_fees),
                "estimated_units": self._round2(estimated_units),
                "estimated_units_raw": self._round2(estimated_units_raw),
                "municipal_dcs": self._round2(muni_dcs),
                "finance_marketing": self._round2(finance_marketing),  # legacy key preserved
                "marketing_cost": self._round2(marketing),
                "finance_cost": self._round2(finance),
                "finance_base": self._round2(finance_base),
                "finance_mode": finance_mode,
                "incentives_total": self._round2(incentives_total),
                "target_profit": self._round2(target_profit),
                "profit_basis": self.overlays.profit_basis,
                "total_costs_ex_profit": self._round2(total_costs_ex_profit),
                "acquisition_costs": self._round2(acquisition_costs),
            },
            "outputs": {
                "residual_land_value": self._round2(residual_land_value),
                "max_offer_for_land": self._round2(residual_land_value),
                "is_viable": residual_land_value > 0,
                "residual_land_value_after_acquisition_costs": self._round2(residual_land_value_after_acq),
                "is_viable_after_acquisition_costs": residual_land_value_after_acq > 0,
            },
        }

        return {
            "gross_development_value": self._round2(gdv),
            "total_construction_cost": self._round2(total_construction),
            "inclusionary_hit": inclusionary_hit,
            "residual_land_value": self._round2(residual_land_value),
            "audit": audit,
        }


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="IH + RLV Calculator", page_icon="🏗️", layout="wide")

st.title("🏗️ Inclusionary Housing + Residual Land Value Calculator")
st.caption("Overlay-aware feasibility engine with audit trail (patched build).")

# Sidebar inputs
st.sidebar.header("Inputs")

with st.sidebar.expander("Site & Zoning", expanded=True):
    plot_size = st.number_input("Plot size (m²)", min_value=1.0, value=1000.0, step=10.0)
    floor_factor = st.number_input("Floor factor (FAR)", min_value=0.01, value=2.0, step=0.05)
    coverage = st.number_input("Coverage (0–1)", min_value=0.01, max_value=1.0, value=0.75, step=0.01)
    assumed_floors = st.number_input("Assumed floors (for coverage cap)", min_value=1, value=3, step=1)
    base_efficiency = st.number_input("Base efficiency (sellable % of GFA)", min_value=0.01, max_value=1.0, value=0.85, step=0.01)

with st.sidebar.expander("Revenues & Build Costs", expanded=True):
    exit_price_sqm = st.number_input("Market exit price (R/m²)", min_value=1.0, value=42000.0, step=500.0)
    build_cost_sqm = st.number_input("Base build cost (R/m² GFA)", min_value=1.0, value=19000.0, step=250.0)

with st.sidebar.expander("Overlays & Policy", expanded=True):
    inclusionary_enabled = st.checkbox("Inclusionary enabled", value=True)
    inclusionary_share = st.slider("Inclusionary share of sellable area (%)", 0.0, 30.0, 10.0, 0.5) / 100.0
    inclusionary_cap = st.number_input("Affordable cap price (R/m²)", min_value=0.0, value=12000.0, step=250.0)

    heritage_cost_multiplier = st.slider("Heritage cost multiplier", 1.00, 1.30, 1.05, 0.01)
    coastal_spec_multiplier = st.slider("Coastal spec multiplier", 1.00, 1.30, 1.00, 0.01)
    efficiency_multiplier = st.slider("Efficiency multiplier", 0.70, 1.05, 1.00, 0.01)

    affordable_cost_multiplier = st.slider("Affordable cost multiplier", 0.85, 1.10, 1.00, 0.01)

with st.sidebar.expander("Costs, Risk, DCs", expanded=True):
    prof_fees_rate = st.slider("Professional fees (% of construction)", 0.0, 25.0, 15.0, 0.5) / 100.0

    contingency_rate = st.slider("Contingency (% of construction)", 0.0, 15.0, 7.0, 0.5) / 100.0
    escalation_rate = st.slider("Escalation (% of construction)", 0.0, 15.0, 0.0, 0.5) / 100.0

    avg_unit_size_sqm = st.number_input("Avg unit size (m²) for DCs", min_value=10.0, value=60.0, step=5.0)
    dc_per_unit = st.number_input("DC per unit (R)", min_value=0.0, value=45000.0, step=5000.0)
    round_units = st.checkbox("Round units for DCs", value=False)

with st.sidebar.expander("Finance/Marketing & Profit", expanded=True):
    # Recommended UI is percent -> convert to decimal
    marketing_rate = st.slider("Marketing (% of market GDV)", 0.0, 10.0, 3.0, 0.1) / 100.0
    finance_rate_on_costs = st.slider(
        "Finance proxy on cost outflows (%)",
        0.0,
        5.0,
        1.5,  # ✅ calibrated default
        0.1,
    ) / 100.0

    profit_basis = st.selectbox("Profit basis", options=["gdv", "cost"], index=0)
    target_profit_rate = st.slider("Target profit (%)", 0.0, 35.0, 20.0, 0.5) / 100.0

    legacy_finance_marketing_rate = st.slider(
        "Legacy blended finance/marketing (% of GDV) (optional)",
        0.0,
        15.0,
        8.0,
        0.5,
        help="Only used if both Marketing and Finance sliders are set to 0.0%.",
    ) / 100.0

with st.sidebar.expander("Incentives & Acquisition (optional)", expanded=False):
    municipal_incentive_amount = st.number_input("Municipal incentive (R)", value=0.0, step=50000.0)
    rates_rebate_amount = st.number_input("Rates rebate (R)", value=0.0, step=50000.0)

    acquisition_cost_rate = st.slider("Acquisition friction (% of land value)", 0.0, 10.0, 0.0, 0.5) / 100.0
    acquisition_cost_lump_sum = st.number_input("Acquisition friction (lump sum R)", value=0.0, step=25000.0)

show_debug = st.sidebar.checkbox("Show debug (recommended while testing)", value=False)

# Build overlays + run model
zoning_rules = {"floor_factor": floor_factor, "coverage": coverage}

overlays = Overlays(
    heritage_cost_multiplier=heritage_cost_multiplier,
    coastal_spec_multiplier=coastal_spec_multiplier,
    efficiency_multiplier=efficiency_multiplier,
    inclusionary_enabled=inclusionary_enabled,
    inclusionary_share=inclusionary_share,
    inclusionary_cap_price_sqm=inclusionary_cap,
    affordable_cost_multiplier=affordable_cost_multiplier,
    prof_fees_rate=prof_fees_rate,
    marketing_rate=marketing_rate,
    finance_rate_on_costs=finance_rate_on_costs,
    finance_marketing_rate=legacy_finance_marketing_rate,  # legacy fallback
    profit_basis=profit_basis,  # type: ignore[arg-type]
    target_profit_rate=target_profit_rate,
    contingency_rate=contingency_rate,
    escalation_rate=escalation_rate,
    avg_unit_size_sqm=avg_unit_size_sqm,
    dc_per_unit=dc_per_unit,
    round_units=round_units,
    municipal_incentive_amount=municipal_incentive_amount,
    rates_rebate_amount=rates_rebate_amount,
    acquisition_cost_rate=acquisition_cost_rate,
    acquisition_cost_lump_sum=acquisition_cost_lump_sum,
)

if show_debug:
    st.sidebar.caption(f"DEBUG finance_rate_on_costs = {overlays.finance_rate_on_costs:.4f}")
    st.sidebar.caption(f"DEBUG marketing_rate = {overlays.marketing_rate:.4f}")
    st.sidebar.caption(f"DEBUG legacy finance_marketing_rate = {overlays.finance_marketing_rate:.4f}")

try:
    model = RLVMachine(
        plot_size=plot_size,
        zoning_rules=zoning_rules,
        base_efficiency=base_efficiency,
        assumed_floors=assumed_floors,
        overlays=overlays,
    )
    result = model.calculate_rlv(exit_price_sqm=exit_price_sqm, build_cost_sqm=build_cost_sqm)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

audit = result["audit"]

# -----------------------------
# Dashboard
# -----------------------------
colA, colB, colC, colD = st.columns(4)
colA.metric("Gross Development Value (GDV)", _money(result["gross_development_value"]))
colB.metric("Construction Cost", _money(result["total_construction_cost"]))
colC.metric("Residual Land Value (before acquisition)", _money(result["residual_land_value"]))
colD.metric("Inclusionary", result["inclusionary_hit"])

# Feasibility Lens
st.subheader("🔎 The Feasibility Lens")

lens1, lens2, lens3 = st.columns(3)

# Useful computed lens metrics
total_bulk = float(audit["bulk_calcs"]["total_bulk_gfa"])
sellable_area = float(audit["area_split"]["sellable_area"])
rlv = float(audit["outputs"]["residual_land_value"])
rlv_after = float(audit["outputs"]["residual_land_value_after_acquisition_costs"])
is_viable = bool(audit["outputs"]["is_viable"])
binding = audit["bulk_calcs"].get("binding_constraint", "—")

lens1.metric("Binding constraint", binding)
lens2.metric("Sellable area (m²)", f"{sellable_area:,.0f}")
lens3.metric("Viable (before acquisition)", "Yes ✅" if is_viable else "No ❌")

tray1, tray2, tray3 = st.columns(3)
tray1.metric("RLV per m² plot", _money(rlv / plot_size))
tray2.metric("RLV per m² bulk", _money(rlv / total_bulk if total_bulk > 0 else 0))
tray3.metric("RLV after acquisition costs", _money(rlv_after))

# Breakdown
st.subheader("📊 Key Breakdown")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Market GDV", _money(audit["gdv"]["gdv_market"]))
k2.metric("Affordable GDV", _money(audit["gdv"]["gdv_affordable"]))
k3.metric("Finance + Marketing", _money(audit["costs"]["finance_marketing"]))
k4.metric("Target Profit", _money(audit["costs"]["target_profit"]))

with st.expander("Full audit trail (JSON)"):
    st.json(audit)

# -----------------------------
# No demo prints in Streamlit
# -----------------------------
if __name__ == "__main__" and not _running_in_streamlit():
    # Safe for local debugging only.
    pass
