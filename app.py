from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Literal


@dataclass
class Overlays:
    # Construction cost modifiers (multipliers)
    heritage_cost_multiplier: float = 1.00
    coastal_spec_multiplier: float = 1.00

    # Efficiency modifiers (sellable %)
    efficiency_multiplier: float = 1.00

    # Inclusionary housing policy
    inclusionary_enabled: bool = True
    inclusionary_share: float = 0.10
    inclusionary_cap_price_sqm: float = 12000.0

    # Optional: affordable spec/cost difference
    affordable_cost_multiplier: float = 1.00

    # Fees/cost assumptions
    prof_fees_rate: float = 0.15

    # Split marketing and finance (more realistic than a single blended %)
    marketing_rate: float = 0.03                 # % of market GDV
    finance_rate_on_costs: float = 0.015         # ✅ calibrated proxy (NOT 0.10)

    # Keep legacy field for backwards compatibility (not used when marketing/finance provided)
    finance_marketing_rate: float = 0.08

    # Profit
    profit_basis: Literal["gdv", "cost"] = "gdv"
    target_profit_rate: float = 0.20

    # Risk allowances
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
    acquisition_cost_rate: float = 0.00
    acquisition_cost_lump_sum: float = 0.00


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

        if "floor_factor" not in self.zoning:
            raise KeyError("zoning_rules must include 'floor_factor'")
        if "coverage" not in self.zoning:
            raise KeyError("zoning_rules must include 'coverage'")

        ff = float(self.zoning["floor_factor"])
        cov = float(self.zoning["coverage"])
        if ff <= 0:
            raise ValueError("zoning_rules['floor_factor'] must be > 0")
        if not (0 < cov <= 1):
            raise ValueError("zoning_rules['coverage'] must be between 0 and 1")

        # sanity for rates
        rate_fields = [
            "prof_fees_rate",
            "marketing_rate",
            "finance_rate_on_costs",
            "target_profit_rate",
            "contingency_rate",
            "escalation_rate",
            "acquisition_cost_rate",
        ]
        for name in rate_fields:
            val = float(getattr(self.overlays, name))
            if not (0 <= val < 1):
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

        # 1) GFA allowed by FAR
        gfa_by_far = self.plot_size * floor_factor

        # 2) GFA cap by coverage (coverage per floor * assumed floors)
        gfa_by_coverage = self.plot_size * coverage * max(self.assumed_floors, 1)

        # 3) Total allowable bulk (choose binding constraint)
        if gfa_by_far <= gfa_by_coverage:
            total_bulk = gfa_by_far
            binding_constraint = "FAR binds"
        else:
            total_bulk = gfa_by_coverage
            binding_constraint = "Coverage binds"

        # 4) Net sellable area (clamp)
        effective_efficiency = self.base_efficiency * self.overlays.efficiency_multiplier
        effective_efficiency = max(0.0, min(1.0, effective_efficiency))
        sellable_area = total_bulk * effective_efficiency

        # 5) GDV (with inclusionary split)
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

        # 6) Hard costs (apply modifiers) + affordable cost multiplier + contingency/escalation
        build_cost_multiplier = (
            self.overlays.heritage_cost_multiplier * self.overlays.coastal_spec_multiplier
        )
        adjusted_build_cost_sqm = build_cost_sqm * build_cost_multiplier

        # Proxy allocation of GFA to affordable vs market based on sellable split
        if sellable_area > 0:
            affordable_gfa = total_bulk * (affordable_area / sellable_area)
        else:
            affordable_gfa = 0.0
        market_gfa = total_bulk - affordable_gfa

        base_construction_market = market_gfa * adjusted_build_cost_sqm
        base_construction_affordable = (
            affordable_gfa * adjusted_build_cost_sqm * self.overlays.affordable_cost_multiplier
        )
        base_construction = base_construction_market + base_construction_affordable

        total_construction = base_construction * (
            1.0 + self.overlays.contingency_rate + self.overlays.escalation_rate
        )

        # 7) Soft costs
        prof_fees = total_construction * self.overlays.prof_fees_rate

        # 8) Municipal DCs (optionally round)
        estimated_units_raw = (
            sellable_area / self.overlays.avg_unit_size_sqm
            if self.overlays.avg_unit_size_sqm > 0
            else 0.0
        )
        estimated_units = float(round(estimated_units_raw)) if self.overlays.round_units else float(estimated_units_raw)
        muni_dcs = estimated_units * self.overlays.dc_per_unit

        # 9) Marketing + finance proxy (keep legacy combined key)
        marketing = gdv_market * self.overlays.marketing_rate

        finance_base = total_construction + prof_fees + muni_dcs
        finance = finance_base * self.overlays.finance_rate_on_costs

        finance_marketing = marketing + finance

        # 10) Incentives
        incentives_total = float(self.overlays.municipal_incentive_amount + self.overlays.rates_rebate_amount)

        total_costs_ex_profit = (
            total_construction + prof_fees + muni_dcs + finance_marketing - incentives_total
        )

        # 11) Profit (basis switch)
        if self.overlays.profit_basis == "cost":
            target_profit = total_costs_ex_profit * self.overlays.target_profit_rate
        else:
            target_profit = gdv * self.overlays.target_profit_rate

        # 12) Residual (before acquisition costs)
        residual_land_value = gdv - (total_costs_ex_profit + target_profit)

        # 13) Acquisition friction costs (optional clarity)
        acquisition_costs = max(0.0, residual_land_value) * self.overlays.acquisition_cost_rate + self.overlays.acquisition_cost_lump_sum
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
                "binding_constraint": binding_constraint,  # added (non-breaking)
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

                # Legacy key preserved + split shown
                "finance_marketing": self._round2(finance_marketing),
                "marketing_cost": self._round2(marketing),
                "finance_cost": self._round2(finance),
                "finance_base": self._round2(finance_base),

                "incentives_total": self._round2(incentives_total),

                "profit_basis": self.overlays.profit_basis,
                "target_profit": self._round2(target_profit),
                "total_costs_ex_profit": self._round2(total_costs_ex_profit),

                "acquisition_costs": self._round2(acquisition_costs),
            },
            "outputs": {
                "residual_land_value": self._round2(residual_land_value),
                "max_offer_for_land": self._round2(residual_land_value),
                "is_viable": residual_land_value > 0,

                # Added clarity; existing keys kept intact
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
