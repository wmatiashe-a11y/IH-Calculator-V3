from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class Overlays:
    # Construction cost modifiers (multipliers)
    heritage_cost_multiplier: float = 1.00
    coastal_spec_multiplier: float = 1.00

    # Efficiency modifiers (sellable %)
    efficiency_multiplier: float = 1.00  # e.g., heritage may reduce net efficiency

    # Inclusionary housing policy
    inclusionary_enabled: bool = True
    inclusionary_share: float = 0.10          # 10% of sellable area
    inclusionary_cap_price_sqm: float = 12000 # R/m² cap for affordable area

    # Fees/cost assumptions
    prof_fees_rate: float = 0.15
    finance_marketing_rate: float = 0.08
    target_profit_rate: float = 0.20

    # Municipal DCs
    avg_unit_size_sqm: float = 60.0
    dc_per_unit: float = 45000.0

    # Incentives (subtract from costs; add to value)
    municipal_incentive_amount: float = 0.0   # e.g., grant/rebate in R
    rates_rebate_amount: float = 0.0          # optional lump sum rebate


class RLVMachine:
    def __init__(
        self,
        plot_size: float,
        zoning_rules: Dict[str, Any],
        base_efficiency: float = 0.85,
        assumed_floors: int = 1,
        overlays: Optional[Overlays] = None,
    ):
        """
        plot_size: site area in m²
        zoning_rules: expects at least {'floor_factor': x, 'coverage': y}
        base_efficiency: net sellable % of GFA (before overlays)
        assumed_floors: used to translate coverage into a GFA cap (coverage * plot_size * floors)
        overlays: Overlays() for policy + modifiers
        """
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

        # Rates sanity checks
        for name in ["prof_fees_rate", "finance_marketing_rate", "target_profit_rate"]:
            val = getattr(self.overlays, name)
            if not (0 <= val < 1):
                raise ValueError(f"{name} must be between 0 and 1")

        if not (0 <= self.overlays.inclusionary_share < 1):
            raise ValueError("inclusionary_share must be between 0 and 1")

        if self.overlays.avg_unit_size_sqm <= 0:
            raise ValueError("avg_unit_size_sqm must be > 0")

    def _round2(self, x: float) -> float:
        return round(float(x), 2)

    def calculate_rlv(
        self,
        exit_price_sqm: float,
        build_cost_sqm: float,
        include_affordable: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        exit_price_sqm: R/m² for market component
        build_cost_sqm: R/m² on GFA (before modifiers)
        include_affordable: override overlays.inclusionary_enabled if provided
        """
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
        total_bulk = min(gfa_by_far, gfa_by_coverage)

        # 4) Net sellable area
        effective_efficiency = self.base_efficiency * self.overlays.efficiency_multiplier
        sellable_area = total_bulk * effective_efficiency

        # 5) GDV (with inclusionary split)
        if inclusionary_on and self.overlays.inclusionary_share > 0:
            affordable_area = sellable_area * self.overlays.inclusionary_share
            market_area = sellable_area - affordable_area
            gdv_market = market_area * exit_price_sqm
            gdv_affordable = affordable_area * self.overlays.inclusionary_cap_price_sqm
            gdv = gdv_market + gdv_affordable
            inclusionary_hit = f"{int(self.overlays.inclusionary_share*100)}% Area Capped @ R{self.overlays.inclusionary_cap_price_sqm:,.0f}/m²"
        else:
            affordable_area = 0.0
            market_area = sellable_area
            gdv_market = sellable_area * exit_price_sqm
            gdv_affordable = 0.0
            gdv = gdv_market
            inclusionary_hit = "None"

        # 6) Hard costs (apply modifiers)
        build_cost_multiplier = self.overlays.heritage_cost_multiplier * self.overlays.coastal_spec_multiplier
        adjusted_build_cost_sqm = build_cost_sqm * build_cost_multiplier
        total_construction = total_bulk * adjusted_build_cost_sqm

        # 7) Soft costs
        prof_fees = total_construction * self.overlays.prof_fees_rate

        # 8) Municipal DCs
        estimated_units = sellable_area / self.overlays.avg_unit_size_sqm
        muni_dcs = estimated_units * self.overlays.dc_per_unit

        # 9) Finance & Marketing
        finance_marketing = gdv * self.overlays.finance_marketing_rate

        # 10) Developer Profit
        target_profit = gdv * self.overlays.target_profit_rate

        # 11) Incentives (deduct from costs)
        incentives_total = float(self.overlays.municipal_incentive_amount + self.overlays.rates_rebate_amount)

        total_costs_ex_profit = total_construction + prof_fees + muni_dcs + finance_marketing - incentives_total
        residual_land_value = gdv - (total_costs_ex_profit + target_profit)

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
                "total_construction_cost": self._round2(total_construction),
                "professional_fees": self._round2(prof_fees),
                "estimated_units": self._round2(estimated_units),
                "municipal_dcs": self._round2(muni_dcs),
                "finance_marketing": self._round2(finance_marketing),
                "incentives_total": self._round2(incentives_total),
                "target_profit": self._round2(target_profit),
                "total_costs_ex_profit": self._round2(total_costs_ex_profit),
            },
            "outputs": {
                "residual_land_value": self._round2(residual_land_value),
                "max_offer_for_land": self._round2(residual_land_value),
                "is_viable": residual_land_value > 0,
            },
        }

        # Return a compact summary + full audit trail
        return {
            "gross_development_value": self._round2(gdv),
            "total_construction_cost": self._round2(total_construction),
            "inclusionary_hit": inclusionary_hit,
            "residual_land_value": self._round2(residual_land_value),
            "audit": audit,
        }


# --- EXAMPLE USAGE (Woodstock Plot) ---
woodstock_zoning = {"floor_factor": 2.0, "coverage": 0.75}  # MU1 example

overlays = Overlays(
    inclusionary_enabled=True,
    inclusionary_share=0.10,
    inclusionary_cap_price_sqm=12000,
    heritage_cost_multiplier=1.05,     # example: +5% heritage compliance
    coastal_spec_multiplier=1.00,
    municipal_incentive_amount=0.0,
)

app = RLVMachine(
    plot_size=1000,
    zoning_rules=woodstock_zoning,
    base_efficiency=0.85,
    assumed_floors=3,   # important if coverage binds
    overlays=overlays,
)

result = app.calculate_rlv(
    exit_price_sqm=42000,
    build_cost_sqm=19000,
)

print(f"Max Offer for Land: R{result['residual_land_value']:,}")
# If you want full breakdown:
# import json; print(json.dumps(result["audit"], indent=2))
