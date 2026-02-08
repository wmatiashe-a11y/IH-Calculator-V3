import streamlit as st

class RLVMachine:
    def __init__(self, plot_size, zoning_rules):
        self.plot_size = plot_size
        self.zoning = zoning_rules

    def calculate_rlv(self, exit_price_sqm, build_cost_sqm, include_affordable=True):
        total_bulk = self.plot_size * self.zoning["floor_factor"]
        sellable_area = total_bulk * 0.85

        if include_affordable:
            affordable_area = sellable_area * 0.10
            market_area = sellable_area * 0.90
            gdv = (market_area * exit_price_sqm) + (affordable_area * 12000)
        else:
            gdv = sellable_area * exit_price_sqm

        total_construction = total_bulk * build_cost_sqm
        prof_fees = total_construction * 0.15

        estimated_units = sellable_area / 60
        muni_dcs = estimated_units * 45000

        finance_marketing = gdv * 0.08
        target_profit = gdv * 0.20

        residual_land_value = gdv - (
            total_construction + prof_fees + muni_dcs + finance_marketing + target_profit
        )

        return {
            "gross_development_value": round(gdv, 2),
            "total_construction_cost": round(total_construction, 2),
            "inclusionary_hit": "10% Area Capped" if include_affordable else "None",
            "residual_land_value": round(residual_land_value, 2),
        }


def main():
    st.set_page_config(page_title="IH / RLV Calculator", layout="wide")
    st.title("IH / Residual Land Value Calculator")

    with st.sidebar:
        st.header("Inputs")
        plot_size = st.number_input("Plot size (m²)", min_value=1.0, value=1000.0, step=50.0)
        floor_factor = st.number_input("Floor factor (FAR)", min_value=0.1, value=2.0, step=0.1)
        coverage = st.number_input("Coverage", min_value=0.1, max_value=1.0, value=0.75, step=0.05)

        exit_price_sqm = st.number_input("Exit price (R/m²)", min_value=1.0, value=42000.0, step=500.0)
        build_cost_sqm = st.number_input("Build cost (R/m²)", min_value=1.0, value=19000.0, step=500.0)

        include_affordable = st.checkbox("Include inclusionary housing (10% @ cap)", value=True)

        run = st.button("Calculate", type="primary")

    if not run:
        st.info("Enter inputs on the left, then click **Calculate**.")
        return

    zoning = {"floor_factor": float(floor_factor), "coverage": float(coverage)}
    model = RLVMachine(plot_size=float(plot_size), zoning_rules=zoning)
    result = model.calculate_rlv(
        exit_price_sqm=float(exit_price_sqm),
        build_cost_sqm=float(build_cost_sqm),
        include_affordable=bool(include_affordable),
    )

    st.subheader("Result")
    st.metric("Max Offer for Land (Residual Land Value)", f"R{result['residual_land_value']:,.2f}")

    with st.expander("Show full calculation"):
        st.json(result)


if __name__ == "__main__":
    main()
