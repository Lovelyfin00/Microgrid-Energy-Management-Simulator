"""
Microgrid Energy Management Simulator — interactive dashboard.

Run locally:    streamlit run app.py
Deploy free:    Streamlit Community Cloud (share.streamlit.io) or
                 Hugging Face Spaces (Streamlit SDK) — see README.md
"""

import matplotlib.pyplot as plt
import streamlit as st

from simulation import SCENARIOS, run_simulation

st.set_page_config(page_title="Microgrid EMS Simulator", layout="wide")

st.title("☀️ Renewable Microgrid Energy Management Simulator")
st.caption(
    "A PV + battery + grid/generator microgrid, dispatched by a rule-based "
    "Energy Management System. Adjust the system, run a scenario, and see "
    "how it responds."
)

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("System configuration")
    pv_capacity_kw = st.slider("PV capacity (kW)", 1, 50, 10)
    battery_capacity_kwh = st.slider("Battery capacity (kWh)", 1, 100, 20)
    base_load_kw = st.slider("Average load (kW)", 1, 30, 6)
    days = st.slider("Simulation length (days)", 1, 7, 3)

    st.header("Scenario")
    scenario = st.selectbox(
        "Choose a disturbance to test",
        SCENARIOS,
        format_func=lambda s: {
            "baseline": "Baseline (normal operation)",
            "cloud_event": "Cloud event (PV drops mid-simulation)",
            "demand_spike": "Demand spike (load doubles for 6h)",
            "grid_outage": "Grid outage (grid unavailable for 6h)",
        }[s],
    )

    st.header("Cost assumptions")
    grid_price = st.number_input("Grid price ($/kWh)", 0.01, 2.0, 0.15, step=0.01)
    diesel_price = st.number_input("Generator/diesel price ($/kWh)", 0.01, 2.0, 0.35, step=0.01)

    run_clicked = st.button("▶ Run simulation", type="primary", use_container_width=True)
    compare_clicked = st.button("⇄ Compare vs baseline", use_container_width=True)

# --------------------------------------------------------------------------
# Run + display
# --------------------------------------------------------------------------

def render_run(df, summary, title):
    st.subheader(title)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Renewable penetration", f"{summary['renewable_penetration_pct']}%")
    c2.metric("Unmet demand", f"{summary['unmet_demand_kwh']} kWh")
    c3.metric("Operating cost", f"${summary['total_operating_cost']}")
    c4.metric("Final battery SOC", f"{summary['final_soc_pct']}%")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(df.index, df["pv_kw"], label="PV output", color="#f2a900")
    axes[0].plot(df.index, df["load_kw"], label="Load", color="#333333")
    axes[0].fill_between(df.index, 0, df["grid_to_load"] + df["generator_to_load"],
                          label="Grid + generator supply", color="#c0392b", alpha=0.3)
    axes[0].set_ylabel("kW")
    axes[0].set_title("Power flow")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(df.index, df["soc_percent"], color="#2471a3")
    axes[1].axhline(20, color="grey", linestyle="--", linewidth=1, label="Min SOC")
    axes[1].set_ylabel("Battery SOC (%)")
    axes[1].set_xlabel("Hour")
    axes[1].set_title("Battery state of charge")
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    st.pyplot(fig)

    if summary["unmet_demand_kwh"] > 0:
        st.warning(
            f"⚠ {summary['unmet_demand_kwh']} kWh of demand went unmet during this run "
            "— the system could not fully cover the load with the current sizing."
        )

    with st.expander("Raw hourly data"):
        st.dataframe(df, use_container_width=True)


if compare_clicked:
    df_base, summary_base = run_simulation(
        pv_capacity_kw, battery_capacity_kwh, base_load_kw, days,
        "baseline", grid_price, diesel_price,
    )
    df_scn, summary_scn = run_simulation(
        pv_capacity_kw, battery_capacity_kwh, base_load_kw, days,
        scenario, grid_price, diesel_price,
    )
    left, right = st.columns(2)
    with left:
        render_run(df_base, summary_base, "Baseline")
    with right:
        render_run(df_scn, summary_scn, scenario.replace("_", " ").title())

elif run_clicked or True:
    # Renders on first load too, so the app isn't blank before any click
    df, summary = run_simulation(
        pv_capacity_kw, battery_capacity_kwh, base_load_kw, days,
        scenario, grid_price, diesel_price,
    )
    render_run(df, summary, scenario.replace("_", " ").title())

st.divider()
st.caption(
    "Engine: rule-based dispatch (PV → load → battery → grid → generator). "
    "See simulation.py for the model. Built as a lightweight, fully "
    "software-based microgrid EMS — no physical hardware involved."
)
