import matplotlib.pyplot as plt
import streamlit as st
from datetime import date

from irradiance import fetch_lagos_irradiance
from simulation import SCENARIOS, run_simulation
from forecasting import train_forecast_model

st.set_page_config(page_title="Microgrid EMS Simulator", layout="wide")

st.title("Hybrid Microgrid Energy Management Simulator")
st.caption(
    "A hybrid microgrid combining solar PV, battery storage, and backup power. "
    "Change the system settings, run a scenario, and see how the Energy "
    "Management System responds to changes in demand and available power. "
    "The simulator uses a reproducible synthetic profile; the irradiance tab "
    "retrieves optional Lagos data from NASA POWER and PVGIS."
)

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("System configuration")
    pv_capacity_kw = st.slider("PV capacity (kW)", 1, 50, 10)
    battery_capacity_kwh = st.slider("Battery capacity (kWh)", 1, 100, 20)
    base_load_kw = st.slider("Average load (kW)", 1, 30, 6)
    generator_capacity_kw = st.slider("Generator capacity (kW)", 0, 50, 10)
    days = st.slider("Simulation length (days)", 1, 7, 3)

    st.header("Scenario")
    scenario = st.selectbox(
        "Choose a disturbance to test",
        SCENARIOS,
        format_func=lambda s: {
            "baseline": "Baseline (normal operation)",
            "cloud_event": "Cloud event (PV dims for 6h)",
            "demand_spike": "Demand spike (load doubles for 6h)",
            "grid_outage": "Grid outage (grid unavailable for 6h)",
            "inverter_failure": "Inverter failure (PV unavailable for 6h)",
        }[s],
    )

    st.header("Dispatch strategy")
    dispatch_mode = st.radio(
        "How should the EMS decide?",
        ["rule_based", "optimized"],
        format_func=lambda m: "Rule-based" if m == "rule_based" else "Optimized (cost-minimizing)",
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


@st.cache_data(ttl=3600, show_spinner=False)
def load_irradiance_data(start_date: date, end_date: date):
    """Cache external irradiance data for one hour per date range."""
    return fetch_lagos_irradiance(start_date.isoformat(), end_date.isoformat())


sim_tab, forecast_tab, irradiance_tab = st.tabs(
    ["Simulation", "PV Forecasting", "Lagos Irradiance"]
)

with sim_tab:
    if compare_clicked:
        df_base, summary_base = run_simulation(
            pv_capacity_kw, battery_capacity_kwh, base_load_kw, days,
            "baseline", grid_price, diesel_price, dispatch_mode,
            generator_capacity_kw=generator_capacity_kw,
        )
        df_scn, summary_scn = run_simulation(
            pv_capacity_kw, battery_capacity_kwh, base_load_kw, days,
            scenario, grid_price, diesel_price, dispatch_mode,
            generator_capacity_kw=generator_capacity_kw,
        )
        left, right = st.columns(2)
        with left:
            render_run(df_base, summary_base, "Baseline")
        with right:
            render_run(df_scn, summary_scn, scenario.replace("_", " ").title())
        df = df_scn  # feed the forecasting tab with the scenario run

    else:
        # Renders on first load too, so the app isn't blank before any click
        df, summary = run_simulation(
            pv_capacity_kw, battery_capacity_kwh, base_load_kw, days,
            scenario, grid_price, diesel_price, dispatch_mode,
            generator_capacity_kw=generator_capacity_kw,
        )
        render_run(df, summary, scenario.replace("_", " ").title())


with forecast_tab:
    st.subheader("Next-Hour PV Forecast with Random Forest")
    st.caption(
        "A Random Forest model predicts the next hour of PV generation using "
        "the previous 3 hours of PV output and the time of day. The forecast "
        "is trained on the synthetic PV data generated by the scenario above. "
        "Downloaded NASA POWER or PVGIS data is available in the irradiance tab "
        "but is not silently mixed into this reproducible forecast."
    )
    try:
        model, X_test, y_test, y_pred, r2 = train_forecast_model(df["pv_kw"])

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(y_test)), y_test, label="Actual PV", color="#f2a900")
        ax.plot(range(len(y_pred)), y_pred, label="Predicted PV", color="#2471a3", linestyle="--")
        ax.set_xlabel("Test hour")
        ax.set_ylabel("PV output (kW)")
        ax.legend()
        st.pyplot(fig)

        st.metric("Test R² Score", f"{r2:.3f}")

        st.caption(
            "An R² closer to 1 indicates that the model captures the PV generation "
            "pattern well. Since the simulation uses a clean synthetic PV profile, "
            "the score is expected to be high. In practice, the model would be "
            "validated using measured PV and weather data."
        )
    except ValueError as e:
        st.warning(str(e))


with irradiance_tab:
    st.subheader("Lagos hourly irradiance")
    st.caption(
        "Fetches historical irradiance from PVGIS 5.3 and NASA POWER for "
        "Lagos (6.5244, 3.3792). Results are cached for one hour, so normal "
        "dashboard reruns do not repeatedly call the external services. "
        "Use Refresh to request new data explicitly."
    )

    date_col1, date_col2 = st.columns(2)
    with date_col1:
        irradiance_start = st.date_input(
            "Start date", value=date(2020, 1, 1), key="irradiance_start"
        )
    with date_col2:
        irradiance_end = st.date_input(
            "End date", value=date(2020, 1, 2), key="irradiance_end"
        )

    fetch_col, refresh_col = st.columns(2)
    with fetch_col:
        fetch_irradiance = st.button("Fetch irradiance", type="primary")
    with refresh_col:
        refresh_irradiance = st.button("Refresh cached data")

    if fetch_irradiance or refresh_irradiance:
        if irradiance_end < irradiance_start:
            st.error("End date must be on or after start date.")
        else:
            if refresh_irradiance:
                load_irradiance_data.clear()
            try:
                with st.spinner("Requesting NASA POWER and PVGIS data..."):
                    irradiance_df = load_irradiance_data(
                        irradiance_start, irradiance_end
                    )
                st.metric("Records returned", len(irradiance_df))
                st.line_chart(
                    irradiance_df.pivot(
                        index="timestamp", columns="source", values="ghi_wm2"
                    ),
                    y_label="Global irradiance (W/m²)",
                )
                st.download_button(
                    "Download fetched data as CSV",
                    irradiance_df.to_csv(index=False),
                    file_name="lagos_irradiance.csv",
                    mime="text/csv",
                )
                with st.expander("Raw irradiance data"):
                    st.dataframe(irradiance_df, use_container_width=True)
            except (RuntimeError, ValueError, KeyError) as error:
                st.error(f"Unable to fetch irradiance data: {error}")
