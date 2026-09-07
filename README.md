# Renewable Microgrid Energy Management Simulator

A Python-based, fully software simulation of a solar PV + battery +
grid/generator microgrid. It models hourly power generation and demand,
dispatches power through an Energy Management System (EMS) with rules, and
lets a user interactively test how the system responds to disturbances such as
a cloudy day, a demand spike, or a grid outage.

## What it does

- **Generation model** — synthetic hourly PV output following a daylight
  curve, scalable by capacity.
- **Load model** — synthetic hourly demand with morning/evening peaks,
  scalable by average load.
- **Battery model** — SOC tracking, charge/discharge efficiency, min/max
  SOC limits, and charge/discharge rate limits.
- **EMS dispatch logic** — priority order PV → load, surplus PV → battery,
  battery → load, grid → load, generator → load, with any shortfall
  logged as unmet demand.
- **Disturbance scenarios** — cloud event, demand spike, and grid outage,
  each comparable against a baseline run.
- **Interactive dashboard** — sliders for PV size, battery size, load, and
  simulation length; a scenario picker; a side-by-side "compare vs
  baseline" view; live plots of power flow and battery SOC; and summary
  metrics (renewable penetration %, unmet demand, operating cost).

## To Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Move the sliders, pick a scenario, and
click **Run simulation** or **Compare vs baseline**.

You can also run the engine standalone without the UI:

```bash
python simulation.py
```
