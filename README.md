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
- **Generator model** — backup generator dispatch constrained by a configurable
  maximum output capacity, with remaining demand reported as unmet demand.
- **EMS dispatch logic** — priority order PV → load, surplus PV → battery,
  battery → load, grid → load, generator → load, with any shortfall
  logged as unmet demand.
- **Disturbance scenarios** — cloud event, demand spike, and grid outage,
  each comparable against a baseline run.
- **Interactive dashboard** — sliders for PV size, battery size, load,
  generator capacity, and
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

## Lagos irradiance data

`irradiance.py` provides opt-in clients for the official NASA POWER and
European Commission PVGIS 5.3 services at Lagos coordinates (`6.5244, 3.3792`).
It returns normalized hourly records with irradiance, temperature, wind speed,
and a `source` column. PVGIS global irradiance is reconstructed as beam plus
diffuse plus reflected irradiance, and its original six-minute timestamp
offset is preserved.

The Streamlit irradiance tab fetches data only when requested and caches the
result for one hour. This avoids repeated API calls on normal dashboard
reruns while still supporting a manual refresh for updated NASA POWER data.
PVGIS is primarily a historical solar-resource dataset, so it should not be
treated as a live sensor feed. The simulation and forecasting tabs continue
to use the synthetic profile until a deliberate modelling decision connects
the external irradiance series to PV generation.

```python
from irradiance import fetch_lagos_irradiance

data = fetch_lagos_irradiance("2020-01-01", "2020-01-31")
data.to_csv("lagos_irradiance_2020-01.csv", index=False)
```

The services require an internet connection when the fetch function is
called. NASA POWER uses UTC timestamps; PVGIS returns its service timestamps
as supplied by the API. The module keeps the two sources in long format so
they are not accidentally treated as identical measurements. CSV export is
explicitly user-triggered from the dashboard rather than used as an
unmanaged automatic data store.
