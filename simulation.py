"""
Microgrid Energy Management Simulator — core engine.

Models an hourly PV + battery + grid/generator microgrid, dispatches
power with a rule-based Energy Management System (EMS), and reports
renewable penetration, unmet demand, and operating cost.

No external data files required — PV and load profiles are generated
synthetically so the simulator runs standalone.
"""

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Profile generation
# --------------------------------------------------------------------------

def generate_pv_profile(hours: int, pv_capacity_kw: float,
                         cloud_event_window: tuple | None = None) -> pd.Series:
    """
    Synthetic hourly PV output over `hours` hours.

    Daylight follows a clipped sine curve peaking at solar noon each day
    (06:00-18:00 window), scaled to pv_capacity_kw. Small random noise is
    added for realism. If cloud_event_window=(start_hour, end_hour) is
    given, PV output is cut to ~10% of normal within that window.
    """
    t = np.arange(hours)
    hour_of_day = t % 24
    # Daylight factor: 0 outside 6-18, sine-shaped peak at 12:00
    daylight = np.clip(np.sin(np.pi * (hour_of_day - 6) / 12), 0, 1)
    noise = np.random.normal(1.0, 0.05, size=hours).clip(0.85, 1.15)
    pv = pv_capacity_kw * daylight * noise
    pv = np.clip(pv, 0, None)

    if cloud_event_window is not None:
        start, end = cloud_event_window
        pv[start:end] *= 0.1

    return pd.Series(pv, name="pv_kw")


def generate_load_profile(hours: int, base_load_kw: float,
                           spike_window: tuple | None = None,
                           spike_multiplier: float = 2.0) -> pd.Series:
    """
    Synthetic hourly load over `hours` hours.

    Uses a simple two-peak daily residential/commercial shape (morning and
    evening peaks) scaled to base_load_kw, plus noise. If spike_window=
    (start_hour, end_hour) is given, load is multiplied by spike_multiplier
    in that window (simulates a demand surge).
    """
    t = np.arange(hours)
    hour_of_day = t % 24
    morning_peak = 0.6 * np.exp(-((hour_of_day - 8) ** 2) / (2 * 2.0 ** 2))
    evening_peak = 1.0 * np.exp(-((hour_of_day - 19) ** 2) / (2 * 2.5 ** 2))
    base = 0.4 + morning_peak + evening_peak  # relative shape, ~0.4-1.4x
    noise = np.random.normal(1.0, 0.05, size=hours).clip(0.85, 1.15)
    load = base_load_kw * base * noise

    if spike_window is not None:
        start, end = spike_window
        load[start:end] *= spike_multiplier

    return pd.Series(load, name="load_kw")


# --------------------------------------------------------------------------
# Battery model
# --------------------------------------------------------------------------

class Battery:
    """Simple battery model with SOC limits, efficiency, and rate limits."""

    def __init__(self, capacity_kwh: float, initial_soc_frac: float = 0.5,
                 min_soc_frac: float = 0.2, max_soc_frac: float = 1.0,
                 charge_eff: float = 0.95, discharge_eff: float = 0.95,
                 max_charge_kw: float | None = None,
                 max_discharge_kw: float | None = None):
        self.capacity_kwh = capacity_kwh
        self.soc_kwh = capacity_kwh * initial_soc_frac
        self.min_soc_kwh = capacity_kwh * min_soc_frac
        self.max_soc_kwh = capacity_kwh * max_soc_frac
        self.charge_eff = charge_eff
        self.discharge_eff = discharge_eff
        # Default rate limit: 0.5C if not specified
        self.max_charge_kw = max_charge_kw or (0.5 * capacity_kwh)
        self.max_discharge_kw = max_discharge_kw or (0.5 * capacity_kwh)

    @property
    def soc_percent(self) -> float:
        return 100 * self.soc_kwh / self.capacity_kwh

    def available_charge_kw(self) -> float:
        """Max kW the battery could absorb right now, given SOC headroom and rate limit."""
        headroom_kwh = self.max_soc_kwh - self.soc_kwh
        return min(self.max_charge_kw, headroom_kwh / self.charge_eff)

    def available_discharge_kw(self) -> float:
        """Max kW the battery could deliver right now, given SOC floor and rate limit."""
        deliverable_kwh = (self.soc_kwh - self.min_soc_kwh) * self.discharge_eff
        return min(self.max_discharge_kw, max(deliverable_kwh, 0))

    def charge(self, kw: float, dt_hours: float = 1.0) -> float:
        """Charge at up to `kw` for dt_hours; returns actual kW accepted."""
        kw = max(0.0, min(kw, self.available_charge_kw()))
        self.soc_kwh += kw * self.charge_eff * dt_hours
        self.soc_kwh = min(self.soc_kwh, self.max_soc_kwh)
        return kw

    def discharge(self, kw: float, dt_hours: float = 1.0) -> float:
        """Discharge at up to `kw` for dt_hours; returns actual kW delivered."""
        kw = max(0.0, min(kw, self.available_discharge_kw()))
        self.soc_kwh -= (kw / self.discharge_eff) * dt_hours
        self.soc_kwh = max(self.soc_kwh, self.min_soc_kwh)
        return kw


# --------------------------------------------------------------------------
# Rule-based Energy Management System
# --------------------------------------------------------------------------

def dispatch(pv_kw: float, load_kw: float, battery: Battery,
             grid_available: bool = True, generator_available: bool = True,
             dt_hours: float = 1.0) -> dict:
    """
    One timestep of rule-based dispatch.

    Priority: PV covers load first, surplus PV charges battery.
    If PV is short, battery covers the gap, then grid, then generator.
    Anything still unmet is logged as unmet demand (a mini blackout).
    """
    pv_to_load = min(pv_kw, load_kw)
    remaining_load = load_kw - pv_to_load
    pv_surplus = pv_kw - pv_to_load

    pv_to_battery = battery.charge(pv_surplus, dt_hours) if pv_surplus > 0 else 0.0

    battery_to_load = 0.0
    if remaining_load > 0:
        battery_to_load = battery.discharge(remaining_load, dt_hours)
        remaining_load -= battery_to_load

    grid_to_load = 0.0
    if remaining_load > 0 and grid_available:
        grid_to_load = remaining_load
        remaining_load = 0.0

    generator_to_load = 0.0
    if remaining_load > 0 and generator_available:
        generator_to_load = remaining_load
        remaining_load = 0.0

    unmet_demand = max(remaining_load, 0.0)

    return {
        "pv_to_load": pv_to_load,
        "pv_to_battery": pv_to_battery,
        "pv_curtailed": pv_surplus - pv_to_battery,
        "battery_to_load": battery_to_load,
        "grid_to_load": grid_to_load,
        "generator_to_load": generator_to_load,
        "unmet_demand": unmet_demand,
        "soc_percent": battery.soc_percent,
    }


# --------------------------------------------------------------------------
# Scenario runner
# --------------------------------------------------------------------------

SCENARIOS = ["baseline", "cloud_event", "demand_spike", "grid_outage"]


def run_simulation(pv_capacity_kw: float, battery_capacity_kwh: float,
                    base_load_kw: float, days: int = 3,
                    scenario: str = "baseline",
                    grid_price_per_kwh: float = 0.15,
                    diesel_price_per_kwh: float = 0.35,
                    seed: int | None = 42) -> tuple[pd.DataFrame, dict]:
    """
    Run the microgrid simulation for `days` days at hourly resolution
    under the given scenario. Returns (timeseries_df, summary_dict).
    """
    if seed is not None:
        np.random.seed(seed)

    hours = days * 24
    disturbance_start = hours // 2  # place any disturbance mid-simulation
    disturbance_len = 6

    cloud_window = (disturbance_start, disturbance_start + disturbance_len) \
        if scenario == "cloud_event" else None
    spike_window = (disturbance_start, disturbance_start + disturbance_len) \
        if scenario == "demand_spike" else None
    grid_outage_window = (disturbance_start, disturbance_start + disturbance_len) \
        if scenario == "grid_outage" else None

    pv = generate_pv_profile(hours, pv_capacity_kw, cloud_event_window=cloud_window)
    load = generate_load_profile(hours, base_load_kw, spike_window=spike_window)

    battery = Battery(capacity_kwh=battery_capacity_kwh)

    records = []
    for h in range(hours):
        grid_available = not (grid_outage_window and
                               grid_outage_window[0] <= h < grid_outage_window[1])
        result = dispatch(pv.iloc[h], load.iloc[h], battery,
                           grid_available=grid_available, generator_available=True)
        result["hour"] = h
        result["pv_kw"] = pv.iloc[h]
        result["load_kw"] = load.iloc[h]
        result["grid_available"] = grid_available
        records.append(result)

    df = pd.DataFrame(records).set_index("hour")

    # Summary metrics
    total_load = df["load_kw"].sum()
    renewable_supplied = df["pv_to_load"].sum() + df["battery_to_load"].sum()
    renewable_penetration = 100 * renewable_supplied / total_load if total_load > 0 else 0
    unmet_demand_kwh = df["unmet_demand"].sum()
    total_cost = (df["grid_to_load"].sum() * grid_price_per_kwh +
                  df["generator_to_load"].sum() * diesel_price_per_kwh)

    summary = {
        "scenario": scenario,
        "renewable_penetration_pct": round(renewable_penetration, 1),
        "unmet_demand_kwh": round(unmet_demand_kwh, 2),
        "total_operating_cost": round(total_cost, 2),
        "total_load_kwh": round(total_load, 1),
        "final_soc_pct": round(df["soc_percent"].iloc[-1], 1),
    }

    return df, summary


if __name__ == "__main__":
    # Quick smoke test when run directly: python simulation.py
    for s in SCENARIOS:
        _, summary = run_simulation(
            pv_capacity_kw=10, battery_capacity_kwh=20, base_load_kw=6,
            days=3, scenario=s,
        )
        print(summary)
