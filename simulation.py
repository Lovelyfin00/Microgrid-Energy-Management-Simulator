"""
Microgrid Energy Management Simulator — core engine.

Models an hourly PV + battery + grid/generator microgrid, dispatches
power with either a rule-based or a cost-optimizing Energy Management
System (EMS), and reports renewable penetration, unmet demand, and
operating cost.

No external data files required — PV and load profiles are generated
synthetically so the simulator runs standalone.
"""

import numpy as np
import pandas as pd
import pulp


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
        if capacity_kwh <= 0:
            raise ValueError("Battery capacity must be greater than zero.")
        if not 0 <= min_soc_frac <= initial_soc_frac <= max_soc_frac <= 1:
            raise ValueError(
                "SOC fractions must satisfy 0 <= min <= initial <= max <= 1."
            )
        if not 0 < charge_eff <= 1 or not 0 < discharge_eff <= 1:
            raise ValueError("Battery efficiencies must be greater than zero and at most one.")
        if max_charge_kw is not None and max_charge_kw < 0:
            raise ValueError("Maximum charge rate cannot be negative.")
        if max_discharge_kw is not None and max_discharge_kw < 0:
            raise ValueError("Maximum discharge rate cannot be negative.")

        self.capacity_kwh = capacity_kwh
        self.soc_kwh = capacity_kwh * initial_soc_frac
        self.renewable_soc_kwh = 0.0
        self.last_discharge_renewable_kw = 0.0
        self.min_soc_kwh = capacity_kwh * min_soc_frac
        self.max_soc_kwh = capacity_kwh * max_soc_frac
        self.charge_eff = charge_eff
        self.discharge_eff = discharge_eff
        # Default rate limit: 0.5C if not specified
        self.max_charge_kw = (0.5 * capacity_kwh
                              if max_charge_kw is None else max_charge_kw)
        self.max_discharge_kw = (0.5 * capacity_kwh
                                 if max_discharge_kw is None else max_discharge_kw)

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
        stored_kwh = kw * self.charge_eff * dt_hours
        self.soc_kwh += stored_kwh
        self.renewable_soc_kwh += stored_kwh
        self.soc_kwh = min(self.soc_kwh, self.max_soc_kwh)
        self.renewable_soc_kwh = min(self.renewable_soc_kwh, self.soc_kwh)
        return kw

    def discharge(self, kw: float, dt_hours: float = 1.0) -> float:
        """Discharge at up to `kw` for dt_hours; returns actual kW delivered."""
        kw = max(0.0, min(kw, self.available_discharge_kw()))
        self.last_discharge_renewable_kw = 0.0
        energy_removed_kwh = (kw / self.discharge_eff) * dt_hours
        renewable_fraction = (
            self.renewable_soc_kwh / self.soc_kwh if self.soc_kwh > 0 else 0.0
        )
        self.soc_kwh -= energy_removed_kwh
        self.renewable_soc_kwh -= energy_removed_kwh * renewable_fraction
        self.soc_kwh = max(self.soc_kwh, self.min_soc_kwh)
        self.renewable_soc_kwh = max(0.0, min(self.renewable_soc_kwh, self.soc_kwh))
        if dt_hours > 0:
            self.last_discharge_renewable_kw = (
                energy_removed_kwh * renewable_fraction / dt_hours
                * self.discharge_eff
            )
        return kw


# --------------------------------------------------------------------------
# Rule-based Energy Management System
# --------------------------------------------------------------------------

def dispatch(pv_kw: float, load_kw: float, battery: Battery,
             grid_available: bool = True, generator_available: bool = True,
             dt_hours: float = 1.0,
             generator_capacity_kw: float = 10.0) -> dict:
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
    renewable_battery_to_load = 0.0
    if remaining_load > 0:
        battery_to_load = battery.discharge(remaining_load, dt_hours)
        renewable_battery_to_load = battery.last_discharge_renewable_kw
        remaining_load -= battery_to_load

    grid_to_load = 0.0
    if remaining_load > 0 and grid_available:
        grid_to_load = remaining_load
        remaining_load = 0.0

    generator_to_load = 0.0
    if remaining_load > 0 and generator_available:
        generator_to_load = min(remaining_load, generator_capacity_kw)
        remaining_load -= generator_to_load

    unmet_demand = max(remaining_load, 0.0)

    return {
        "pv_to_load": pv_to_load,
        "pv_to_battery": pv_to_battery,
        "pv_curtailed": pv_surplus - pv_to_battery,
        "battery_to_load": battery_to_load,
        "renewable_battery_to_load": renewable_battery_to_load,
        "grid_to_load": grid_to_load,
        "generator_to_load": generator_to_load,
        "unmet_demand": unmet_demand,
        "soc_percent": battery.soc_percent,
    }


def optimize_dispatch(pv_kw: float, load_kw: float, battery: Battery,
                       grid_price: float, diesel_price: float,
                       grid_available: bool = True,
                       generator_available: bool = True,
                       dt_hours: float = 1.0,
                       generator_capacity_kw: float = 10.0) -> dict:
    """
    One timestep of cost-minimizing dispatch, solved with PuLP.

    PV is applied to load first (it's free, so it's never worth curtailing
    in favor of a paid source). The solver then decides how the battery,
    grid, and generator split covering whatever load PV didn't cover,
    minimizing (grid_to_load * grid_price + generator_to_load * diesel_price),
    subject to the battery's available charge/discharge headroom and the
    requirement that all sources sum to exactly the remaining load (or to
    unmet demand if nothing can cover it).

    Falls back to leaving the deficit as unmet_demand if grid, generator,
    and battery combined can't cover it (this is expected during a
    grid-outage scenario with a small battery).
    """
    pv_to_load = min(pv_kw, load_kw)
    remaining_load = load_kw - pv_to_load
    pv_surplus = pv_kw - pv_to_load

    # Free surplus PV always goes to the battery if there's headroom;
    # this isn't part of the cost optimization since it's free energy.
    pv_to_battery = battery.charge(pv_surplus, dt_hours) if pv_surplus > 0 else 0.0
    pv_curtailed = pv_surplus - pv_to_battery

    battery_max = battery.available_discharge_kw() if remaining_load > 0 else 0.0
    grid_max = remaining_load if grid_available else 0.0
    generator_max = (min(remaining_load, generator_capacity_kw)
                     if generator_available else 0.0)

    if remaining_load <= 1e-9:
        battery_to_load = renewable_battery_to_load = 0.0
        grid_to_load = generator_to_load = unmet = 0.0
    else:
        prob = pulp.LpProblem("dispatch_cost_min", pulp.LpMinimize)
        b = pulp.LpVariable("battery_to_load", 0, battery_max)
        g = pulp.LpVariable("grid_to_load", 0, grid_max)
        d = pulp.LpVariable("generator_to_load", 0, generator_max)
        u = pulp.LpVariable("unmet_demand", 0, remaining_load)

        # Unmet demand carries a heavy penalty so the solver only accepts
        # it when grid + generator + battery genuinely can't cover the load.
        unmet_penalty = 100 * max(grid_price, diesel_price, 0.01)
        prob += g * grid_price + d * diesel_price + u * unmet_penalty
        prob += b + g + d + u == remaining_load

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        battery_to_load = battery.discharge(b.value() or 0.0, dt_hours)
        renewable_battery_to_load = battery.last_discharge_renewable_kw
        grid_to_load = g.value() or 0.0
        generator_to_load = d.value() or 0.0
        unmet = u.value() or 0.0

    return {
        "pv_to_load": pv_to_load,
        "pv_to_battery": pv_to_battery,
        "pv_curtailed": pv_curtailed,
        "battery_to_load": battery_to_load,
        "renewable_battery_to_load": renewable_battery_to_load,
        "grid_to_load": grid_to_load,
        "generator_to_load": generator_to_load,
        "unmet_demand": unmet,
        "soc_percent": battery.soc_percent,
    }


# --------------------------------------------------------------------------
# Scenario runner
# --------------------------------------------------------------------------

SCENARIOS = ["baseline", "cloud_event", "demand_spike", "grid_outage", "inverter_failure"]


def run_simulation(pv_capacity_kw: float, battery_capacity_kwh: float,
                    base_load_kw: float, days: int = 3,
                    scenario: str = "baseline",
                    grid_price_per_kwh: float = 0.15,
                    diesel_price_per_kwh: float = 0.35,
                    dispatch_mode: str = "rule_based",
                    seed: int | None = 42,
                    generator_capacity_kw: float = 10.0) -> tuple[pd.DataFrame, dict]:
    """
    Run the microgrid simulation for `days` days at hourly resolution
    under the given scenario and dispatch_mode ("rule_based" or
    "optimized"). Returns (timeseries_df, summary_dict).
    """
    if not isinstance(days, (int, np.integer)) or isinstance(days, bool) or days < 1:
        raise ValueError("Simulation days must be a positive integer.")
    if pv_capacity_kw < 0 or battery_capacity_kwh <= 0 or base_load_kw <= 0:
        raise ValueError("PV capacity cannot be negative; battery capacity and base load must be greater than zero.")
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Choose one of {SCENARIOS}.")
    if dispatch_mode not in ("rule_based", "optimized"):
        raise ValueError("Dispatch mode must be 'rule_based' or 'optimized'.")
    if grid_price_per_kwh < 0 or diesel_price_per_kwh < 0:
        raise ValueError("Energy prices cannot be negative.")
    if generator_capacity_kw < 0:
        raise ValueError("Generator capacity cannot be negative.")

    if seed is not None:
        np.random.seed(seed)

    hours = days * 24
    disturbance_start = hours // 2  # place any disturbance mid-simulation
    disturbance_len = 6

    # inverter_failure zeroes PV the same way a cloud event dims it, so it
    # reuses the same generation-time window mechanism.
    cloud_window = (disturbance_start, disturbance_start + disturbance_len) \
        if scenario in ("cloud_event", "inverter_failure") else None
    spike_window = (disturbance_start, disturbance_start + disturbance_len) \
        if scenario == "demand_spike" else None
    grid_outage_window = (disturbance_start, disturbance_start + disturbance_len) \
        if scenario == "grid_outage" else None

    pv = generate_pv_profile(hours, pv_capacity_kw, cloud_event_window=cloud_window)
    if scenario == "inverter_failure":
        # Total loss of PV output for the window (inverter down = no PV
        # reaches the bus at all), vs. cloud_event's partial ~10% dimming.
        start, end = cloud_window
        pv.iloc[start:end] = 0.0
    load = generate_load_profile(hours, base_load_kw, spike_window=spike_window)

    battery = Battery(capacity_kwh=battery_capacity_kwh)

    records = []
    for h in range(hours):
        grid_available = not (grid_outage_window and
                               grid_outage_window[0] <= h < grid_outage_window[1])
        if dispatch_mode == "optimized":
            result = optimize_dispatch(
                pv.iloc[h], load.iloc[h], battery,
                grid_price=grid_price_per_kwh, diesel_price=diesel_price_per_kwh,
                grid_available=grid_available, generator_available=True,
                generator_capacity_kw=generator_capacity_kw,
            )
        else:
            result = dispatch(pv.iloc[h], load.iloc[h], battery,
                               grid_available=grid_available, generator_available=True,
                               generator_capacity_kw=generator_capacity_kw)
        result["hour"] = h
        result["pv_kw"] = pv.iloc[h]
        result["load_kw"] = load.iloc[h]
        result["grid_available"] = grid_available
        records.append(result)

    df = pd.DataFrame(records).set_index("hour")

    # Summary metrics
    total_load = df["load_kw"].sum()
    renewable_supplied = (
        df["pv_to_load"].sum() + df["renewable_battery_to_load"].sum()
    )
    renewable_penetration = 100 * renewable_supplied / total_load if total_load > 0 else 0
    unmet_demand_kwh = df["unmet_demand"].sum()
    total_cost = (df["grid_to_load"].sum() * grid_price_per_kwh +
                  df["generator_to_load"].sum() * diesel_price_per_kwh)

    summary = {
        "scenario": scenario,
        "dispatch_mode": dispatch_mode,
        "renewable_penetration_pct": round(renewable_penetration, 1),
        "unmet_demand_kwh": round(unmet_demand_kwh, 2),
        "total_operating_cost": round(total_cost, 2),
        "total_load_kwh": round(total_load, 1),
        "final_soc_pct": round(df["soc_percent"].iloc[-1], 1),
    }

    return df, summary


if __name__ == "__main__":
    # Quick smoke test when run directly: python simulation.py
    for mode in ("rule_based", "optimized"):
        for s in SCENARIOS:
            _, summary = run_simulation(
                pv_capacity_kw=10, battery_capacity_kwh=20, base_load_kw=6,
                days=3, scenario=s, dispatch_mode=mode,
            )
            print(summary)
