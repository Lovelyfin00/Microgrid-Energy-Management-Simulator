"""Hourly irradiance data clients for Lagos, Nigeria.

The functions in this module fetch data from the official NASA POWER and
European Commission PVGIS services. Network access is opt-in: importing this
module does not make a request.
"""

from __future__ import annotations

from datetime import date, datetime
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

LAGOS_LATITUDE = 6.5244
LAGOS_LONGITUDE = 3.3792
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
PVGIS_SERIES_URL = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"


def _validate_dates(start_date: str, end_date: str) -> tuple[date, date]:
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError as error:
        raise ValueError("Dates must use YYYY-MM-DD format.") from error

    if end < start:
        raise ValueError("end_date must be on or after start_date.")
    return start, end


def _get_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "Microgrid-Energy-Management-Simulator/1.0"})
    try:
        with urlopen(request, timeout=60) as response:
            return json.load(response)
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"Irradiance service request failed: {error}") from error


def fetch_nasa_power_hourly(
    start_date: str,
    end_date: str,
    latitude: float = LAGOS_LATITUDE,
    longitude: float = LAGOS_LONGITUDE,
) -> pd.DataFrame:
    """Fetch hourly NASA POWER irradiance and weather data in UTC.

    ``ghi_wm2`` is NASA POWER's hourly all-sky surface shortwave irradiance.
    """
    start, end = _validate_dates(start_date, end_date)
    params = urlencode(
        {
            "parameters": "ALLSKY_SFC_SW_DWN,T2M,WS10M",
            "community": "RE",
            "longitude": longitude,
            "latitude": latitude,
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "format": "JSON",
            "time-standard": "UTC",
        }
    )
    payload = _get_json(f"{NASA_POWER_URL}?{params}")
    parameters = payload["properties"]["parameter"]
    rows = []
    for timestamp, ghi in parameters["ALLSKY_SFC_SW_DWN"].items():
        rows.append(
            {
                "timestamp": pd.to_datetime(timestamp, format="%Y%m%d%H"),
                "ghi_wm2": float(ghi),
                "temperature_c": float(parameters["T2M"][timestamp]),
                "wind_speed_ms": float(parameters["WS10M"][timestamp]),
                "source": "NASA POWER",
            }
        )
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def fetch_pvgis_hourly(
    start_date: str,
    end_date: str,
    latitude: float = LAGOS_LATITUDE,
    longitude: float = LAGOS_LONGITUDE,
) -> pd.DataFrame:
    """Fetch hourly PVGIS irradiance and weather data for the requested dates.

    PVGIS accepts complete years, so the response is filtered to the requested
    date range after retrieval. Global irradiance is reconstructed from the
    returned beam, diffuse, and reflected components.
    """
    start, end = _validate_dates(start_date, end_date)
    params = urlencode(
        {
            "lat": latitude,
            "lon": longitude,
            "startyear": start.year,
            "endyear": end.year,
            "pvcalculation": 0,
            "components": 1,
            "outputformat": "json",
        }
    )
    payload = _get_json(f"{PVGIS_SERIES_URL}?{params}")
    rows = []
    for item in payload["outputs"]["hourly"]:
        timestamp = pd.to_datetime(item["time"], format="%Y%m%d:%H%M")
        if not start <= timestamp.date() <= end:
            continue
        beam = float(item.get("Gb(i)", 0.0))
        diffuse = float(item.get("Gd(i)", 0.0))
        reflected = float(item.get("Gr(i)", 0.0))
        rows.append(
            {
                "timestamp": timestamp,
                "ghi_wm2": beam + diffuse + reflected,
                "beam_wm2": beam,
                "diffuse_wm2": diffuse,
                "reflected_wm2": reflected,
                "temperature_c": float(item["T2m"]),
                "wind_speed_ms": float(item["WS10m"]),
                "source": "PVGIS",
            }
        )
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def fetch_lagos_irradiance(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch normalized hourly records from both services for Lagos.

    The returned long-form dataframe keeps a ``source`` column because NASA
    POWER and PVGIS use different timestamps and data products.
    """
    nasa = fetch_nasa_power_hourly(start_date, end_date)
    pvgis = fetch_pvgis_hourly(start_date, end_date)
    return pd.concat([nasa, pvgis], ignore_index=True, sort=False).sort_values(
        ["timestamp", "source"]
    ).reset_index(drop=True)
