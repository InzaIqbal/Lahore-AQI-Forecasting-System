"""
live_aqi_client.py
==================
PURPOSE : Fetch LIVE AQI for Lahore by averaging multiple monitoring stations.

WHY AVERAGE MULTIPLE STATIONS?
  A single station can be down, malfunctioning, or locally unrepresentative.
  Averaging across stations gives a city-wide reading that is more robust
  and matches how official city AQI values are typically reported.

STATIONS USED (Lahore):
  - lahore              → main city station
  - lahore/us-consulate → US Embassy reference monitor (often most accurate)
  - lahore/gulberg      → residential area monitor

HOW TO RUN:
  pip install requests python-dotenv
  export AQICN_TOKEN=your_token_here
  python live_aqi_client.py
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import requests
from dotenv import load_dotenv

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

_AQICN_TOKEN: str = os.environ["AQICN_TOKEN"]
_BASE_URL = "https://api.waqi.info/feed"

# All known Lahore monitoring stations
LAHORE_STATIONS = [
    "lahore",
    "lahore/us-consulate",
    "lahore/gulberg",
]

_MAX_RETRIES = 3
_BACKOFF_BASE = 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_iaqi(iaqi: dict, field: str) -> Optional[float]:
    raw = iaqi.get(field, {}).get("v")
    if raw is None or raw == "-":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _fetch_single_station(city: str) -> Optional[dict]:
    """
    Fetch AQI data for one station slug. Returns None on any failure
    so the caller can skip it and use the other stations.
    """
    import time
    url = f"{_BASE_URL}/{city}/"
    params = {"token": _AQICN_TOKEN}

    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            wait = _BACKOFF_BASE ** attempt
            logger.warning("Station %s attempt %d failed: %s. Retrying in %ds…",
                           city, attempt, exc, wait)
            time.sleep(wait)
    else:
        logger.error("All retries failed for station %s: %s", city, last_exc)
        return None

    payload = response.json()
    if payload.get("status") != "ok":
        logger.warning("AQICN returned non-ok status for %s: %s", city, payload.get("status"))
        return None

    station = payload["data"]
    iaqi = station.get("iaqi", {})
    raw_aqi = station.get("aqi")
    aqi_value = float(raw_aqi) if isinstance(raw_aqi, (int, float)) else None

    if aqi_value is None:
        logger.warning("Station %s returned no numeric AQI, skipping.", city)
        return None

    return {
        "station":    city,
        "aqi":        aqi_value,
        "pm25":       _safe_iaqi(iaqi, "pm25"),
        "pm10":       _safe_iaqi(iaqi, "pm10"),
        "no2":        _safe_iaqi(iaqi, "no2"),
        "co":         _safe_iaqi(iaqi, "co"),
        "o3":         _safe_iaqi(iaqi, "o3"),
        "so2":        _safe_iaqi(iaqi, "so2"),
        "humidity":   _safe_iaqi(iaqi, "h"),
        "temp":       _safe_iaqi(iaqi, "t"),
        "wind":       _safe_iaqi(iaqi, "w"),
        "pressure":   _safe_iaqi(iaqi, "p"),
        "dominant_pollutant": station.get("dominentpol"),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_lahore_aqi_average(stations: list = None) -> dict:
    """
    Fetch AQI from multiple Lahore stations and return an averaged reading.

    Parameters
    ----------
    stations : list, optional
        List of AQICN station slugs. Defaults to LAHORE_STATIONS.

    Returns
    -------
    dict with keys:
        timestamp_utc, city, aqi, pm25, pm10, no2, co, o3, so2,
        humidity, temp, wind, pressure, dominant_pollutant,
        stations_used, stations_attempted

    Notes
    -----
    - Numeric fields are averaged across all responding stations.
    - Non-numeric fields (dominant_pollutant) use the value from the
      station with the highest AQI reading (most representative).
    - If ALL stations fail, raises RuntimeError.
    """
    if stations is None:
        stations = LAHORE_STATIONS

    logger.info("Fetching AQI from %d Lahore stations: %s", len(stations), stations)

    readings = []
    for station_slug in stations:
        result = _fetch_single_station(station_slug)
        if result is not None:
            readings.append(result)
            logger.info("  %-30s  AQI=%s", station_slug, result["aqi"])

    if not readings:
        raise RuntimeError(
            f"All {len(stations)} Lahore stations failed. "
            "Check your AQICN_TOKEN and internet connection."
        )

    # ── Average numeric fields across all responding stations ─────────────────
    numeric_fields = ["aqi", "pm25", "pm10", "no2", "co", "o3", "so2",
                      "humidity", "temp", "wind", "pressure"]

    averaged = {}
    for field in numeric_fields:
        values = [r[field] for r in readings if r[field] is not None]
        averaged[field] = round(sum(values) / len(values), 1) if values else None

    # dominant_pollutant → from the station reporting highest AQI
    best_station = max(readings, key=lambda r: r["aqi"])
    averaged["dominant_pollutant"] = best_station["dominant_pollutant"]

    result = {
        "timestamp_utc":      datetime.now(timezone.utc).isoformat(),
        "city":               "lahore",
        "stations_used":      len(readings),
        "stations_attempted": len(stations),
        "station_readings":   [{"station": r["station"], "aqi": r["aqi"]} for r in readings],
        **averaged,
    }

    logger.info(
        "Lahore averaged AQI: %.1f  (from %d/%d stations)",
        averaged["aqi"] or 0,
        len(readings),
        len(stations),
    )
    return result


# ── Manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = fetch_lahore_aqi_average()
    print("\n✅ Lahore City-Wide AQI (Multi-Station Average):")
    print(f"  Timestamp : {data['timestamp_utc']}")
    print(f"  Stations  : {data['stations_used']}/{data['stations_attempted']} responded")
    for s in data.get("station_readings", []):
        print(f"    {s['station']:<35} AQI = {s['aqi']}")
    print(f"\n  City Average AQI  : {data['aqi']}")
    print(f"  PM2.5             : {data['pm25']}")
    print(f"  PM10              : {data['pm10']}")
    print(f"  Dominant Pollutant: {data['dominant_pollutant']}")