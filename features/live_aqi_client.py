"""
api_client.py
=============
PURPOSE : Fetch the latest AQI reading from AQICN for a given city.
          Used by the feature pipeline on every hourly run.

DESIGN DECISIONS (senior notes for junior devs):
  - City is a parameter, not a global constant → function is reusable + testable
  - Retry logic with exponential back-off → survives transient network errors
  - All timestamps normalised to UTC ISO-8601 → merges with Open-Meteo cleanly
  - Structured logging (not print) → grep-able in production logs
  - No API key in source code → always via environment variable
  - Returns typed dict via TypedDict → downstream code knows what fields exist

INSTALL:
  pip install requests python-dotenv

USAGE:
  from features.api_client import fetch_aqi
  row = fetch_aqi("lahore")
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import requests
from dotenv import load_dotenv

# ── Logging ───────────────────────────────────────────────────────────────────
# Use module-level logger, not print().
# Callers (pipeline scripts) configure the root logger with level + handler.
# This module just emits — it never decides where logs go.
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

_AQICN_TOKEN: str = os.environ["AQICN_TOKEN"]   # KeyError = fail fast at startup
_BASE_URL = "https://api.waqi.info/feed"

# Retry settings
_MAX_RETRIES = 3
_BACKOFF_BASE = 2   # seconds; waits 2s, 4s, 8s between attempts


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_iaqi(iaqi: dict, field: str) -> Optional[float]:
    """
    Extract a numeric value from AQICN's iaqi sub-dict.

    AQICN returns:  "iaqi": { "pm25": {"v": 145.0}, "no2": {"v": "-"} }
    We want:        145.0  or  None

    WHY Optional[float] and not int?
    AQI sub-indices are floats in the API (e.g. 145.3).
    Using float avoids silent precision loss.
    """
    raw = iaqi.get(field, {}).get("v")
    if raw is None or raw == "-":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("Unexpected iaqi value for %s: %r", field, raw)
        return None


def _parse_station_time(time_str: str) -> str:
    """
    Convert AQICN station timestamp → UTC ISO-8601 string.

    AQICN returns local-time strings like "2024-11-15 14:00:00"
    with NO timezone offset, and the station is in Lahore (PKT = UTC+5).

    If we store this as-is and later merge with Open-Meteo (which we stored
    in UTC), every join will be silently off by 5 hours — a data-quality
    disaster that's hard to debug.

    Fix: assume PKT, convert to UTC.
    """
    from datetime import timedelta
    PKT_OFFSET = timedelta(hours=5)

    try:
        local_dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        utc_dt = local_dt - PKT_OFFSET
        return utc_dt.replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        # If format ever changes, return the raw string and log a warning
        logger.warning("Could not parse station time %r — storing raw", time_str)
        return time_str


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_aqi(city: str) -> dict:
    """
    Fetch the latest AQI data for *city* from AQICN.

    Parameters
    ----------
    city : str
        AQICN city slug, e.g. "lahore", "karachi", "islamabad".

    Returns
    -------
    dict with keys:
        timestamp_utc, fetch_utc, city, aqi, pm25, pm10, no2,
        co, o3, so2, humidity, temp, wind, pressure, dominant_pollutant

    Raises
    ------
    requests.HTTPError   — on 4xx/5xx responses after all retries
    RuntimeError         — if AQICN returns status != "ok"
    """
    url = f"{_BASE_URL}/{city}/"
    params = {"token": _AQICN_TOKEN}

    # ── Retry loop with exponential back-off ──────────────────────────────────
    last_exc: Optional[Exception] = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.debug("Fetching AQI for %s (attempt %d/%d)", city, attempt, _MAX_RETRIES)
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            break                          # success — exit retry loop
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            wait = _BACKOFF_BASE ** attempt
            logger.warning("Request failed (attempt %d): %s. Retrying in %ds…", attempt, exc, wait)
            time.sleep(wait)
    else:
        # All retries exhausted
        raise RuntimeError(f"All {_MAX_RETRIES} attempts failed for {city}") from last_exc

    payload = response.json()

    if payload.get("status") != "ok":
        raise RuntimeError(f"AQICN API error for {city!r}: {payload}")

    station = payload["data"]
    iaqi    = station.get("iaqi", {})

    # AQI can be an int, float, or the string "-" when the station has no data
    raw_aqi = station.get("aqi")
    aqi_value: Optional[float] = float(raw_aqi) if isinstance(raw_aqi, (int, float)) else None

    result = {
        # Always store time in UTC — avoids timezone bugs in every downstream step
        "timestamp_utc":      _parse_station_time(station["time"]["s"]),
        "fetch_utc":          datetime.now(timezone.utc).isoformat(),
        "city":               city,
        "aqi":                aqi_value,
        "pm25":               _safe_iaqi(iaqi, "pm25"),
        "pm10":               _safe_iaqi(iaqi, "pm10"),
        "no2":                _safe_iaqi(iaqi, "no2"),
        "co":                 _safe_iaqi(iaqi, "co"),
        "o3":                 _safe_iaqi(iaqi, "o3"),
        "so2":                _safe_iaqi(iaqi, "so2"),
        "humidity":           _safe_iaqi(iaqi, "h"),
        "temp":               _safe_iaqi(iaqi, "t"),
        "wind":               _safe_iaqi(iaqi, "w"),
        "pressure":           _safe_iaqi(iaqi, "p"),
        "dominant_pollutant": station.get("dominentpol"),  # note: AQICN typo preserved
    }

    logger.info(
        "Fetched AQI for %s: aqi=%s, pm25=%s (at %s)",
        city, result["aqi"], result["pm25"], result["timestamp_utc"]
    )
    return result


# ── Manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # When running directly, configure basic logging so you can see output
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    data = fetch_aqi("lahore")
    print("\n✅ Live Lahore AQI Data:")
    for key, value in data.items():
        print(f"  {key:25s}: {value}")