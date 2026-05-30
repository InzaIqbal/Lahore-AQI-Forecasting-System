"""
live_aqi_client.py
==================
UPDATED: Open-Meteo with PM2.5-based dust correction.

During dust storms, Open-Meteo's atmospheric model inflates AQI
because it counts dust at all altitude levels, not just ground level.
This version detects dust spikes and recalculates AQI from PM2.5
concentration using EPA breakpoints — giving ground-level conditions.
"""

import logging
import os
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

LAHORE_LAT = 31.5497
LAHORE_LON = 74.3436
LAHORE_TZ  = "Asia/Karachi"

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
AQ_HOURLY_VARS = [
    "us_aqi", "pm2_5", "pm10",
    "nitrogen_dioxide", "sulphur_dioxide",
    "carbon_monoxide", "ozone", "dust",
]

# Dust threshold above which we apply PM2.5-based correction (µg/m³)
DUST_SPIKE_THRESHOLD = 150


# ── PM2.5 → US AQI converter (EPA breakpoints) ───────────────────────────────

def pm25_to_us_aqi(pm25: float) -> int:
    """
    Convert PM2.5 concentration (µg/m³) to US AQI using EPA breakpoints.
    This is the standard formula used by AirNow and IQAir.

    Breakpoints source: EPA AQI Technical Assistance Document (2018)
    """
    # (pm25_low, pm25_high, aqi_low, aqi_high)
    breakpoints = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4, 101,  150),
        (55.5, 150.4, 151,  200),
        (150.5, 250.4, 201,  300),
        (250.5, 350.4, 301,  400),
        (350.5, 500.4, 401,  500),
    ]

    pm25 = round(pm25, 1)   # EPA truncates to 1 decimal place

    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
            return round(aqi)

    # Above 500.4 µg/m³ — beyond index
    return 500


# ── Open-Meteo fetch ──────────────────────────────────────────────────────────

def fetch_live_aqi() -> dict:
    """
    Fetch current hour AQI for Lahore from Open-Meteo.

    Dust correction:
      When dust > 150 µg/m³, Open-Meteo's us_aqi is dominated by
      atmospheric dust (not ground-level pollution). In that case we
      recalculate AQI from PM2.5 using EPA breakpoints, which matches
      what ground station networks like IQAir report.
    """
    logger.info("Fetching hourly air quality array from Open-Meteo...")

    resp = requests.get(
        AQ_URL,
        params={
            "latitude":      LAHORE_LAT,
            "longitude":     LAHORE_LON,
            "hourly":        AQ_HOURLY_VARS,
            "timezone":      LAHORE_TZ,
            "past_days":     1,
            "forecast_days": 1,
        },
        timeout=20,
    )
    resp.raise_for_status()
    data   = resp.json()
    hourly = data["hourly"]
    times  = pd.to_datetime(hourly["time"])

    # Find index for current local hour
    now_local = (
        pd.Timestamp.now(tz=LAHORE_TZ)
        .floor("h")
        .tz_localize(None)
    )

    mask = (times == now_local)
    if mask.any():
        idx = int(mask.argmax())
        logger.info("Exact index match at position %d → %s", idx, times[idx])
    else:
        idx = int((times - now_local).abs().argmin())
        logger.warning(
            "No exact match for %s — using nearest hour at index %d → %s",
            now_local, idx, times[idx],
        )

    def _get(field):
        arr = hourly.get(field, [])
        return arr[idx] if idx < len(arr) else None

    raw_aqi  = _get("us_aqi")
    pm25_val = _get("pm2_5")
    dust_val = _get("dust") or 0.0

    # ── Dust correction ───────────────────────────────────────────────────────
    source = "open-meteo-hourly"
    final_aqi = raw_aqi

    if dust_val > DUST_SPIKE_THRESHOLD and pm25_val is not None:
        corrected = pm25_to_us_aqi(pm25_val)
        logger.info(
            "Dust spike detected (dust=%.0f µg/m³ > threshold=%d). "
            "Raw AQI=%s → PM2.5-based AQI=%d (PM2.5=%.1f µg/m³)",
            dust_val, DUST_SPIKE_THRESHOLD, raw_aqi, corrected, pm25_val,
        )
        final_aqi = corrected
        source    = "open-meteo-dust-corrected"
    # ─────────────────────────────────────────────────────────────────────────

    result = {
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
        "data_hour_local": str(times[idx]),
        "aqi":             final_aqi,
        "aqi_raw":         raw_aqi,       # keep raw for debugging
        "dust":            dust_val,
        "pm25":            pm25_val,
        "pm10":            _get("pm10"),
        "no2":             _get("nitrogen_dioxide"),
        "so2":             _get("sulphur_dioxide"),
        "co":              _get("carbon_monoxide"),
        "o3":              _get("ozone"),
        "source":          source,
        "stations_used":   1,
    }

    logger.info(
        "Live AQI for Lahore at %s (PKT) → AQI: %s  PM2.5: %s µg/m³  "
        "Dust: %.0f µg/m³  Source: %s",
        times[idx], final_aqi, pm25_val, dust_val, source,
    )

    return result


# ── Hourly forecast (used by inference pipeline) ──────────────────────────────

def fetch_hourly_forecast(past_days: int = 1, forecast_days: int = 4) -> pd.DataFrame:
    """
    Fetch multi-day hourly AQI + pollutant DataFrame for Lahore.
    Used by feature_pipeline and inference_pipeline — not for live display.
    """
    logger.info(
        "Fetching hourly forecast (past_days=%d, forecast_days=%d)...",
        past_days, forecast_days,
    )
    resp = requests.get(
        AQ_URL,
        params={
            "latitude":      LAHORE_LAT,
            "longitude":     LAHORE_LON,
            "hourly":        AQ_HOURLY_VARS,
            "timezone":      LAHORE_TZ,
            "past_days":     past_days,
            "forecast_days": forecast_days,
        },
        timeout=20,
    )
    resp.raise_for_status()
    data   = resp.json()
    hourly = data["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "us_aqi":    hourly["us_aqi"],
        "pm25":      hourly["pm2_5"],
        "pm10":      hourly["pm10"],
        "no2":       hourly["nitrogen_dioxide"],
        "so2":       hourly["sulphur_dioxide"],
        "co":        hourly["carbon_monoxide"],
        "o3":        hourly["ozone"],
        "dust":      hourly["dust"],
    })

    logger.info(
        "Fetched %d hourly rows (%s to %s)",
        len(df), df["timestamp"].min(), df["timestamp"].max(),
    )
    return df


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Live AQI (current hour) ──────────────────────────────")
    live = fetch_live_aqi()
    for k, v in live.items():
        print(f"  {k:<22} {v}")

    print("\n── Hourly forecast preview (next 5 rows from now) ───────")
    df = fetch_hourly_forecast(past_days=0, forecast_days=2)
    now_naive = pd.Timestamp.now(tz=LAHORE_TZ).floor("h").tz_localize(None)
    future = df[df["timestamp"] >= now_naive].head(5)
    print(future[["timestamp", "us_aqi", "pm25", "dust"]].to_string(index=False))