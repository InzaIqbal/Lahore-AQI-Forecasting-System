"""
backfill_open_meteo.py
======================
PURPOSE : Fetch 2 years of hourly air quality + weather history for Lahore.
          Produces lahore_historical.csv — your training dataset.

WHY Open-Meteo?
  - Completely free, no API key, no credit card
  - Returns BOTH air quality AND weather variables
  - Historical data goes back years (unlike OpenWeather free tier)
  - No rate limits for non-commercial use

HOW TO RUN:
  pip install requests pandas
  python backfill_open_meteo.py

WHAT THIS PRODUCES:
  lahore_historical.csv  — clean, deduplicated, UTC-normalised training data

SENIOR NOTES:
  - All imports at top of file (PEP 8 §imports)
  - Deduplication on re-run — safe to run twice
  - Explicit UTC timestamps throughout
  - Functions do ONE thing each (single-responsibility principle)
  - Config block is the only place you change settings
"""

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
LAT         = 31.5497
LON         = 74.3436
CITY        = "lahore"
DAYS_BACK   = 730        # 2 years — gives model strong seasonal patterns
OUTPUT_FILE = "lahore_historical.csv"
TIMEZONE    = "Asia/Karachi"   # PKT — local time makes seasonal analysis easier

# Open-Meteo returns all timestamps in the requested timezone.
# We convert to UTC after parsing so every column is on the same reference frame.
# ─────────────────────────────────────────────────────────────────────────────


# ── Fetching ──────────────────────────────────────────────────────────────────

def fetch_air_quality_history(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """
    Fetch hourly air quality data from Open-Meteo (free, no API key).

    Variables chosen for Lahore:
      pm2_5, pm10         — the two dominant pollutants (winter smog)
      carbon_monoxide     — vehicle emissions proxy
      nitrogen_dioxide    — traffic + industrial activity
      ozone               — photochemical smog (spikes in summer)
      sulphur_dioxide     — power plant / industrial indicator
      us_aqi              — target variable (0-500 scale, more granular than 1-5)
      us_aqi_pm2_5        — PM2.5 sub-index (usually dominant for Lahore)
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "hourly":     ",".join([
            "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
            "sulphur_dioxide", "ozone", "us_aqi", "us_aqi_pm2_5",
        ]),
        "start_date": start_date,
        "end_date":   end_date,
        "timezone":   TIMEZONE,
    }
    logger.info("Fetching air quality: %s → %s", start_date, end_date)
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_weather_history(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """
    Fetch hourly weather data from Open-Meteo Historical Weather API.

    WHY weather matters for AQI prediction:
      - Temperature inversions in winter trap pollution near ground → AQI spikes
      - Rain washes PM2.5 out of the air → AQI drops sharply after rain
      - Wind speed/direction disperses or concentrates pollutants
      - Humidity affects how particles clump and settle
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "hourly":     ",".join([
            "temperature_2m",       # air temperature at 2 m height (°C)
            "relative_humidity_2m", # relative humidity (%)
            "wind_speed_10m",       # wind speed at 10 m (km/h)
            "wind_direction_10m",   # wind direction (degrees, 0=N, 90=E)
            "precipitation",        # rainfall (mm) — rain cleans air
            "surface_pressure",     # atmospheric pressure (hPa)
            "cloud_cover",          # cloud cover (%) — affects photochemical reactions
        ]),
        "start_date": start_date,
        "end_date":   end_date,
        "timezone":   TIMEZONE,
    }
    logger.info("Fetching weather history: %s → %s", start_date, end_date)
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_to_dataframe(api_response: dict) -> pd.DataFrame:
    """
    Convert Open-Meteo JSON → flat DataFrame with a UTC timestamp column.

    Open-Meteo returns:
      { "hourly": { "time": [...], "pm2_5": [...], ... } }

    We want one row per timestamp, all variables as columns, time in UTC.

    WHY convert to UTC?
    Open-Meteo returns timestamps in the requested local timezone (PKT here).
    The AQICN data we fetch in api_client.py is stored in UTC.
    Merging two DataFrames where one is PKT and one is UTC silently misaligns
    every row by 5 hours. UTC everywhere prevents this class of bug.
    """
    hourly = api_response["hourly"]
    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "timestamp"})

    # Parse as PKT then convert to UTC
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"])
          .dt.tz_localize("Asia/Karachi")   # tell pandas this is PKT
          .dt.tz_convert("UTC")             # convert to UTC
          .dt.tz_localize(None)             # drop tzinfo for cleaner CSV storage
    )
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based ML features from the UTC timestamp.

    WHY these features?
    Raw timestamps are not meaningful to most ML models. These derived
    features let the model learn patterns such as:
      - "AQI is always worse at 8am (rush hour)"
      - "AQI spikes Nov–Jan (winter smog season in Lahore)"
      - "AQI is lower on Sundays (less traffic)"

    WHY sin/cos encoding?
    Without it, the model treats hour=23 and hour=0 as far apart (distance=23)
    when they are actually adjacent (1 hour apart on a clock face).
    Mapping to a circle with sin/cos fixes this — a standard technique for
    any cyclical feature.
    """
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Monday, 6=Sunday
    df["month"]       = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)

    return df


# ── Quality Report ────────────────────────────────────────────────────────────

def print_quality_report(df: pd.DataFrame) -> None:
    """
    Print a data quality summary.

    Extracted into its own function so main() stays readable
    and so you can call this independently during EDA.
    """
    separator = "=" * 55

    logger.info(separator)
    logger.info("  DATA QUALITY REPORT")
    logger.info(separator)
    logger.info("Shape     : %d rows × %d columns", df.shape[0], df.shape[1])
    logger.info(
        "Date range: %s  →  %s",
        df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
    )

    null_pct = (df.isnull().mean() * 100).round(1)
    problems = null_pct[null_pct > 0]
    if problems.empty:
        logger.info("Missing values: none ✅")
    else:
        for col, pct in problems.items():
            level = "⚠️  PROBLEM" if pct > 10 else "note"
            logger.info("  [%s] %s: %.1f%% missing", level, col, pct)

    expected_range = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    gap_count = len(expected_range) - len(df)
    logger.info("Timeline gaps: %d missing hour(s) out of %d", gap_count, len(expected_range))

    logger.info("us_aqi statistics:\n%s", df["us_aqi"].describe().round(1).to_string())

    pollutant_cols = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
    available = [c for c in pollutant_cols if c in df.columns]
    logger.info("Pollutant statistics:\n%s", df[available].describe().round(2).to_string())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    end_date   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")

    logger.info("Starting backfill: %s → %s (~%d rows expected)", start_date, end_date, DAYS_BACK * 24)

    # Fetch
    try:
        aq_raw      = fetch_air_quality_history(LAT, LON, start_date, end_date)
        weather_raw = fetch_weather_history(LAT, LON, start_date, end_date)
    except requests.exceptions.HTTPError as exc:
        logger.error("HTTP error from Open-Meteo: %s", exc)
        raise
    except requests.exceptions.Timeout:
        logger.error("Request timed out. Check internet connection.")
        raise

    # Parse
    df_aq      = parse_to_dataframe(aq_raw)
    df_weather = parse_to_dataframe(weather_raw)
    logger.info("Parsed: %d air quality rows, %d weather rows", len(df_aq), len(df_weather))

    # Merge — inner join: only keep rows where BOTH datasets have data
    df = pd.merge(df_aq, df_weather, on="timestamp", how="inner")
    logger.info("After merge: %d rows", len(df))

    # Add features
    df = add_time_features(df)

    # Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Add city column
    df.insert(0, "city", CITY)

    # ── Deduplication ─────────────────────────────────────────────────────────
    # If you run this script twice (e.g. after a crash), you'd get duplicate
    # rows. Deduplicate on the natural key before saving.
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["city", "timestamp"])
    if len(df) < before_dedup:
        logger.warning("Dropped %d duplicate rows", before_dedup - len(df))

    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info("Saved %d rows to '%s'", len(df), OUTPUT_FILE)

    print_quality_report(df)
    logger.info("Done. Next step: run feature_pipeline.py")


if __name__ == "__main__":
    main()