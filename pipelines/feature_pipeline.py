"""
feature_pipeline.py
===================
PURPOSE : Takes lahore_historical.csv, engineers ML-ready features,
          and uploads to Hopsworks Feature Store.

WHAT THIS ADDS ON TOP OF BACKFILL:
  - Lag features      : AQI 1h, 3h, 6h, 24h, 48h ago (trend awareness)
  - Rolling features  : mean/std/max over past windows (momentum)
  - Change features   : how fast is AQI rising or falling?
  - Target variables  : future AQI 24h, 48h, 72h ahead (what model predicts)

SENIOR NOTES — KEY DESIGN DECISIONS:
  1. No training/serving skew: change features only use lagged values,
     never the current row's AQI (which you won't have at inference time).
  2. Division guard: percentage-change clips denominator at ±1 to avoid inf.
  3. API key via env var only — never hardcoded in source.
  4. Single-responsibility functions — each does exactly one job.
  5. All imports at file top (PEP 8).

HOW TO RUN:
  pip install pandas numpy hopsworks python-dotenv
  export HOPSWORKS_API_KEY=your_key_here
  python feature_pipeline.py
"""

import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "lahore_historical.csv"
OUTPUT_FILE = "lahore_features.csv"     # local backup before Hopsworks upload

# Lag horizons — hours to look back at for AQI history
AQI_LAG_HOURS     = [1, 3, 6, 24, 48]
WEATHER_LAG_HOURS = [1, 3, 6]

# Rolling windows — hours to average/std over for momentum features
ROLLING_MEAN_WINDOWS = [3, 6, 24]
ROLLING_STD_WINDOWS  = [6, 24]

# Forecast horizons — hours ahead we want to predict
TARGET_HOURS = [24, 48, 72]


# ── Step 1: Load ──────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded %d rows from %s", len(df), path)
    logger.info("Date range: %s → %s", df["timestamp"].min(), df["timestamp"].max())
    return df


# ── Step 2: Lag Features ──────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY lags?
    A model seeing only AQI=280 has no sense of direction.
    If we also tell it "AQI was 210 one hour ago", it knows AQI is rising fast.

    WHY 24h specifically?
    Lahore has strong daily cycles (rush hour patterns, industrial shifts).
    AQI at 8am today correlates strongly with AQI at 8am yesterday —
    the 24h lag is often the single most powerful predictor.

    IMPORTANT: We use .shift(N) on *past* values only.
    These features are safe to compute at inference time because
    you will always have the last 48 hours of actual observations.
    """
    for hours in AQI_LAG_HOURS:
        df[f"aqi_lag_{hours}h"] = df["us_aqi"].shift(hours)

    for hours in WEATHER_LAG_HOURS:
        df[f"wind_lag_{hours}h"] = df["wind_speed_10m"].shift(hours)
        df[f"temp_lag_{hours}h"] = df["temperature_2m"].shift(hours)

    logger.info("Added %d lag features", len(AQI_LAG_HOURS) + 2 * len(WEATHER_LAG_HOURS))
    return df


# ── Step 3: Rolling Features ──────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY rolling windows?
    A single lag is noisy — one unusual hour skews it.
    Rolling average smooths noise and captures sustained trend direction.

    WHY .shift(1) before rolling?
    Without shift(1), the rolling mean for row t includes row t itself —
    this is data leakage. The model would be trained using information
    it cannot access at inference time (the current AQI value).
    .shift(1) ensures the window only looks at *past* values.
    """
    lagged = df["us_aqi"].shift(1)   # past values only — no leakage

    for window in ROLLING_MEAN_WINDOWS:
        df[f"aqi_rolling_mean_{window}h"] = (
            lagged.rolling(window=window, min_periods=1).mean().round(2)
        )

    for window in ROLLING_STD_WINDOWS:
        df[f"aqi_rolling_std_{window}h"] = (
            lagged.rolling(window=window, min_periods=1).std().round(2)
        )

    # Rolling max — worst AQI in the last 24h (captures pollution events)
    df["aqi_rolling_max_24h"] = lagged.rolling(window=24, min_periods=1).max()

    logger.info(
        "Added %d rolling features",
        len(ROLLING_MEAN_WINDOWS) + len(ROLLING_STD_WINDOWS) + 1
    )
    return df


# ── Step 4: Change Rate Features ─────────────────────────────────────────────

def add_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY change rate?
    AQI=200 that has been stable all day is very different from
    AQI=200 that jumped from 80 in 3 hours (pollution event unfolding).
    Change rate gives the model that context.

    CRITICAL — NO TRAINING/SERVING SKEW:
    All change features use df["us_aqi"].shift(1) as the "current" value,
    NOT df["us_aqi"] directly.

    Why? At inference time (predicting future AQI), your most recent
    observation IS the lagged value — you don't yet have the current hour.
    If you train with df["us_aqi"] but predict with df["aqi_lag_1h"],
    the feature distribution shifts and model accuracy drops silently.

    DIVISION GUARD:
    aqi_pct_change clips the denominator so it is never smaller than ±1.
    This prevents inf values when AQI is near zero (rare but possible
    with clean-air events or API data gaps).
    """
    prev_aqi = df["us_aqi"].shift(1)    # "current" observation at inference time

    # Raw 1-hour change
    df["aqi_change_1h"] = prev_aqi - df["us_aqi"].shift(2)

    # Percentage change — guarded against division by near-zero
    denom = df["us_aqi"].shift(2).clip(lower=1)   # never divide by < 1
    df["aqi_pct_change_1h"] = ((prev_aqi - df["us_aqi"].shift(2)) / denom * 100).round(2)

    # 3-hour trend direction: +1 rising, -1 falling, 0 flat
    df["aqi_trend_3h"] = np.sign(prev_aqi - df["us_aqi"].shift(4))

    # Wind change — sudden wind typically improves air quality
    df["wind_change_3h"] = df["wind_speed_10m"].shift(1) - df["wind_speed_10m"].shift(4)

    logger.info("Added 4 change rate features (no training/serving skew)")
    return df


# ── Step 5: Target Variables ──────────────────────────────────────────────────

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY shift forward?
    To predict AQI 24h from now, the target at row t = actual AQI at row t+24.
    The last 72 rows will have NaN targets — that is correct and expected.
    Those rows are used for live inference (predicting the future), not training.

    We create three targets so you can train three separate models,
    one for each forecast horizon, as the project requires.
    """
    for hours in TARGET_HOURS:
        df[f"target_aqi_{hours}h"] = df["us_aqi"].shift(-hours)

    logger.info("Added %d target columns (%s)", len(TARGET_HOURS),
                [f"target_aqi_{h}h" for h in TARGET_HOURS])
    return df


# ── Step 6: Clean Up NaN rows ────────────────────────────────────────────────

def drop_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the first N rows where lag features are NaN (the 'warm-up' period).
    Do NOT drop rows where only target columns are NaN — those are used for
    live inference.

    WHY keep NaN-target rows?
    They represent the most recent hours. Your inference pipeline will
    read these rows and fill in predictions once the model runs.
    """
    lag_and_rolling_cols = [c for c in df.columns if "lag_" in c or "rolling_" in c]
    before = len(df)
    df = df.dropna(subset=lag_and_rolling_cols)
    dropped = before - len(df)
    logger.info("Dropped %d warm-up rows with NaN lag/rolling features", dropped)
    logger.info("Dataset after clean: %d rows × %d columns", df.shape[0], df.shape[1])
    return df


# ── Step 7: Save Locally ──────────────────────────────────────────────────────

def save_local(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    engineered_cols = [
        c for c in df.columns
        if any(x in c for x in ["lag_", "rolling_", "change_", "trend_", "target_"])
    ]
    logger.info("Saved to '%s'", path)
    logger.info("Engineered columns (%d): %s", len(engineered_cols), engineered_cols)


# ── Step 8: Upload to Hopsworks ───────────────────────────────────────────────

def upload_to_hopsworks(df: pd.DataFrame) -> None:
    """
    Upload the feature DataFrame to Hopsworks Feature Store.

    WHY Hopsworks instead of just a CSV?
      1. Versioning  — every upload is a new version; old versions are preserved
      2. Sharing     — training pipeline and web app both read the same source
      3. Consistency — no risk of one pipeline using a stale local CSV
      4. Free tier   — no credit card, 10 GB storage, plenty for this project

    HOW TO GET YOUR API KEY (free, no credit card):
      1. Go to https://app.hopsworks.ai and register with your email
      2. Create a project (e.g. "aqi-predictor-lahore")
      3. Account Settings → API Keys → Create key
      4. Add to your .env file: HOPSWORKS_API_KEY=your_key_here

    WHAT IS A FEATURE GROUP?
    A versioned table in Hopsworks. Primary key = (city, timestamp) so each
    city+hour combination is uniquely identified. insert() upserts — it updates
    existing rows and adds new ones, so re-running is safe.
    """
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        logger.warning(
            "HOPSWORKS_API_KEY not set — skipping upload. "
            "Set it in your .env file. Get a free key at https://app.hopsworks.ai"
        )
        return

    try:
        import hopsworks  # imported here so the script runs without hopsworks installed

        logger.info("Connecting to Hopsworks…")
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()

        fg = fs.get_or_create_feature_group(
            name="lahore_aqi_features",
            version=1,
            primary_key=["city", "timestamp"],
            description="Hourly AQI features for Lahore: lag, rolling, change, targets",
            event_time="timestamp",
        )

        fg.insert(df, write_options={"wait_for_job": False})
        logger.info("Uploaded %d rows to Hopsworks feature group 'lahore_aqi_features'", len(df))

    except ImportError:
        logger.error("hopsworks package not installed. Run: pip install hopsworks")
    except Exception as exc:
        logger.error("Hopsworks upload failed: %s", exc)
        raise


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 55)
    logger.info("  FEATURE PIPELINE — Lahore AQI Predictor")
    logger.info("=" * 55)

    df = load_data(INPUT_FILE)

    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_change_features(df)
    df = add_targets(df)
    df = drop_warmup_rows(df)

    save_local(df, OUTPUT_FILE)
    upload_to_hopsworks(df)

    # Final summary
    training_rows   = df["target_aqi_24h"].notna().sum()
    inference_rows  = df["target_aqi_24h"].isna().sum()
    logger.info("=" * 55)
    logger.info("  FEATURE SUMMARY")
    logger.info("=" * 55)
    logger.info("Total features   : %d", df.shape[1])
    logger.info("Training rows    : %d  (have 24h target)", training_rows)
    logger.info("Inference rows   : %d  (most recent — no future yet)", inference_rows)
    logger.info("Next step: run training_pipeline.py")


if __name__ == "__main__":
    main()