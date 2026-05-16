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

FIXES APPLIED (vs original):
  FIX 1 — Timezone-aware timestamps:
    load_data() now localises timestamp to UTC after parsing.
    Hopsworks requires tz-aware datetimes when event_time is set.
    Without this, fg.insert() silently fails or throws a cryptic error.

  FIX 2 — wait_for_job: True:
    Changed from False → True so the script actually waits for the
    Hopsworks Spark job to finish and surfaces any job-level errors.
    With False, the insert appeared to succeed even when the job crashed.

  FIX 3 — Column name sanitisation:
    Hopsworks rejects column names with spaces or special chars beyond _.
    sanitise_column_names() renames any offending columns before upload.

  FIX 4 — Primary key NaN guard:
    Added assertion before insert to catch NaN city/timestamp values
    that would cause a silent partial upload.

  FIX 5 — Schema print before insert:
    Prints dtypes + head so you can visually confirm what Hopsworks sees.

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
OUTPUT_FILE = "lahore_features.csv"

AQI_LAG_HOURS     = [1, 3, 6, 24, 48]
WEATHER_LAG_HOURS = [1, 3, 6]

ROLLING_MEAN_WINDOWS = [3, 6, 24]
ROLLING_STD_WINDOWS  = [6, 24]

TARGET_HOURS = [24, 48, 72]


# ── Step 1: Load ──────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV and ensure timestamp is timezone-aware UTC.

    FIX 1: Hopsworks event_time columns must be tz-aware.
    Previously the timestamp was parsed as naive (no tzinfo), which caused
    Hopsworks to either reject the insert or store garbage time values.

    We localise to UTC here, at the source, so every downstream step
    (feature engineering, upload) works on the same reference frame.
    """
    df = pd.read_csv(path)

    # Parse → localise to UTC  ← FIX 1
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"])
          .dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
    )

    # Drop any rows where timestamp could not be parsed
    bad_ts = df["timestamp"].isna().sum()
    if bad_ts:
        logger.warning("Dropping %d rows with unparseable timestamps", bad_ts)
        df = df.dropna(subset=["timestamp"])

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
    lagged = df["us_aqi"].shift(1)

    for window in ROLLING_MEAN_WINDOWS:
        df[f"aqi_rolling_mean_{window}h"] = (
            lagged.rolling(window=window, min_periods=1).mean().round(2)
        )

    for window in ROLLING_STD_WINDOWS:
        df[f"aqi_rolling_std_{window}h"] = (
            lagged.rolling(window=window, min_periods=1).std().round(2)
        )

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
    """
    prev_aqi = df["us_aqi"].shift(1)

    df["aqi_change_1h"]     = prev_aqi - df["us_aqi"].shift(2)
    denom                   = df["us_aqi"].shift(2).clip(lower=1)
    df["aqi_pct_change_1h"] = ((prev_aqi - df["us_aqi"].shift(2)) / denom * 100).round(2)
    df["aqi_trend_3h"]      = np.sign(prev_aqi - df["us_aqi"].shift(4))
    df["wind_change_3h"]    = df["wind_speed_10m"].shift(1) - df["wind_speed_10m"].shift(4)

    logger.info("Added 4 change rate features (no training/serving skew)")
    return df


# ── Step 5: Target Variables ──────────────────────────────────────────────────

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY shift forward?
    To predict AQI 24h from now, the target at row t = actual AQI at row t+24.
    The last 72 rows will have NaN targets — correct and expected.
    Those rows are used for live inference, not training.
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
    """
    lag_and_rolling_cols = [c for c in df.columns if "lag_" in c or "rolling_" in c]
    before = len(df)
    df = df.dropna(subset=lag_and_rolling_cols)
    dropped = before - len(df)
    logger.info("Dropped %d warm-up rows with NaN lag/rolling features", dropped)
    logger.info("Dataset after clean: %d rows × %d columns", df.shape[0], df.shape[1])
    return df


# ── Step 7: Sanitise Column Names ─────────────────────────────────────────────

def sanitise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX 3: Hopsworks rejects column names that contain characters other than
    lowercase letters, digits, and underscores.

    This function lowercases all names and replaces any illegal character
    with an underscore. Running it is safe even if all names are already clean.

    Common offenders in Open-Meteo data: 'pm2.5' → 'pm2_5' (dot),
    spaces, hyphens.
    """
    import re
    old_cols = df.columns.tolist()
    new_cols = [re.sub(r"[^a-z0-9_]", "_", c.lower()) for c in old_cols]

    renamed = {o: n for o, n in zip(old_cols, new_cols) if o != n}
    if renamed:
        logger.info("Sanitised %d column name(s): %s", len(renamed), renamed)
        df = df.rename(columns=renamed)
    else:
        logger.info("All column names already clean — no renaming needed")

    return df


# ── Step 8: Save Locally ──────────────────────────────────────────────────────

def save_local(df: pd.DataFrame, path: str) -> None:
    # Strip tzinfo for CSV — it serialises as +00:00 which confuses some readers
    df_csv = df.copy()
    df_csv["timestamp"] = df_csv["timestamp"].dt.tz_localize(None)
    df_csv.to_csv(path, index=False)

    engineered_cols = [
        c for c in df.columns
        if any(x in c for x in ["lag_", "rolling_", "change_", "trend_", "target_"])
    ]
    logger.info("Saved to '%s'", path)
    logger.info("Engineered columns (%d): %s", len(engineered_cols), engineered_cols)


# ── Step 9: Upload to Hopsworks ───────────────────────────────────────────────

def upload_to_hopsworks(df: pd.DataFrame) -> None:
    """
    Upload the feature DataFrame to Hopsworks Feature Store.

    FIXES APPLIED:
      FIX 2 — wait_for_job: True  → script waits for the Spark job to finish
               and raises immediately if it fails. With False the job could
               crash silently and you'd never know.

      FIX 4 — Primary key NaN guard: asserts city + timestamp have no NaNs
               before calling insert(). A NaN primary key causes a partial
               or silent failed upload with no clear error message.

      FIX 5 — Schema print: logs dtypes and first 3 rows so you can visually
               confirm the DataFrame Hopsworks is about to receive.
    """
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        logger.warning(
            "HOPSWORKS_API_KEY not set — skipping upload. "
            "Set it in your .env file. Get a free key at https://app.hopsworks.ai"
        )
        return

    # ── Block 1: import check only ────────────────────────────────────────────
    try:
        import hopsworks
    except ImportError:
        logger.error("hopsworks package not installed. Run: pip install hopsworks")
        return

    # ── FIX 4: Primary key NaN guard ─────────────────────────────────────────
    assert df["city"].notna().all(), (
        "city column has NaN values — cannot use as primary key. "
        "Check that 'city' was set before calling upload_to_hopsworks()."
    )
    assert df["timestamp"].notna().all(), (
        "timestamp column has NaN values — cannot use as primary key. "
        "Check load_data() for parsing failures."
    )

    # ── FIX 5: Schema print so you can see exactly what Hopsworks receives ────
    logger.info("─── DataFrame schema being sent to Hopsworks ───")
    logger.info("Shape  : %d rows × %d columns", df.shape[0], df.shape[1])
    logger.info("Dtypes :\n%s", df.dtypes.to_string())
    logger.info("Head   :\n%s", df[["city", "timestamp"]].head(3).to_string())
    logger.info("────────────────────────────────────────────────")

    # ── Block 2: upload logic only ────────────────────────────────────────────
    try:
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

        # FIX 2: wait_for_job True — surfaces job-level failures immediately
        fg.insert(df, write_options={"wait_for_job": True})
        logger.info(
            "✅ Successfully uploaded %d rows to Hopsworks feature group "
            "'lahore_aqi_features'", len(df)
        )

    except Exception as exc:
        logger.error("Hopsworks upload failed: %s", exc, exc_info=True)
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

    # FIX 3: sanitise column names before saving or uploading
    df = sanitise_column_names(df)

    save_local(df, OUTPUT_FILE)
    upload_to_hopsworks(df)

    # Final summary
    training_rows  = df["target_aqi_24h"].notna().sum()
    inference_rows = df["target_aqi_24h"].isna().sum()
    logger.info("=" * 55)
    logger.info("  FEATURE SUMMARY")
    logger.info("=" * 55)
    logger.info("Total features   : %d", df.shape[1])
    logger.info("Training rows    : %d  (have 24h target)", training_rows)
    logger.info("Inference rows   : %d  (most recent — no future yet)", inference_rows)
    logger.info("Next step: run training_pipeline.py")


if __name__ == "__main__":
    main()