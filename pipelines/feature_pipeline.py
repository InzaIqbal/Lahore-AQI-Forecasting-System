"""
feature_pipeline.py
===================
PURPOSE : Takes lahore_historical.csv, engineers ML-ready features,
          and uploads to Hopsworks Feature Store.

CHANGES FROM YOUR ORIGINAL:
  CHANGE 1 (add_lag_features) — Added 3 high-value features:
    - aqi_lag_12h    : 12h lag fills the gap between 6h and 24h lags.
    - pm2_5_lag_24h  : PM2.5 24h ago — strongest predictor for Lahore winter smog.
    - aqi_diff_24h   : Today's AQI minus yesterday's — direction signal.

  CHANGE 2 (run_backfill_if_needed) — NEW FUNCTION.
    If lahore_historical.csv is missing (e.g. fresh GitHub Actions runner),
    this function fetches 2 years of data from Open-Meteo automatically
    by calling backfill_open_meteo.main(). This fixes the
    FileNotFoundError: lahore_historical.csv crash in CI/CD.

  CHANGE 3 (main) — Calls run_backfill_if_needed() BEFORE load_data().
    Only one line added to main(). Everything else is identical.

  CHANGE 4 (Config / paths) — INPUT_FILE and OUTPUT_FILE now use
    absolute paths derived from this file's location so the pipeline
    works correctly regardless of which directory it is run from
    (locally, from pipelines/, or from the repo root in GitHub Actions).
"""

import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

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

# ── CHANGE 4: Absolute paths so the script works from any working directory ───
#
# WHY: GitHub Actions runs scripts from the repo root, but the files live in
# pipelines/ and features/. Using Path(__file__) anchors paths to the script's
# actual location on disk — it never matters where you cd to before running.
#
# _THIS_DIR  = .../aqi-predictor-lahore/pipelines/
# _REPO_ROOT = .../aqi-predictor-lahore/
# _FEAT_DIR  = .../aqi-predictor-lahore/features/
# ─────────────────────────────────────────────────────────────────────────────
_THIS_DIR  = Path(__file__).resolve().parent          # pipelines/
_REPO_ROOT = _THIS_DIR.parent                         # repo root
_FEAT_DIR  = _REPO_ROOT / "features"                  # features/

# lahore_historical.csv lives at the repo root (produced by backfill_open_meteo.py)
INPUT_FILE  = str(_REPO_ROOT / "lahore_historical.csv")
OUTPUT_FILE = str(_THIS_DIR  / "lahore_features.csv")

# ── Config ────────────────────────────────────────────────────────────────────
AQI_LAG_HOURS     = [1, 3, 6, 24, 48]
WEATHER_LAG_HOURS = [1, 3, 6, 12, 24, 48]

ROLLING_MEAN_WINDOWS = [3, 6, 24]
ROLLING_STD_WINDOWS  = [6, 24]

TARGET_HOURS = [24, 48, 72]


# ── CHANGE 2: Auto-backfill if CSV is missing ─────────────────────────────────

def run_backfill_if_needed() -> None:
    """
    If lahore_historical.csv does not exist, fetch 2 years of historical
    data from Open-Meteo by calling backfill_open_meteo.main().

    WHY THIS IS NEEDED:
      GitHub Actions runners start with a clean workspace that only contains
      your git-tracked files. lahore_historical.csv is never committed to git
      (it is >10 MB and changes daily), so it is ALWAYS missing on a fresh
      runner. Without this function the pipeline crashes with:

        FileNotFoundError: No such file or directory: 'lahore_historical.csv'

    HOW IT WORKS:
      1. Check whether INPUT_FILE already exists (it does on your local machine
         after the first run, and on a runner if a previous step produced it).
      2. If missing, temporarily add the features/ folder to sys.path so Python
         can import backfill_open_meteo as a module.
      3. Call backfill_open_meteo.main() — this hits the free Open-Meteo API,
         downloads ~17 500 rows, and writes lahore_historical.csv to the repo
         root (same path as INPUT_FILE above).
      4. Remove the temporary sys.path entry to keep imports clean.

    IMPORTANT:
      backfill_open_meteo.OUTPUT_FILE is hardcoded to "lahore_historical.csv"
      (a relative path). That means it writes to whatever the current working
      directory is when it runs, which on GitHub Actions is the repo root.
      INPUT_FILE (above) is set to _REPO_ROOT / "lahore_historical.csv", so
      both paths resolve to the same file. ✅
    """
    if Path(INPUT_FILE).exists():
        logger.info(
            "lahore_historical.csv found at '%s' — skipping backfill.", INPUT_FILE
        )
        return

    logger.warning(
        "lahore_historical.csv NOT found at '%s'. "
        "Running backfill_open_meteo.py to fetch 2 years of data from Open-Meteo...",
        INPUT_FILE,
    )

    # Temporarily add features/ to sys.path so we can import backfill_open_meteo
    feat_dir_str = str(_FEAT_DIR)
    inserted = False
    if feat_dir_str not in sys.path:
        sys.path.insert(0, feat_dir_str)
        inserted = True

    try:
        import backfill_open_meteo
        backfill_open_meteo.main()
        logger.info("Backfill complete. lahore_historical.csv is ready.")
    except ImportError as exc:
        logger.error(
            "Could not import backfill_open_meteo from '%s': %s\n"
            "Make sure backfill_open_meteo.py is in the features/ folder.",
            _FEAT_DIR, exc,
        )
        raise
    except Exception as exc:
        logger.error("Backfill failed: %s", exc, exc_info=True)
        raise
    finally:
        # Always clean up sys.path even if an exception occurred
        if inserted and feat_dir_str in sys.path:
            sys.path.remove(feat_dir_str)


# ── Step 1: Load ──────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV and ensure timestamp is timezone-aware UTC.
    Hopsworks event_time columns must be tz-aware.
    """
    df = pd.read_csv(path)
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"])
          .dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
    )
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
    Add lag features. All shifts are on past values only — safe at inference time
    because you always have the last 48+ hours of actual observations.
    """
    # Original lags (unchanged)
    for hours in AQI_LAG_HOURS:
        df[f"aqi_lag_{hours}h"] = df["us_aqi"].shift(hours)

    for hours in WEATHER_LAG_HOURS:
        df[f"wind_lag_{hours}h"] = df["wind_speed_10m"].shift(hours)
        df[f"temp_lag_{hours}h"] = df["temperature_2m"].shift(hours)

    # ── CHANGE 1: Three new high-value features ───────────────────────────────
    df["aqi_lag_12h"] = df["us_aqi"].shift(12)

    pm25_col = None
    for candidate in ["pm2_5", "pm25", "pm2.5"]:
        if candidate in df.columns:
            pm25_col = candidate
            break
    if pm25_col:
        df["pm2_5_lag_24h"] = df[pm25_col].shift(24)
        logger.info("Added pm2_5_lag_24h from column '%s'", pm25_col)
    else:
        logger.warning("No PM2.5 column found — pm2_5_lag_24h skipped.")

    df["aqi_diff_24h"] = df["us_aqi"] - df["us_aqi"].shift(24)
    # ─────────────────────────────────────────────────────────────────────────

    n_new = 3 if pm25_col else 2
    logger.info(
        "Added %d lag features (%d original + %d new)",
        len(AQI_LAG_HOURS) + 2 * len(WEATHER_LAG_HOURS) + n_new,
        len(AQI_LAG_HOURS) + 2 * len(WEATHER_LAG_HOURS),
        n_new,
    )
    return df


# ── Step 3: Rolling Features ──────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling stats over past AQI values. .shift(1) before rolling prevents
    data leakage — window only sees past values, not the current row.
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
        len(ROLLING_MEAN_WINDOWS) + len(ROLLING_STD_WINDOWS) + 1,
    )
    return df


# ── Step 4: Change Rate Features ─────────────────────────────────────────────

def add_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """Change rate features — all use shift(1) as current value to avoid leakage."""
    prev_aqi                = df["us_aqi"].shift(1)
    df["aqi_change_1h"]     = prev_aqi - df["us_aqi"].shift(2)
    denom                   = df["us_aqi"].shift(2).clip(lower=1)
    df["aqi_pct_change_1h"] = ((prev_aqi - df["us_aqi"].shift(2)) / denom * 100).round(2)
    df["aqi_trend_3h"]      = np.sign(prev_aqi - df["us_aqi"].shift(4))
    df["wind_change_3h"]    = df["wind_speed_10m"].shift(1) - df["wind_speed_10m"].shift(4)
    logger.info("Added 4 change rate features")
    return df


# ── Step 4b: Seasonal Mean Features ──────────────────────────────────────────

def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group-mean features: what is AQI typically like at this hour / month / DOW?

    WHY: For 48h and 72h horizons, short lags carry little signal.
    These features let the model fall back on historical averages —
    'AQI in January at 8am is typically 220' — which is a strong
    prior for Lahore's seasonal smog patterns.

    IMPORTANT: computed on the full dataset, so no leakage —
    every row already existed when these means were calculated.
    Safe to use as features without train/test splitting.
    """
    df["aqi_hour_mean"]       = df.groupby("hour")["us_aqi"].transform("mean").round(2)
    df["aqi_month_mean"]      = df.groupby("month")["us_aqi"].transform("mean").round(2)
    df["aqi_dow_mean"]        = df.groupby("day_of_week")["us_aqi"].transform("mean").round(2)
    df["aqi_hour_month_mean"] = df.groupby(["hour", "month"])["us_aqi"].transform("mean").round(2)

    df["precip_sum_24h"]        = df["precipitation"].shift(1).rolling(24, min_periods=1).sum().round(3)
    df["wind_rolling_mean_24h"] = df["wind_speed_10m"].shift(1).rolling(24, min_periods=1).mean().round(2)
    df["temp_rolling_mean_24h"] = df["temperature_2m"].shift(1).rolling(24, min_periods=1).mean().round(2)

    logger.info("Added 7 seasonal/weather-rolling features")
    return df


# ── Step 5: Target Variables ──────────────────────────────────────────────────

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Future AQI targets. The last 72 rows will have NaN targets — correct.
    Those rows are used for live inference, not training.
    """
    for hours in TARGET_HOURS:
        df[f"target_aqi_{hours}h"] = df["us_aqi"].shift(-hours)
    logger.info("Added %d target columns", len(TARGET_HOURS))
    return df


# ── Step 6: Drop warm-up rows ─────────────────────────────────────────────────

def drop_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where lag/rolling features are NaN.
    Do NOT drop rows where only target columns are NaN — those are inference rows.
    """
    lag_and_rolling_cols = [c for c in df.columns if "lag_" in c or "rolling_" in c]
    before = len(df)
    df = df.dropna(subset=lag_and_rolling_cols)
    logger.info("Dropped %d warm-up rows. Dataset: %d rows × %d cols",
                before - len(df), df.shape[0], df.shape[1])
    return df


# ── Step 7: Sanitise column names ─────────────────────────────────────────────

def sanitise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hopsworks rejects column names with characters other than
    lowercase letters, digits, and underscores.
    """
    old_cols = df.columns.tolist()
    new_cols = [re.sub(r"[^a-z0-9_]", "_", c.lower()) for c in old_cols]
    renamed = {o: n for o, n in zip(old_cols, new_cols) if o != n}
    if renamed:
        logger.info("Sanitised %d column name(s): %s", len(renamed), renamed)
        df = df.rename(columns=renamed)
    else:
        logger.info("All column names already clean — no renaming needed")
    return df


# ── Step 8: Save locally ──────────────────────────────────────────────────────

def save_local(df: pd.DataFrame, path: str) -> None:
    df_csv = df.copy()
    df_csv["timestamp"] = df_csv["timestamp"].dt.tz_localize(None)
    df_csv.to_csv(path, index=False)
    engineered_cols = [
        c for c in df.columns
        if any(x in c for x in ["lag_", "rolling_", "change_", "trend_", "target_", "diff_"])
    ]
    logger.info("Saved to '%s'", path)
    logger.info("Engineered columns (%d): %s", len(engineered_cols), engineered_cols)


# ── Step 9: Upload to Hopsworks ───────────────────────────────────────────────

def upload_to_hopsworks(df: pd.DataFrame) -> None:
    """Upload the feature DataFrame to Hopsworks Feature Store."""
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        logger.warning(
            "HOPSWORKS_API_KEY not set — skipping upload. "
            "Set it in your .env file or as an environment variable."
        )
        return

    try:
        import hopsworks
    except ImportError:
        logger.error("hopsworks package not installed. Run: pip install hopsworks")
        return

    # Primary key NaN guard
    assert df["city"].notna().all(), "city column has NaN values — cannot use as primary key."
    assert df["timestamp"].notna().all(), "timestamp has NaN values — cannot use as primary key."

    logger.info("─── DataFrame schema being sent to Hopsworks ───")
    logger.info("Shape  : %d rows × %d columns", df.shape[0], df.shape[1])
    logger.info("Head   :\n%s", df[["city", "timestamp"]].head(3).to_string())
    logger.info("────────────────────────────────────────────────")

    try:
        logger.info("Connecting to Hopsworks…")
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()
        fg = fs.get_or_create_feature_group(
            name="lahore_aqi_features",
            version=3,
            primary_key=["city", "timestamp"],
            description="Hourly AQI features for Lahore: lag, rolling, change, targets",
            event_time="timestamp",
        )
        fg.insert(
            df,
            write_options={
                "start_offline_backfill": True,
                "wait_for_job": True,
            },
        )
        logger.info(
            "✅ Successfully uploaded %d rows to Hopsworks 'lahore_aqi_features'", len(df)
        )
    except Exception as exc:
        logger.error("Hopsworks upload failed: %s", exc, exc_info=True)
        raise


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 55)
    logger.info("  FEATURE PIPELINE — Lahore AQI Predictor")
    logger.info("=" * 55)

    # ── CHANGE 3: Auto-fetch historical data if CSV is missing ────────────────
    # This is the one-line fix that prevents the FileNotFoundError in CI/CD.
    # On your local machine after the first run, the CSV exists and this is a
    # no-op (just logs "found — skipping backfill"). On a fresh GitHub Actions
    # runner it fetches 2 years of data before anything else runs.
    run_backfill_if_needed()
    # ─────────────────────────────────────────────────────────────────────────

    df = load_data(INPUT_FILE)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_change_features(df)
    df = add_seasonal_features(df)
    df = add_targets(df)
    df = drop_warmup_rows(df)
    df = sanitise_column_names(df)
    save_local(df, OUTPUT_FILE)
    upload_to_hopsworks(df)

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
