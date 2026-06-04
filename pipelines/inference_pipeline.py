"""
inference_pipeline.py
=====================
FIX APPLIED — Feature shape mismatch (expected 43 got 59)

ROOT CAUSE:
  The XGBoost model in Hopsworks was trained on 43 features (old schema).
  After adding new features, feature_cols.json grew to 59 columns.
  At inference time, the local 59-col JSON was loaded and passed to the
  43-feature model → crash.

WHAT CHANGED (4 places, all marked ── FIX ──):

  FIX 1 — load_model_from_hopsworks() now also reads feature_cols.json
           from the SAME download folder as the model. This is the exact
           file that was saved together with the model at training time,
           so it always has the right column count for that model.
           Returns 5 values: (model_type, model, scaler_X, scaler_y, artifact_cols)

  FIX 2 — load_model_from_local() does the same for locally saved models.
           Reads feature_cols.json from models/ alongside the model file.
           Also returns 5 values.

  FIX 3 — main() unpacks 5 values from both loaders and stores
           artifact_cols per horizon in models dict.

  FIX 4 — main() uses artifact_cols from the 24h model as feature_cols
           BEFORE falling back to get_feature_cols_from_model() or
           load_feature_cols(). This is the Option A fix — artifact cols
           are always in sync with the model because they were saved
           together at training time.

No other logic changed.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ── Path fixes ────────────────────────────────────────────────────────────────
_PIPELINES_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT  = _PIPELINES_DIR.parent
_MODELS_DIR    = _PIPELINES_DIR / "models"
_DATA_DIR      = _PIPELINES_DIR / "data"
_FEATURES_DIR  = _PROJECT_ROOT / "features"
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_FEATURES_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
USE_HOPSWORKS         = os.environ.get("USE_HOPSWORKS", "true").lower() == "true"
HOPSWORKS_API_KEY     = os.environ.get("HOPSWORKS_API_KEY", "")
FEATURE_GROUP_NAME    = "lahore_aqi_features"
FEATURE_GROUP_VERSION = 3
PRED_GROUP_NAME       = "lahore_aqi_predictions"
PRED_GROUP_VERSION    = 3

LOCAL_FEATURES_CSV    = str(_PIPELINES_DIR / "lahore_features.csv")
LOCAL_PREDICTIONS_CSV = str(_DATA_DIR / "predictions.csv")
FEATURE_COLS_JSON     = str(_MODELS_DIR / "feature_cols.json")

SEQ_LEN = 24

AQI_CATEGORIES = [
    (0,   50,  "Good",                    "🟢"),
    (51,  100, "Moderate",                "🟡"),
    (101, 150, "Unhealthy for Sensitive", "🟠"),
    (151, 200, "Unhealthy",               "🔴"),
    (201, 300, "Very Unhealthy",          "🟣"),
    (301, 500, "Hazardous",               "⚫"),
]


# ── Step 1: Load feature columns ──────────────────────────────────────────────

def load_feature_cols() -> list:
    """
    LAST RESORT fallback only — called when artifact_cols is unavailable.
    Priority:
      1. Local models/feature_cols.json
      2. Hopsworks model artifact download
      3. Derive from local CSV
    """
    # 1. Local file
    if Path(FEATURE_COLS_JSON).exists():
        with open(FEATURE_COLS_JSON) as f:
            cols = json.load(f)
        logger.info("Loaded %d feature columns from local %s", len(cols), FEATURE_COLS_JSON)
        return cols

    # 2. Download from Hopsworks model artifact
    if USE_HOPSWORKS and HOPSWORKS_API_KEY:
        try:
            import hopsworks
            logger.info("feature_cols.json not found locally — downloading from Hopsworks model artifact...")
            project  = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
            mr       = project.get_model_registry()
            hw_model = mr.get_model(name="lahore_aqi_best_24h", version=1)
            save_dir = hw_model.download()
            cols_path = os.path.join(save_dir, "feature_cols.json")
            if Path(cols_path).exists():
                with open(cols_path) as f:
                    cols = json.load(f)
                Path(FEATURE_COLS_JSON).parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy(cols_path, FEATURE_COLS_JSON)
                logger.info("Loaded %d feature columns from Hopsworks model artifact", len(cols))
                return cols
            else:
                logger.warning("feature_cols.json not found inside Hopsworks model artifact at %s", save_dir)
        except Exception as exc:
            logger.warning("Could not load feature_cols.json from Hopsworks: %s", exc)

    # 3. Derive from local CSV
    if Path(LOCAL_FEATURES_CSV).exists():
        df = pd.read_csv(LOCAL_FEATURES_CSV, nrows=1)
        exclude = {"timestamp", "city", "target_aqi_24h", "target_aqi_48h",
                   "target_aqi_72h", "us_aqi", "us_aqi_pm2_5"}
        cols = [c for c in df.columns if c not in exclude]
        logger.info("Derived %d feature columns from CSV (last resort)", len(cols))
        return cols

    raise FileNotFoundError(
        f"Cannot load feature columns:\n"
        f"  {FEATURE_COLS_JSON} — missing\n"
        f"  Hopsworks download failed\n"
        f"  {LOCAL_FEATURES_CSV} — missing\n"
        "Run training_pipeline.py first, or ensure Hopsworks credentials are set."
    )


def get_feature_cols_from_model(model_type: str, model) -> list:
    """
    Try to extract feature names stored inside the model object itself.
    XGBoost stores names only if trained with a DataFrame (not numpy array).
    RandomForest stores them in feature_names_in_.
    Returns None for Ridge/LSTM — caller uses artifact_cols instead.
    """
    try:
        if model_type == "xgb":
            names = model.get_booster().feature_names
            if names:
                logger.info(
                    "Got %d feature names from XGBoost booster (authoritative)",
                    len(names),
                )
                return names

        if model_type == "rf":
            if hasattr(model, "feature_names_in_"):
                names = list(model.feature_names_in_)
                logger.info(
                    "Got %d feature names from RandomForest.feature_names_in_",
                    len(names),
                )
                return names

    except Exception as exc:
        logger.warning("Could not extract feature names from model: %s", exc)

    return None


# ── Step 2: Load model ────────────────────────────────────────────────────────

# ── FIX 1 ─────────────────────────────────────────────────────────────────────
# load_model_from_hopsworks() now reads feature_cols.json from the SAME
# temp folder where the model was downloaded. This file was saved together
# with the model at training time, so it always matches the model's feature
# count exactly — regardless of what your local feature_cols.json says.
# Returns 5 values: model_type, model, scaler_X, scaler_y, artifact_cols
# artifact_cols is None if the JSON file wasn't found in the artifact.
# ──────────────────────────────────────────────────────────────────────────────

def load_model_from_hopsworks(api_key: str, horizon_h: int):
    """
    Download the best registered model for this horizon from Hopsworks.
    Also reads feature_cols.json from the artifact folder (FIX 1).
    Returns: (model_type, model, scaler_X, scaler_y, artifact_cols)
    """
    import hopsworks
    suffix     = f"_{horizon_h}h"
    model_name = f"lahore_aqi_best{suffix}"

    try:
        logger.info("Loading '%s' from Hopsworks Model Registry...", model_name)
        project  = hopsworks.login(api_key_value=api_key)
        mr       = project.get_model_registry()
        hw_model = mr.get_model(name=model_name, version=1)
        save_dir = hw_model.download()
        logger.info("Downloaded '%s' to: %s", model_name, save_dir)

        # ── FIX 1: Read feature cols from this artifact folder ────────────────
        # Try horizon-specific file first (feature_cols_24h.json), then global.
        # These were copied into the artifact by save_best_to_hopsworks() in
        # training_pipeline.py — they have exactly the right column count.
        artifact_cols = None
        for cols_filename in [f"feature_cols_{horizon_h}h.json", "feature_cols.json"]:
            cols_path = os.path.join(save_dir, cols_filename)
            if Path(cols_path).exists():
                with open(cols_path) as f:
                    artifact_cols = json.load(f)
                logger.info(
                    "FIX 1 ✅ Loaded %d feature cols from artifact file '%s' "
                    "(these match the model exactly)",
                    len(artifact_cols), cols_filename,
                )
                break
        if artifact_cols is None:
            logger.warning(
                "FIX 1 ⚠️  No feature_cols*.json found in artifact at %s — "
                "will fall back to local JSON. Re-register the model to fix permanently.",
                save_dir,
            )
        # ─────────────────────────────────────────────────────────────────────

        # Try XGBoost first
        xgb_file = os.path.join(save_dir, f"xgb_model{suffix}.pkl")
        if Path(xgb_file).exists():
            model = joblib.load(xgb_file)
            logger.info("Loaded XGBoost model for %dh from Hopsworks", horizon_h)
            return "xgb", model, None, None, artifact_cols

        # Try RF
        rf_file = os.path.join(save_dir, f"random_forest_model{suffix}.pkl")
        if Path(rf_file).exists():
            model = joblib.load(rf_file)
            logger.info("Loaded RF model for %dh from Hopsworks", horizon_h)
            return "rf", model, None, None, artifact_cols

        # Try LSTM
        lstm_file = os.path.join(save_dir, f"lstm_model{suffix}.keras")
        if Path(lstm_file).exists():
            from tensorflow import keras as tf_keras
            model    = tf_keras.models.load_model(lstm_file)
            scaler_X = joblib.load(os.path.join(save_dir, f"scaler_X{suffix}.pkl"))
            scaler_y = joblib.load(os.path.join(save_dir, f"scaler_y{suffix}.pkl"))
            logger.info("Loaded LSTM model for %dh from Hopsworks", horizon_h)
            return "lstm", model, scaler_X, scaler_y, artifact_cols

        # Try Ridge
        ridge_file = os.path.join(save_dir, f"ridge_model{suffix}.pkl")
        if Path(ridge_file).exists():
            model = joblib.load(ridge_file)
            logger.info("Loaded Ridge model for %dh from Hopsworks", horizon_h)
            return "ridge", model, None, None, artifact_cols

    except Exception as exc:
        logger.warning(
            "Hopsworks model load failed for %dh (%s) — falling back to local.",
            horizon_h, exc,
        )

    return load_model_from_local(horizon_h)


# ── FIX 2 ─────────────────────────────────────────────────────────────────────
# load_model_from_local() also reads feature_cols.json from the models/
# folder alongside the model file. Same idea as FIX 1 but for local models.
# Returns 5 values to match load_model_from_hopsworks().
# ──────────────────────────────────────────────────────────────────────────────

def load_model_from_local(horizon_h: int):
    """
    Fallback: load from local models/ folder.
    Also reads feature_cols.json from models/ (FIX 2).
    Returns: (model_type, model, scaler_X, scaler_y, artifact_cols)
    """
    suffix = f"_{horizon_h}h"

    # ── FIX 2: Read feature cols from local models/ folder ───────────────────
    artifact_cols = None
    for cols_filename in [f"feature_cols_{horizon_h}h.json", "feature_cols.json"]:
        cols_path = _MODELS_DIR / cols_filename
        if cols_path.exists():
            with open(cols_path) as f:
                artifact_cols = json.load(f)
            logger.info(
                "FIX 2 ✅ Loaded %d feature cols from local models/%s",
                len(artifact_cols), cols_filename,
            )
            break
    if artifact_cols is None:
        logger.warning(
            "FIX 2 ⚠️  No feature_cols*.json found in %s", _MODELS_DIR
        )
    # ─────────────────────────────────────────────────────────────────────────

    xgb_path = str(_MODELS_DIR / f"xgb_model{suffix}.pkl")
    if Path(xgb_path).exists():
        model = joblib.load(xgb_path)
        logger.info("Loaded local XGBoost model for %dh", horizon_h)
        return "xgb", model, None, None, artifact_cols

    rf_path = str(_MODELS_DIR / f"random_forest_model{suffix}.pkl")
    if Path(rf_path).exists():
        model = joblib.load(rf_path)
        logger.info("Loaded local RF model for %dh", horizon_h)
        return "rf", model, None, None, artifact_cols

    lstm_path     = str(_MODELS_DIR / f"lstm_model{suffix}.keras")
    scaler_x_path = str(_MODELS_DIR / f"scaler_X{suffix}.pkl")
    scaler_y_path = str(_MODELS_DIR / f"scaler_y{suffix}.pkl")
    if Path(lstm_path).exists():
        try:
            from tensorflow import keras as tf_keras
            model    = tf_keras.models.load_model(lstm_path)
            scaler_X = joblib.load(scaler_x_path)
            scaler_y = joblib.load(scaler_y_path)
            logger.info("Loaded local LSTM model for %dh", horizon_h)
            return "lstm", model, scaler_X, scaler_y, artifact_cols
        except Exception as exc:
            logger.warning("Local LSTM load failed for %dh (%s), trying Ridge.", horizon_h, exc)

    ridge_path = str(_MODELS_DIR / f"ridge_model{suffix}.pkl")
    if Path(ridge_path).exists():
        model = joblib.load(ridge_path)
        logger.info("Loaded local Ridge model for %dh", horizon_h)
        return "ridge", model, None, None, artifact_cols

    raise FileNotFoundError(
        f"No model found for {horizon_h}h in {_MODELS_DIR}.\n"
        f"Run training_pipeline.py first."
    )


# ── Step 3: Load latest features ──────────────────────────────────────────────

def load_latest_features() -> pd.DataFrame:
    """Load features — Hopsworks primary, local CSV fallback."""
    if USE_HOPSWORKS and HOPSWORKS_API_KEY:
        try:
            import hopsworks
            logger.info("Loading latest features from Hopsworks Feature Store...")
            project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
            fs      = project.get_feature_store()
            fg      = fs.get_feature_group(name=FEATURE_GROUP_NAME,
                                            version=FEATURE_GROUP_VERSION)
            df = fg.read()
            df = df.sort_values("timestamp").tail(SEQ_LEN + 10)
            logger.info("Loaded %d rows from Hopsworks '%s'", len(df), FEATURE_GROUP_NAME)
            return df
        except Exception as exc:
            logger.warning("Hopsworks feature load failed (%s). Trying local CSV.", exc)

    if not Path(LOCAL_FEATURES_CSV).exists():
        raise FileNotFoundError(
            f"Features CSV not found at {LOCAL_FEATURES_CSV}.\n"
            "Run feature_pipeline.py first."
        )
    df = pd.read_csv(LOCAL_FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").tail(SEQ_LEN + 10)
    logger.info("Loaded last %d rows from local CSV", len(df))
    return df


# ── Step 4: Prepare feature matrix ───────────────────────────────────────────

def prepare_X(df_features: pd.DataFrame, feature_cols: list) -> np.ndarray:
    missing = [c for c in feature_cols if c not in df_features.columns]
    if missing:
        logger.warning("Missing %d feature cols (filling with 0): %s", len(missing), missing[:5])
        for col in missing:
            df_features[col] = 0.0
    X = df_features[feature_cols].ffill().fillna(0).values
    return X


# ── Step 5: Predict ───────────────────────────────────────────────────────────

def predict_one_horizon(horizon_h, model_type, model, scaler_X, scaler_y, X_all) -> float:
    if model_type == "lstm":
        if len(X_all) < SEQ_LEN:
            raise ValueError(f"LSTM needs {SEQ_LEN} rows, only {len(X_all)} available.")
        X_scaled = scaler_X.transform(X_all)
        seq      = X_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, X_scaled.shape[1])
        y_scaled = model.predict(seq, verbose=0).ravel()[0]
        pred     = float(scaler_y.inverse_transform([[y_scaled]])[0][0])
    else:
        pred = float(model.predict(X_all[-1:])[0])

    pred = max(0.0, min(pred, 500.0))
    logger.info("%dh prediction (%s): %.1f", horizon_h, model_type, pred)
    return round(pred, 1)


# ── Step 6: Store predictions ─────────────────────────────────────────────────

def store_predictions_hopsworks(api_key: str, prediction_row: dict) -> None:
    import hopsworks
    logger.info("Storing predictions to Hopsworks Feature Store...")
    project = hopsworks.login(api_key_value=api_key)
    fs      = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=PRED_GROUP_NAME,
        version=PRED_GROUP_VERSION,
        primary_key=["city", "forecast_created_utc"],
        description="AQI 24h/48h/72h predictions for Lahore",
        event_time="forecast_created_utc",
    )
    df_pred = pd.DataFrame([prediction_row])
    df_pred["forecast_created_utc"] = pd.to_datetime(df_pred["forecast_created_utc"])
    fg.insert(df_pred, write_options={"start_offline_backfill": True, "wait_for_job": True})
    logger.info("Stored prediction to Hopsworks '%s'", PRED_GROUP_NAME)


def store_predictions_csv(prediction_row: dict) -> None:
    Path(LOCAL_PREDICTIONS_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([prediction_row])
    if Path(LOCAL_PREDICTIONS_CSV).exists():
        df_existing = pd.read_csv(LOCAL_PREDICTIONS_CSV)
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(LOCAL_PREDICTIONS_CSV, index=False)
    logger.info("Saved prediction to %s (%d rows total)",
                LOCAL_PREDICTIONS_CSV, len(df_out))


# ── Step 7: AQI alert ─────────────────────────────────────────────────────────

def classify_aqi(value: float) -> tuple:
    for lo, hi, label, emoji in AQI_CATEGORIES:
        if lo <= value <= hi:
            return label, emoji
    return "Unknown", "❓"


def print_alert(preds: dict, live_aqi: float) -> None:
    print("\n" + "=" * 60)
    print("  LAHORE AQI FORECAST ALERT")
    print("=" * 60)
    label, emoji = classify_aqi(live_aqi)
    print(f"  Current AQI : {live_aqi:>6.1f}  {emoji}  {label}")
    print()
    for hours, key in [(24, "pred_24h"), (48, "pred_48h"), (72, "pred_72h")]:
        v = preds[key]
        label, emoji = classify_aqi(v)
        print(f"  {hours}h forecast: {v:>6.1f}  {emoji}  {label}")
    print()
    worst = max(preds["pred_24h"], preds["pred_48h"], preds["pred_72h"])
    if worst > 200:
        print("  HAZARD WARNING: Very Unhealthy / Hazardous levels forecast!")
        print("     → Avoid ALL outdoor activities")
        print("     → Keep windows closed; use air purifiers")
    elif worst > 150:
        print("  WARNING: Unhealthy AQI forecast.")
        print("     → Sensitive groups should stay indoors")
    else:
        print("  Air quality forecast within acceptable range.")
    print("=" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("  INFERENCE PIPELINE — Lahore AQI Predictor")
    logger.info("  %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 60)

    # Live AQI
    try:
        from live_aqi_client import fetch_live_aqi
        live_data = fetch_live_aqi()
        live_aqi  = live_data["aqi"] or 0.0
        logger.info("Live AQI: %.1f", live_aqi)
    except Exception as exc:
        logger.warning("Could not fetch live AQI: %s. Using 0.", exc)
        live_aqi  = 0.0
        live_data = {}

    # ── FIX 3 ─────────────────────────────────────────────────────────────────
    # Unpack 5 values from both loaders (added artifact_cols as 5th value).
    # Store artifact_cols per horizon inside the models dict.
    # ──────────────────────────────────────────────────────────────────────────
    models = {}
    for horizon_h in [24, 48, 72]:
        logger.info("Loading model for %dh...", horizon_h)
        if USE_HOPSWORKS and HOPSWORKS_API_KEY:
            model_type, model, scaler_X, scaler_y, artifact_cols = load_model_from_hopsworks(
                HOPSWORKS_API_KEY, horizon_h
            )
        else:
            model_type, model, scaler_X, scaler_y, artifact_cols = load_model_from_local(horizon_h)

        # Store all 5 values including artifact_cols
        models[horizon_h] = (model_type, model, scaler_X, scaler_y, artifact_cols)
        logger.info("  %dh → %s  (artifact_cols: %s)",
                    horizon_h, model_type,
                    f"{len(artifact_cols)} cols" if artifact_cols else "not found")
    # ──────────────────────────────────────────────────────────────────────────

    # ── FIX 4 ─────────────────────────────────────────────────────────────────
    # Feature column resolution — priority order:
    #
    #   1. artifact_cols from the 24h model download  ← THIS IS THE FIX
    #      These were saved TOGETHER with the model at training time.
    #      They are guaranteed to match the model's expected feature count.
    #
    #   2. Feature names stored inside the model object itself
    #      (works for XGBoost only if trained with DataFrame, not numpy)
    #
    #   3. Local feature_cols.json / CSV fallback
    #      DANGEROUS — this file grows when you add features but doesn't
    #      update old registered models. This caused the 43 vs 59 crash.
    #
    # By putting artifact_cols first we always use the right column count
    # for the model that's actually registered in Hopsworks.
    # ──────────────────────────────────────────────────────────────────────────
    _mt, _m, _, _, _artifact_cols = models[24]

    if _artifact_cols is not None:
        feature_cols = _artifact_cols
        logger.info(
            "FIX 4 ✅ Using %d feature cols from model artifact (Option A fix — "
            "these are guaranteed to match the registered model)",
            len(feature_cols),
        )
    else:
        # artifact_cols not available — try extracting from model object
        feature_cols = get_feature_cols_from_model(_mt, _m)
        if feature_cols is not None:
            logger.info(
                "Using %d feature cols extracted from %s model object",
                len(feature_cols), _mt,
            )
        else:
            # Last resort — local JSON/CSV (risky if features changed after training)
            logger.warning(
                "FIX 4 ⚠️  artifact_cols not found and model doesn't store feature names. "
                "Falling back to local JSON/CSV — this may cause a shape mismatch "
                "if features were added after the model was registered. "
                "Re-register the model to fix permanently."
            )
            feature_cols = load_feature_cols()

    logger.info("Final feature count: %d", len(feature_cols))
    # ──────────────────────────────────────────────────────────────────────────

    # Load features
    df_features = load_latest_features()
    X_all = prepare_X(df_features, feature_cols)
    logger.info("Feature matrix shape: %s", X_all.shape)

    # Predict — unpack 5 values now
    preds = {}
    for horizon_h in [24, 48, 72]:
        model_type, model, scaler_X, scaler_y, _ = models[horizon_h]
        preds[f"pred_{horizon_h}h"] = predict_one_horizon(
            horizon_h, model_type, model, scaler_X, scaler_y, X_all
        )

    logger.info("Predictions → 24h: %.1f  48h: %.1f  72h: %.1f",
                preds["pred_24h"], preds["pred_48h"], preds["pred_72h"])

    # Build prediction record
    now_utc = datetime.now(timezone.utc).isoformat()
    prediction_row = {
        "city":                 "lahore",
        "forecast_created_utc": now_utc,
        "model_used_24h":       models[24][0],
        "model_used_48h":       models[48][0],
        "model_used_72h":       models[72][0],
        "live_aqi":             float(live_aqi),
        "stations_used":        live_data.get("stations_used", 1),
        "pred_aqi_24h":         preds["pred_24h"],
        "pred_aqi_48h":         preds["pred_48h"],
        "pred_aqi_72h":         preds["pred_72h"],
        "target_date_24h":      str(pd.Timestamp(now_utc) + pd.Timedelta(hours=24)),
        "target_date_48h":      str(pd.Timestamp(now_utc) + pd.Timedelta(hours=48)),
        "target_date_72h":      str(pd.Timestamp(now_utc) + pd.Timedelta(hours=72)),
    }

    # Hopsworks is PRIMARY storage for predictions
    if USE_HOPSWORKS and HOPSWORKS_API_KEY:
        try:
            store_predictions_hopsworks(HOPSWORKS_API_KEY, prediction_row)
            logger.info("Predictions stored in Hopsworks Feature Store (primary).")
        except Exception as exc:
            logger.warning("Hopsworks store failed (%s). Saving to local CSV (fallback).", exc)
            store_predictions_csv(prediction_row)
    else:
        store_predictions_csv(prediction_row)
        logger.info("Predictions saved locally (USE_HOPSWORKS=false).")

    print_alert(
        {"pred_24h": preds["pred_24h"], "pred_48h": preds["pred_48h"],
         "pred_72h": preds["pred_72h"]},
        live_aqi,
    )
    logger.info("Inference pipeline complete.")


if __name__ == "__main__":
    main()
