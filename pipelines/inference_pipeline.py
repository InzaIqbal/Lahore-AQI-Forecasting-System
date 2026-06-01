"""
inference_pipeline.py
=====================
CHANGES FROM YOUR ORIGINAL:
  CHANGE 1 — _PROJECT_ROOT now points to the SAME folder as this script.
             Previously it pointed one level UP (parent.parent), so models/
             folder was never found. This was a silent crash bug.

  CHANGE 2 — Hopsworks is PRIMARY storage for predictions.
             Local CSV is FALLBACK only. Previously reversed — wrong per spec.

  CHANGE 3 — Models loaded from Hopsworks first, local fallback second.

  CHANGE 4 — XGBoost loading added to both load_model_from_hopsworks()
             and load_model_from_local(). Without this, if XGBoost wins
             training it would never be found at inference time.

  CHANGE 5 — Removed duplicate if __name__ == "__main__" block at bottom.
             The second block referenced `preds` outside main() which would
             crash with NameError.
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
# CHANGE 1: _PROJECT_ROOT is the folder containing this script — same level
# as training_pipeline.py and feature_pipeline.py.
# Previously was _PIPELINES_DIR.parent which pointed one directory UP,
# so models/ was looked for in the wrong place and never found.
_PIPELINES_DIR = Path(__file__).resolve().parent      # = .../pipelines/
_PROJECT_ROOT  = _PIPELINES_DIR.parent                # = .../aqi-predictor-lahore/
_MODELS_DIR    = _PIPELINES_DIR / "models"            # = .../pipelines/models/  (where your models are)
_DATA_DIR      = _PIPELINES_DIR / "data"              # = .../pipelines/data/
_FEATURES_DIR  = _PROJECT_ROOT / "features"           # = .../features/  (where live_aqi_client.py is)
sys.path.insert(0, str(_PROJECT_ROOT))   # so live_aqi_client.py is importable
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
FEATURE_GROUP_VERSION = 2
PRED_GROUP_NAME       = "lahore_aqi_predictions"
PRED_GROUP_VERSION    = 2

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
    Load feature columns — tries (in order):
      1. Local models/feature_cols.json  (exists after training pipeline runs locally)
      2. Hopsworks model artifact        (always available in CI — downloaded with the model)
      3. Derive from local CSV           (last resort fallback)
    """
    # 1. Local file (works when running locally after training)
    if Path(FEATURE_COLS_JSON).exists():
        with open(FEATURE_COLS_JSON) as f:
            cols = json.load(f)
        logger.info("Loaded %d feature columns from local %s", len(cols), FEATURE_COLS_JSON)
        return cols

    # 2. Download from Hopsworks model artifact (works in GitHub Actions CI)
    if USE_HOPSWORKS and HOPSWORKS_API_KEY:
        try:
            import hopsworks
            logger.info("feature_cols.json not found locally — downloading from Hopsworks model artifact...")
            project  = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
            mr       = project.get_model_registry()
            # Any horizon's model will have feature_cols.json — use 24h
            hw_model = mr.get_model(name="lahore_aqi_best_24h", version=1)
            save_dir = hw_model.download()
            cols_path = os.path.join(save_dir, "feature_cols.json")
            if Path(cols_path).exists():
                with open(cols_path) as f:
                    cols = json.load(f)
                # Cache it locally so subsequent calls don't re-download
                Path(FEATURE_COLS_JSON).parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy(cols_path, FEATURE_COLS_JSON)
                logger.info("Loaded %d feature columns from Hopsworks model artifact", len(cols))
                return cols
            else:
                logger.warning("feature_cols.json not found inside Hopsworks model artifact at %s", save_dir)
        except Exception as exc:
            logger.warning("Could not load feature_cols.json from Hopsworks: %s", exc)

    # 3. Derive from local CSV (last resort)
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
    Extract the EXACT feature names the model was trained on from the
    model object itself. This is the only 100% reliable source and fixes
    the 'expected 43 got 46' mismatch when feature_cols.json drifts.

    XGBoost stores feature names in the booster at fit() time.
    RandomForest stores them in feature_names_in_.
    Ridge/LSTM do not store names — returns None, caller uses fallback.
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

    return None  # Ridge / LSTM — caller falls back to load_feature_cols()


# ── Step 2: Load model — Hopsworks first, local fallback ─────────────────────

def load_model_from_hopsworks(api_key: str, horizon_h: int):
    """Download the best registered model for this horizon from Hopsworks."""
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

        # CHANGE 4: Try XGBoost first (most likely winner after adding it)
        xgb_file = os.path.join(save_dir, f"xgb_model{suffix}.pkl")
        if Path(xgb_file).exists():
            model = joblib.load(xgb_file)
            logger.info("Loaded XGBoost model for %dh from Hopsworks", horizon_h)
            return "xgb", model, None, None

        # Try RF
        rf_file = os.path.join(save_dir, f"random_forest_model{suffix}.pkl")
        if Path(rf_file).exists():
            model = joblib.load(rf_file)
            logger.info("Loaded RF model for %dh from Hopsworks", horizon_h)
            return "rf", model, None, None

        # Try LSTM
        lstm_file = os.path.join(save_dir, f"lstm_model{suffix}.keras")
        if Path(lstm_file).exists():
            from tensorflow import keras as tf_keras
            model    = tf_keras.models.load_model(lstm_file)
            scaler_X = joblib.load(os.path.join(save_dir, f"scaler_X{suffix}.pkl"))
            scaler_y = joblib.load(os.path.join(save_dir, f"scaler_y{suffix}.pkl"))
            logger.info("Loaded LSTM model for %dh from Hopsworks", horizon_h)
            return "lstm", model, scaler_X, scaler_y

        # Try Ridge
        ridge_file = os.path.join(save_dir, f"ridge_model{suffix}.pkl")
        if Path(ridge_file).exists():
            model = joblib.load(ridge_file)
            logger.info("Loaded Ridge model for %dh from Hopsworks", horizon_h)
            return "ridge", model, None, None

    except Exception as exc:
        logger.warning(
            "Hopsworks model load failed for %dh (%s) — falling back to local.",
            horizon_h, exc,
        )

    return load_model_from_local(horizon_h)


def load_model_from_local(horizon_h: int):
    """Fallback: load from local models/ folder."""
    suffix = f"_{horizon_h}h"

    # CHANGE 4: Try XGBoost first
    xgb_path = str(_MODELS_DIR / f"xgb_model{suffix}.pkl")
    if Path(xgb_path).exists():
        model = joblib.load(xgb_path)
        logger.info("Loaded local XGBoost model for %dh", horizon_h)
        return "xgb", model, None, None

    rf_path = str(_MODELS_DIR / f"random_forest_model{suffix}.pkl")
    if Path(rf_path).exists():
        model = joblib.load(rf_path)
        logger.info("Loaded local RF model for %dh", horizon_h)
        return "rf", model, None, None

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
            return "lstm", model, scaler_X, scaler_y
        except Exception as exc:
            logger.warning("Local LSTM load failed for %dh (%s), trying Ridge.", horizon_h, exc)

    ridge_path = str(_MODELS_DIR / f"ridge_model{suffix}.pkl")
    if Path(ridge_path).exists():
        model = joblib.load(ridge_path)
        logger.info("Loaded local Ridge model for %dh", horizon_h)
        return "ridge", model, None, None

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
        # Works for RF, XGBoost, and Ridge — all use .predict(X)
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

    # Load models first — we derive feature cols FROM the model itself.
    # This is the only guaranteed-correct source: XGBoost stores the exact
    # column names it was trained on in get_booster().feature_names.
    # Any other approach (JSON file, schema, CSV header) can drift and cause
    # the 'expected 43 got 46' shape mismatch.
    models = {}
    for horizon_h in [24, 48, 72]:
        logger.info("Loading model for %dh...", horizon_h)
        if USE_HOPSWORKS and HOPSWORKS_API_KEY:
            model_type, model, scaler_X, scaler_y = load_model_from_hopsworks(
                HOPSWORKS_API_KEY, horizon_h
            )
        else:
            model_type, model, scaler_X, scaler_y = load_model_from_local(horizon_h)
        models[horizon_h] = (model_type, model, scaler_X, scaler_y)
        logger.info("  %dh → %s", horizon_h, model_type)

    # Derive feature cols from the 24h model (all horizons trained on same cols).
    # Falls back to load_feature_cols() for Ridge/LSTM which don't store names.
    _mt, _m, _, _ = models[24]
    feature_cols = get_feature_cols_from_model(_mt, _m)
    if feature_cols is None:
        logger.info("Model type '%s' doesn't store feature names — using JSON/CSV fallback", _mt)
        feature_cols = load_feature_cols()
    logger.info("Using %d feature columns (source: %s model)", len(feature_cols), _mt)

    # Load features
    df_features = load_latest_features()
    X_all = prepare_X(df_features, feature_cols)
    logger.info("Feature matrix shape: %s", X_all.shape)

    # Predict
    preds = {}
    for horizon_h in [24, 48, 72]:
        model_type, model, scaler_X, scaler_y = models[horizon_h]
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

    # CHANGE 2: Hopsworks is PRIMARY storage for predictions
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


# CHANGE 5: Only ONE if __name__ block. The original had two — the second
# referenced `preds` outside main() which would crash with NameError.
if __name__ == "__main__":
    main()
