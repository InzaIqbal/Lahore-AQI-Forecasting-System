"""
inference_pipeline.py
=====================
PURPOSE : Runs every hour via GitHub Actions.
          1. Fetches live AQI + weather for Lahore (multi-station average)
          2. Loads the latest feature row from Hopsworks Feature Store
          3. Loads the best trained model FROM Hopsworks Model Registry
          4. Runs the model to predict 24h/48h/72h AQI
          5. Writes predictions back to Hopsworks (predictions feature group)
          6. Prints an AQI alert if levels are hazardous

HOW TO RUN:
  pip install requests pandas numpy hopsworks python-dotenv joblib tensorflow
  Set your .env file with HOPSWORKS_API_KEY and AQICN_TOKEN
  python pipelines/inference_pipeline.py

FIXES APPLIED:
  FIX 1 — sys.path fix so live_aqi_client.py is found in features/ folder
  FIX 2 — Model loaded FROM Hopsworks Model Registry (not local .pkl files)
  FIX 3 — All file paths are relative to THIS file's location (not cwd)
  FIX 4 — feature_cols.json looked up in models/ folder first
  FIX 5 — LOCAL_FEATURES_CSV points to data/ folder
  FIX 6 — .fillna(method="ffill") → .ffill() for pandas 2.0+
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
# This file lives in pipelines/ — we need to reach features/ and models/
_PIPELINES_DIR = Path(__file__).resolve().parent        # .../pipelines/
_PROJECT_ROOT  = _PIPELINES_DIR.parent                  # .../aqi-predictor-lahore/
_FEATURES_DIR  = _PROJECT_ROOT / "features"             # .../features/
_MODELS_DIR    = _PROJECT_ROOT / "models"               # .../models/
_DATA_DIR      = _PROJECT_ROOT / "data"                 # .../data/

# Add features/ to path so live_aqi_client can be imported
sys.path.insert(0, str(_FEATURES_DIR))

# ── Logging ───────────────────────────────────────────────────────────────────
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
FEATURE_GROUP_VERSION = 1
PRED_GROUP_NAME       = "lahore_aqi_predictions"
PRED_GROUP_VERSION    = 1

# Local file paths (all relative to project root)
LOCAL_FEATURES_CSV    = str(_PROJECT_ROOT / "lahore_features.csv")
LOCAL_PREDICTIONS_CSV = str(_DATA_DIR / "predictions.csv")
FEATURE_COLS_JSON     = str(_MODELS_DIR / "feature_cols.json")

# Local model paths (fallback if Hopsworks is not available)
RF_MODEL_PATH         = str(_MODELS_DIR / "random_forest_model.pkl")
RF_48H_MODEL_PATH     = str(_MODELS_DIR / "random_forest_model_48h.pkl")
RF_72H_MODEL_PATH     = str(_MODELS_DIR / "random_forest_model_72h.pkl")
LSTM_MODEL_PATH       = str(_MODELS_DIR / "lstm_model.keras")
SCALER_X_PATH         = str(_MODELS_DIR / "scaler_X.pkl")
SCALER_Y_PATH         = str(_MODELS_DIR / "scaler_y.pkl")
RIDGE_MODEL_PATH      = str(_MODELS_DIR / "ridge_model.pkl")

SEQ_LEN = 24  # must match training_pipeline SEQ_LEN

AQI_CATEGORIES = [
    (0,   50,  "Good",                    "🟢"),
    (51,  100, "Moderate",                "🟡"),
    (101, 150, "Unhealthy for Sensitive", "🟠"),
    (151, 200, "Unhealthy",               "🔴"),
    (201, 300, "Very Unhealthy",          "🟣"),
    (301, 500, "Hazardous",               "⚫"),
]


# ── Step 1: Load feature columns list ─────────────────────────────────────────

def load_feature_cols() -> list:
    """
    Load the list of feature column names.
    Looks in models/ folder first, then falls back to deriving from CSV.
    """
    if Path(FEATURE_COLS_JSON).exists():
        with open(FEATURE_COLS_JSON) as f:
            cols = json.load(f)
        logger.info("Loaded %d feature columns from %s", len(cols), FEATURE_COLS_JSON)
        return cols

    logger.warning("%s not found — deriving feature cols from CSV.", FEATURE_COLS_JSON)

    if not Path(LOCAL_FEATURES_CSV).exists():
        raise FileNotFoundError(
            f"Neither {FEATURE_COLS_JSON} nor {LOCAL_FEATURES_CSV} found.\n"
            "Run feature_pipeline.py first to generate the features CSV."
        )

    df = pd.read_csv(LOCAL_FEATURES_CSV, nrows=1)
    exclude = {
        "timestamp", "city",
        "target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
        "us_aqi", "us_aqi_pm2_5",
    }
    cols = [c for c in df.columns if c not in exclude]
    logger.info("Derived %d feature columns from CSV", len(cols))
    return cols


# ── Step 2A: Load model FROM Hopsworks Model Registry ─────────────────────────

def load_best_model_from_hopsworks(api_key: str):
    """
    Download the best trained model directly from Hopsworks Model Registry.
    Tries LSTM first, then Random Forest, then Ridge.
    Returns: (model_type, model, scaler_X, scaler_y)
    """
    import hopsworks

    logger.info("Connecting to Hopsworks Model Registry...")
    project = hopsworks.login(api_key_value=api_key)
    mr = project.get_model_registry()

    # Try models in priority order: LSTM > RF > Ridge
    model_candidates = [
        ("lahore_aqi_lstm",  "lstm"),
        ("lahore_aqi_rf",    "rf"),
        ("lahore_aqi_ridge", "ridge"),
    ]

    for model_name, model_type in model_candidates:
        try:
            logger.info("Trying to load '%s' from Hopsworks...", model_name)
            hw_model = mr.get_model(name=model_name, version=1)
            save_dir = hw_model.download()  # downloads all files to a temp directory
            logger.info("Downloaded '%s' to: %s", model_name, save_dir)

            if model_type == "lstm":
                from tensorflow import keras
                model_file = os.path.join(save_dir, "lstm_model.keras")
                scaler_x_file = os.path.join(save_dir, "scaler_X.pkl")
                scaler_y_file = os.path.join(save_dir, "scaler_y.pkl")

                if not Path(model_file).exists():
                    logger.warning("lstm_model.keras not found in downloaded dir, skipping.")
                    continue

                model    = keras.models.load_model(model_file)
                scaler_X = joblib.load(scaler_x_file)
                scaler_y = joblib.load(scaler_y_file)
                logger.info("✅ Loaded LSTM model from Hopsworks Registry")
                return "lstm", model, scaler_X, scaler_y

            elif model_type == "rf":
                model_file = os.path.join(save_dir, "random_forest_model.pkl")
                if not Path(model_file).exists():
                    logger.warning("random_forest_model.pkl not found in downloaded dir, skipping.")
                    continue
                model = joblib.load(model_file)
                logger.info("✅ Loaded Random Forest model from Hopsworks Registry")
                return "rf", model, None, None

            elif model_type == "ridge":
                model_file = os.path.join(save_dir, "ridge_model.pkl")
                if not Path(model_file).exists():
                    logger.warning("ridge_model.pkl not found in downloaded dir, skipping.")
                    continue
                model = joblib.load(model_file)
                logger.info("✅ Loaded Ridge model from Hopsworks Registry")
                return "ridge", model, None, None

        except Exception as exc:
            logger.warning("Could not load '%s' from Hopsworks: %s", model_name, exc)
            continue

    raise RuntimeError(
        "No model found in Hopsworks Model Registry.\n"
        "Make sure training_pipeline_colab.ipynb ran successfully and "
        "the model was saved to Hopsworks."
    )


# ── Step 2B: Load model from local files (fallback) ───────────────────────────

def load_best_model_local():
    """
    Fallback: load the best available model from local files in models/ folder.
    Priority: LSTM > Random Forest > Ridge.
    """
    if Path(LSTM_MODEL_PATH).exists():
        try:
            from tensorflow import keras
            model    = keras.models.load_model(LSTM_MODEL_PATH)
            scaler_X = joblib.load(SCALER_X_PATH)
            scaler_y = joblib.load(SCALER_Y_PATH)
            logger.info("Loaded LSTM model from local: %s", LSTM_MODEL_PATH)
            return "lstm", model, scaler_X, scaler_y
        except Exception as exc:
            logger.warning("Local LSTM load failed (%s), trying Random Forest.", exc)

    if Path(RF_MODEL_PATH).exists():
        model = joblib.load(RF_MODEL_PATH)
        logger.info("Loaded Random Forest model from local: %s", RF_MODEL_PATH)
        return "rf", model, None, None

    if Path(RIDGE_MODEL_PATH).exists():
        model = joblib.load(RIDGE_MODEL_PATH)
        logger.info("Loaded Ridge model from local: %s", RIDGE_MODEL_PATH)
        return "ridge", model, None, None

    raise FileNotFoundError(
        f"No trained model found locally in {_MODELS_DIR}.\n"
        "Either:\n"
        "  1. Set USE_HOPSWORKS=true and provide HOPSWORKS_API_KEY, OR\n"
        "  2. Run training_pipeline_colab.ipynb and download the .pkl files to models/"
    )


def load_aux_model_local(path: str):
    """
    Load an auxiliary model for 48h or 72h prediction from local files.
    Returns None if the file does not exist.
    """
    if Path(path).exists():
        model = joblib.load(path)
        logger.info("Loaded auxiliary model from %s", path)
        return model
    logger.info("Auxiliary model %s not found — will use persistence estimate.", path)
    return None


# ── Step 3: Load latest features ──────────────────────────────────────────────

def load_latest_features_from_hopsworks(api_key: str) -> pd.DataFrame:
    import hopsworks
    logger.info("Loading latest features from Hopsworks Feature Store...")
    project = hopsworks.login(api_key_value=api_key)
    fs      = project.get_feature_store()
    fg      = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df      = fg.read()
    df      = df.sort_values("timestamp").tail(SEQ_LEN + 10)
    logger.info("Fetched %d rows from Hopsworks feature group '%s'", len(df), FEATURE_GROUP_NAME)
    return df


def load_latest_features_from_csv() -> pd.DataFrame:
    if not Path(LOCAL_FEATURES_CSV).exists():
        raise FileNotFoundError(
            f"Features CSV not found at {LOCAL_FEATURES_CSV}.\n"
            "Run feature_pipeline.py first."
        )
    df = pd.read_csv(LOCAL_FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").tail(SEQ_LEN + 10)
    logger.info("Loaded last %d rows from %s", len(df), LOCAL_FEATURES_CSV)
    return df


# ── Step 4: Prepare feature matrix ────────────────────────────────────────────

def prepare_X(df_features: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """
    Select and clean feature columns from the DataFrame.
    FIX 6: .fillna(method="ffill") → .ffill() for pandas 2.0+
    """
    missing = [c for c in feature_cols if c not in df_features.columns]
    if missing:
        logger.warning(
            "Missing %d feature cols (filling with 0): %s",
            len(missing), missing[:5]
        )
        for col in missing:
            df_features[col] = 0.0

    available_cols = [c for c in feature_cols if c in df_features.columns]
    X = df_features[available_cols].ffill().fillna(0).values
    return X


# ── Step 5: Run inference ──────────────────────────────────────────────────────

def _predict_tabular(model, X_all: np.ndarray) -> float:
    """Run a single-row tabular prediction (RF or Ridge)."""
    return float(model.predict(X_all[-1:])[0])


def _predict_lstm(model, scaler_X, scaler_y, X_all: np.ndarray) -> float:
    """Run an LSTM sequence prediction."""
    if len(X_all) < SEQ_LEN:
        raise ValueError(
            f"LSTM needs at least {SEQ_LEN} rows but only {len(X_all)} are available."
        )
    X_scaled = scaler_X.transform(X_all)
    seq      = X_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, X_scaled.shape[1])
    y_scaled = model.predict(seq, verbose=0).ravel()[0]
    return float(scaler_y.inverse_transform([[y_scaled]])[0][0])


def predict(
    model_type: str,
    model,
    scaler_X,
    scaler_y,
    df_features: pd.DataFrame,
    feature_cols: list,
    aux_model_48h=None,
    aux_model_72h=None,
) -> dict:
    """
    Run the loaded model on the latest feature row(s).
    Returns dict with keys: pred_24h, pred_48h, pred_72h.
    """
    X_all = prepare_X(df_features, feature_cols)

    # 24h prediction
    if model_type == "lstm":
        pred_24h = _predict_lstm(model, scaler_X, scaler_y, X_all)
    else:
        pred_24h = _predict_tabular(model, X_all)

    # 48h prediction — use dedicated model if available
    if aux_model_48h is not None:
        pred_48h = _predict_tabular(aux_model_48h, X_all)
        logger.info("48h prediction from dedicated model: %.1f", pred_48h)
    else:
        pred_48h = pred_24h * 1.02
        logger.info("48h prediction via persistence fallback: %.1f", pred_48h)

    # 72h prediction — use dedicated model if available
    if aux_model_72h is not None:
        pred_72h = _predict_tabular(aux_model_72h, X_all)
        logger.info("72h prediction from dedicated model: %.1f", pred_72h)
    else:
        pred_72h = pred_24h * 1.03
        logger.info("72h prediction via persistence fallback: %.1f", pred_72h)

    # Clip to valid AQI range [0, 500]
    pred_24h = max(0.0, min(pred_24h, 500.0))
    pred_48h = max(0.0, min(pred_48h, 500.0))
    pred_72h = max(0.0, min(pred_72h, 500.0))

    logger.info(
        "Final predictions → 24h: %.1f  48h: %.1f  72h: %.1f",
        pred_24h, pred_48h, pred_72h,
    )
    return {
        "pred_24h": round(pred_24h, 1),
        "pred_48h": round(pred_48h, 1),
        "pred_72h": round(pred_72h, 1),
    }


# ── Step 6: Store predictions ──────────────────────────────────────────────────

def store_predictions_hopsworks(api_key: str, prediction_row: dict) -> None:
    import hopsworks
    logger.info("Storing predictions to Hopsworks...")
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
    fg.insert(df_pred, write_options={"wait_for_job": True})
    logger.info("✅ Stored prediction to Hopsworks '%s'", PRED_GROUP_NAME)


def store_predictions_csv(prediction_row: dict) -> None:
    # Make sure data/ directory exists
    Path(LOCAL_PREDICTIONS_CSV).parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame([prediction_row])
    if Path(LOCAL_PREDICTIONS_CSV).exists():
        df_existing = pd.read_csv(LOCAL_PREDICTIONS_CSV)
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(LOCAL_PREDICTIONS_CSV, index=False)
    logger.info(
        "Saved prediction to %s (%d rows total)",
        LOCAL_PREDICTIONS_CSV, len(df_out),
    )


# ── Step 7: AQI alert ─────────────────────────────────────────────────────────

def classify_aqi(value: float) -> tuple:
    for lo, hi, label, emoji in AQI_CATEGORIES:
        if lo <= value <= hi:
            return label, emoji
    return "Unknown", "❓"


def print_alert(preds: dict, live_aqi: float) -> None:
    print("\n" + "=" * 60)
    print("  🌫️  LAHORE AQI FORECAST ALERT")
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
        print("  ⚠️  HAZARD WARNING: Very Unhealthy / Hazardous levels forecast!")
        print("     → Avoid ALL outdoor activities")
        print("     → Keep windows closed; use air purifiers")
        print("     → Wear N95 masks if going outside is unavoidable")
    elif worst > 150:
        print("  ⚠️  WARNING: Unhealthy AQI forecast.")
        print("     → Sensitive groups (elderly, children, asthma) should stay indoors")
        print("     → Limit strenuous outdoor activity")
    else:
        print("  ✅  Air quality forecast within acceptable range.")
    print("=" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("  INFERENCE PIPELINE — Lahore AQI Predictor")
    logger.info("  %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 60)
    logger.info("  Project root : %s", _PROJECT_ROOT)
    logger.info("  Models dir   : %s", _MODELS_DIR)
    logger.info("  Data dir     : %s", _DATA_DIR)
    logger.info("=" * 60)

    # ── Live AQI ──────────────────────────────────────────────────────────────
    try:
        from live_aqi_client import fetch_lahore_aqi_average
        live_data = fetch_lahore_aqi_average()
        live_aqi  = live_data["aqi"] or 0.0
        logger.info(
            "Live AQI (averaged): %.1f from %d stations",
            live_aqi, live_data["stations_used"],
        )
    except Exception as exc:
        logger.warning("Could not fetch live AQI: %s. Using 0 as fallback.", exc)
        live_aqi  = 0.0
        live_data = {}

    # ── Feature columns ───────────────────────────────────────────────────────
    feature_cols = load_feature_cols()

    # ── Load primary model ────────────────────────────────────────────────────
    # Tries Hopsworks Model Registry first, falls back to local files
    if USE_HOPSWORKS and HOPSWORKS_API_KEY:
        try:
            model_type, model, scaler_X, scaler_y = load_best_model_from_hopsworks(
                HOPSWORKS_API_KEY
            )
        except Exception as exc:
            logger.warning(
                "Hopsworks model load failed (%s). Falling back to local files.", exc
            )
            model_type, model, scaler_X, scaler_y = load_best_model_local()
    else:
        logger.info("USE_HOPSWORKS=false — loading model from local files.")
        model_type, model, scaler_X, scaler_y = load_best_model_local()

    # ── Load auxiliary 48h / 72h models (local only, optional) ───────────────
    aux_model_48h = load_aux_model_local(RF_48H_MODEL_PATH)
    aux_model_72h = load_aux_model_local(RF_72H_MODEL_PATH)

    # ── Load latest features ──────────────────────────────────────────────────
    if USE_HOPSWORKS and HOPSWORKS_API_KEY:
        try:
            df_features = load_latest_features_from_hopsworks(HOPSWORKS_API_KEY)
        except Exception as exc:
            logger.warning(
                "Hopsworks feature load failed (%s). Falling back to local CSV.", exc
            )
            df_features = load_latest_features_from_csv()
    else:
        df_features = load_latest_features_from_csv()

    # ── Run inference ─────────────────────────────────────────────────────────
    preds = predict(
        model_type, model, scaler_X, scaler_y,
        df_features, feature_cols,
        aux_model_48h=aux_model_48h,
        aux_model_72h=aux_model_72h,
    )

    # ── Build prediction record ───────────────────────────────────────────────
    now_utc = datetime.now(timezone.utc).isoformat()
    prediction_row = {
        "city":                 "lahore",
        "forecast_created_utc": now_utc,
        "model_used":           model_type,
        "live_aqi":             live_aqi,
        "stations_used":        live_data.get("stations_used", 0),
        "pred_aqi_24h":         preds["pred_24h"],
        "pred_aqi_48h":         preds["pred_48h"],
        "pred_aqi_72h":         preds["pred_72h"],
        "target_date_24h":      str(pd.Timestamp(now_utc) + pd.Timedelta(hours=24)),
        "target_date_48h":      str(pd.Timestamp(now_utc) + pd.Timedelta(hours=48)),
        "target_date_72h":      str(pd.Timestamp(now_utc) + pd.Timedelta(hours=72)),
    }

    # ── Store predictions ─────────────────────────────────────────────────────
    if USE_HOPSWORKS and HOPSWORKS_API_KEY:
        try:
            store_predictions_hopsworks(HOPSWORKS_API_KEY, prediction_row)
        except Exception as exc:
            logger.warning(
                "Hopsworks prediction store failed (%s). Saving to local CSV.", exc
            )
            store_predictions_csv(prediction_row)
    else:
        store_predictions_csv(prediction_row)

    # ── Print alert ───────────────────────────────────────────────────────────
    print_alert(preds, live_aqi)
    logger.info("Inference pipeline complete. Predictions: %s", preds)


if __name__ == "__main__":
    main()