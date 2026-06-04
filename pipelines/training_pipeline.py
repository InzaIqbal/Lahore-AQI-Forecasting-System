

import warnings
warnings.filterwarnings("ignore")

import os
import json
import logging
import joblib
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone
from pathlib import Path
from sklearn.linear_model  import Ridge
from sklearn.ensemble      import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.metrics       import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import shap
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.dpi"]     = 120
plt.rcParams["figure.figsize"] = (14, 5)

USE_HOPSWORKS         = os.environ.get("USE_HOPSWORKS", "true").lower() == "true"
HOPSWORKS_API_KEY     = os.environ.get("HOPSWORKS_API_KEY", "")
FEATURE_GROUP_NAME    = "lahore_aqi_features"
FEATURE_GROUP_VERSION = 3   # CHANGE 10: bumped from 1→3 to match new feature schema
LOCAL_FEATURES_CSV    = "lahore_features.csv"

HORIZONS = [24, 48, 72]
SEQ_LEN  = 24

# CHANGE 1: All models go into models/ subfolder
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
logger.info("Models will be saved to: %s", MODELS_DIR)

EXCLUDE_ALWAYS = [
    "timestamp", "city",
    "target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
    "us_aqi",
    "us_aqi_pm2_5",
]


# ── Step 1: Load data ─────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if USE_HOPSWORKS and HOPSWORKS_API_KEY != "YOUR_KEY_HERE":
        try:
            import hopsworks
            logger.info("Connecting to Hopsworks feature store...")
            project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
            fs      = project.get_feature_store()
            fg      = fs.get_feature_group(name=FEATURE_GROUP_NAME,
                                            version=FEATURE_GROUP_VERSION)
            df = fg.read()
            logger.info("Loaded %d rows from Hopsworks '%s'", len(df), FEATURE_GROUP_NAME)
            return df
        except Exception as exc:
            logger.warning("Hopsworks load failed (%s) — falling back to CSV.", exc)

    if not Path(LOCAL_FEATURES_CSV).exists():
        raise FileNotFoundError(
            f"'{LOCAL_FEATURES_CSV}' not found.\n"
            "Either set USE_HOPSWORKS=true or place the CSV in the same folder."
        )
    df = pd.read_csv(LOCAL_FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info("Loaded %d rows from CSV '%s'", len(df), LOCAL_FEATURES_CSV)
    return df


# ── Step 2: Helpers ───────────────────────────────────────────────────────────

def evaluate(name: str, y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    print(f"  {name:<35}  RMSE={rmse:7.2f}  MAE={mae:7.2f}  R²={r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


def chronological_split(df: pd.DataFrame, feature_cols: list, target_col: str):
    df_model = (
        df[feature_cols + [target_col, "timestamp"]]
        .dropna(subset=[target_col])
        .dropna(subset=feature_cols)
        .reset_index(drop=True)
    )
    n         = len(df_model)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)
    df_train  = df_model.iloc[:train_end]
    df_val    = df_model.iloc[train_end:val_end]
    df_test   = df_model.iloc[val_end:]
    logger.info(
        "Split for '%s': train=%d  val=%d  test=%d",
        target_col, len(df_train), len(df_val), len(df_test),
    )
    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values
    X_val   = df_val[feature_cols].values
    y_val   = df_val[target_col].values
    X_test  = df_test[feature_cols].values
    y_test  = df_test[target_col].values
    return X_train, y_train, X_val, y_val, X_test, y_test


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len: i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


# ── Step 2b: Per-horizon feature selection ────────────────────────────────────
# CHANGE 6

def get_feature_cols_for_horizon(all_cols: list, horizon_h: int) -> list:
    """
    Drop short-lag features that are irrelevant (and add noise) at longer horizons.

    WHY this matters:
      aqi_lag_1h tells you what AQI was 1 hour ago — useful for 24h but
      essentially random noise for 72h. Keeping it confuses tree models
      because they waste splits on meaningless correlations.
      Removing these features forces the model to rely on the stronger
      long-range signals: seasonal means, 24h/48h lags, weather rolling stats.
    """
    if horizon_h == 72:
        # Drop everything short-range: 1h, 3h, 6h lags and derived change features
        drop = {
            "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h",
            "wind_lag_1h", "wind_lag_3h", "temp_lag_1h", "temp_lag_3h",
            "aqi_change_1h", "aqi_pct_change_1h",
        }
    elif horizon_h == 48:
        # Drop only the very shortest lags
        drop = {
            "aqi_lag_1h", "aqi_lag_3h",
            "wind_lag_1h", "temp_lag_1h",
            "aqi_change_1h",
        }
    else:
        drop = set()  # 24h: use all features

    filtered = [c for c in all_cols if c not in drop]
    logger.info(
        "Horizon %dh: using %d / %d features (dropped %d short-lag cols)",
        horizon_h, len(filtered), len(all_cols), len(all_cols) - len(filtered),
    )
    return filtered


# ── Step 3: Train one horizon ─────────────────────────────────────────────────

def train_one_horizon(df_raw: pd.DataFrame, feature_cols: list, horizon_h: int) -> dict:
    target_col = f"target_aqi_{horizon_h}h"
    suffix     = f"_{horizon_h}h"

    print()
    print("=" * 70)
    print(f"  HORIZON: {horizon_h}h   TARGET: {target_col}")
    print("=" * 70)

    # CHANGE 8: each horizon gets its own optimal feature set
    horizon_feature_cols = get_feature_cols_for_horizon(feature_cols, horizon_h)

    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(
        df_raw, horizon_feature_cols, target_col
    )

    results = []

    # ── Ridge ─────────────────────────────────────────────────────────────────
    print(f"\n  --- Ridge Regression ({horizon_h}h) ---")
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0, random_state=SEED)),
    ])
    ridge_pipe.fit(X_train, y_train)
    results.append(evaluate(f"Ridge Val  {horizon_h}h", y_val,  ridge_pipe.predict(X_val)))
    ridge_test = evaluate(f"Ridge Test {horizon_h}h", y_test, ridge_pipe.predict(X_test))
    results.append(ridge_test)
    ridge_path = str(MODELS_DIR / f"ridge_model{suffix}.pkl")
    joblib.dump(ridge_pipe, ridge_path)
    logger.info("Saved: %s", ridge_path)

    # ── Random Forest ─────────────────────────────────────────────────────────
    print(f"\n  --- Random Forest ({horizon_h}h) ---")
    rf_model = RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_leaf=4,
        n_jobs=-1, random_state=SEED,
    )
    rf_model.fit(X_train, y_train)
    results.append(evaluate(f"RF Val  {horizon_h}h", y_val,  rf_model.predict(X_val)))
    rf_test = evaluate(f"RF Test {horizon_h}h", y_test, rf_model.predict(X_test))
    results.append(rf_test)
    rf_path = str(MODELS_DIR / f"random_forest_model{suffix}.pkl")
    joblib.dump(rf_model, rf_path)
    logger.info("Saved: %s", rf_path)

    # CHANGE 9: use horizon_feature_cols so column count matches X_test shape
    importances = pd.Series(rf_model.feature_importances_, index=horizon_feature_cols)
    top15 = importances.sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    top15.plot(kind="barh", color="steelblue", ax=ax)
    ax.set_title(f"RF Feature Importances — {horizon_h}h horizon", fontweight="bold")
    plt.tight_layout()
    imp_path = str(MODELS_DIR / f"rf_importance{suffix}.png")
    plt.savefig(imp_path, bbox_inches="tight")
    plt.close()

    # ── XGBoost ───────────────────────────────────────────────────────────────
    # CHANGE 5 + CHANGE 7: XGBoost with per-horizon hyperparameters.
    #
    # WHY different params per horizon?
    #   24h: deep trees (max_depth=6) are fine — the target is predictable.
    #   48h: shallower trees (max_depth=4) + more regularisation because the
    #        target is noisier and deep trees overfit.
    #   72h: very shallow (max_depth=3) + strong regularisation — the 72h
    #        target is mostly driven by seasonal patterns, not short-range
    #        interactions, so complex trees just memorise training noise.
    print(f"\n  --- XGBoost ({horizon_h}h) ---")
    xgb_test = {"model": f"XGB Test {horizon_h}h", "rmse": 9999.0, "mae": 9999.0, "r2": -1.0}
    try:
        from xgboost import XGBRegressor

        _xgb_params = {
            24: dict(max_depth=6, n_estimators=500,  learning_rate=0.05,
                     min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0),
            48: dict(max_depth=4, n_estimators=800,  learning_rate=0.03,
                     min_child_weight=5, reg_alpha=0.5, reg_lambda=2.0),
            72: dict(max_depth=3, n_estimators=1000, learning_rate=0.02,
                     min_child_weight=7, reg_alpha=1.0, reg_lambda=3.0),
        }

        xgb_model = XGBRegressor(
            **_xgb_params[horizon_h],
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            tree_method="hist",
            early_stopping_rounds=30,
            eval_metric="rmse",
            verbosity=0,
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        results.append(evaluate(f"XGB Val  {horizon_h}h", y_val,  xgb_model.predict(X_val)))
        xgb_test = evaluate(f"XGB Test {horizon_h}h", y_test, xgb_model.predict(X_test))
        results.append(xgb_test)
        xgb_path = str(MODELS_DIR / f"xgb_model{suffix}.pkl")
        joblib.dump(xgb_model, xgb_path)
        logger.info("Saved: %s", xgb_path)
    except ImportError:
        logger.warning("xgboost not installed — skipping. Run: pip install xgboost")

    # ── LSTM ──────────────────────────────────────────────────────────────────
    print(f"\n  --- LSTM ({horizon_h}h) ---")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s   = scaler_X.transform(X_val)
    X_test_s  = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s   = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    X_train_seq, y_train_seq = make_sequences(X_train_s, y_train_s, SEQ_LEN)
    X_val_seq,   y_val_seq   = make_sequences(X_val_s,   y_val_s,   SEQ_LEN)
    X_test_seq,  y_test_seq  = make_sequences(X_test_s,  y_test,    SEQ_LEN)

    n_features = X_train_seq.shape[2]
    lstm_model = keras.Sequential([
        layers.Input(shape=(SEQ_LEN, n_features)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64,  return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ], name=f"AQI_LSTM_{horizon_h}h")

    lstm_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse", metrics=["mae"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, verbose=1,
        ),
    ]
    lstm_model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=60, batch_size=128, callbacks=callbacks, verbose=1,
    )

    lstm_pred_scaled = lstm_model.predict(X_test_seq, verbose=0).ravel()
    lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).ravel()
    lstm_test = evaluate(f"LSTM Test {horizon_h}h", y_test_seq, lstm_pred)
    results.append(lstm_test)

    lstm_path     = str(MODELS_DIR / f"lstm_model{suffix}.keras")
    scaler_x_path = str(MODELS_DIR / f"scaler_X{suffix}.pkl")
    scaler_y_path = str(MODELS_DIR / f"scaler_y{suffix}.pkl")
    lstm_model.save(lstm_path)
    joblib.dump(scaler_X, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)
    logger.info("Saved: %s  %s  %s", lstm_path, scaler_x_path, scaler_y_path)

    # ── SHAP on RF ────────────────────────────────────────────────────────────
    # CHANGE 9: feature_names uses horizon_feature_cols (matches X_test shape)
    print(f"\n  --- SHAP ({horizon_h}h) ---")
    X_shap      = X_test[:min(500, len(X_test))]
    explainer   = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_shap)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, feature_names=horizon_feature_cols,
                      show=False, max_display=20)
    plt.title(f"SHAP Summary — RF {horizon_h}h", fontsize=13, fontweight="bold")
    plt.tight_layout()
    shap_path = str(MODELS_DIR / f"shap_summary{suffix}.png")
    plt.savefig(shap_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved: %s", shap_path)

    # Pick best test model
    test_results = [r for r in results if "Test" in r["model"]]
    best = min(test_results, key=lambda r: r["rmse"])
    print(f"\n  Best for {horizon_h}h: {best['model']}  "
          f"RMSE={best['rmse']:.2f}  R²={best['r2']:.4f}")

    return {
        "horizon_h":            horizon_h,
        "best_model":           best["model"],
        "rmse":                 best["rmse"],
        "mae":                  best["mae"],
        "r2":                   best["r2"],
        "rf_rmse":              rf_test["rmse"],
        "lstm_rmse":            lstm_test["rmse"],
        "ridge_rmse":           ridge_test["rmse"],
        "xgb_rmse":             xgb_test["rmse"],
        "horizon_feature_cols": horizon_feature_cols,   # needed for per-horizon JSON save
    }


# ── Step 4: Upload BEST model to Hopsworks Model Registry ────────────────────

def save_best_to_hopsworks(
    model_path: str,
    model_name: str,
    metrics: dict,
    feature_cols: list,
    horizon_h: int,
    extra_files: list = None,
) -> None:
    """Upload the single best model for one horizon to Hopsworks Model Registry."""
    try:
        import hopsworks
    except ImportError:
        logger.error("hopsworks not installed — skipping registry upload.")
        return

    logger.info("Uploading '%s' to Hopsworks Model Registry...", model_name)
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    mr      = project.get_model_registry()

    staging = f"/tmp/{model_name}_staging"
    os.makedirs(staging, exist_ok=True)
    shutil.copy(model_path, staging)

    # CHANGE 3 + CHANGE 10: copy BOTH the global feature_cols.json AND
    # the per-horizon one so inference_pipeline can load the right columns.
    global_cols_path  = str(MODELS_DIR / "feature_cols.json")
    horizon_cols_path = str(MODELS_DIR / f"feature_cols_{horizon_h}h.json")
    shutil.copy(global_cols_path,  staging)
    if Path(horizon_cols_path).exists():
        shutil.copy(horizon_cols_path, staging)

    for fpath in (extra_files or []):
        if os.path.exists(fpath):
            shutil.copy(fpath, staging)

    model_obj = mr.python.create_model(
        name=model_name,
        metrics=metrics,
        description=f"Best AQI model for {model_name} — lowest test RMSE",
    )
    model_obj.save(staging)
    logger.info("Registered '%s' v%s in Hopsworks Model Registry",
                model_name, model_obj.version)


# ── Step 5: Comparison plot ───────────────────────────────────────────────────

def plot_horizon_comparison(all_results: list) -> None:
    df = pd.DataFrame(all_results)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#4477aa", "#ee6677", "#228833"]
    for ax, metric, label in zip(
        axes,
        ["rmse", "mae", "r2"],
        ["RMSE (lower is better)", "MAE (lower is better)", "R² (higher is better)"],
    ):
        bars = ax.bar([f"{h}h" for h in df["horizon_h"]], df[metric], color=colors)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Forecast horizon")
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    plt.suptitle("Model Performance Across Forecast Horizons",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    cmp_path = str(MODELS_DIR / "comparison_all_horizons.png")
    plt.savefig(cmp_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", cmp_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("  TRAINING PIPELINE — Lahore AQI Predictor")
    logger.info("  %s", datetime.now(timezone.utc).isoformat())
    logger.info("  Models directory: %s", MODELS_DIR)
    logger.info("=" * 70)

    df_raw = load_data()
    df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)

    feature_cols = [c for c in df_raw.columns if c not in EXCLUDE_ALWAYS]
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")

    # CHANGE 2: save global feature_cols.json to MODELS_DIR
    feature_cols_path = str(MODELS_DIR / "feature_cols.json")
    with open(feature_cols_path, "w") as fh:
        json.dump(feature_cols, fh)
    logger.info("Saved global feature cols: %s", feature_cols_path)

    all_results = []
    for h in HORIZONS:
        result = train_one_horizon(df_raw, feature_cols, h)
        all_results.append(result)

        # CHANGE 10: also save per-horizon feature_cols_<h>h.json
        # inference_pipeline.py will prefer this over the global one
        horizon_cols_path = str(MODELS_DIR / f"feature_cols_{h}h.json")
        with open(horizon_cols_path, "w") as fh:
            json.dump(result["horizon_feature_cols"], fh)
        logger.info("Saved per-horizon feature cols (%dh): %s", h, horizon_cols_path)

    plot_horizon_comparison(all_results)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY — Best model per horizon")
    print("=" * 70)
    print(f"  {'Horizon':<10} {'Best model':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("  " + "-" * 63)
    for r in all_results:
        print(f"  {str(r['horizon_h'])+'h':<10} {r['best_model']:<25} "
              f"{r['rmse']:>8.2f} {r['mae']:>8.2f} {r['r2']:>8.4f}")

    print("\n  Saved model files:")
    for h in HORIZONS:
        for fname in [
            f"random_forest_model_{h}h.pkl",
            f"ridge_model_{h}h.pkl",
            f"xgb_model_{h}h.pkl",
            f"lstm_model_{h}h.keras",
            f"scaler_X_{h}h.pkl",
            f"scaler_y_{h}h.pkl",
            f"feature_cols_{h}h.json",   # CHANGE 10: per-horizon feature list
        ]:
            fpath = MODELS_DIR / fname
            tag = "OK     " if fpath.exists() else "MISSING"
            print(f"    [{tag}]  models/{fname}")

    # ── CHANGE 4: Upload ONLY the best model per horizon to Hopsworks ─────────
    if USE_HOPSWORKS and HOPSWORKS_API_KEY != "YOUR_KEY_HERE":
        print("\n  Uploading best model per horizon to Hopsworks Model Registry...")
        for r in all_results:
            h      = r["horizon_h"]
            suffix = f"_{h}h"

            rmse_scores = {
                "rf":    r["rf_rmse"],
                "lstm":  r["lstm_rmse"],
                "ridge": r["ridge_rmse"],
                "xgb":   r["xgb_rmse"],
            }
            winner = min(rmse_scores, key=rmse_scores.get)
            logger.info("Horizon %dh winner: %s (RMSE=%.2f)", h, winner, rmse_scores[winner])

            if winner == "rf":
                best_path = str(MODELS_DIR / f"random_forest_model{suffix}.pkl")
                extra     = []
            elif winner == "xgb":
                best_path = str(MODELS_DIR / f"xgb_model{suffix}.pkl")
                extra     = []
            elif winner == "lstm":
                best_path = str(MODELS_DIR / f"lstm_model{suffix}.keras")
                extra     = [str(MODELS_DIR / f"scaler_X{suffix}.pkl"),
                              str(MODELS_DIR / f"scaler_y{suffix}.pkl")]
            else:
                best_path = str(MODELS_DIR / f"ridge_model{suffix}.pkl")
                extra     = []

            model_name = f"lahore_aqi_best_{h}h"
            save_best_to_hopsworks(
                model_path   = best_path,
                model_name   = model_name,
                metrics      = {"rmse": r["rmse"], "mae": r["mae"], "r2": r["r2"]},
                feature_cols = r["horizon_feature_cols"],
                horizon_h    = h,
                extra_files  = extra,
            )
    else:
        print("\n  Hopsworks upload skipped (USE_HOPSWORKS=false or no API key).")
        print(f"  All models saved locally in: {MODELS_DIR}")

    print("\n" + "=" * 70)
    print("  Training pipeline complete! Next: run inference_pipeline.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
