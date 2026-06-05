# 🌫️ Lahore AQI Forecasting System

> A fully serverless, end-to-end Machine Learning system that predicts Air Quality Index (AQI) for Lahore, Pakistan at **24h, 48h, and 72h horizons** — with a live Streamlit dashboard, automated CI/CD pipelines, and a Hopsworks Feature Store backend.

---

## 🔗 Live Demo

| Resource | Link |
|----------|------|
| 🖥️ Live Dashboard | [ https://lahore-aqi-forecasting-system-vyn8ysdekqpafskbvubspd.streamlit.app]
| 📦 Feature Store | Hopsworks — `lahore_aqi_features` v3 |
| 🤖 Model Registry | Hopsworks — `lahore_aqi_best_24h/48h/72h` v1 |

---

## 📌 Project Overview

Lahore consistently ranks among the most polluted cities in the world, especially during the **winter smog season (October–February)** driven by crop burning, vehicle exhaust, and brick kilns. This project builds a complete MLOps pipeline to forecast AQI so residents can plan outdoor activities and take protective measures in advance.

**What makes this project unique:**
- 100% free and serverless stack — no paid APIs, no servers to manage
- Dust spike correction using EPA PM2.5 breakpoints (prevents overestimated AQI during sandstorms)
- Automated hourly feature pipeline + daily retraining via GitHub Actions
- Four model types compared per horizon: Ridge, Random Forest, XGBoost, LSTM

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                            │
│  Open-Meteo Air Quality API  +  Open-Meteo Weather API      │
│         (free, no API key, 2 years historical)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE PIPELINE                           │
│  backfill_open_meteo.py  →  feature_pipeline.py             │
│  • 53 engineered features (lag, rolling, cyclic, seasonal)  │
│  • Uploads to Hopsworks Feature Store v3                    │
│  • Runs every hour via GitHub Actions                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  TRAINING PIPELINE                          │
│  training_pipeline.py                                       │
│  • Trains Ridge / Random Forest / XGBoost / LSTM            │
│  • Per-horizon feature selection (drops short lags at 72h)  │
│  • Registers best model per horizon in Hopsworks Registry   │
│  • Runs every day at 2 AM UTC via GitHub Actions            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 INFERENCE PIPELINE                          │
│  inference_pipeline.py                                      │
│  • Loads best model from Hopsworks Model Registry           │
│  • Predicts AQI for next 24h / 48h / 72h                   │
│  • Stores predictions to Hopsworks Feature Store            │
│  • Runs every hour via GitHub Actions                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  STREAMLIT DASHBOARD                        │
│  streamlit_app.py                                           │
│  • Live AQI with dust correction                            │
│  • 24h / 48h / 72h forecast cards                          │
│  • Pollutant breakdown (PM2.5, PM10, NO₂, SO₂, O₃, Dust)  │
│  • 7-day historical chart + prediction history              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
aqi-predictor-lahore/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml    # runs every hour
│       └── training_pipeline.yml   # runs every day at 2 AM UTC
│
├── app/
│   └── streamlit_app.py            # live dashboard
│
├── data/
│   └── predictions.csv             # local fallback for predictions
│
├── features/
│   ├── backfill_open_meteo.py      # fetches 2 years historical data
│   └── live_aqi_client.py          # fetches current hour AQI + dust correction
│
├── notebooks/
│   └── eda_lahore_aqi.ipynb        # 13-section exploratory data analysis
│
├── pipelines/
│   ├── feature_pipeline.py         # engineers features + uploads to Hopsworks
│   ├── training_pipeline.py        # trains all models + registers best
│   ├── inference_pipeline.py       # predicts + stores results
│   └── models/                     # saved model files (.pkl, .keras, .json)
│
├── .env.example                    # environment variable template
├── requirements.txt
└── README.md
```

---

## 🔬 Feature Engineering

The feature pipeline engineers **53 columns** from raw hourly air quality and weather data:

| Feature Group | Examples | Why |
|--------------|----------|-----|
| **AQI Lags** | `aqi_lag_1h`, `aqi_lag_24h`, `aqi_lag_48h` | Past AQI is the strongest predictor of future AQI |
| **12h Lag** | `aqi_lag_12h` | Fills the 6h→24h gap; captures mid-day smog build-up |
| **PM2.5 Lag** | `pm2_5_lag_24h` | PM2.5 yesterday is the strongest winter smog predictor |
| **Direction** | `aqi_diff_24h` | Whether pollution is worsening or improving |
| **Weather Lags** | `wind_lag_6h`, `temp_lag_24h` | Wind disperses; cold traps pollution near ground |
| **Rolling Stats** | `aqi_rolling_mean_24h`, `aqi_rolling_std_6h` | Trend and volatility signals |
| **Change Rate** | `aqi_change_1h`, `aqi_pct_change_1h` | Rate of deterioration |
| **Cyclic Time** | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` | Prevents 23:00→00:00 discontinuity |
| **Seasonal Means** | `aqi_hour_mean`, `aqi_month_mean` | "January at 8am is typically AQI 220" |
| **Weather Rolling** | `precip_sum_24h`, `wind_rolling_mean_24h` | Rain cleans air; sustained wind disperses smog |
| **Targets** | `target_aqi_24h/48h/72h` | What we are predicting |

---

## 🤖 Models

Four model types are trained and compared for each forecast horizon:

| Model | 24h RMSE | 24h R² | Notes |
|-------|----------|--------|-------|
| **XGBoost** ✅ | 19.37 | 0.71 | Winner for all horizons |
| Random Forest | ~22 | ~0.65 | Strong but slower |
| LSTM | ~25 | ~0.60 | Best for long sequences |
| Ridge Regression | ~35 | ~0.40 | Baseline |

**Per-horizon feature selection:** Short-lag features (`aqi_lag_1h`, `aqi_lag_3h`) that are informative at 24h become noise at 72h. The pipeline automatically drops them per horizon to prevent overfitting.

**Per-horizon XGBoost hyperparameters:**
- 24h: `max_depth=6`, `n_estimators=500` — deep trees fine for near-term
- 48h: `max_depth=4`, `n_estimators=800` — shallower + more regularisation
- 72h: `max_depth=3`, `n_estimators=1000` — minimal depth, strong regularisation

---

## 🌡️ Dust Spike Correction

Open-Meteo's atmospheric model counts dust at all altitude levels. During dust storms this inflates the AQI far above what ground stations measure. This project corrects for this:

```python
# When dust > 150 µg/m³, recalculate AQI from PM2.5 using EPA breakpoints
if dust > DUST_SPIKE_THRESHOLD and pm25 is not None:
    final_aqi = pm25_to_us_aqi(pm25)   # ground-level conditions
```

This matches what IQAir and AirNow report during dust events.

---

## 📊 EDA Highlights

The `eda_lahore_aqi.ipynb` notebook covers 13 sections. Key findings:

- **Worst months:** November–January (smog season mean AQI ~280 vs summer mean ~90)
- **Peak hour:** 8:00 AM PKT — rush hour + temperature inversion
- **24h autocorrelation:** r = 0.82 — yesterday's AQI strongly predicts today's
- **Wind is the biggest AQI reducer** — r = −0.41 with AQI
- **PM2.5 is the dominant pollutant** — r = 0.96 with AQI
- Over **40% of hours** are in the Unhealthy range (AQI > 150)

---

## ⚙️ CI/CD Pipelines

### Hourly Feature Pipeline (`.github/workflows/feature_pipeline.yml`)
Runs every hour at `:00`:
1. Checks out repo on a fresh Ubuntu runner
2. Installs dependencies
3. Runs `feature_pipeline.py` — auto-fetches historical data if missing, engineers features, uploads to Hopsworks
4. Runs `inference_pipeline.py` — loads best model, predicts next 24/48/72h, stores to Hopsworks

### Daily Training Pipeline (`.github/workflows/training_pipeline.yml`)
Runs every day at 2:00 AM UTC (7:00 AM PKT):
1. Fetches latest 2 years of data from Open-Meteo
2. Re-engineers all features
3. Retrains all 4 model types × 3 horizons = 12 models
4. Registers the best model per horizon in Hopsworks Model Registry
5. Uploads model artifacts as GitHub Actions artifacts (7-day retention)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- A free [Hopsworks account](https://app.hopsworks.ai) — get your API key from Project Settings

### 1. Clone the repo
```bash
git clone https://github.com/your-username/aqi-predictor-lahore.git
cd aqi-predictor-lahore
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
```
Edit `.env`:
```
HOPSWORKS_API_KEY=your_key_here
USE_HOPSWORKS=true
```

### 4. Run the full pipeline locally
```bash
# Step 1: Fetch 2 years of historical data (one-time, ~30 seconds)
python features/backfill_open_meteo.py

# Step 2: Engineer features + upload to Hopsworks
python pipelines/feature_pipeline.py

# Step 3: Train all models + register best in Hopsworks
python pipelines/training_pipeline.py

# Step 4: Run inference + store predictions
python pipelines/inference_pipeline.py

# Step 5: Launch the dashboard
streamlit run app/streamlit_app.py
```

### 5. Set up GitHub Actions
Add these secrets to your GitHub repo (Settings → Secrets → Actions):
```
HOPSWORKS_API_KEY    your Hopsworks API key
```
The two workflow files will then run automatically on schedule.

---

## 🌐 Deploy to Streamlit Cloud

1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set main file to `app/streamlit_app.py`
4. Under **Advanced settings → Secrets**, add:
```toml
HOPSWORKS_API_KEY = "your_key_here"
USE_HOPSWORKS = "true"
```
5. Click **Deploy**

---

## 📦 Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Data source | [Open-Meteo](https://open-meteo.com) | Free, no API key, historical + forecast |
| Feature Store | [Hopsworks](https://hopsworks.ai) | Free tier, versioned feature groups |
| Model Registry | Hopsworks | Stores best model + feature schema together |
| ML models | scikit-learn, XGBoost, TensorFlow/Keras | Best-in-class for tabular + sequential data |
| Explainability | SHAP | Feature importance for each horizon |
| Dashboard | Streamlit | Fast Python-native dashboards |
| CI/CD | GitHub Actions | Free, runs on schedule |
| Hosting | Streamlit Cloud | Free tier, connects to GitHub |

---

## 📄 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HOPSWORKS_API_KEY` | Yes | Your Hopsworks project API key |
| `USE_HOPSWORKS` | No | Set to `false` to use local CSV fallback (default: `true`) |

---

## 🗂️ Key Design Decisions

**Why Open-Meteo instead of AQICN?**
AQICN ground stations go stale frequently. Open-Meteo provides consistent hourly data going back years, which is essential for training. Live AQI is fetched from Open-Meteo with a dust correction layer.

**Why feature group v3?**
v1 had a schema mismatch after adding seasonal features. v2 fixed column names. v3 is stable with the full 53-column schema.

**Why per-horizon feature selection?**
`aqi_lag_1h` tells you what happened 1 hour ago — useful at 24h but random noise at 72h. Removing short lags at longer horizons forces models to rely on seasonal means and 24h/48h lags — the actual strong signals.

**Why save `feature_cols.json` inside the model artifact?**
Models and feature schemas must stay in sync. If features are added after a model is registered, the local JSON grows but the registered model still expects the old column count. Saving the exact feature list inside the artifact guarantees they always match.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Open-Meteo](https://open-meteo.com) for the free air quality and weather API
- [Hopsworks](https://hopsworks.ai) for the free-tier Feature Store and Model Registry
- EPA AQI Technical Assistance Document (2018) for PM2.5 breakpoints
