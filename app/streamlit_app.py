"""
streamlit_app.py
================
CHANGES FROM YOUR ORIGINAL:
  CHANGE 1 (load_predictions) — Hopsworks is now PRIMARY source for
            predictions. Local CSV is FALLBACK only for dev.
            Previously local CSV was primary — wrong per project spec.
            Project page 7: "Loads the model and features from the
            Feature Store" — everything comes from Hopsworks.

  CHANGE 2 (load_historical) — Same priority flip. Hopsworks first,
            local CSV only as fallback.

  CHANGE 3 (show_model_info sidebar) — New: shows which model version
            is registered in Hopsworks Model Registry so users and
            your mentor can see it's actually using the registry.

No other logic changed — dashboard layout, gauges, charts all identical.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

_APP_DIR      = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
_FEATURES_DIR = _PROJECT_ROOT / "features"
_DATA_DIR     = _PROJECT_ROOT / "data"

sys.path.insert(0, str(_FEATURES_DIR))

LOCAL_PRED_CSV = str(_DATA_DIR / "predictions.csv")
LOCAL_FEAT_CSV = str(_PROJECT_ROOT / "lahore_features.csv")

st.set_page_config(
    page_title="Lahore AQI Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

USE_HOPSWORKS     = os.environ.get("USE_HOPSWORKS", "true").lower() == "true"
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY", "")
PRED_GROUP_NAME   = "lahore_aqi_predictions"
FEAT_GROUP_NAME   = "lahore_aqi_features"

AQI_BANDS = [
    (0,   50,  "Good",                    "#00e400", "#1a1a1a"),
    (51,  100, "Moderate",                "#ffff00", "#1a1a1a"),
    (101, 150, "Unhealthy for Sensitive", "#ff7e00", "#ffffff"),
    (151, 200, "Unhealthy",               "#ff0000", "#ffffff"),
    (201, 300, "Very Unhealthy",          "#8f3f97", "#ffffff"),
    (301, 500, "Hazardous",               "#7e0023", "#ffffff"),
]

HEALTH_ADVICE = {
    "Good":                    "Air quality is satisfactory. Enjoy outdoor activities! ✅",
    "Moderate":                "Unusually sensitive people should consider limiting prolonged outdoor activity. 🟡",
    "Unhealthy for Sensitive": "Sensitive groups should limit outdoor exertion. 🟠",
    "Unhealthy":               "Everyone may experience health effects. Limit outdoor activity. 🔴",
    "Very Unhealthy":          "Health alert! Avoid all outdoor activity. Keep windows closed. 🟣",
    "Hazardous":               "EMERGENCY: Remain indoors. Wear N95 if you must go outside. ⚫",
}


def classify_aqi(value: float):
    for lo, hi, label, bg, fg in AQI_BANDS:
        if lo <= value <= hi:
            return label, bg, fg
    return "Unknown", "#cccccc", "#000000"


# ── CHANGE 1: load_predictions — Hopsworks PRIMARY, local CSV fallback ────────

@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame:
    """
    Load predictions from Hopsworks Feature Store (primary).
    Project page 7: web app loads features and model from Feature Store.
    Local CSV used only as fallback when Hopsworks is unavailable.
    """
    # PRIMARY: Hopsworks Feature Store
    if USE_HOPSWORKS and HOPSWORKS_API_KEY:
        try:
            import hopsworks
            project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
            fs  = project.get_feature_store()
            fg  = fs.get_feature_group(name=PRED_GROUP_NAME, version=2)
            df  = fg.read()
            df["forecast_created_utc"] = pd.to_datetime(df["forecast_created_utc"])
            df  = df.sort_values("forecast_created_utc").tail(168)
            return df
        except Exception as exc:
            st.warning(f"⚠️ Hopsworks predictions unavailable ({exc}). Trying local CSV...")

    # FALLBACK: local CSV (dev/offline only)
    if os.path.exists(LOCAL_PRED_CSV):
        df = pd.read_csv(LOCAL_PRED_CSV)
        df["forecast_created_utc"] = pd.to_datetime(df["forecast_created_utc"])
        return df.sort_values("forecast_created_utc").tail(168)

    return pd.DataFrame()


# ── CHANGE 2: load_historical — Hopsworks PRIMARY, local CSV fallback ─────────

@st.cache_data(ttl=1800)
def load_historical() -> pd.DataFrame:
    """Load last 7 days of hourly feature data — Hopsworks primary."""
    # PRIMARY: Hopsworks
    if USE_HOPSWORKS and HOPSWORKS_API_KEY:
        try:
            import hopsworks
            project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
            fs  = project.get_feature_store()
            fg  = fs.get_feature_group(name=FEAT_GROUP_NAME, version=1)
            df  = fg.read()
            df  = df.sort_values("timestamp").tail(7 * 24)
            return df
        except Exception:
            pass

    # FALLBACK: local CSV
    if os.path.exists(LOCAL_FEAT_CSV):
        df = pd.read_csv(LOCAL_FEAT_CSV)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").tail(7 * 24)

    return pd.DataFrame()


@st.cache_data(ttl=1800)
def get_live_aqi_data() -> dict:
    try:
        from live_aqi_client import fetch_live_aqi
        return fetch_live_aqi()
    except Exception as exc:
        return {
            "aqi": None, "pm25": None, "pm10": None,
            "no2": None, "so2": None, "co": None, "o3": None, "dust": None,
            "stations_used": 0, "error": str(exc),
        }


# ── CHANGE 3: show model registry info in sidebar ─────────────────────────────

def show_model_registry_info():
    """
    Show which model versions are registered in Hopsworks Model Registry.
    Demonstrates to your mentor that models flow from training → registry → app.
    """
    if not (USE_HOPSWORKS and HOPSWORKS_API_KEY):
        st.sidebar.info("Hopsworks not configured.")
        return
    try:
        import hopsworks
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        mr = project.get_model_registry()
        st.sidebar.markdown("### Model Registry")
        for h in [24, 48, 72]:
            try:
                m = mr.get_model(name=f"lahore_aqi_best_{h}h", version=1)
                metrics = m.training_metrics or {}
                rmse = metrics.get("rmse", "—")
                r2   = metrics.get("r2",   "—")
                st.sidebar.success(f"**{h}h model** v{m.version}  \nRMSE: {rmse}  R²: {r2}")
            except Exception:
                st.sidebar.warning(f"{h}h model — not registered yet")
    except Exception as exc:
        st.sidebar.error(f"Registry unavailable: {exc}")


# ── Chart builders (unchanged from your original) ─────────────────────────────

def make_aqi_gauge(value: float, title: str = "AQI") -> go.Figure:
    label, bg, _ = classify_aqi(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={"text": title, "font": {"size": 18}},
        delta={"reference": 100,
               "increasing": {"color": "#ff4444"},
               "decreasing": {"color": "#44bb44"}},
        gauge={
            "axis": {"range": [0, 300], "tickwidth": 1},
            "bar":  {"color": bg, "thickness": 0.3},
            "steps": [
                {"range": [0,   50],  "color": "rgba(0,228,0,0.2)"},
                {"range": [51,  100], "color": "rgba(255,255,0,0.2)"},
                {"range": [101, 150], "color": "rgba(255,126,0,0.2)"},
                {"range": [151, 200], "color": "rgba(255,0,0,0.2)"},
                {"range": [201, 300], "color": "rgba(143,63,151,0.2)"},
            ],
            "threshold": {"line": {"color": bg, "width": 4},
                          "thickness": 0.75, "value": value},
        },
        number={"font": {"size": 48, "color": bg}},
    ))
    fig.update_layout(
        height=280, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#ffffff"},
    )
    return fig


def make_forecast_chart(df_pred: pd.DataFrame, df_hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not df_hist.empty and "us_aqi" in df_hist.columns:
        fig.add_trace(go.Scatter(
            x=df_hist["timestamp"], y=df_hist["us_aqi"],
            name="Historical AQI", line=dict(color="#4a9eff", width=2), mode="lines",
        ))
    if not df_pred.empty:
        latest = df_pred.iloc[-1]
        now    = pd.to_datetime(latest["forecast_created_utc"])
        future_times = [now + pd.Timedelta(hours=h) for h in [24, 48, 72]]
        future_vals  = [float(latest.get(f"pred_aqi_{h}h", 0)) for h in [24, 48, 72]]
        colours      = [classify_aqi(v)[1] for v in future_vals]
        fig.add_trace(go.Scatter(
            x=future_times, y=future_vals, name="Forecast AQI",
            mode="lines+markers", line=dict(color="#ff9f43", width=2, dash="dash"),
            marker=dict(size=12, color=colours, line=dict(width=2, color="#ffffff")),
        ))
        fig.add_vline(x=now.timestamp() * 1000, line_dash="dot",
                      line_color="rgba(255,255,255,0.4)",
                      annotation_text="Now", annotation_position="top left")
    for threshold, label, colour in [(150, "Unhealthy", "#ff0000"),
                                      (200, "Very Unhealthy", "#8f3f97")]:
        fig.add_hline(y=threshold, line_dash="dash", line_color=colour,
                      line_width=1, annotation_text=label, annotation_position="right")
    fig.update_layout(
        title="AQI — Historical (7 days) + 72-Hour Forecast",
        xaxis_title="Time (PKT)", yaxis_title="US AQI",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.04)",
        font={"color": "#ffffff"},
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15),
        height=380, margin=dict(l=40, r=40, t=50, b=60),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    )
    return fig


def make_prediction_history_chart(df_pred: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df_pred.empty:
        return fig
    fig.add_trace(go.Scatter(
        x=df_pred["forecast_created_utc"], y=df_pred["pred_aqi_24h"],
        name="24h Prediction", line=dict(color="#ff9f43", width=2),
    ))
    if "live_aqi" in df_pred.columns:
        fig.add_trace(go.Scatter(
            x=df_pred["forecast_created_utc"], y=df_pred["live_aqi"],
            name="Live AQI at prediction time",
            line=dict(color="#4a9eff", width=1, dash="dot"),
        ))
    fig.update_layout(
        title="24h Forecast vs Live AQI Over Time",
        xaxis_title="Prediction created at (UTC)", yaxis_title="AQI",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.04)",
        font={"color": "#ffffff"}, height=280, margin=dict(l=40, r=20, t=50, b=40),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    )
    return fig


def make_pollutant_bar(live_data: dict) -> go.Figure:
    pollutants = {
        "PM2.5 (µg/m³)": live_data.get("pm25"),
        "PM10 (µg/m³)":  live_data.get("pm10"),
        "NO₂ (µg/m³)":   live_data.get("no2"),
        "SO₂ (µg/m³)":   live_data.get("so2"),
        "O₃ (µg/m³)":    live_data.get("o3"),
        "Dust (µg/m³)":  live_data.get("dust"),
    }
    labels = [k for k, v in pollutants.items() if v is not None]
    values = [pollutants[k] for k in labels]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color="#4a9eff",
        text=[f"{v:.1f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title="Current Pollutant Levels",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.04)",
        font={"color": "#ffffff"}, height=300, margin=dict(l=20, r=20, t=50, b=60),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"), showlegend=False,
    )
    return fig


# ── CSS (unchanged) ───────────────────────────────────────────────────────────

st.markdown("""
<style>
body, .stApp { background-color: #0d1117; color: #c9d1d9; }
.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 20px 24px; margin-bottom: 12px;
}
.aqi-badge {
    display: inline-block; padding: 6px 14px; border-radius: 20px;
    font-weight: 700; font-size: 0.85rem; letter-spacing: 0.05em; margin-top: 4px;
}
.section-header {
    font-size: 0.7rem; letter-spacing: 0.15em; text-transform: uppercase;
    color: #8b949e; margin-bottom: 4px;
}
h1 { color: #e6edf3 !important; }
.stPlotlyChart { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Main dashboard ────────────────────────────────────────────────────────────

def main():
    col_title, col_refresh = st.columns([4, 1])
    with col_title:
        st.markdown("# 🌫️ Lahore AQI Forecast")
        from zoneinfo import ZoneInfo
        now_pkt = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Karachi"))
        st.markdown(
            f"<span style='color:#8b949e; font-size:0.85rem;'>Last updated: "
            f"{now_pkt.strftime('%Y-%m-%d %H:%M PKT')}</span>",
            unsafe_allow_html=True,
        )
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # CHANGE 3: show model registry info in sidebar
    with st.sidebar:
        show_model_registry_info()

    st.divider()

    with st.spinner("Loading data from Hopsworks..."):
        live_data = get_live_aqi_data()
        df_pred   = load_predictions()    # CHANGE 1: from Hopsworks
        df_hist   = load_historical()     # CHANGE 2: from Hopsworks

    if live_data.get("error") and live_data.get("aqi") is None:
        st.warning(f"⚠️ Could not fetch live AQI: {live_data['error']}")

    live_aqi = float(live_data.get("aqi") or 0.0)
    live_label, live_bg, live_fg = classify_aqi(live_aqi)

    # Row 1: Live AQI + 3 forecast gauges
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="section-header">Live City AQI (Open-Meteo)</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:3.5rem; font-weight:800; color:{live_bg};'>"
            f"{live_aqi:.0f}</div>"
            f"<div class='aqi-badge' style='background:{live_bg}; color:{live_fg};'>"
            f"{live_label}</div>",
            unsafe_allow_html=True,
        )
        data_hour = live_data.get("data_hour_local", "")
        st.markdown(
            f"<div style='color:#8b949e; font-size:0.75rem; margin-top:8px;'>"
            f"Data hour (PKT): {data_hour}</div>",
            unsafe_allow_html=True,
        )

    if not df_pred.empty:
        latest = df_pred.iloc[-1]
        for col, hours, key in [
            (col2, 24, "pred_aqi_24h"),
            (col3, 48, "pred_aqi_48h"),
            (col4, 72, "pred_aqi_72h"),
        ]:
            val = float(latest.get(key, 0))
            with col:
                st.plotly_chart(make_aqi_gauge(val, f"{hours}h Forecast"),
                                use_container_width=True)
    else:
        for col, hours in [(col2, 24), (col3, 48), (col4, 72)]:
            with col:
                st.info(f"No {hours}h forecast yet.\nRun inference_pipeline.py first.")

    # Health advice banner
    advice = HEALTH_ADVICE.get(live_label, "")
    if advice:
        colour_map = {
            "Good": "success", "Moderate": "warning",
            "Unhealthy for Sensitive": "warning", "Unhealthy": "error",
            "Very Unhealthy": "error", "Hazardous": "error",
        }
        level = colour_map.get(live_label, "info")
        getattr(st, level)(f"**Health Advice:** {advice}")

    # Forecast chart
    st.plotly_chart(make_forecast_chart(df_pred, df_hist), use_container_width=True)

    # Row 2: Pollutants + prediction history
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### 🧪 Current Pollutant Breakdown")
        if live_data.get("pm25") is not None:
            st.plotly_chart(make_pollutant_bar(live_data), use_container_width=True)
            pm25 = live_data.get("pm25") or 0.0
            dust = live_data.get("dust") or 0.0
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"<div class='metric-card'><div class='section-header'>PM2.5</div>"
                    f"<span style='font-size:1.8rem; font-weight:700; color:#ff7e00;'>"
                    f"{pm25:.1f}</span> µg/m³</div>", unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"<div class='metric-card'><div class='section-header'>Dust</div>"
                    f"<span style='font-size:1.8rem; font-weight:700; color:#ff9f43;'>"
                    f"{dust:.1f}</span> µg/m³</div>", unsafe_allow_html=True,
                )
        else:
            st.info("Pollutant data unavailable.")

    with col_right:
        st.markdown("### 📈 Prediction History")
        if not df_pred.empty:
            st.plotly_chart(make_prediction_history_chart(df_pred), use_container_width=True)
        else:
            st.info("No prediction history yet.\nRun inference_pipeline.py to populate this.")

    # AQI reference guide
    with st.expander("📋 AQI Reference Guide"):
        for lo, hi, label, bg, fg in AQI_BANDS:
            advice_text = HEALTH_ADVICE.get(label, "")
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:16px; padding:8px 0; "
                f"border-bottom: 1px solid #21262d;'>"
                f"<div class='aqi-badge' style='background:{bg}; color:{fg}; "
                f"min-width:60px; text-align:center;'>{lo}–{hi}</div>"
                f"<div><strong style='color:{bg};'>{label}</strong><br>"
                f"<span style='font-size:0.82rem; color:#8b949e;'>{advice_text}</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown(
        "<div style='text-align:center; color:#8b949e; font-size:0.75rem;'>"
        "Lahore AQI Predictor &nbsp;•&nbsp; Data: Open-Meteo &nbsp;•&nbsp; "
        "Models: Hopsworks Model Registry &nbsp;•&nbsp; Refreshes every 30 min"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<script>setTimeout(() => window.location.reload(), 1800000);</script>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()