

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
    (0,   50,  "Good",                    "#22c55e", "#ffffff"),
    (51,  100, "Moderate",                "#eab308", "#ffffff"),
    (101, 150, "Unhealthy for Sensitive", "#f97316", "#ffffff"),
    (151, 200, "Unhealthy",               "#ef4444", "#ffffff"),
    (201, 300, "Very Unhealthy",          "#a855f7", "#ffffff"),
    (301, 500, "Hazardous",               "#7f1d1d", "#ffffff"),
]

HEALTH_ADVICE = {
    "Good":                    "Air quality is satisfactory. Enjoy outdoor activities!",
    "Moderate":                "Unusually sensitive people should consider limiting prolonged outdoor activity.",
    "Unhealthy for Sensitive": "Sensitive groups should limit outdoor exertion.",
    "Unhealthy":               "Everyone may experience health effects. Limit outdoor activity.",
    "Very Unhealthy":          "Health alert! Avoid all outdoor activity. Keep windows closed.",
    "Hazardous":               "EMERGENCY: Remain indoors. Wear N95 if you must go outside.",
}


def classify_aqi(value: float):
    for lo, hi, label, bg, fg in AQI_BANDS:
        if lo <= value <= hi:
            return label, bg, fg
    return "Unknown", "#94a3b8", "#ffffff"


# ── CHANGE 1: load_predictions — Hopsworks PRIMARY, local CSV fallback ────────

@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame:
    """
    Load predictions from Hopsworks Feature Store (primary).
    Project page 7: web app loads features and model from Feature Store.
    Local CSV used only as fallback when Hopsworks is unavailable.
    """
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

    if os.path.exists(LOCAL_PRED_CSV):
        df = pd.read_csv(LOCAL_PRED_CSV)
        df["forecast_created_utc"] = pd.to_datetime(df["forecast_created_utc"])
        return df.sort_values("forecast_created_utc").tail(168)

    return pd.DataFrame()


# ── CHANGE 2: load_historical — Hopsworks PRIMARY, local CSV fallback ─────────

@st.cache_data(ttl=1800)
def load_historical() -> pd.DataFrame:
    """Load last 7 days of hourly feature data — Hopsworks primary."""
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


# ── Chart builders — light theme versions ─────────────────────────────────────

def make_aqi_gauge(value: float, title: str = "AQI") -> go.Figure:
    label, bg, _ = classify_aqi(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 13, "color": "#64748b", "family": "DM Sans"}},
        gauge={
            "axis": {"range": [0, 300], "tickwidth": 1, "tickcolor": "#cbd5e1",
                     "tickfont": {"size": 9, "color": "#94a3b8"}},
            "bar":  {"color": bg, "thickness": 0.25},
            "bgcolor": "#f8fafc",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   50],  "color": "#dcfce7"},
                {"range": [51,  100], "color": "#fef9c3"},
                {"range": [101, 150], "color": "#ffedd5"},
                {"range": [151, 200], "color": "#fee2e2"},
                {"range": [201, 300], "color": "#f3e8ff"},
            ],
            "threshold": {"line": {"color": bg, "width": 3},
                          "thickness": 0.75, "value": value},
        },
        number={"font": {"size": 36, "color": bg, "family": "DM Sans"}, "suffix": ""},
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1e293b", "family": "DM Sans"},
    )
    return fig


def make_forecast_chart(df_pred: pd.DataFrame, df_hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not df_hist.empty and "us_aqi" in df_hist.columns:
        fig.add_trace(go.Scatter(
            x=df_hist["timestamp"], y=df_hist["us_aqi"],
            name="Historical AQI",
            line=dict(color="#3b82f6", width=2),
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.07)",
        ))
    if not df_pred.empty:
        latest = df_pred.iloc[-1]
        now    = pd.to_datetime(latest["forecast_created_utc"])
        future_times = [now + pd.Timedelta(hours=h) for h in [24, 48, 72]]
        future_vals  = [float(latest.get(f"pred_aqi_{h}h", 0)) for h in [24, 48, 72]]
        colours      = [classify_aqi(v)[1] for v in future_vals]
        fig.add_trace(go.Scatter(
            x=future_times, y=future_vals, name="Forecast AQI",
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2, dash="dash"),
            marker=dict(size=10, color=colours, line=dict(width=2, color="#ffffff")),
        ))
        fig.add_vline(x=now.timestamp() * 1000, line_dash="dot",
                      line_color="#94a3b8",
                      annotation_text="Now",
                      annotation_font_color="#64748b",
                      annotation_position="top left")
    for threshold, label, colour in [(150, "Unhealthy", "#ef4444"),
                                      (200, "Very Unhealthy", "#a855f7")]:
        fig.add_hline(y=threshold, line_dash="dash", line_color=colour,
                      line_width=1,
                      annotation_text=label,
                      annotation_font_color=colour,
                      annotation_position="right")
    fig.update_layout(
        title=dict(text="AQI — Historical (7 days) + 72-Hour Forecast",
                   font=dict(size=14, color="#1e293b", family="DM Sans"),
                   x=0),
        xaxis_title="Time (PKT)",
        yaxis_title="US AQI",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        font={"color": "#475569", "family": "DM Sans"},
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18,
                    font=dict(color="#475569")),
        height=340,
        margin=dict(l=40, r=40, t=50, b=60),
        yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0", color="#94a3b8"),
        xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0", color="#94a3b8"),
    )
    return fig


def make_prediction_history_chart(df_pred: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df_pred.empty:
        return fig
    fig.add_trace(go.Scatter(
        x=df_pred["forecast_created_utc"], y=df_pred["pred_aqi_24h"],
        name="24h Prediction",
        line=dict(color="#f59e0b", width=2),
        fill="tozeroy",
        fillcolor="rgba(245,158,11,0.07)",
    ))
    if "live_aqi" in df_pred.columns:
        fig.add_trace(go.Scatter(
            x=df_pred["forecast_created_utc"], y=df_pred["live_aqi"],
            name="Live AQI at prediction time",
            line=dict(color="#3b82f6", width=1.5, dash="dot"),
        ))
    fig.update_layout(
        title=dict(text="24h Forecast vs Live AQI Over Time",
                   font=dict(size=13, color="#1e293b", family="DM Sans"), x=0),
        xaxis_title="Prediction created at (UTC)",
        yaxis_title="AQI",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        font={"color": "#475569", "family": "DM Sans"},
        height=260,
        margin=dict(l=40, r=20, t=50, b=40),
        yaxis=dict(gridcolor="#e2e8f0", color="#94a3b8"),
        xaxis=dict(gridcolor="#e2e8f0", color="#94a3b8"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#475569")),
    )
    return fig


def make_pollutant_bar(live_data: dict) -> go.Figure:
    pollutants = {
        "PM2.5": live_data.get("pm25"),
        "PM10":  live_data.get("pm10"),
        "NO₂":   live_data.get("no2"),
        "SO₂":   live_data.get("so2"),
        "O₃":    live_data.get("o3"),
        "Dust":  live_data.get("dust"),
    }
    labels = [k for k, v in pollutants.items() if v is not None]
    values = [pollutants[k] for k in labels]
    bar_colors = ["#3b82f6", "#06b6d4", "#8b5cf6", "#f59e0b", "#22c55e", "#f97316"]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=bar_colors[:len(labels)],
        marker_line_width=0,
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
        textfont=dict(color="#475569", size=11, family="DM Sans"),
    ))
    fig.update_layout(
        title=dict(text="Current Pollutant Levels (µg/m³)",
                   font=dict(size=13, color="#1e293b", family="DM Sans"), x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        font={"color": "#475569", "family": "DM Sans"},
        height=270,
        margin=dict(l=20, r=20, t=50, b=40),
        yaxis=dict(gridcolor="#e2e8f0", color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", color="#475569"),
        showlegend=False,
    )
    return fig


# ── CSS — Light AeroForecast-style theme ──────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: #f1f5f9 !important;
    color: #1e293b !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stHeader"] {
    background-color: #0f172a !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.stDeployButton { display: none; }

/* ── Main content padding ── */
[data-testid="stAppViewBlockContainer"] {
    padding: 0 !important;
    max-width: 100% !important;
}
.block-container {
    padding: 0 2rem 2rem 2rem !important;
    max-width: 100% !important;
}

/* ── Top nav bar ── */
.nav-bar {
    background: #0f172a;
    padding: 0 2rem;
    height: 52px;
    display: flex;
    align-items: center;
    gap: 2rem;
    margin: 0 -2rem 2rem -2rem;
    position: sticky;
    top: 0;
    z-index: 100;
    border-bottom: 1px solid #1e293b;
}
.nav-brand {
    font-size: 0.95rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.02em;
}
.nav-brand span { color: #3b82f6; }
.nav-links {
    display: flex;
    gap: 1.5rem;
    margin-left: 1rem;
}
.nav-link {
    font-size: 0.8rem;
    color: #94a3b8;
    font-weight: 500;
    cursor: pointer;
    transition: color 0.15s;
}
.nav-link:hover, .nav-link.active { color: #f8fafc; }
.nav-right {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.nav-location {
    font-size: 0.78rem;
    color: #94a3b8;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Page hero ── */
.page-hero {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 1.5rem;
}
.page-hero-left h1 {
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    letter-spacing: -0.04em !important;
    margin: 0 0 0.2rem 0 !important;
    line-height: 1.1 !important;
}
.page-hero-left .subtitle {
    font-size: 0.8rem;
    color: #94a3b8;
    font-weight: 400;
}
.hero-aqi-badge {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: #0f172a;
    border-radius: 12px;
    padding: 0.75rem 1.25rem;
}
.hero-aqi-value {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    font-family: 'DM Mono', monospace;
}
.hero-aqi-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.15rem;
}
.hero-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 4px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Metric cards (forecast horizon row) ── */
.metric-row-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0;
    transition: box-shadow 0.2s, transform 0.2s;
}
.metric-row-card:hover {
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    transform: translateY(-1px);
}
.metric-card-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.5rem;
}
.metric-card-value {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    font-family: 'DM Mono', monospace;
    line-height: 1;
    margin-bottom: 0.35rem;
}
.metric-aqi-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Section cards (content panels) ── */
.section-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
}
.section-card-title {
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Pollutant mini-cards ── */
.pollutant-mini {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    text-align: center;
}
.pollutant-mini-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.25rem;
}
.pollutant-mini-value {
    font-size: 1.3rem;
    font-weight: 800;
    font-family: 'DM Mono', monospace;
    letter-spacing: -0.02em;
}
.pollutant-mini-unit {
    font-size: 0.65rem;
    color: #94a3b8;
    font-weight: 400;
}

/* ── AQI Reference table ── */
.aqi-ref-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid #f1f5f9;
}
.aqi-ref-row:last-child { border-bottom: none; }
.aqi-ref-badge {
    min-width: 72px;
    text-align: center;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.02em;
}
.aqi-ref-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #1e293b;
    min-width: 190px;
}
.aqi-ref-advice {
    font-size: 0.78rem;
    color: #64748b;
    font-weight: 400;
}

/* ── Health advice banner ── */
.health-banner {
    border-radius: 10px;
    padding: 0.75rem 1.25rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 0.85rem;
    font-weight: 500;
    border-left: 4px solid;
}
.health-banner.good    { background: #f0fdf4; border-color: #22c55e; color: #166534; }
.health-banner.moderate { background: #fefce8; border-color: #eab308; color: #713f12; }
.health-banner.warning { background: #fff7ed; border-color: #f97316; color: #7c2d12; }
.health-banner.danger  { background: #fef2f2; border-color: #ef4444; color: #7f1d1d; }
.health-banner.critical{ background: #fdf4ff; border-color: #a855f7; color: #581c87; }

/* ── Divider ── */
hr { border-color: #e2e8f0 !important; margin: 1.5rem 0 !important; }

/* ── Plotly chart containers ── */
.stPlotlyChart {
    border-radius: 10px;
    overflow: hidden;
}

/* ── Streamlit component overrides ── */
.stButton > button {
    background: #0f172a !important;
    color: #f8fafc !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    padding: 0.4rem 1rem !important;
    transition: background 0.15s !important;
}
.stButton > button:hover {
    background: #1e293b !important;
}

[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: #1e293b !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}

.stSpinner > div { color: #3b82f6 !important; }
.stWarning, .stInfo { border-radius: 10px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Footer ── */
.app-footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.72rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


# ── Main dashboard ────────────────────────────────────────────────────────────

def main():
    # ── Top navigation bar ──
    st.markdown("""
    <div class="nav-bar">
        <div class="nav-brand">Aero<span>Forecast</span></div>
        <div class="nav-links">
            <div class="nav-link active">Dashboard</div>
            <div class="nav-link">Forecast</div>
            <div class="nav-link">Insights</div>
        </div>
        <div class="nav-right">
            <div class="nav-location">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
                     stroke="#94a3b8" stroke-width="2">
                    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"/>
                    <circle cx="12" cy="10" r="3"/>
                </svg>
                Lahore, PK
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CHANGE 3: show model registry info in sidebar
    with st.sidebar:
        show_model_registry_info()

    # ── Load data ──
    with st.spinner("Loading data from Hopsworks..."):
        live_data = get_live_aqi_data()
        df_pred   = load_predictions()    # CHANGE 1: from Hopsworks
        df_hist   = load_historical()     # CHANGE 2: from Hopsworks

    if live_data.get("error") and live_data.get("aqi") is None:
        st.warning(f"⚠️ Could not fetch live AQI: {live_data['error']}")

    live_aqi = float(live_data.get("aqi") or 0.0)
    live_label, live_bg, live_fg = classify_aqi(live_aqi)
    data_hour = live_data.get("data_hour_local", "")

    from zoneinfo import ZoneInfo
    now_pkt = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Karachi"))

    # ── Page hero ──
    col_hero_l, col_hero_r = st.columns([3, 1])
    with col_hero_l:
        st.markdown(f"""
        <div class="page-hero-left">
            <h1>Lahore</h1>
            <div class="subtitle">
                <span class="hero-status-dot" style="background:{live_bg};"></span>
                Live AQI data · {now_pkt.strftime('%d %b %Y, %H:%M PKT')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_hero_r:
        st.markdown(f"""
        <div style="display:flex; justify-content:flex-end;">
            <div class="hero-aqi-badge">
                <div>
                    <div class="hero-aqi-label">Current AQI</div>
                    <div class="hero-aqi-value" style="color:{live_bg};">{live_aqi:.0f}</div>
                </div>
                <div>
                    <div style="margin-bottom:0.4rem;">
                        <span class="metric-aqi-pill"
                              style="background:{live_bg}22; color:{live_bg}; border:1px solid {live_bg}55;">
                            <span class="hero-status-dot" style="background:{live_bg}; width:6px; height:6px;"></span>
                            {live_label}
                        </span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        # Refresh button aligned right
        if st.button("↺ Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown("<div style='margin-bottom:1.25rem'></div>", unsafe_allow_html=True)

    # ── Health advice banner ──
    advice = HEALTH_ADVICE.get(live_label, "")
    if advice:
        banner_class_map = {
            "Good": "good", "Moderate": "moderate",
            "Unhealthy for Sensitive": "warning", "Unhealthy": "danger",
            "Very Unhealthy": "critical", "Hazardous": "critical",
        }
        icon_map = {
            "Good": "✅", "Moderate": "🟡",
            "Unhealthy for Sensitive": "🟠", "Unhealthy": "🔴",
            "Very Unhealthy": "🟣", "Hazardous": "⚫",
        }
        cls  = banner_class_map.get(live_label, "good")
        icon = icon_map.get(live_label, "")
        st.markdown(
            f'<div class="health-banner {cls}">{icon} <strong>{live_label}:</strong>&nbsp;{advice}</div>',
            unsafe_allow_html=True,
        )

    # ── Forecast metric cards ──
    if not df_pred.empty:
        latest = df_pred.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        card_data = [
            (col1, "Current AQI", live_aqi, live_label, live_bg),
        ]
        for col, hours, key in [
            (col2, 24, "pred_aqi_24h"),
            (col3, 48, "pred_aqi_48h"),
            (col4, 72, "pred_aqi_72h"),
        ]:
            val = float(latest.get(key, 0))
            lbl, clr, _ = classify_aqi(val)
            card_data.append((col, f"{hours}h Forecast", val, lbl, clr))

        for col, label, val, lbl, clr in card_data:
            with col:
                st.markdown(f"""
                <div class="metric-row-card">
                    <div class="metric-card-label">{label}</div>
                    <div class="metric-card-value" style="color:{clr};">{val:.0f}</div>
                    <span class="metric-aqi-pill"
                          style="background:{clr}18; color:{clr}; border:1px solid {clr}44;">
                        {lbl}
                    </span>
                </div>
                """, unsafe_allow_html=True)
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-row-card">
                <div class="metric-card-label">Current AQI</div>
                <div class="metric-card-value" style="color:{live_bg};">{live_aqi:.0f}</div>
                <span class="metric-aqi-pill"
                      style="background:{live_bg}18; color:{live_bg}; border:1px solid {live_bg}44;">
                    {live_label}
                </span>
            </div>
            """, unsafe_allow_html=True)
        for col, hours in [(col2, 24), (col3, 48), (col4, 72)]:
            with col:
                st.info(f"No {hours}h forecast.\nRun inference_pipeline.py first.")

    st.markdown("<div style='margin-bottom:1.25rem'></div>", unsafe_allow_html=True)

    # ── Two-column middle section ──
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Current Pollutant Breakdown
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-card-title">🧪 Current Pollutant Breakdown</div>', unsafe_allow_html=True)
        if live_data.get("pm25") is not None:
            st.plotly_chart(make_pollutant_bar(live_data), use_container_width=True)

            pm25 = live_data.get("pm25") or 0.0
            pm10 = live_data.get("pm10") or 0.0
            no2  = live_data.get("no2")  or 0.0
            so2  = live_data.get("so2")  or 0.0
            o3   = live_data.get("o3")   or 0.0
            dust = live_data.get("dust") or 0.0

            mini_cols = st.columns(3)
            mini_data = [
                ("PM2.5", pm25, "#3b82f6"),
                ("PM10",  pm10, "#06b6d4"),
                ("NO₂",   no2,  "#8b5cf6"),
                ("SO₂",   so2,  "#f59e0b"),
                ("O₃",    o3,   "#22c55e"),
                ("Dust",  dust, "#f97316"),
            ]
            for i, (name, val, clr) in enumerate(mini_data):
                with mini_cols[i % 3]:
                    st.markdown(f"""
                    <div class="pollutant-mini" style="border-top:3px solid {clr};">
                        <div class="pollutant-mini-label">{name}</div>
                        <div class="pollutant-mini-value" style="color:{clr};">{val:.1f}</div>
                        <div class="pollutant-mini-unit">µg/m³</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Pollutant data unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        # Prediction History
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-card-title">📈 Prediction History</div>', unsafe_allow_html=True)
        if not df_pred.empty:
            st.plotly_chart(make_prediction_history_chart(df_pred), use_container_width=True)
        else:
            st.info("No prediction history yet.\nRun inference_pipeline.py to populate.")

        # Sub-location AQI summary (static cards matching screenshot layout)
        st.markdown('<div style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="section-card-title" style="margin-bottom:0.5rem;">📍 Air Quality Summary</div>', unsafe_allow_html=True)
        pm25_val = live_data.get("pm25") or 0.0
        pm10_val = live_data.get("pm10") or 0.0
        stat_cols = st.columns(2)
        with stat_cols[0]:
            st.markdown(f"""
            <div class="pollutant-mini" style="border-top:3px solid #3b82f6; margin-bottom:0.5rem;">
                <div class="pollutant-mini-label">PM2.5 Index</div>
                <div class="pollutant-mini-value" style="color:#3b82f6;">{pm25_val:.0f}</div>
                <div class="pollutant-mini-unit">µg/m³ · live</div>
            </div>
            """, unsafe_allow_html=True)
        with stat_cols[1]:
            st.markdown(f"""
            <div class="pollutant-mini" style="border-top:3px solid #06b6d4; margin-bottom:0.5rem;">
                <div class="pollutant-mini-label">PM10 Index</div>
                <div class="pollutant-mini-value" style="color:#06b6d4;">{pm10_val:.0f}</div>
                <div class="pollutant-mini-unit">µg/m³ · live</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Full-width forecast chart ──
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.plotly_chart(make_forecast_chart(df_pred, df_hist), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── AQI Reference Guide ──
    with st.expander("📋 AQI Reference Guide"):
        for lo, hi, label, bg, fg in AQI_BANDS:
            advice_text = HEALTH_ADVICE.get(label, "")
            st.markdown(f"""
            <div class="aqi-ref-row">
                <div class="aqi-ref-badge" style="background:{bg}; color:{fg};">{lo}–{hi}</div>
                <div class="aqi-ref-label" style="color:{bg};">{label}</div>
                <div class="aqi-ref-advice">{advice_text}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ──
    st.markdown(f"""
    <div class="app-footer">
        AeroForecast &nbsp;·&nbsp; Lahore AQI Predictor &nbsp;·&nbsp;
        Data: Open-Meteo &nbsp;·&nbsp; Models: Hopsworks Model Registry &nbsp;·&nbsp;
        Refreshes every 30 min &nbsp;·&nbsp;
        Last updated: {now_pkt.strftime('%Y-%m-%d %H:%M PKT')}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<script>setTimeout(() => window.location.reload(), 1800000);</script>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
