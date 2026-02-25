"""
AI-Based Smart Energy Usage Prediction for Sustainable Living
=============================================================
Streamlit Web App  |  Harshal — Capstone Project

Run with:
    pip install streamlit plotly pandas numpy scikit-learn
    streamlit run app.py
"""

import os, zipfile, urllib.request, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

# ─────────────────────────────────────────────
#  PAGE CONFIG & GLOBAL STYLES
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Energy Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] { background: #F0F7FF; }
[data-testid="stSidebar"]          { background: #021B2E; }
[data-testid="stSidebar"] *        { color: #E2EAF4 !important; }
[data-testid="stSidebar"] .stRadio label span { color: #E2EAF4 !important; }
[data-testid="stSidebar"] hr       { border-color: #1C7293; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 16px 20px;
    border-left: 5px solid #02C39A;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
[data-testid="stMetricLabel"]  { font-weight: 600; color: #065A82; }
[data-testid="stMetricValue"]  { color: #021B2E; }
[data-testid="stMetricDelta"]  { font-size: 13px; }

/* ── Section headings ── */
.section-header {
    background: linear-gradient(90deg, #065A82, #1C7293);
    color: white;
    padding: 12px 22px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: 700;
    margin: 14px 0 18px 0;
    letter-spacing: 0.3px;
}

/* ── Stat card ── */
.stat-card {
    background: white;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    text-align: center;
}
.stat-card .value  { font-size: 32px; font-weight: 800; color: #065A82; }
.stat-card .label  { font-size: 13px; color: #64748B; margin-top: 4px; }
.stat-card .accent { font-size: 28px; margin-bottom: 6px; }

/* ── Peak / Low cards ── */
.peak-card {
    background: #FEE2E2; border-left: 6px solid #EF4444;
    padding: 16px 18px; border-radius: 10px; margin: 6px 0;
}
.low-card {
    background: #DCFCE7; border-left: 6px solid #22C55E;
    padding: 16px 18px; border-radius: 10px; margin: 6px 0;
}
.peak-card h4 { color: #991B1B; margin:0 0 8px 0; }
.low-card  h4 { color: #166534; margin:0 0 8px 0; }
.peak-card p, .low-card p { margin: 2px 0; font-size: 14px; color: #374151; }

/* ── Info box ── */
.info-box {
    background: #EBF5FB;
    border-left: 5px solid #1C7293;
    padding: 14px 18px;
    border-radius: 8px;
    margin: 12px 0;
    font-size: 14px;
    color: #1E293B;
}

/* ── Step pill ── */
.step-pill {
    display: inline-block;
    background: #065A82;
    color: white;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 13px;
    font-weight: 700;
    margin-right: 8px;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: white;
    border-radius: 10px;
    padding: 6px 8px;
    gap: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 700;
    font-size: 15px;
    color: #065A82 !important;
    background: transparent;
    border: none;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    background: #EBF5FB;
    color: #021B2E !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, #065A82, #1C7293) !important;
    color: white !important;
    box-shadow: 0 3px 10px rgba(6,90,130,0.3);
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    display: none;
}

/* ── Slider label ── */
[data-testid="stSlider"] label p {
    color: #065A82 !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

/* ── Checkbox label ── */
[data-testid="stCheckbox"] label p {
    color: #065A82 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* ── All widget labels on light pages ── */
[data-testid="stAppViewContainer"] label {
    color: #065A82 !important;
    font-weight: 600 !important;
}

/* ── Sub-headers inside pages ── */
[data-testid="stAppViewContainer"] h3 {
    color: #065A82 !important;
    font-weight: 700 !important;
}
[data-testid="stAppViewContainer"] h4 {
    color: #021B2E !important;
}

/* ── Bold text ── */
[data-testid="stAppViewContainer"] strong {
    color: #021B2E !important;
}

/* ── Caption text ── */
[data-testid="stCaptionContainer"] p {
    color: #475569 !important;
    font-size: 13px !important;
}

/* ── Dataframe header ── */
[data-testid="stDataFrame"] th {
    background: #065A82 !important;
    color: white !important;
    font-weight: 700 !important;
}

/* ── Tab content text ── */
[data-testid="stTabsContent"] p,
[data-testid="stTabsContent"] span,
[data-testid="stTabsContent"] div {
    color: #1E293B;
}

/* ── Hide streamlit footer ── */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA & MODEL PIPELINE  (cached)
# ─────────────────────────────────────────────
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
ZIP_PATH = "household_power_consumption.zip"
TXT_PATH = "household_power_consumption.txt"


@st.cache_data(show_spinner=False)
def load_data():
    """Download (if needed) and load the UCI dataset."""
    if not os.path.exists(TXT_PATH):
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(".")

    df = pd.read_csv(TXT_PATH, sep=";", low_memory=False, na_values=["?", "nan"])
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
    df = df.drop(columns=["Date", "Time"]).dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()

    # Keep last 365 days
    df = df.loc[df.index >= (df.index.max() - pd.Timedelta(days=365))].copy()
    return df


@st.cache_data(show_spinner=False)
def build_hourly(_df):
    hourly_raw  = _df["Global_active_power"].resample("1h").mean()
    full_idx    = pd.date_range(hourly_raw.index.min(), hourly_raw.index.max(), freq="h")
    hourly_full = hourly_raw.reindex(full_idx)
    hourly      = hourly_full.interpolate(limit=3, limit_direction="both")
    return hourly


@st.cache_data(show_spinner=False)
def build_features(_hourly):
    data = pd.DataFrame({"y": _hourly})
    data["hour"]         = data.index.hour
    data["dayofweek"]    = data.index.dayofweek
    data["month"]        = data.index.month
    data["is_weekend"]   = (data["dayofweek"] >= 5).astype(int)
    data["lag_1"]        = data["y"].shift(1)
    data["lag_24"]       = data["y"].shift(24)
    data["lag_168"]      = data["y"].shift(168)
    data["roll_mean_24"] = data["y"].rolling(24).mean()
    data["roll_std_24"]  = data["y"].rolling(24).std()
    data["roll_mean_168"]= data["y"].rolling(168).mean()
    data["target"]       = data["y"].shift(-1)
    return data.dropna()


@st.cache_resource(show_spinner=False)
def train_models(_data):
    X = _data.drop(columns=["y", "target"])
    y = _data["target"]
    split = int(len(_data) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Baseline
    y_base = X_test["lag_24"]
    mae_base  = mean_absolute_error(y_test, y_base)
    rmse_base = np.sqrt(mean_squared_error(y_test, y_base))

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_lr  = lr.predict(X_test)
    mae_lr  = mean_absolute_error(y_test, y_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_lr))

    # Gradient Boosting
    gbr = HistGradientBoostingRegressor(random_state=42)
    gbr.fit(X_train, y_train)
    y_gbr  = gbr.predict(X_test)
    mae_gbr  = mean_absolute_error(y_test, y_gbr)
    rmse_gbr = np.sqrt(mean_squared_error(y_test, y_gbr))

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_base": y_base, "y_lr": y_lr, "y_gbr": y_gbr,
        "mae_base": mae_base, "rmse_base": rmse_base,
        "mae_lr": mae_lr,   "rmse_lr": rmse_lr,
        "mae_gbr": mae_gbr, "rmse_gbr": rmse_gbr,
        "lr": lr, "gbr": gbr,
    }


@st.cache_data(show_spinner=False)
def make_forecast(_hourly, _X_train, _gbr, steps=24):
    y_hist = _hourly.copy()
    t = y_hist.index.max()
    preds = []
    for _ in range(steps):
        feats = {
            "hour":          t.hour,
            "dayofweek":     t.dayofweek,
            "month":         t.month,
            "is_weekend":    int(t.dayofweek >= 5),
            "lag_1":         float(y_hist.loc[t]),
            "lag_24":        float(y_hist.loc[t - pd.Timedelta(hours=24)]),
            "lag_168":       float(y_hist.loc[t - pd.Timedelta(hours=168)]),
            "roll_mean_24":  float(y_hist.loc[:t].tail(24).mean()),
            "roll_std_24":   float(y_hist.loc[:t].tail(24).std()),
            "roll_mean_168": float(y_hist.loc[:t].tail(168).mean()),
        }
        X_next = pd.DataFrame([feats])[_X_train.columns]
        y_next = _gbr.predict(X_next)[0]
        next_t = t + pd.Timedelta(hours=1)
        preds.append((next_t, y_next))
        y_hist.loc[next_t] = y_next
        t = next_t
    return pd.Series([v for _, v in preds], index=[ts for ts, _ in preds], name="Forecast_24h")


@st.cache_data(show_spinner=False)
def get_feature_importance(_gbr, _X_test, _y_test):
    imp = permutation_importance(_gbr, _X_test, _y_test, n_repeats=5, random_state=42)
    return pd.Series(imp.importances_mean, index=_X_test.columns).sort_values(ascending=False)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Smart Energy")
    st.markdown("### Prediction Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠  Home", "📊  Data Explorer", "🤖  Model Comparison",
         "🔮  24h Forecast", "🔍  Feature Importance"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Project:** Capstone — AI & ML")
    st.markdown("**Author:** Harshal")
    st.markdown("**Dataset:** UCI Household Power")
    st.markdown("---")
    st.markdown(
        '<div style="font-size:12px;color:#94A3B8;">Powered by Scikit-learn<br>& Streamlit</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("⚡ Loading & processing data..."):
    df      = load_data()
    hourly  = build_hourly(df)
    data    = build_features(hourly)
    models  = train_models(data)

X_train = models["X_train"]
X_test  = models["X_test"]
y_test  = models["y_test"]
gbr     = models["gbr"]


# ═══════════════════════════════════════════════
#  PAGE 1 — HOME
# ═══════════════════════════════════════════════
if page == "🏠  Home":

    # Hero banner
    st.markdown("""
    <div style='background:linear-gradient(135deg,#021B2E,#065A82);
                border-radius:16px;padding:36px 40px;margin-bottom:28px;'>
        <h1 style='color:white;margin:0;font-size:36px;'>
            ⚡ AI-Based Smart Energy Usage Prediction
        </h1>
        <p style='color:#02C39A;font-size:18px;margin:10px 0 6px 0;font-weight:600;'>
            for Sustainable Living
        </p>
        <p style='color:#94A3B8;font-size:15px;margin:0;'>
            Machine Learning system that forecasts household electricity consumption
            for the next 24 hours — helping you save energy & money.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI strip
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📅 Data Points", f"{len(df):,}", "minute-level readings")
    with c2:
        st.metric("⏰ Hourly Records", f"{len(hourly):,}", "after resampling")
    with c3:
        st.metric("🏆 Best MAE", f"{models['mae_gbr']:.3f} kW", f"vs baseline {models['mae_base']:.3f} kW")
    with c4:
        st.metric("🔮 Forecast Horizon", "24 Hours", "recursive prediction")

    st.markdown("<br>", unsafe_allow_html=True)

    # How it works
    st.markdown('<div class="section-header">🔄 How This Works</div>', unsafe_allow_html=True)

    s1, s2, s3, s4, s5, s6, s7 = st.columns(7)
    steps = [
        ("1", "📥", "Data\nCollection"),
        ("2", "🧹", "Pre-\nprocessing"),
        ("3", "⚙️", "Feature\nEngineering"),
        ("4", "🤖", "Model\nTraining"),
        ("5", "📊", "Evaluation"),
        ("6", "🔮", "24h\nForecast"),
        ("7", "💡", "Actionable\nInsights"),
    ]
    for col, (n, em, lbl) in zip([s1, s2, s3, s4, s5, s6, s7], steps):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="accent">{em}</div>
                <div class="value" style="font-size:22px;">#{n}</div>
                <div class="label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model comparison mini
    st.markdown('<div class="section-header">📈 Model Performance Summary</div>', unsafe_allow_html=True)

    fig = go.Figure()
    models_names = ["Baseline (lag_24)", "Linear Regression", "Gradient Boosting"]
    maes  = [models["mae_base"], models["mae_lr"], models["mae_gbr"]]
    rmses = [models["rmse_base"], models["rmse_lr"], models["rmse_gbr"]]
    colors = ["#94A3B8", "#60A5FA", "#02C39A"]

    fig.add_trace(go.Bar(name="MAE",  x=models_names, y=maes,  marker_color=colors,
                         text=[f"{v:.3f}" for v in maes], textposition="outside"))
    fig.add_trace(go.Bar(name="RMSE", x=models_names, y=rmses, marker_color=colors,
                         text=[f"{v:.3f}" for v in rmses], textposition="outside",
                         marker_pattern_shape="x", opacity=0.7))
    fig.update_layout(
        barmode="group", height=360,
        paper_bgcolor="white", plot_bgcolor="white",
        yaxis_title="Error (kW)", legend=dict(orientation="h", y=1.12),
        margin=dict(t=30, b=10, l=40, r=20),
        font=dict(color="#111111"),
    )
    fig.update_yaxes(gridcolor="#E2E8F0", tickfont=dict(color="#111111"), title_font=dict(color="#111111"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-box">
        🏆 <strong>Gradient Boosting Regressor</strong> achieves the best performance with the lowest
        MAE and RMSE — capturing non-linear patterns that Linear Regression misses.
        Navigate to <strong>24h Forecast</strong> to see peak hour predictions.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  PAGE 2 — DATA EXPLORER
# ═══════════════════════════════════════════════
elif page == "📊  Data Explorer":
    st.markdown('<div class="section-header">📊 Data Explorer</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Hourly Usage", "📅 Patterns & Heatmap", "🗂️ Raw Data"])

    # ── Tab 1: Hourly usage ──
    with tab1:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            days = st.slider("Show last N days:", 7, 90, 30)
        with col_b:
            show_ma = st.checkbox("Show 24h Moving Avg", value=True)

        tail = hourly.tail(days * 24)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tail.index, y=tail.values,
            mode="lines", name="Hourly Usage",
            line=dict(color="#1C7293", width=1.2), opacity=0.8,
        ))
        if show_ma:
            ma = tail.rolling(24).mean()
            fig.add_trace(go.Scatter(
                x=ma.index, y=ma.values,
                mode="lines", name="24h Moving Avg",
                line=dict(color="#02C39A", width=2.5),
            ))
        fig.update_layout(
            title=f"Hourly Global Active Power — Last {days} Days",
            xaxis_title="Datetime", yaxis_title="Power (kW)",
            paper_bgcolor="white", plot_bgcolor="white", height=420,
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=50, b=30),
            font=dict(color="#111111"),
        )
        fig.update_yaxes(gridcolor="#E2E8F0", tickfont=dict(color="#111111"), title_font=dict(color="#111111"))
        fig.update_xaxes(gridcolor="#E2E8F0", tickfont=dict(color="#111111"), title_font=dict(color="#111111"))
        st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean Usage",  f"{tail.mean():.3f} kW")
        m2.metric("Peak Usage",  f"{tail.max():.3f} kW")
        m3.metric("Min Usage",   f"{tail.min():.3f} kW")
        m4.metric("Std Dev",     f"{tail.std():.3f} kW")

    # ── Tab 2: Patterns ──
    with tab2:
        col1, col2 = st.columns(2)

        # Average by hour of day
        with col1:
            avg_by_hour = hourly.groupby(hourly.index.hour).mean()
            fig2 = go.Figure(go.Bar(
                x=avg_by_hour.index, y=avg_by_hour.values,
                marker_color=["#EF4444" if v == avg_by_hour.max()
                               else "#22C55E" if v == avg_by_hour.min()
                               else "#1C7293" for v in avg_by_hour.values],
                text=[f"{v:.2f}" for v in avg_by_hour.values],
                textposition="outside",
            ))
            fig2.update_layout(
                title="⏰ Average Usage by Hour of Day",
                xaxis_title="Hour", yaxis_title="Avg Power (kW)",
                paper_bgcolor="white", plot_bgcolor="white", height=350,
                margin=dict(t=50, b=30),
                font=dict(color="#111111"),
            )
            fig2.update_yaxes(gridcolor="#E2E8F0")
            st.plotly_chart(fig2, use_container_width=True)

        # Average by day of week
        with col2:
            avg_by_dow = hourly.groupby(hourly.index.dayofweek).mean()
            dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            fig3 = go.Figure(go.Bar(
                x=[dow_labels[i] for i in avg_by_dow.index],
                y=avg_by_dow.values,
                marker_color=["#F97316" if i >= 5 else "#065A82" for i in avg_by_dow.index],
                text=[f"{v:.2f}" for v in avg_by_dow.values],
                textposition="outside",
            ))
            fig3.update_layout(
                title="📅 Average Usage by Day of Week",
                xaxis_title="Day", yaxis_title="Avg Power (kW)",
                paper_bgcolor="white", plot_bgcolor="white", height=350,
                margin=dict(t=50, b=30),
                font=dict(color="#111111"),
            )
            fig3.update_yaxes(gridcolor="#E2E8F0")
            st.plotly_chart(fig3, use_container_width=True)

        # Heatmap: hour vs day-of-week
        st.markdown("**🗓️ Usage Heatmap — Hour × Day of Week**")
        heat_df = pd.DataFrame({
            "hour": hourly.index.hour,
            "dow":  hourly.index.dayofweek,
            "val":  hourly.values,
        }).groupby(["dow", "hour"])["val"].mean().unstack()

        fig4 = go.Figure(go.Heatmap(
            z=heat_df.values,
            x=[f"{h}:00" for h in heat_df.columns],
            y=[dow_labels[i] for i in heat_df.index],
            colorscale="Blues",
            colorbar=dict(title="kW"),
        ))
        fig4.update_layout(
            title="Average Power Consumption (kW) by Hour & Day",
            paper_bgcolor="white", height=320,
            margin=dict(t=50, b=30),
            font=dict(color="#111111"),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Tab 3: Raw data ──
    with tab3:
        n_rows = st.slider("Show last N rows:", 50, 500, 100)
        st.dataframe(
            df.tail(n_rows).style.format("{:.4f}"),
            use_container_width=True, height=400,
        )
        st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns  |  "
                   f"Date range: {df.index.min().date()} → {df.index.max().date()}")


# ═══════════════════════════════════════════════
#  PAGE 3 — MODEL COMPARISON
# ═══════════════════════════════════════════════
elif page == "🤖  Model Comparison":
    st.markdown('<div class="section-header">🤖 Model Training & Comparison</div>', unsafe_allow_html=True)

    # Metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="stat-card">
            <div class="accent">📏</div>
            <div style="font-size:13px;font-weight:700;color:#64748B;margin-bottom:8px;">
                BASELINE (lag_24)
            </div>
            <div style="font-size:22px;font-weight:800;color:#94A3B8;">
                MAE: {:.4f} kW
            </div>
            <div style="font-size:15px;color:#94A3B8;">RMSE: {:.4f} kW</div>
            <div style="font-size:12px;color:#94A3B8;margin-top:8px;">
                Naïve — predicts same hour yesterday
            </div>
        </div>""".format(models["mae_base"], models["rmse_base"]), unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="stat-card">
            <div class="accent">📐</div>
            <div style="font-size:13px;font-weight:700;color:#64748B;margin-bottom:8px;">
                LINEAR REGRESSION
            </div>
            <div style="font-size:22px;font-weight:800;color:#3B82F6;">
                MAE: {:.4f} kW
            </div>
            <div style="font-size:15px;color:#3B82F6;">RMSE: {:.4f} kW</div>
            <div style="font-size:12px;color:#64748B;margin-top:8px;">
                Learns linear feature weights
            </div>
        </div>""".format(models["mae_lr"], models["rmse_lr"]), unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="stat-card">
            <div class="accent">🏆</div>
            <div style="font-size:13px;font-weight:700;color:#064E3B;margin-bottom:8px;">
                GRADIENT BOOSTING ★
            </div>
            <div style="font-size:22px;font-weight:800;color:#02C39A;">
                MAE: {:.4f} kW
            </div>
            <div style="font-size:15px;color:#02C39A;">RMSE: {:.4f} kW</div>
            <div style="font-size:12px;color:#064E3B;margin-top:8px;">
                Best model — non-linear patterns ✓
            </div>
        </div>""".format(models["mae_gbr"], models["rmse_gbr"]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    col_left, col_right = st.columns(2)

    with col_left:
        # Bar chart comparison
        fig = go.Figure()
        mnames = ["Baseline", "Linear Reg.", "Grad. Boost"]
        maes   = [models["mae_base"],  models["mae_lr"],  models["mae_gbr"]]
        rmses  = [models["rmse_base"], models["rmse_lr"], models["rmse_gbr"]]
        clrs   = ["#94A3B8", "#60A5FA", "#02C39A"]

        fig.add_trace(go.Bar(name="MAE",  x=mnames, y=maes,
                             marker_color=clrs,
                             text=[f"{v:.4f}" for v in maes], textposition="outside"))
        fig.add_trace(go.Bar(name="RMSE", x=mnames, y=rmses,
                             marker_color=clrs, opacity=0.6,
                             text=[f"{v:.4f}" for v in rmses], textposition="outside"))
        fig.update_layout(
            title="MAE & RMSE — All Models", barmode="group",
            paper_bgcolor="white", plot_bgcolor="white", height=380,
            yaxis_title="Error (kW)", font=dict(color="#111111"),
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=50, b=20),
        )
        fig.update_yaxes(gridcolor="#E2E8F0", tickfont=dict(color="#111111"), title_font=dict(color="#111111"))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Improvement over baseline
        imp_mae  = (1 - models["mae_gbr"]  / models["mae_base"])  * 100
        imp_rmse = (1 - models["rmse_gbr"] / models["rmse_base"]) * 100

        fig2 = go.Figure()
        fig2.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=round(models["mae_gbr"], 4),
            title={"text": "GBR MAE (kW)", "font": {"size": 16}},
            delta={"reference": models["mae_base"], "relative": True,
                   "decreasing": {"color": "#02C39A"}},
            gauge={
                "axis":  {"range": [0, models["mae_base"] * 1.5]},
                "bar":   {"color": "#02C39A"},
                "steps": [
                    {"range": [0, models["mae_gbr"]],  "color": "#DCFCE7"},
                    {"range": [models["mae_gbr"], models["mae_base"]], "color": "#FEE2E2"},
                ],
                "threshold": {"line": {"color": "#EF4444", "width": 3},
                               "thickness": 0.75, "value": models["mae_base"]},
            },
            domain={"row": 0, "column": 0}
        ))
        fig2.update_layout(
            height=380, paper_bgcolor="white",
            title=dict(text=f"GBR improves MAE by {imp_mae:.1f}% over baseline",
                       font=dict(size=13, color="#111111")),
            margin=dict(t=60, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Actual vs Predicted plot
    st.markdown('<div class="section-header">📉 Actual vs Predicted — 300 Continuous Test Hours</div>', unsafe_allow_html=True)

    pred_gbr = pd.Series(models["y_gbr"], index=y_test.index)
    pred_lr  = pd.Series(models["y_lr"],  index=y_test.index)

    idx   = y_test.index.to_series()
    group = (idx.diff() != pd.Timedelta(hours=1)).cumsum()
    sizes = group.value_counts().sort_index()
    good  = sizes[sizes >= 300].index[0]

    y_blk = y_test[group == good].iloc[:300]
    g_blk = pred_gbr[group == good].iloc[:300]
    l_blk = pred_lr[group == good].iloc[:300]

    show_lr = st.checkbox("Also show Linear Regression predictions", value=False)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=y_blk.index, y=y_blk.values,
                              mode="lines", name="Actual",
                              line=dict(color="#1E293B", width=2)))
    fig3.add_trace(go.Scatter(x=g_blk.index, y=g_blk.values,
                              mode="lines", name="Gradient Boosting",
                              line=dict(color="#02C39A", width=1.8, dash="solid")))
    if show_lr:
        fig3.add_trace(go.Scatter(x=l_blk.index, y=l_blk.values,
                                  mode="lines", name="Linear Regression",
                                  line=dict(color="#60A5FA", width=1.5, dash="dot")))
    fig3.update_layout(
        xaxis_title="Datetime", yaxis_title="Power (kW)",
        paper_bgcolor="white", plot_bgcolor="white", height=400,
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=20, b=20),
        font=dict(color="#111111"),
    )
    fig3.update_yaxes(gridcolor="#E2E8F0")
    fig3.update_xaxes(gridcolor="#E2E8F0")
    st.plotly_chart(fig3, use_container_width=True)

    # Training data info
    st.markdown(f"""
    <div class="info-box">
        📐 <strong>Train/Test Split:</strong>
        Training on <strong>{len(models['X_train']):,} samples (80%)</strong> —
        Testing on <strong>{len(X_test):,} samples (20%)</strong> —
        Time-based split (no data leakage) ✓
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  PAGE 4 — 24h FORECAST
# ═══════════════════════════════════════════════
elif page == "🔮  24h Forecast":
    st.markdown('<div class="section-header">🔮 24-Hour Energy Forecast</div>', unsafe_allow_html=True)

    with st.spinner("Running recursive 24-hour forecast..."):
        forecast = make_forecast(hourly, X_train, gbr)

    top_peaks = forecast.sort_values(ascending=False).head(3)
    low_hours = forecast.sort_values(ascending=True).head(3)

    # Peak / Low cards
    col_p, col_l = st.columns(2)
    with col_p:
        st.markdown("### ⚡ Peak Hours — Avoid Heavy Usage")
        for ts, val in top_peaks.items():
            st.markdown(f"""
            <div class="peak-card">
                <h4>🔴 {ts.strftime('%I:%M %p')} on {ts.strftime('%d %b %Y')}</h4>
                <p><strong>{val:.3f} kW</strong> predicted — avoid running AC, washing machine, EV charger</p>
            </div>""", unsafe_allow_html=True)

    with col_l:
        st.markdown("### 🌿 Low-Usage Hours — Best for Heavy Tasks")
        for ts, val in low_hours.items():
            st.markdown(f"""
            <div class="low-card">
                <h4>🟢 {ts.strftime('%I:%M %p')} on {ts.strftime('%d %b %Y')}</h4>
                <p><strong>{val:.3f} kW</strong> predicted — ideal time to run dishwasher, charge EV, do laundry</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Forecast plot
    history = hourly.tail(24 * 7)

    fig = go.Figure()

    # History
    fig.add_trace(go.Scatter(
        x=history.index, y=history.values,
        mode="lines", name="Last 7 Days (History)",
        line=dict(color="#1C7293", width=1.5),
        fill="tozeroy", fillcolor="rgba(28,114,147,0.07)",
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast.values,
        mode="lines+markers", name="24-Hour Forecast",
        line=dict(color="#02C39A", width=2.5),
        marker=dict(size=6, color="#02C39A"),
        fill="tozeroy", fillcolor="rgba(2,195,154,0.1)",
    ))

    # Peak markers
    fig.add_trace(go.Scatter(
        x=top_peaks.index, y=top_peaks.values,
        mode="markers", name="Peak Hours",
        marker=dict(symbol="triangle-up", size=14, color="#EF4444", line=dict(color="white", width=1.5)),
    ))

    # Low markers
    fig.add_trace(go.Scatter(
        x=low_hours.index, y=low_hours.values,
        mode="markers", name="Low-Usage Hours",
        marker=dict(symbol="triangle-down", size=14, color="#22C55E", line=dict(color="white", width=1.5)),
    ))

    # Divider line at forecast start (add_shape is more compatible than add_vline)
    forecast_start_str = str(history.index[-1])
    fig.add_shape(
        type="line",
        x0=forecast_start_str, x1=forecast_start_str,
        y0=0, y1=1, yref="paper",
        line=dict(color="#F97316", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=forecast_start_str, y=1, yref="paper",
        text="Forecast Start",
        showarrow=False,
        font=dict(color="#F97316", size=12),
        xanchor="right", yanchor="bottom",
        bgcolor="white", bordercolor="#F97316", borderwidth=1,
    )

    fig.update_layout(
        title="Energy Usage: Last 7 Days History + 24h AI Forecast",
        xaxis_title="Datetime", yaxis_title="Power (kW)",
        paper_bgcolor="white", plot_bgcolor="white", height=480,
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=60, b=20),
        font=dict(color="#111111"),
        xaxis=dict(color="#111111"), yaxis=dict(color="#111111"),
    )
    fig.update_yaxes(gridcolor="#E2E8F0", tickfont=dict(color="#111111"), title_font=dict(color="#111111"))
    fig.update_xaxes(gridcolor="#E2E8F0", tickfont=dict(color="#111111"), title_font=dict(color="#111111"))
    st.plotly_chart(fig, use_container_width=True)

    # Hourly forecast table
    st.markdown("### 📋 Full 24-Hour Forecast Table")
    forecast_df = pd.DataFrame({
        "⏰ Time":        forecast.index.strftime("%I:%M %p"),
        "📅 Date":        forecast.index.strftime("%d %b %Y"),
        "⚡ Predicted (kW)": forecast.values.round(3),
        "📊 Status": ["🔴  PEAK — Avoid Heavy Use"  if t in top_peaks.index else
                      "🟢  LOW — Best Time to Run Appliances" if t in low_hours.index else
                      "🟡  Normal" for t in forecast.index],
    })

    def row_style(row):
        status = row["📊 Status"]
        # Peak row — light red background, dark red bold text
        if "PEAK" in status:
            bg = "background-color: #FECACA"   # light red
            tc = "color: #7F1D1D"              # very dark red
            fw = "font-weight: 800"
            return [f"{bg}; {tc}; {fw}; font-size:14px;"] * 4
        # Low row — light green background, dark green bold text
        elif "LOW" in status:
            bg = "background-color: #BBF7D0"   # light green
            tc = "color: #14532D"              # very dark green
            fw = "font-weight: 800"
            return [f"{bg}; {tc}; {fw}; font-size:14px;"] * 4
        # Normal row — white background, dark text
        return ["background-color: #FFFFFF; color: #1E293B; font-size:14px;"] * 4

    styled = (
        forecast_df
        .style
        .apply(row_style, axis=1)
        .set_table_styles([
            {"selector": "thead th",
             "props": [("background-color", "#065A82"),
                       ("color", "white"),
                       ("font-weight", "bold"),
                       ("font-size", "14px"),
                       ("padding", "10px 14px")]},
            {"selector": "tbody td",
             "props": [("padding", "9px 14px")]},
        ])
    )

    st.dataframe(
        styled,
        use_container_width=True, hide_index=True, height=420,
    )

    st.markdown("""
    <div class="info-box">
        🔄 <strong>Recursive Forecasting:</strong> Each hour's prediction is fed back as a lag feature
        for the next — creating a continuous 24-step forecast chain using the trained
        Gradient Boosting model. Red ▲ = peak hours, Green ▼ = best time to run heavy appliances.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  PAGE 5 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════
elif page == "🔍  Feature Importance":
    st.markdown('<div class="section-header">🔍 Feature Importance Analysis</div>', unsafe_allow_html=True)

    with st.spinner("Computing permutation importance (this takes ~30 seconds)..."):
        feat_imp = get_feature_importance(gbr, X_test, y_test)

    col_chart, col_info = st.columns([3, 2])

    with col_chart:
        colors = ["#02C39A" if i < 3 else "#1C7293" if i < 6 else "#94A3B8"
                  for i in range(len(feat_imp))]
        fig = go.Figure(go.Bar(
            x=feat_imp.values,
            y=feat_imp.index,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.4f}" for v in feat_imp.values],
            textposition="outside",
        ))
        fig.update_layout(
            title="Permutation Feature Importance (Gradient Boosting)",
            xaxis_title="Mean Importance Score",
            paper_bgcolor="white", plot_bgcolor="white", height=450,
            yaxis=dict(autorange="reversed"),
            margin=dict(t=50, b=20, l=120, r=60),
            font=dict(color="#111111"),
        )
        fig.update_xaxes(gridcolor="#E2E8F0", tickfont=dict(color="#111111"), title_font=dict(color="#111111"))
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.markdown("### 💡 What Each Feature Means")
        features_info = {
            "lag_1":         ("⚡ Lag 1h",   "Energy used 1 hour ago — the single most powerful predictor"),
            "lag_24":        ("🕐 Lag 24h",  "Same hour yesterday — captures daily patterns"),
            "lag_168":       ("📅 Lag 168h", "Same hour last week — captures weekly patterns"),
            "roll_mean_24":  ("📊 Roll 24h", "Average over last 24 hrs — trend indicator"),
            "roll_std_24":   ("📉 Std 24h",  "Variability over last 24 hrs — volatility signal"),
            "roll_mean_168": ("📊 Roll 168h","Average over last week — baseline level"),
            "hour":          ("⏰ Hour",     "Hour of day — morning/evening peaks"),
            "dayofweek":     ("📆 DayOfWeek","Day pattern — weekday vs weekend"),
            "is_weekend":    ("🏖️ Weekend",  "Binary: is it a weekend?"),
            "month":         ("🗓️ Month",    "Seasonal patterns — winter vs summer"),
        }
        for feat, (icon, desc) in features_info.items():
            rank = list(feat_imp.index).index(feat) + 1 if feat in feat_imp.index else "-"
            score = feat_imp.get(feat, 0)
            color = "#02C39A" if rank <= 3 else "#1C7293" if rank <= 6 else "#94A3B8"
            st.markdown(f"""
            <div style="padding:8px 12px;border-left:4px solid {color};
                        background:white;border-radius:6px;margin:5px 0;
                        box-shadow:0 1px 4px rgba(0,0,0,0.06);">
                <span style="font-weight:700;color:{color};">#{rank} {icon}</span>
                <span style="font-size:12px;color:#64748B;margin-left:6px;">score: {score:.4f}</span><br>
                <span style="font-size:12px;color:#374151;">{desc}</span>
            </div>""", unsafe_allow_html=True)

    # Key insight callout
    st.markdown("<br>", unsafe_allow_html=True)
    top3 = list(feat_imp.index[:3])
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#021B2E,#065A82);
                border-radius:12px;padding:22px 28px;color:white;">
        <h3 style="margin:0 0 10px 0;color:#02C39A;">🔑 Key Insight</h3>
        <p style="margin:0;font-size:15px;line-height:1.6;">
            The top 3 most important features are <strong style="color:#02C39A;">{", ".join(top3)}</strong>.
            This confirms that <strong>energy consumption is highly autocorrelated</strong> —
            the best predictor of what you'll use next is what you used recently.
            Time-of-day patterns (hour, dayofweek) add additional signal.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔗 Feature Correlation Matrix")
    corr = X_test.corr()
    fig2 = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="r"),
    ))
    fig2.update_layout(
        title="Pearson Correlation Between Features",
        paper_bgcolor="white", height=420,
        margin=dict(t=50, b=20),
        font=dict(color="#111111"),
    )
    st.plotly_chart(fig2, use_container_width=True)
