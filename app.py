"""
app.py — Austin Traffic Prediction Dashboard
Run with: streamlit run app.py
"""

import json
import os
import pickle

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Austin Traffic Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ──────────────────────────────────────────────────────────────────
GRAPH_DIR = "data/graph"
PROC_DIR  = "data/processed"
CKPT_DIR  = "checkpoints"
STEP_MIN  = 5
K_CONTEXT = 15   # matches checkpoint weight shape [128, 180] = [128, 12*15]

SENSOR_NAMES = {
    1:  "Lamar / Manchaca",
    3:  "Lamar / Shoal Creek",
    8:  "Loop 360 / Walsh Tarlton",
    10: "Lamar / Broken Spoke",
    11: "Burnet / Rutland",
    15: "Cesar Chavez / Reynolds",
    20: "Cesar Chavez / IH-35",
    22: "Lamar / Collier",
    23: "Burnet / Palm Way",
    24: "Loop 360 / Lakewood",
}

RESULT_FILES = {
    "ARIMA(2,1,2)":               "results/arima_20260411_193632.json",
    "Chronos-T5-Base (zero-shot)":"results/chronos_20260418_022611.json",
    "LSTM — Sensor Only":         "results/lstm_only_20260411_174822.json",
    "LSTM + Weather":             "results/lstm_context_20260411_170936.json",
    "LSTM + Weather + Events":    "results/lstm_context_20260411_173212.json",
}

# Context scaler params (feature 0-based, K=15)
# temp, precip, wind, humidity, visibility, weather_code,
# hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, is_weekend,
# event_active, n_concurrent
CTX_MEAN  = np.array([68.858, 0.129, 8.519, 65.959, 0.0, 9.265,
                       0.0, 0.0, 0.003, -0.001, -0.005, -0.002,
                       0.285, 0.0002, 0.0001], dtype=np.float32)
CTX_SCALE = np.array([16.214, 0.805, 4.092, 20.673, 1.0, 19.208,
                       0.7071, 0.7071, 0.7080, 0.7062, 0.7056, 0.7086,
                       0.4514, 0.0151, 0.0050], dtype=np.float32)

WEATHER_PRESETS = {
    "Clear / Dry":         dict(temp=72,  precip=0.0,  wind=5,   humidity=45, event=False),
    "Light Rain":          dict(temp=65,  precip=0.3,  wind=8,   humidity=75, event=False),
    "Heavy Rain / Storm":  dict(temp=60,  precip=2.5,  wind=20,  humidity=90, event=False),
    "Winter Storm":        dict(temp=28,  precip=0.8,  wind=25,  humidity=85, event=False),
    "Extreme Heat":        dict(temp=105, precip=0.0,  wind=5,   humidity=30, event=False),
    "Event Day + Rain":    dict(temp=68,  precip=1.0,  wind=10,  humidity=70, event=True),
}

# ── data & model loading ───────────────────────────────────────────────────────
@st.cache_resource
def load_data():
    traffic    = np.load(os.path.join(GRAPH_DIR, "traffic.npy"),    mmap_mode="r")
    context    = np.load(os.path.join(GRAPH_DIR, "context.npy"),    mmap_mode="r")
    timestamps = np.load(os.path.join(GRAPH_DIR, "timestamps.npy"), allow_pickle=True)
    with open(os.path.join(GRAPH_DIR, "split_indices.json")) as f:
        splits = json.load(f)
    with open(os.path.join(GRAPH_DIR, "scaler_traffic.pkl"), "rb") as f:
        scaler_t = pickle.load(f)
    with open(os.path.join(GRAPH_DIR, "sensor_order.json")) as f:
        sensor_order = json.load(f)
    sensor_locs = pd.read_csv(os.path.join(PROC_DIR, "sensor_locations.csv"))
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    return traffic, context, ts, splits, scaler_t, sensor_order, sensor_locs


@st.cache_resource
def load_models():
    from src.models.lstm_baseline import LSTMBaseline
    from src.models.lstm_context  import LSTMWithContext

    ck = torch.load(os.path.join(CKPT_DIR, "lstm_only_best.pt"),
                    map_location="cpu", weights_only=False)
    m_base = LSTMBaseline(F=3, hidden_dim=64, n_layers=2, H=12, dropout=0.1)
    m_base.load_state_dict(ck["model_state"])
    m_base.eval()

    ck_ctx = torch.load(os.path.join(CKPT_DIR, "lstm_context_best.pt"),
                        map_location="cpu", weights_only=False)
    m_ctx = LSTMWithContext(F=3, K=K_CONTEXT, T=12, hidden_dim=64,
                            n_layers=2, H=12, dropout=0.1)
    m_ctx.load_state_dict(ck_ctx["model_state"])
    m_ctx.eval()

    return m_base, m_ctx


def norm_traffic(speed_mph, scaler_t):
    """Normalize speed back to model input scale."""
    return (speed_mph - scaler_t.mean_[0]) / scaler_t.scale_[0]


def denorm(arr, scaler_t):
    return arr * scaler_t.scale_[0] + scaler_t.mean_[0]


def build_context_tensor(temp, precip, wind, humidity, event_active,
                          hour, dow, month, T=12):
    """
    Build a (1, T, K) context tensor from user-provided weather/event values.
    Calendar features are computed from the selected time.
    """
    import math
    hour_sin  = math.sin(2 * math.pi * hour  / 24)
    hour_cos  = math.cos(2 * math.pi * hour  / 24)
    dow_sin   = math.sin(2 * math.pi * dow   / 7)
    dow_cos   = math.cos(2 * math.pi * dow   / 7)
    month_sin = math.sin(2 * math.pi * (month - 1) / 12)
    month_cos = math.cos(2 * math.pi * (month - 1) / 12)
    is_weekend = 1.0 if dow >= 5 else 0.0
    n_conc     = 1.0 if event_active else 0.0

    raw = np.array([
        temp, precip, wind, humidity,
        0.0,   # visibility (use mean)
        0.0,   # weather_code (use mean)
        hour_sin, hour_cos, dow_sin, dow_cos,
        month_sin, month_cos, is_weekend,
        float(event_active), n_conc,
    ], dtype=np.float32)

    normed = (raw - CTX_MEAN) / CTX_SCALE
    ctx    = np.tile(normed, (T, 1))   # (T, K)
    return torch.from_numpy(ctx).unsqueeze(0)   # (1, T, K)


def run_prediction(traffic_window_norm, ctx_tensor, m_base, m_ctx):
    """Run both models and return (pred_base, pred_ctx) in mph."""
    traffic_t = torch.from_numpy(traffic_window_norm).unsqueeze(0)  # (1,T,N,F)
    with torch.no_grad():
        pred_base = m_base(traffic_t, target=None, teacher_forcing_ratio=0.0)
        pred_ctx  = m_ctx(traffic_t, ctx_tensor,   target=None, teacher_forcing_ratio=0.0)
    return pred_base.squeeze(0).numpy(), pred_ctx.squeeze(0).numpy()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🚦 Austin Traffic")
    st.markdown("**Prediction Dashboard**")
    st.divider()
    page = st.radio("View",
                    ["What-If Simulator", "Model Comparison", "Data Analytics"],
                    label_visibility="collapsed")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
if page == "What-If Simulator":
    traffic, context, ts, splits, scaler_t, sensor_order, sensor_locs = load_data()
    m_base, m_ctx = load_models()

    st.title("What-If Weather & Event Simulator")
    st.markdown(
        "Select a sensor and a real traffic window, then adjust weather and event conditions "
        "to see how the **Sensor-Only** and **Weather + Events** models respond differently."
    )

    # ── step 1: pick sensor & window ─────────────────────────────────────────
    col_s, col_w = st.columns([1, 2])

    with col_s:
        st.subheader("1 — Pick Sensor & Window")
        sensor_label = st.selectbox(
            "Sensor",
            [SENSOR_NAMES[sid] for sid in sensor_order],
        )
        sensor_idx = [SENSOR_NAMES[sid] for sid in sensor_order].index(sensor_label)

        preset_window = st.selectbox(
            "Start from a scenario",
            [
                "Rush Hour (Weekday 8am)",
                "Late Night (Low Traffic)",
                "Mid-Day Steady Flow",
                "Custom",
            ],
        )

        test_s = splits["test"][0]
        PRESET_WINDOWS = {
            "Rush Hour (Weekday 8am)":   9900,
            "Late Night (Low Traffic)":  192,
            "Mid-Day Steady Flow":       2160,
        }

        if preset_window == "Custom":
            n_windows = splits["test"][1] - test_s - 24
            window_idx = st.slider("Test window index", 0, n_windows, 0)
        else:
            window_idx = PRESET_WINDOWS[preset_window]

        window_time = ts[test_s + window_idx]
        st.caption(f"Window: **{window_time.strftime('%a, %b %d %Y  %H:%M')}**")

    # ── step 2: weather & events ──────────────────────────────────────────────
    with col_w:
        st.subheader("2 — Set Weather & Events")

        preset_wx = st.selectbox("Quick preset", ["Custom"] + list(WEATHER_PRESETS.keys()))
        if preset_wx != "Custom":
            wp = WEATHER_PRESETS[preset_wx]
        else:
            wp = dict(temp=72, precip=0.0, wind=8, humidity=60, event=False)

        c1, c2 = st.columns(2)
        with c1:
            temp     = st.slider("Temperature (°F)",   0,   120,  wp["temp"])
            precip   = st.slider("Precipitation (in)", 0.0, 5.0,  float(wp["precip"]), step=0.1)
            wind     = st.slider("Wind speed (mph)",   0,   60,   wp["wind"])
        with c2:
            humidity = st.slider("Humidity (%)",       0,   100,  wp["humidity"])
            event    = st.checkbox("Active permitted event nearby", value=wp["event"])

        # Plain-English summary
        conditions = []
        if precip >= 2.0:   conditions.append("🌧 Heavy rain")
        elif precip >= 0.3: conditions.append("🌦 Light rain")
        else:               conditions.append("☀️ Clear")
        if temp <= 32:      conditions.append("❄️ Freezing")
        elif temp >= 95:    conditions.append("🌡 Extreme heat")
        if wind >= 20:      conditions.append("💨 High winds")
        if event:           conditions.append("🎪 Event active")
        st.info("  ·  ".join(conditions))

    st.divider()

    # ── load real traffic history for this window ─────────────────────────────
    T = 12
    t0 = test_s + window_idx
    t1 = t0 + T
    t2 = t1 + T

    traffic_win_norm = np.array(traffic[t0:t1], dtype=np.float32)  # (T, N, F)
    actual_speed_mph = denorm(np.array(traffic[t1:t2, :, 0]), scaler_t)  # (H, N)
    input_speed_mph  = denorm(traffic_win_norm[:, :, 0], scaler_t)       # (T, N)

    ctx_tensor = build_context_tensor(
        temp=temp, precip=precip, wind=wind, humidity=humidity,
        event_active=event,
        hour=window_time.hour, dow=window_time.dayofweek,
        month=window_time.month, T=T,
    )

    pred_base_norm, pred_ctx_norm = run_prediction(traffic_win_norm, ctx_tensor, m_base, m_ctx)
    pred_base_mph = denorm(pred_base_norm[:, :, 0], scaler_t)   # (H, N)
    pred_ctx_mph  = denorm(pred_ctx_norm[:, :, 0],  scaler_t)   # (H, N)

    # ── step 3: results ───────────────────────────────────────────────────────
    st.subheader("3 — Forecast Results")

    hist_x   = [-STEP_MIN * (T - j) for j in range(T)]
    future_x = [s * STEP_MIN for s in range(1, T + 1)]

    # ── focused single-sensor chart ───────────────────────────────────────────
    col_chart, col_delta = st.columns([2, 1])

    with col_chart:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_x, y=input_speed_mph[:, sensor_idx],
            mode="lines", name="History (actual)",
            line=dict(color="#94A3B8", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=future_x, y=actual_speed_mph[:, sensor_idx],
            mode="lines+markers", name="Ground truth",
            line=dict(color="#1e293b", width=2.5),
            marker=dict(size=6),
        ))
        fig.add_trace(go.Scatter(
            x=future_x, y=pred_base_mph[:, sensor_idx],
            mode="lines+markers", name="Sensor Only",
            line=dict(color="#2563EB", width=2, dash="dash"),
            marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=future_x, y=pred_ctx_mph[:, sensor_idx],
            mode="lines+markers", name="+ Weather & Events",
            line=dict(color="#EA580C", width=2, dash="dot"),
            marker=dict(size=5),
        ))
        fig.add_vline(x=0, line_dash="dot", line_color="#64748B", line_width=1.2,
                      annotation_text="now", annotation_position="top right")
        fig.update_layout(
            title=f"{sensor_label} — 60-min Forecast",
            xaxis_title="Minutes from now",
            yaxis_title="Speed (mph)",
            height=380,
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis=dict(zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_delta:
        st.markdown("**Model difference at each horizon**")
        st.caption("Sensor-Only  vs  +Weather & Events")

        for h_step, label in [(3, "15 min"), (6, "30 min"), (12, "60 min")]:
            h_idx = h_step - 1
            base_v = float(pred_base_mph[h_idx, sensor_idx])
            ctx_v  = float(pred_ctx_mph[h_idx, sensor_idx])
            diff   = ctx_v - base_v
            actual = float(actual_speed_mph[h_idx, sensor_idx])

            st.markdown(f"**{label}**")
            c1, c2 = st.columns(2)
            c1.metric("Sensor Only",        f"{base_v:.1f} mph",
                      delta=f"{base_v - actual:+.1f} vs actual", delta_color="inverse")
            c2.metric("+ Weather & Events", f"{ctx_v:.1f} mph",
                      delta=f"{diff:+.1f} vs sensor-only",
                      delta_color="inverse" if diff < 0 else "normal")

        st.divider()
        mae_base = float(np.mean(np.abs(pred_base_mph[:, sensor_idx] - actual_speed_mph[:, sensor_idx])))
        mae_ctx  = float(np.mean(np.abs(pred_ctx_mph[:, sensor_idx]  - actual_speed_mph[:, sensor_idx])))
        st.metric("MAE — Sensor Only",       f"{mae_base:.2f} mph")
        st.metric("MAE — +Weather & Events", f"{mae_ctx:.2f} mph",
                  delta=f"{mae_ctx - mae_base:+.2f} vs sensor-only",
                  delta_color="inverse")

    # ── all-sensor network view ───────────────────────────────────────────────
    st.divider()
    st.subheader("All Sensors — Network Impact")

    # Bar chart: difference between models at 60 min
    h_idx = 11
    diffs = pred_ctx_mph[h_idx] - pred_base_mph[h_idx]
    names = [SENSOR_NAMES.get(sid, f"{sid}").split("/")[-1].strip() for sid in sensor_order]
    bar_colors = ["#16A34A" if d >= 0 else "#DC2626" for d in diffs]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=names, y=diffs,
        marker_color=bar_colors,
        text=[f"{d:+.2f}" for d in diffs],
        textposition="outside",
    ))
    fig2.add_hline(y=0, line_color="#1e293b", line_width=1)
    fig2.update_layout(
        title="Speed change: +Weather & Events  vs  Sensor Only  @  60 min",
        yaxis_title="Δ Speed (mph)",
        height=300,
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
        xaxis_tickangle=-20,
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Green = weather model predicts higher speed. Red = weather model predicts lower speed. "
        "The magnitude shows how much the weather/event context shifts the forecast."
    )

    # ── map colored by predicted speed difference ─────────────────────────────
    col_m1, col_m2 = st.columns(2)
    for col, pred_mph, title in [
        (col_m1, pred_base_mph, "Sensor Only @ 60 min"),
        (col_m2, pred_ctx_mph,  "+Weather & Events @ 60 min"),
    ]:
        with col:
            st.markdown(f"**{title}**")
            m = folium.Map(location=[30.295, -97.76], zoom_start=12,
                           tiles="CartoDB positron")
            for i, sid in enumerate(sensor_order):
                row = sensor_locs[sensor_locs["int_id"] == sid]
                if row.empty: continue
                lat, lon = float(row["lat"].iloc[0]), float(row["lon"].iloc[0])
                spd = float(pred_mph[11, i])
                r = max(0, min(220, int(220 * (65 - spd) / 45)))
                g = max(0, min(180, int(180 * (spd - 20) / 45)))
                color = f"#{r:02x}{g:02x}30"
                name  = SENSOR_NAMES.get(sid, f"Sensor {sid}")
                folium.CircleMarker(
                    location=[lat, lon], radius=14,
                    color="#1e293b", weight=1.5,
                    fill=True, fill_color=color, fill_opacity=0.9,
                    tooltip=f"{name}: {spd:.1f} mph",
                    popup=folium.Popup(f"<b>{name}</b><br>{spd:.1f} mph predicted", max_width=180),
                ).add_to(m)
            st_folium(m, width=400, height=300, returned_objects=[],
                      key=f"map_{title}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    traffic, context, ts, splits, scaler_t, sensor_order, sensor_locs = load_data()
    m_base, m_ctx = load_models()

    st.title("Model Comparison")

    # Aggregate metrics
    st.subheader("Test Set Results — full evaluation")
    rows = []
    for label, path in RESULT_FILES.items():
        if not os.path.exists(path): continue
        with open(path) as f: res = json.load(f)
        m = res.get("test_metrics", {})
        rows.append({
            "Model":       label,
            "MAE @15min":  m.get("15min", {}).get("mae"),
            "MAE @30min":  m.get("30min", {}).get("mae"),
            "MAE @60min":  m.get("60min", {}).get("mae"),
            "RMSE @15min": m.get("15min", {}).get("rmse"),
            "MAPE @15min": m.get("15min", {}).get("mape"),
        })
    df_res = pd.DataFrame(rows)
    best_mae = df_res["MAE @15min"].min()
    st.dataframe(
        df_res.style.map(
            lambda v: "background-color:#DCFCE7;font-weight:bold" if v == best_mae else "",
            subset=["MAE @15min"],
        ),
        use_container_width=True, hide_index=True,
    )

    # MAE by horizon bar chart
    st.subheader("MAE by Forecast Horizon")
    BAR_COLORS = ["#94A3B8", "#7C3AED", "#2563EB", "#0EA5E9", "#16A34A"]
    fig = go.Figure()
    for idx, row in enumerate(rows):
        fig.add_trace(go.Bar(
            name=row["Model"],
            x=["15 min", "30 min", "60 min"],
            y=[row["MAE @15min"], row["MAE @30min"], row["MAE @60min"]],
            marker_color=BAR_COLORS[idx % len(BAR_COLORS)],
        ))
    fig.update_layout(
        barmode="group", yaxis_title="MAE (mph)", height=360,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Live window comparison
    st.divider()
    st.subheader("Live Window Comparison")
    test_s, test_e = splits["test"]
    n_windows = (test_e - test_s) - 24
    window_idx = st.slider("Test window", 0, n_windows, 500)
    window_time = ts[test_s + window_idx]
    st.caption(f"**{window_time.strftime('%a, %b %d %Y  %H:%M')}**")

    T = 12
    t0 = test_s + window_idx
    t1 = t0 + T
    t2 = t1 + T
    traffic_win  = np.array(traffic[t0:t1], dtype=np.float32)
    actual_mph   = denorm(np.array(traffic[t1:t2, :, 0]), scaler_t)
    input_mph    = denorm(traffic_win[:, :, 0], scaler_t)
    ctx_real     = torch.from_numpy(
        np.array(context[t0:t1, :K_CONTEXT], dtype=np.float32)
    ).unsqueeze(0)

    pb, pc = run_prediction(traffic_win, ctx_real, m_base, m_ctx)
    pb_mph = denorm(pb[:, :, 0], scaler_t)
    pc_mph = denorm(pc[:, :, 0], scaler_t)

    hist_x   = [-STEP_MIN * (T - j) for j in range(T)]
    future_x = [s * STEP_MIN for s in range(1, T + 1)]

    # Network average chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_x,   y=input_mph.mean(axis=1),
                             mode="lines", name="History",
                             line=dict(color="#94A3B8", width=2)))
    fig.add_trace(go.Scatter(x=future_x, y=actual_mph.mean(axis=1),
                             mode="lines+markers", name="Actual",
                             line=dict(color="#1e293b", width=2.5),
                             marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=future_x, y=pb_mph.mean(axis=1),
                             mode="lines+markers", name="Sensor Only",
                             line=dict(color="#2563EB", width=2, dash="dash"),
                             marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=future_x, y=pc_mph.mean(axis=1),
                             mode="lines+markers", name="+Weather & Events",
                             line=dict(color="#EA580C", width=2, dash="dot"),
                             marker=dict(size=5)))
    fig.add_vline(x=0, line_dash="dot", line_color="#64748B", line_width=1.2)
    fig.update_layout(
        title="Network-average speed — Actual vs Both Models",
        xaxis_title="min", yaxis_title="Speed (mph)", height=340,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-sensor trajectories
    st.subheader("Per-Sensor Forecast")
    sensor_cols = st.columns(5)
    for i, sid in enumerate(sensor_order):
        sname = SENSOR_NAMES.get(sid, f"Sensor {sid}").split("/")[-1].strip()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_x, y=input_mph[:, i],
                                 mode="lines", name="History",
                                 line=dict(color="#94A3B8", width=1.5)))
        fig.add_trace(go.Scatter(x=future_x, y=actual_mph[:, i],
                                 mode="lines+markers", name="Actual",
                                 line=dict(color="#1e293b", width=2),
                                 marker=dict(size=4)))
        fig.add_trace(go.Scatter(x=future_x, y=pb_mph[:, i],
                                 mode="lines+markers", name="Sensor",
                                 line=dict(color="#2563EB", width=1.8, dash="dash"),
                                 marker=dict(size=3)))
        fig.add_trace(go.Scatter(x=future_x, y=pc_mph[:, i],
                                 mode="lines+markers", name="+Wx",
                                 line=dict(color="#EA580C", width=1.8, dash="dot"),
                                 marker=dict(size=3)))
        fig.add_vline(x=0, line_dash="dot", line_color="#94A3B8", line_width=1)
        fig.update_layout(
            title=dict(text=sname, font_size=10),
            margin=dict(l=5, r=5, t=28, b=20), height=200,
            showlegend=(i == 0),
            xaxis=dict(title="min", zeroline=False, tickfont_size=8),
            yaxis=dict(title="mph", tickfont_size=8),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.4, font_size=8),
        )
        sensor_cols[i % 5].plotly_chart(fig, use_container_width=True,
                                         config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data Analytics":
    st.title("Data Analytics")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Speed Overview", "STL Decomposition", "Daily / Weekly Patterns", "Anomaly Periods"
    ])
    FIG_DIR = "docs/figures"

    with tab1:
        st.image(os.path.join(FIG_DIR, "speed_overview.png"), use_container_width=True)
        st.markdown("""
**Reading this chart:** The blue line is the network average across all 10 sensors.
Grey traces show individual sensors. The bottom panel shows daily speed variability —
higher variance in mid-2021 as COVID restrictions lifted.
        """)
    with tab2:
        st.image(os.path.join(FIG_DIR, "stl_decomposition.png"), use_container_width=True)
        st.markdown("""
- **Trend**: slow decline from ~47 mph (Apr 2020, COVID) to ~41 mph (mid-2021), then recovery.
- **Seasonal (weekly)**: ±4 mph cycle — weekends faster, Mon–Fri shows rush-hour congestion.
- **Residual**: anomalies around the expected pattern.
        """)
    with tab3:
        st.image(os.path.join(FIG_DIR, "daily_weekly_pattern.png"), use_container_width=True)
        st.markdown("""
- **Weekday rush hours**: speeds dip to ~35 mph at 8am and 5–6pm.
- **Weekend**: smooth ~55 mph all day.
- **Heatmap**: darkest cells are Mon–Fri 7–9am and 4–7pm.
        """)
    with tab4:
        st.image(os.path.join(FIG_DIR, "anomaly_periods.png"), use_container_width=True)
        st.markdown("""
| Cause | Why it appears |
|---|---|
| COVID lockdown (Apr 2020) | Speeds above trend — roads empty |
| Heavy rain | Volume drops → free-flow, or drivers slow in severe rain |
| Winter Storm Uri (Feb 15, 2021) | ~15°F, icy roads, city shut down |
| Holidays | Break the weekly pattern |
| High wind | Gusts slowed traffic |
| Event + rain | ACE events coinciding with heavy rain |
        """)
