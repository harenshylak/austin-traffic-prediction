"""
app.py — Austin Traffic Prediction Dashboard
Run with: streamlit run app.py
"""

import json
import math
import os
import pickle

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import torch
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Austin Traffic Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── force all plotly text to black on white background ────────────────────────
_black = "black"
_tpl = go.layout.Template()
_tpl.layout.update(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color=_black, size=12, family="sans-serif"),
    title=dict(font=dict(color=_black)),
    xaxis=dict(
        color=_black,
        tickfont=dict(color=_black),
        title_font=dict(color=_black),
        linecolor="#cbd5e1",
        gridcolor="#f1f5f9",
        zeroline=False,
    ),
    yaxis=dict(
        color=_black,
        tickfont=dict(color=_black),
        title_font=dict(color=_black),
        linecolor="#cbd5e1",
        gridcolor="#f1f5f9",
    ),
    legend=dict(
        font=dict(color=_black),
        title_font=dict(color=_black),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#cbd5e1",
        borderwidth=1,
    ),
    coloraxis=dict(colorbar=dict(tickfont=dict(color=_black),
                                  title_font=dict(color=_black))),
    annotationdefaults=dict(font=dict(color=_black)),
)
pio.templates["austin"] = _tpl
pio.templates.default   = "austin"

PLOT_LAYOUT = dict(plot_bgcolor="white", paper_bgcolor="white")

# ── constants ──────────────────────────────────────────────────────────────────
GRAPH_DIR = "data/graph"
PROC_DIR  = "data/processed"
CKPT_DIR  = "checkpoints"
STEP_MIN  = 5
K_CONTEXT = 15

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
    "ARIMA(2,1,2)":                "results/arima_20260411_193632.json",
    "Chronos-T5-Base (zero-shot)": "results/chronos_20260418_022611.json",
    "LSTM — Sensor Only":          "results/lstm_only_20260411_174822.json",
    "LSTM + Weather":              "results/lstm_context_20260411_170936.json",
    "LSTM + Weather + Events":     "results/lstm_context_20260411_173212.json",
}

CTX_MEAN  = np.array([68.858, 0.129, 8.519, 65.959, 0.0, 9.265,
                       0.0, 0.0, 0.003, -0.001, -0.005, -0.002,
                       0.285, 0.0002, 0.0001], dtype=np.float32)
CTX_SCALE = np.array([16.214, 0.805, 4.092, 20.673, 1.0, 19.208,
                       0.7071, 0.7071, 0.7080, 0.7062, 0.7056, 0.7086,
                       0.4514, 0.0151, 0.0050], dtype=np.float32)

WEATHER_PRESETS = {
    "Clear / Dry":        dict(temp=72,  precip=0.0, wind=5,  humidity=45, event=False),
    "Light Rain":         dict(temp=65,  precip=0.3, wind=8,  humidity=75, event=False),
    "Heavy Rain / Storm": dict(temp=60,  precip=2.5, wind=20, humidity=90, event=False),
    "Winter Storm":       dict(temp=28,  precip=0.8, wind=25, humidity=85, event=False),
    "Extreme Heat":       dict(temp=105, precip=0.0, wind=5,  humidity=30, event=False),
    "Event Day + Rain":   dict(temp=68,  precip=1.0, wind=10, humidity=70, event=True),
}

PRESET_WINDOWS = {
    "Rush Hour (Weekday 8am)":  9900,
    "Late Night (Low Traffic)": 192,
    "Mid-Day Steady Flow":      2160,
}

# ── helpers ────────────────────────────────────────────────────────────────────
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


def denorm(arr, scaler_t):
    return arr * scaler_t.scale_[0] + scaler_t.mean_[0]


def build_context_tensor(temp, precip, wind, humidity, event_active, hour, dow, month, T=12):
    hour_sin  = math.sin(2 * math.pi * hour  / 24)
    hour_cos  = math.cos(2 * math.pi * hour  / 24)
    dow_sin   = math.sin(2 * math.pi * dow   / 7)
    dow_cos   = math.cos(2 * math.pi * dow   / 7)
    month_sin = math.sin(2 * math.pi * (month - 1) / 12)
    month_cos = math.cos(2 * math.pi * (month - 1) / 12)
    raw = np.array([
        temp, precip, wind, humidity, 0.0, 0.0,
        hour_sin, hour_cos, dow_sin, dow_cos,
        month_sin, month_cos,
        1.0 if dow >= 5 else 0.0,
        float(event_active), 1.0 if event_active else 0.0,
    ], dtype=np.float32)
    normed = (raw - CTX_MEAN) / CTX_SCALE
    return torch.from_numpy(np.tile(normed, (T, 1))).unsqueeze(0)   # (1,T,K)


def run_prediction(traffic_win_norm, ctx_tensor, m_base, m_ctx):
    t = torch.from_numpy(traffic_win_norm).unsqueeze(0)
    with torch.no_grad():
        pb = m_base(t, target=None, teacher_forcing_ratio=0.0)
        pc = m_ctx(t, ctx_tensor, target=None, teacher_forcing_ratio=0.0)
    return pb.squeeze(0).numpy(), pc.squeeze(0).numpy()


def speed_color(spd, vmin=20, vmax=65):
    r = max(0, min(220, int(220 * (vmax - spd) / (vmax - vmin))))
    g = max(0, min(180, int(180 * (spd - vmin) / (vmax - vmin))))
    return f"#{r:02x}{g:02x}30"


def make_map(sensor_order, sensor_locs, pred_mph, h_idx, key):
    m = folium.Map(location=[30.295, -97.76], zoom_start=12, tiles="CartoDB positron")
    for i, sid in enumerate(sensor_order):
        row = sensor_locs[sensor_locs["int_id"] == sid]
        if row.empty:
            continue
        lat, lon = float(row["lat"].iloc[0]), float(row["lon"].iloc[0])
        spd  = float(pred_mph[h_idx, i])
        name = SENSOR_NAMES.get(sid, f"Sensor {sid}")
        folium.CircleMarker(
            location=[lat, lon], radius=14,
            color="#1e293b", weight=1.5,
            fill=True, fill_color=speed_color(spd), fill_opacity=0.9,
            tooltip=f"{name}: {spd:.1f} mph",
            popup=folium.Popup(f"<b>{name}</b><br>{spd:.1f} mph", max_width=180),
        ).add_to(m)
    return m


def date_to_window(selected_dt, ts, test_s, T=12):
    """Find the nearest test window index for a given datetime."""
    target = pd.Timestamp(selected_dt)
    diffs  = np.abs((ts[test_s:] - target).total_seconds())
    return int(np.argmin(diffs))


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

traffic, context, ts, splits, scaler_t, sensor_order, sensor_locs = load_data()
m_base, m_ctx = load_models()
test_s, test_e = splits["test"]
T = 12

# Test period bounds for date pickers
test_start_dt = ts[test_s].to_pydatetime()
test_end_dt   = ts[test_e - T - T].to_pydatetime()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
if page == "What-If Simulator":
    st.title("What-If Weather & Event Simulator")
    st.markdown(
        "Pick a sensor and a real traffic window from the test period, then adjust "
        "weather and event conditions to see how the two models respond differently."
    )

    col_s, col_w = st.columns([1, 2])

    # ── col 1: sensor + window ────────────────────────────────────────────────
    with col_s:
        st.subheader("1 — Sensor & Time Window")

        sensor_label = st.selectbox("Sensor", [SENSOR_NAMES[sid] for sid in sensor_order])
        sensor_idx   = [SENSOR_NAMES[sid] for sid in sensor_order].index(sensor_label)

        scenario = st.selectbox("Quick scenario", list(PRESET_WINDOWS.keys()) + ["Custom date/time"])

        if scenario == "Custom date/time":
            sel_date = st.date_input(
                "Date  (Jul 1 – Sep 30, 2021)",
                value=test_start_dt.date(),
                min_value=test_start_dt.date(),
                max_value=test_end_dt.date(),
            )
            sel_hour = st.selectbox("Hour", list(range(24)),
                                    index=8, format_func=lambda h: f"{h:02d}:00")
            selected_dt = pd.Timestamp(year=sel_date.year, month=sel_date.month,
                                       day=sel_date.day, hour=sel_hour)
            window_idx  = date_to_window(selected_dt, ts, test_s)
        else:
            window_idx = PRESET_WINDOWS[scenario]

        window_time = ts[test_s + window_idx]
        st.info(f"📅 {window_time.strftime('%A, %b %d %Y  %H:%M')}")

    # ── col 2: weather sliders ────────────────────────────────────────────────
    with col_w:
        st.subheader("2 — Weather & Events")

        preset_wx = st.selectbox("Quick weather preset", ["Custom"] + list(WEATHER_PRESETS.keys()))
        wp = WEATHER_PRESETS[preset_wx] if preset_wx != "Custom" else \
             dict(temp=72, precip=0.0, wind=8, humidity=60, event=False)

        c1, c2 = st.columns(2)
        with c1:
            temp     = st.slider("Temperature (°F)",   0,   120, wp["temp"])
            precip   = st.slider("Precipitation (in)", 0.0, 5.0, float(wp["precip"]), step=0.1)
            wind     = st.slider("Wind speed (mph)",   0,   60,  wp["wind"])
        with c2:
            humidity = st.slider("Humidity (%)",       0,   100, wp["humidity"])
            event    = st.checkbox("Active permitted event nearby", value=wp["event"])

        conds = []
        if precip >= 2.0:   conds.append("🌧 Heavy rain")
        elif precip >= 0.3: conds.append("🌦 Light rain")
        else:               conds.append("☀️ Clear")
        if temp <= 32:      conds.append("❄️ Freezing")
        elif temp >= 95:    conds.append("🌡 Extreme heat")
        if wind >= 20:      conds.append("💨 High winds")
        if event:           conds.append("🎪 Event active")
        st.info("  ·  ".join(conds))

    st.divider()

    # ── run inference ─────────────────────────────────────────────────────────
    t0 = test_s + window_idx
    t1, t2 = t0 + T, t0 + T + T

    traffic_win  = np.array(traffic[t0:t1], dtype=np.float32)
    input_mph    = denorm(traffic_win[:, :, 0], scaler_t)

    ctx_tensor   = build_context_tensor(
        temp=temp, precip=precip, wind=wind, humidity=humidity,
        event_active=event,
        hour=window_time.hour, dow=window_time.dayofweek, month=window_time.month,
    )

    pb_norm, pc_norm = run_prediction(traffic_win, ctx_tensor, m_base, m_ctx)
    pb_mph = denorm(pb_norm[:, :, 0], scaler_t)
    pc_mph = denorm(pc_norm[:, :, 0], scaler_t)

    hist_x   = [-STEP_MIN * (T - j) for j in range(T)]
    future_x = [s * STEP_MIN for s in range(1, T + 1)]

    st.subheader("3 — Forecast Results")

    col_chart, col_delta = st.columns([2, 1])

    with col_chart:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_x, y=input_mph[:, sensor_idx],
            mode="lines", name="History (observed)",
            line=dict(color="#94A3B8", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=future_x, y=pb_mph[:, sensor_idx],
            mode="lines+markers", name="Sensor Only",
            line=dict(color="#2563EB", width=2.5, dash="dash"),
            marker=dict(size=6),
        ))
        fig.add_trace(go.Scatter(
            x=future_x, y=pc_mph[:, sensor_idx],
            mode="lines+markers", name="+ Weather & Events",
            line=dict(color="#EA580C", width=2.5, dash="dot"),
            marker=dict(size=6),
        ))
        fig.add_vline(x=0, line_dash="dot", line_color="#64748B", line_width=1.2,
                      annotation_text="now", annotation_position="top right",
                      annotation_font_color="#1e293b")
        fig.update_layout(
            **PLOT_LAYOUT,
            title=f"{sensor_label} — 60-min Forecast under current conditions",
            xaxis_title="Minutes from now",
            yaxis_title="Speed (mph)",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        font=dict(color="#1e293b")),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_delta:
        st.markdown("**Impact of weather & events at each horizon**")
        st.caption("How much does the +Weather & Events model differ from Sensor Only?")

        for h_step, label in [(3, "@ 15 min"), (6, "@ 30 min"), (12, "@ 60 min")]:
            base_v = float(pb_mph[h_step - 1, sensor_idx])
            ctx_v  = float(pc_mph[h_step - 1, sensor_idx])
            diff   = ctx_v - base_v
            st.markdown(f"**{label}**")
            c1, c2 = st.columns(2)
            c1.metric("Sensor Only",        f"{base_v:.1f} mph")
            c2.metric("+ Weather & Events", f"{ctx_v:.1f} mph",
                      delta=f"{diff:+.2f} mph",
                      delta_color="normal" if diff >= 0 else "inverse")

    # ── all-sensor delta bar ──────────────────────────────────────────────────
    st.divider()
    st.subheader("All Sensors — Weather Impact @ 60 min")

    diffs  = pc_mph[11] - pb_mph[11]
    names  = [SENSOR_NAMES.get(sid, str(sid)).split("/")[-1].strip() for sid in sensor_order]
    colors = ["#16A34A" if d >= 0 else "#DC2626" for d in diffs]

    fig2 = go.Figure(go.Bar(
        x=names, y=diffs,
        marker_color=colors,
        text=[f"{d:+.2f}" for d in diffs],
        textposition="outside",
        textfont=dict(color="#1e293b"),
    ))
    fig2.add_hline(y=0, line_color="#1e293b", line_width=1)
    fig2.update_layout(
        **PLOT_LAYOUT,
        title="+Weather & Events  vs  Sensor Only  (Δ mph at 60 min)",
        yaxis_title="Δ Speed (mph)",
        height=310,
        showlegend=False,
        xaxis_tickangle=-20,
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Green = weather model predicts higher speed · Red = lower speed · "
               "Bar height shows how much conditions shift the forecast.")

    # ── side-by-side maps ─────────────────────────────────────────────────────
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("**Sensor Only @ 60 min**")
        st_folium(make_map(sensor_order, sensor_locs, pb_mph, 11, "map_base"),
                  width=400, height=300, returned_objects=[], key="map_base")
    with col_m2:
        st.markdown("**+ Weather & Events @ 60 min**")
        st_folium(make_map(sensor_order, sensor_locs, pc_mph, 11, "map_ctx"),
                  width=400, height=300, returned_objects=[], key="map_ctx")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.title("Model Comparison")

    # ── aggregate results table ───────────────────────────────────────────────
    st.subheader("Full Test Set — Evaluation Results")
    rows = []
    for label, path in RESULT_FILES.items():
        if not os.path.exists(path):
            continue
        with open(path) as f:
            res = json.load(f)
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
            lambda v: "background-color:#DCFCE7;font-weight:bold;color:#166534"
                      if v == best_mae else "color:#1e293b",
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
        **PLOT_LAYOUT,
        barmode="group",
        yaxis_title="MAE (mph)",
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    font=dict(color="#1e293b")),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── interactive: pick a window + weather, compare both models ────────────
    st.divider()
    st.subheader("Live Comparison — Sensor Only vs +Weather & Events")
    st.markdown("Pick a time window and adjust weather conditions to compare how the two "
                "models diverge under different scenarios.")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("**Time Window**")
        scenario2 = st.selectbox("Scenario", list(PRESET_WINDOWS.keys()) + ["Custom date/time"],
                                 key="comp_scenario")
        if scenario2 == "Custom date/time":
            sel_date2 = st.date_input("Date", value=test_start_dt.date(),
                                      min_value=test_start_dt.date(),
                                      max_value=test_end_dt.date(), key="comp_date")
            sel_hour2 = st.selectbox("Hour", list(range(24)), index=8,
                                     format_func=lambda h: f"{h:02d}:00", key="comp_hour")
            selected_dt2 = pd.Timestamp(year=sel_date2.year, month=sel_date2.month,
                                        day=sel_date2.day, hour=sel_hour2)
            window_idx2  = date_to_window(selected_dt2, ts, test_s)
        else:
            window_idx2 = PRESET_WINDOWS[scenario2]

        window_time2 = ts[test_s + window_idx2]
        st.info(f"📅 {window_time2.strftime('%A, %b %d %Y  %H:%M')}")

    with col_b:
        st.markdown("**Weather Conditions**")
        preset_wx2 = st.selectbox("Preset", ["Custom"] + list(WEATHER_PRESETS.keys()),
                                  key="comp_wx")
        wp2 = WEATHER_PRESETS[preset_wx2] if preset_wx2 != "Custom" else \
              dict(temp=72, precip=0.0, wind=8, humidity=60, event=False)
        temp2     = st.slider("Temperature (°F)",   0,   120, wp2["temp"],          key="t2")
        precip2   = st.slider("Precipitation (in)", 0.0, 5.0, float(wp2["precip"]), key="p2", step=0.1)
        wind2     = st.slider("Wind speed (mph)",   0,   60,  wp2["wind"],           key="w2")
        humidity2 = st.slider("Humidity (%)",       0,   100, wp2["humidity"],       key="h2")
        event2    = st.checkbox("Active event nearby", value=wp2["event"],           key="e2")

    t0b = test_s + window_idx2
    traffic_win2 = np.array(traffic[t0b:t0b + T], dtype=np.float32)
    input_mph2   = denorm(traffic_win2[:, :, 0], scaler_t)
    ctx2         = build_context_tensor(
        temp=temp2, precip=precip2, wind=wind2, humidity=humidity2,
        event_active=event2,
        hour=window_time2.hour, dow=window_time2.dayofweek, month=window_time2.month,
    )
    pb2_norm, pc2_norm = run_prediction(traffic_win2, ctx2, m_base, m_ctx)
    pb2_mph = denorm(pb2_norm[:, :, 0], scaler_t)
    pc2_mph = denorm(pc2_norm[:, :, 0], scaler_t)

    hist_x   = [-STEP_MIN * (T - j) for j in range(T)]
    future_x = [s * STEP_MIN for s in range(1, T + 1)]

    # Network-average chart: Sensor Only vs +Weather (no actual)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=hist_x, y=input_mph2.mean(axis=1),
        mode="lines", name="History (observed)",
        line=dict(color="#94A3B8", width=2),
    ))
    fig3.add_trace(go.Scatter(
        x=future_x, y=pb2_mph.mean(axis=1),
        mode="lines+markers", name="Sensor Only",
        line=dict(color="#2563EB", width=2.5, dash="dash"),
        marker=dict(size=6),
    ))
    fig3.add_trace(go.Scatter(
        x=future_x, y=pc2_mph.mean(axis=1),
        mode="lines+markers", name="+ Weather & Events",
        line=dict(color="#EA580C", width=2.5, dash="dot"),
        marker=dict(size=6),
    ))
    fig3.add_vline(x=0, line_dash="dot", line_color="#64748B", line_width=1.2,
                   annotation_text="now", annotation_font_color="#1e293b")
    fig3.update_layout(
        **PLOT_LAYOUT,
        title="Network-Average Speed — Sensor Only vs +Weather & Events",
        xaxis_title="Minutes from now", yaxis_title="Speed (mph)", height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    font=dict(color="#1e293b")),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Per-sensor charts
    st.subheader("Per-Sensor Forecast")
    sensor_cols = st.columns(5)
    for i, sid in enumerate(sensor_order):
        sname = SENSOR_NAMES.get(sid, str(sid)).split("/")[-1].strip()
        fig4  = go.Figure()
        fig4.add_trace(go.Scatter(x=hist_x, y=input_mph2[:, i], mode="lines",
                                  name="History", line=dict(color="#94A3B8", width=1.5)))
        fig4.add_trace(go.Scatter(x=future_x, y=pb2_mph[:, i], mode="lines+markers",
                                  name="Sensor", line=dict(color="#2563EB", width=1.8, dash="dash"),
                                  marker=dict(size=3)))
        fig4.add_trace(go.Scatter(x=future_x, y=pc2_mph[:, i], mode="lines+markers",
                                  name="+Wx", line=dict(color="#EA580C", width=1.8, dash="dot"),
                                  marker=dict(size=3)))
        fig4.add_vline(x=0, line_dash="dot", line_color="#94A3B8", line_width=1)
        fig4.update_layout(
            **PLOT_LAYOUT,
            title=dict(text=sname, font=dict(size=10, color="#1e293b")),
            margin=dict(l=5, r=5, t=28, b=20), height=200,
            showlegend=(i == 0),
            xaxis=dict(title="min", zeroline=False,
                       tickfont=dict(size=8, color="#1e293b"),
                       title_font=dict(color="#1e293b")),
            yaxis=dict(title="mph",
                       tickfont=dict(size=8, color="#1e293b"),
                       title_font=dict(color="#1e293b")),
            legend=dict(orientation="h", y=1.4, font=dict(size=8, color="#1e293b")),
        )
        sensor_cols[i % 5].plotly_chart(fig4, use_container_width=True,
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
higher variance in mid-2021 as COVID restrictions lifted and traffic returned.
        """)
    with tab2:
        st.image(os.path.join(FIG_DIR, "stl_decomposition.png"), use_container_width=True)
        st.markdown("""
- **Trend**: slow decline from ~47 mph (Apr 2020, COVID) to ~41 mph (mid-2021), then recovery.
- **Seasonal (weekly)**: ±4 mph cycle — weekends faster, Mon–Fri shows rush-hour congestion.
- **Residual**: deviation from trend + seasonal. Large spikes = unusual traffic days.
        """)
    with tab3:
        st.image(os.path.join(FIG_DIR, "daily_weekly_pattern.png"), use_container_width=True)
        st.markdown("""
- **Weekday rush hours**: speeds dip to ~35 mph at 8am and 5–6pm.
- **Weekend**: smooth ~55 mph profile all day — no morning dip.
- **Heatmap**: darkest cells (slowest) are Mon–Fri 7–9am and 4–7pm.
        """)
    with tab4:
        st.image(os.path.join(FIG_DIR, "anomaly_periods.png"), use_container_width=True)
        st.markdown("""
| Cause | Why it appears |
|---|---|
| COVID lockdown (Apr 2020) | Speeds *above* trend — roads empty, no rush-hour congestion |
| Heavy rain | Volume drops → free-flow, or drivers slow in severe rain |
| Winter Storm Uri (Feb 15, 2021) | ~15°F, icy roads, city effectively shut down |
| Holidays | Break the weekly pattern (Thanksgiving, Christmas, MLK Day, Labor Day) |
| High wind | Gusts slowed traffic |
| Event + rain | ACE permitted events coinciding with heavy rain |
        """)
