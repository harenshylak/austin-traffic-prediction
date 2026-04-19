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
import torch.nn as nn
from streamlit_folium import st_folium

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Austin Traffic Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ──────────────────────────────────────────────────────────────────
GRAPH_DIR   = "data/graph"
PROC_DIR    = "data/processed"
CKPT_DIR    = "checkpoints"
HORIZONS    = [3, 6, 12]          # steps → 15, 30, 60 min
STEP_MIN    = 5
SENSOR_NAMES = {
    1:  "Lamar / Manchaca",
    3:  "Lamar / Shoal Creek",
    8:  "Loop 360 / Walsh Tarlton",
    10: "Lamar / Broken Spoke",
    11: "Burnet / Rutland",
    15: "Cesar Chavez / BR Reynolds",
    20: "Cesar Chavez / IH-35",
    22: "Lamar / Collier",
    23: "Burnet / Palm Way",
    24: "Loop 360 / Lakewood",
}

MODEL_LABELS = {
    "LSTM — Sensor Only":        ("lstm_only",     False),
    "LSTM + Weather":            ("lstm_context",   True),   # K=6
    "LSTM + Weather + Events":   ("lstm_context",   True),   # K=17
}

RESULT_FILES = {
    "ARIMA(2,1,2)":                 "results/arima_20260411_193632.json",
    "Chronos-T5-Base (zero-shot)":  "results/chronos_20260418_022611.json",
    "LSTM — Sensor Only":           "results/lstm_only_20260411_174822.json",
    "LSTM + Weather":               "results/lstm_context_20260411_170936.json",
    "LSTM + Weather + Events":      "results/lstm_context_20260411_173212.json",
}

SPEED_COLORS = ["#16A34A", "#65A30D", "#CA8A04", "#EA580C", "#DC2626"]

# ── caching data loads ─────────────────────────────────────────────────────────
@st.cache_resource
def load_data():
    traffic  = np.load(os.path.join(GRAPH_DIR, "traffic.npy"),   mmap_mode="r")
    context  = np.load(os.path.join(GRAPH_DIR, "context.npy"),   mmap_mode="r")
    timestamps = np.load(os.path.join(GRAPH_DIR, "timestamps.npy"), allow_pickle=True)
    with open(os.path.join(GRAPH_DIR, "split_indices.json")) as f:
        splits = json.load(f)
    with open(os.path.join(GRAPH_DIR, "scaler_traffic.pkl"), "rb") as f:
        scaler_t = pickle.load(f)
    with open(os.path.join(GRAPH_DIR, "scaler_context.pkl"), "rb") as f:
        scaler_c = pickle.load(f)
    with open(os.path.join(GRAPH_DIR, "sensor_order.json")) as f:
        sensor_order = json.load(f)

    sensor_locs = pd.read_csv(os.path.join(PROC_DIR, "sensor_locations.csv"))

    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    return traffic, context, ts, splits, scaler_t, scaler_c, sensor_order, sensor_locs


@st.cache_resource
def load_models():
    from src.models.lstm_baseline import LSTMBaseline
    from src.models.lstm_context  import LSTMWithContext

    models = {}
    # LSTM sensor only
    ck = torch.load(os.path.join(CKPT_DIR, "lstm_only_best.pt"),
                    map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = LSTMBaseline(
        n_sensors=cfg["data"]["N"],
        n_features=cfg["data"]["F"],
        H=cfg["data"]["H"],
        hidden_dim=cfg["model"]["d_model"],
        n_layers=cfg["model"]["patch_transformer"]["n_layers"],
        dropout=cfg["model"]["dropout"],
    )
    m.load_state_dict(ck["model_state"])
    m.eval()
    models["LSTM — Sensor Only"] = ("baseline", m, None)

    # LSTM + Weather (K=6)
    ck6 = torch.load(os.path.join(CKPT_DIR, "lstm_context_best.pt"),
                     map_location="cpu", weights_only=False)
    cfg6 = ck6["cfg"]
    m6 = LSTMWithContext(
        n_sensors=cfg6["data"]["N"],
        n_features=cfg6["data"]["F"],
        K=6,
        H=cfg6["data"]["H"],
        hidden_dim=cfg6["model"]["d_model"],
        n_layers=cfg6["model"]["patch_transformer"]["n_layers"],
        dropout=cfg6["model"]["dropout"],
    )
    m6.load_state_dict(ck6["model_state"])
    m6.eval()
    models["LSTM + Weather"] = ("context", m6, 6)

    # LSTM + Weather + Events (K=17)
    m17 = LSTMWithContext(
        n_sensors=cfg6["data"]["N"],
        n_features=cfg6["data"]["F"],
        K=17,
        H=cfg6["data"]["H"],
        hidden_dim=cfg6["model"]["d_model"],
        n_layers=cfg6["model"]["patch_transformer"]["n_layers"],
        dropout=cfg6["model"]["dropout"],
    )
    m17.load_state_dict(ck6["model_state"])
    m17.eval()
    models["LSTM + Weather + Events"] = ("context", m17, 17)

    return models


def denorm_speed(arr, scaler):
    return arr * scaler.scale_[0] + scaler.mean_[0]


@st.cache_data
def run_inference(model_name, window_idx):
    traffic, context, ts, splits, scaler_t, scaler_c, sensor_order, _ = load_data()
    models = load_models()

    T, H = 12, 12
    test_s = splits["test"][0]
    t0 = test_s + window_idx
    t1 = t0 + T
    t2 = t1 + H

    traffic_win = torch.from_numpy(np.array(traffic[t0:t1], dtype=np.float32)).unsqueeze(0)  # (1,T,N,F)
    target_win  = np.array(traffic[t1:t2, :, 0])  # (H, N)

    kind, model, K = models[model_name]

    with torch.no_grad():
        if kind == "baseline":
            pred = model(traffic_win, target=None, teacher_forcing_ratio=0.0)
        else:
            ctx_win = torch.from_numpy(
                np.array(context[t0:t1, :K], dtype=np.float32)
            ).unsqueeze(0)  # (1,T,K)
            pred = model(traffic_win, ctx_win, target=None, teacher_forcing_ratio=0.0)

    pred_np = pred.squeeze(0).numpy()          # (H, N, 1)
    pred_speed = denorm_speed(pred_np[:, :, 0], scaler_t)   # (H, N)
    actual_speed = denorm_speed(target_win, scaler_t)        # (H, N)
    input_speed  = denorm_speed(np.array(traffic[t0:t1, :, 0]), scaler_t)  # (T, N)

    return pred_speed, actual_speed, input_speed, ts[t0:t2]


def speed_to_color(speed, vmin=20, vmax=65):
    ratio = max(0.0, min(1.0, (speed - vmin) / (vmax - vmin)))
    r = int(220 * (1 - ratio))
    g = int(180 * ratio)
    return f"#{r:02x}{g:02x}30"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🚦 Austin Traffic")
    st.markdown("**Prediction Dashboard**")
    st.divider()

    page = st.radio(
        "View",
        ["Forecast Explorer", "Model Comparison", "Data Analytics"],
        label_visibility="collapsed",
    )
    st.divider()

    if page == "Forecast Explorer":
        st.subheader("Settings")
        model_name = st.selectbox(
            "Model",
            list(MODEL_LABELS.keys()),
        )
        _, _, splits_raw, _, _, _, _, _ = (
            load_data()[0], load_data()[1], load_data()[3],
            load_data()[4], load_data()[5], load_data()[6], load_data()[7],
            None,
        )

        traffic, context, ts, splits, scaler_t, scaler_c, sensor_order, sensor_locs = load_data()
        test_s, test_e = splits["test"]
        n_windows = (test_e - test_s) - 12 - 12 + 1

        window_idx = st.slider(
            "Test window",
            0, n_windows - 1, 0,
            help="Each step = 5 minutes in the Jul–Sep 2021 test set",
        )

        window_time = ts[test_s + window_idx]
        st.caption(f"Window start: **{window_time.strftime('%b %d, %Y  %H:%M')}**")

        horizon_step = st.select_slider(
            "Forecast horizon",
            options=[3, 6, 12],
            value=12,
            format_func=lambda x: f"{x*5} min",
        )

    elif page == "Model Comparison":
        traffic, context, ts, splits, scaler_t, scaler_c, sensor_order, sensor_locs = load_data()
        test_s, test_e = splits["test"]
        n_windows = (test_e - test_s) - 12 - 12 + 1
        window_idx = st.slider("Test window", 0, n_windows - 1, 500)
        window_time = ts[test_s + window_idx]
        st.caption(f"Window start: **{window_time.strftime('%b %d, %Y  %H:%M')}**")
    else:
        traffic, context, ts, splits, scaler_t, scaler_c, sensor_order, sensor_locs = load_data()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — FORECAST EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
if page == "Forecast Explorer":
    st.title("Forecast Explorer")
    st.caption(f"Model: **{model_name}** | Window: {window_time.strftime('%A, %b %d %Y %H:%M')} | Horizon: {horizon_step*5} min")

    with st.spinner("Running inference..."):
        pred_speed, actual_speed, input_speed, win_ts = run_inference(model_name, window_idx)

    # ── top row: map + horizon metrics ────────────────────────────────────────
    col_map, col_metrics = st.columns([1.4, 1])

    with col_map:
        st.subheader("Sensor Map")
        m = folium.Map(
            location=[30.295, -97.76],
            zoom_start=12,
            tiles="CartoDB positron",
        )

        h_idx = horizon_step - 1
        for i, sid in enumerate(sensor_order):
            row = sensor_locs[sensor_locs["int_id"] == sid]
            if row.empty:
                continue
            lat, lon = float(row["lat"].iloc[0]), float(row["lon"].iloc[0])
            pred_v   = float(pred_speed[h_idx, i])
            actual_v = float(actual_speed[h_idx, i])
            name     = SENSOR_NAMES.get(sid, f"Sensor {sid}")
            color    = speed_to_color(pred_v)

            folium.CircleMarker(
                location=[lat, lon],
                radius=14,
                color="#1e293b",
                weight=1.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.88,
                popup=folium.Popup(
                    f"<b>{name}</b><br>"
                    f"Predicted: <b>{pred_v:.1f} mph</b><br>"
                    f"Actual: {actual_v:.1f} mph<br>"
                    f"Error: {abs(pred_v - actual_v):.1f} mph",
                    max_width=200,
                ),
                tooltip=f"{name}: {pred_v:.1f} mph",
            ).add_to(m)

        st_folium(m, width=520, height=370, returned_objects=[])

    with col_metrics:
        st.subheader(f"Network Summary @ {horizon_step*5} min")
        h_idx = horizon_step - 1
        mae   = float(np.mean(np.abs(pred_speed[h_idx] - actual_speed[h_idx])))
        mape  = float(np.mean(np.abs(pred_speed[h_idx] - actual_speed[h_idx]) /
                              np.clip(actual_speed[h_idx], 1, None)) * 100)
        avg_pred   = float(pred_speed[h_idx].mean())
        avg_actual = float(actual_speed[h_idx].mean())

        c1, c2 = st.columns(2)
        c1.metric("MAE",          f"{mae:.2f} mph")
        c2.metric("MAPE",         f"{mape:.1f}%")
        c1.metric("Avg Predicted", f"{avg_pred:.1f} mph")
        c2.metric("Avg Actual",    f"{avg_actual:.1f} mph")

        st.divider()
        st.subheader("Per-sensor @ horizon")
        rows = []
        for i, sid in enumerate(sensor_order):
            p = float(pred_speed[h_idx, i])
            a = float(actual_speed[h_idx, i])
            rows.append({
                "Sensor": SENSOR_NAMES.get(sid, f"Sensor {sid}"),
                "Predicted": round(p, 1),
                "Actual": round(a, 1),
                "Error": round(abs(p - a), 1),
            })
        df_tbl = pd.DataFrame(rows).sort_values("Error", ascending=False)
        st.dataframe(df_tbl, use_container_width=True, hide_index=True,
                     column_config={
                         "Predicted": st.column_config.NumberColumn(format="%.1f mph"),
                         "Actual":    st.column_config.NumberColumn(format="%.1f mph"),
                         "Error":     st.column_config.NumberColumn(format="%.1f mph"),
                     })

    # ── bottom: per-sensor forecast chart ────────────────────────────────────
    st.subheader("Forecast vs Actual — All Sensors")
    n_sensors = len(sensor_order)
    n_cols = 5
    n_rows = (n_sensors + n_cols - 1) // n_cols
    sensor_cols = st.columns(n_cols)

    T = 12
    future_steps = list(range(1, 13))
    future_min   = [s * STEP_MIN for s in future_steps]

    for i, sid in enumerate(sensor_order):
        col = sensor_cols[i % n_cols]
        name = SENSOR_NAMES.get(sid, f"Sensor {sid}").split("/")[-1].strip()

        fig = go.Figure()
        # input history
        hist_x = [-STEP_MIN * (T - j) for j in range(T)]
        fig.add_trace(go.Scatter(
            x=hist_x, y=input_speed[:, i],
            mode="lines", name="History",
            line=dict(color="#94A3B8", width=1.5),
        ))
        # actual future
        fig.add_trace(go.Scatter(
            x=future_min, y=actual_speed[:, i],
            mode="lines+markers", name="Actual",
            line=dict(color="#2563EB", width=2),
            marker=dict(size=4),
        ))
        # predicted future
        fig.add_trace(go.Scatter(
            x=future_min, y=pred_speed[:, i],
            mode="lines+markers", name="Predicted",
            line=dict(color="#EA580C", width=2, dash="dash"),
            marker=dict(size=4),
        ))
        # vertical divider
        fig.add_vline(x=0, line_dash="dot", line_color="#64748B", line_width=1)

        fig.update_layout(
            title=dict(text=name, font_size=11),
            margin=dict(l=10, r=10, t=30, b=20),
            height=200,
            showlegend=False,
            xaxis=dict(title="min", tickfont_size=9, zeroline=False),
            yaxis=dict(title="mph", tickfont_size=9),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        col.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.title("Model Comparison")

    # ── aggregate metrics table from saved results ─────────────────────────────
    st.subheader("Test Set Results (full evaluation)")
    rows = []
    for label, path in RESULT_FILES.items():
        if not os.path.exists(path):
            continue
        with open(path) as f:
            res = json.load(f)
        m = res.get("test_metrics", {})
        rows.append({
            "Model":       label,
            "MAE @15min":  m.get("15min", {}).get("mae",  "—"),
            "MAE @30min":  m.get("30min", {}).get("mae",  "—"),
            "MAE @60min":  m.get("60min", {}).get("mae",  "—"),
            "RMSE @15min": m.get("15min", {}).get("rmse", "—"),
            "MAPE @15min": m.get("15min", {}).get("mape", "—"),
        })

    df_res = pd.DataFrame(rows)
    best_mae = df_res["MAE @15min"].min()

    def highlight_best(val):
        if val == best_mae:
            return "background-color: #DCFCE7; font-weight: bold"
        return ""

    st.dataframe(
        df_res.style.applymap(highlight_best, subset=["MAE @15min"]),
        use_container_width=True,
        hide_index=True,
    )

    # ── MAE by horizon bar chart ───────────────────────────────────────────────
    st.subheader("MAE by Forecast Horizon")
    fig = go.Figure()
    colors = ["#94A3B8", "#7C3AED", "#2563EB", "#0EA5E9", "#16A34A"]
    for idx, row in enumerate(rows):
        mae_vals = [row["MAE @15min"], row["MAE @30min"], row["MAE @60min"]]
        fig.add_trace(go.Bar(
            name=row["Model"],
            x=["15 min", "30 min", "60 min"],
            y=mae_vals,
            marker_color=colors[idx % len(colors)],
        ))
    fig.update_layout(
        barmode="group",
        yaxis_title="MAE (mph)",
        height=380,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── live side-by-side forecast for selected window ─────────────────────────
    st.subheader(f"Live Forecasts on Window {window_idx}  ({window_time.strftime('%b %d %H:%M')})")
    st.caption("Sensor-level predictions from the three LSTM variants on the same test window.")

    lstm_models = ["LSTM — Sensor Only", "LSTM + Weather", "LSTM + Weather + Events"]
    result_cols = st.columns(3)

    for col, mname in zip(result_cols, lstm_models):
        with col:
            with st.spinner(f"Running {mname}..."):
                pred_s, actual_s, input_s, _ = run_inference(mname, window_idx)

            mae_val = float(np.mean(np.abs(pred_s - actual_s)))
            col.metric(mname, f"MAE {mae_val:.2f} mph")

            # network-average forecast chart
            avg_pred   = pred_s.mean(axis=1)
            avg_actual = actual_s.mean(axis=1)
            avg_input  = input_s.mean(axis=1)
            T = 12
            hist_x   = [-STEP_MIN * (T - j) for j in range(T)]
            future_x = [s * STEP_MIN for s in range(1, 13)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_x, y=avg_input,
                                     mode="lines", name="History",
                                     line=dict(color="#94A3B8", width=1.5)))
            fig.add_trace(go.Scatter(x=future_x, y=avg_actual,
                                     mode="lines+markers", name="Actual",
                                     line=dict(color="#2563EB", width=2),
                                     marker=dict(size=4)))
            fig.add_trace(go.Scatter(x=future_x, y=avg_pred,
                                     mode="lines+markers", name="Predicted",
                                     line=dict(color="#EA580C", width=2, dash="dash"),
                                     marker=dict(size=4)))
            fig.add_vline(x=0, line_dash="dot", line_color="#64748B", line_width=1)
            fig.update_layout(
                height=220, margin=dict(l=10, r=10, t=10, b=30),
                showlegend=(mname == lstm_models[0]),
                xaxis=dict(title="min", zeroline=False),
                yaxis=dict(title="mph"),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True,
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
        st.image(os.path.join(FIG_DIR, "speed_overview.png"),
                 caption="Raw speed time series — all 10 sensors (grey) and network average (blue). "
                         "Shaded bands show train / validation / test splits.",
                 use_container_width=True)
        st.markdown("""
**Reading this chart:**
- The blue line is the average of all 10 sensors — a stable ~43 mph with clear daily oscillation.
- The grey traces show individual sensors; some run faster (highway) and some slower (city intersections).
- Daily variability (bottom panel) is higher in mid-2021 as COVID restrictions lifted and traffic patterns became less predictable.
        """)

    with tab2:
        st.image(os.path.join(FIG_DIR, "stl_decomposition.png"),
                 caption="STL decomposition: observed, trend, weekly seasonal component, and residual.",
                 use_container_width=True)
        st.markdown("""
**Components:**
- **Trend**: slow decline from ~47 mph (Apr 2020, COVID low traffic) down to ~41 mph (mid-2021, traffic returning), then slight recovery.
- **Seasonal (weekly, period=7 days)**: ±4 mph weekly cycle — weekends have faster free-flow speeds, Mon–Fri show rush-hour congestion.
- **Residual**: noise around trend + seasonal. Large spikes are unusual traffic days (storms, holidays, events).
        """)

    with tab3:
        st.image(os.path.join(FIG_DIR, "daily_weekly_pattern.png"),
                 caption="Hourly speed profiles, day-of-week averages, and hour × day heatmap.",
                 use_container_width=True)
        st.markdown("""
**Key patterns:**
- **Weekday rush hours**: speeds dip to ~35 mph at 8am and 5–6pm, matching Austin's well-documented commute congestion.
- **Weekend**: smooth high-speed profile (~55 mph) all day with no morning dip.
- **Heatmap** (right): darkest cells (lowest speed) are Mon–Fri 7–9am and 4–7pm. Saturday night shows slightly elevated speeds.
        """)

    with tab4:
        st.image(os.path.join(FIG_DIR, "anomaly_periods.png"),
                 caption="Anomaly days identified by STL residual > 95th percentile, color-coded by cause.",
                 use_container_width=True)
        st.markdown("""
**Anomaly causes:**
| Cause | Why it appears |
|---|---|
| COVID lockdown (Apr 2020) | Speeds *above* trend — roads empty, no rush-hour congestion |
| Heavy rain | Reduces volume → free-flow speeds, or slows traffic depending on severity |
| Winter Storm Uri (Feb 15, 2021) | Most extreme single day — ~15°F, icy roads, city effectively shut down |
| Holidays | Break the weekly pattern — Thanksgiving, Christmas, MLK Day, Labor Day |
| High wind (Nov 15, 2020) | 20+ mph gusts → drivers slow down |
| Event + rain (Aug 2021) | ACE permitted events coinciding with heavy rainfall |

**Note:** Most anomalies have *positive* residuals — meaning speeds were higher than expected. This is the volume effect: extreme weather or holidays keep drivers off the road, reducing congestion at monitored intersections.
        """)
