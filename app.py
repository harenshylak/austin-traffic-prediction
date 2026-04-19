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

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Austin Traffic Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ──────────────────────────────────────────────────────────────────
GRAPH_DIR  = "data/graph"
PROC_DIR   = "data/processed"
CKPT_DIR   = "checkpoints"
STEP_MIN   = 5

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

# K=15: weight shape in checkpoint is [128, 180] = [128, 12*15]
# context columns: weather(6) + calendar(7) + event_active + n_concurrent
K_CONTEXT = 15

RESULT_FILES = {
    "ARIMA(2,1,2)":               "results/arima_20260411_193632.json",
    "Chronos-T5-Base (zero-shot)":"results/chronos_20260418_022611.json",
    "LSTM — Sensor Only":         "results/lstm_only_20260411_174822.json",
    "LSTM + Weather":             "results/lstm_context_20260411_170936.json",
    "LSTM + Weather + Events":    "results/lstm_context_20260411_173212.json",
}

LIVE_MODELS = ["LSTM — Sensor Only", "LSTM + Weather + Events"]

MODEL_COLORS = {
    "LSTM — Sensor Only":       "#2563EB",
    "LSTM + Weather + Events":  "#16A34A",
}

# ── data loading ───────────────────────────────────────────────────────────────
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

    # ── LSTM sensor only ──────────────────────────────────────────────────────
    ck = torch.load(os.path.join(CKPT_DIR, "lstm_only_best.pt"),
                    map_location="cpu", weights_only=False)
    m_base = LSTMBaseline(
        F=3, hidden_dim=64, n_layers=2, H=12, dropout=0.1,
    )
    m_base.load_state_dict(ck["model_state"])
    m_base.eval()

    # ── LSTM + context (K=15, as saved in checkpoint) ─────────────────────────
    ck_ctx = torch.load(os.path.join(CKPT_DIR, "lstm_context_best.pt"),
                        map_location="cpu", weights_only=False)
    m_ctx = LSTMWithContext(
        F=3, K=K_CONTEXT, T=12, hidden_dim=64, n_layers=2, H=12, dropout=0.1,
    )
    m_ctx.load_state_dict(ck_ctx["model_state"])
    m_ctx.eval()

    return {
        "LSTM — Sensor Only":       ("baseline", m_base),
        "LSTM + Weather + Events":  ("context",  m_ctx),
    }


def denorm(arr, scaler):
    return arr * scaler.scale_[0] + scaler.mean_[0]


@st.cache_data
def run_inference(model_name: str, window_idx: int):
    traffic, context, ts, splits, scaler_t, sensor_order, _ = load_data()
    models = load_models()

    T, H = 12, 12
    test_s = splits["test"][0]
    t0 = test_s + window_idx
    t1 = t0 + T
    t2 = t1 + H

    traffic_win = torch.from_numpy(
        np.array(traffic[t0:t1], dtype=np.float32)
    ).unsqueeze(0)   # (1,T,N,F)

    kind, model = models[model_name]

    with torch.no_grad():
        if kind == "baseline":
            pred = model(traffic_win, target=None, teacher_forcing_ratio=0.0)
        else:
            ctx_win = torch.from_numpy(
                np.array(context[t0:t1, :K_CONTEXT], dtype=np.float32)
            ).unsqueeze(0)   # (1,T,K)
            pred = model(traffic_win, ctx_win, target=None, teacher_forcing_ratio=0.0)

    pred_np    = pred.squeeze(0).numpy()                        # (H,N,1)
    pred_speed = denorm(pred_np[:, :, 0], scaler_t)             # (H,N)
    actual_speed = denorm(np.array(traffic[t1:t2, :, 0]), scaler_t)  # (H,N)
    input_speed  = denorm(np.array(traffic[t0:t1, :, 0]), scaler_t)  # (T,N)

    return pred_speed, actual_speed, input_speed, ts[t0:t2]


def speed_to_hex(speed, vmin=20, vmax=65):
    r = max(0.0, min(1.0, (vmax - speed) / (vmax - vmin)))
    g = max(0.0, min(1.0, (speed - vmin) / (vmax - vmin)))
    return f"#{int(220*r):02x}{int(180*g):02x}30"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🚦 Austin Traffic")
    st.markdown("**Prediction Dashboard**")
    st.divider()

    page = st.radio("View", ["Forecast Explorer", "Model Comparison", "Data Analytics"],
                    label_visibility="collapsed")
    st.divider()

    traffic, context, ts, splits, scaler_t, sensor_order, sensor_locs = load_data()
    test_s, test_e = splits["test"]
    n_windows = (test_e - test_s) - 12 - 12 + 1

    if page in ("Forecast Explorer", "Model Comparison"):
        st.subheader("Test Window")
        window_idx = st.slider("", 0, n_windows - 1, 0,
                               help="Each step = 5 min in Jul–Sep 2021 test set")
        window_time = ts[test_s + window_idx]
        st.caption(f"**{window_time.strftime('%a, %b %d %Y  %H:%M')}**")

    if page == "Forecast Explorer":
        st.subheader("Settings")
        model_name = st.selectbox("Model", LIVE_MODELS)
        horizon_step = st.select_slider(
            "Forecast horizon",
            options=[3, 6, 12],
            value=12,
            format_func=lambda x: f"{x*5} min",
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — FORECAST EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
if page == "Forecast Explorer":
    st.title("Forecast Explorer")

    with st.spinner("Running inference..."):
        pred_speed, actual_speed, input_speed, win_ts = run_inference(model_name, window_idx)

    h_idx = horizon_step - 1
    T = 12

    # ── top row: map + metrics ────────────────────────────────────────────────
    col_map, col_metrics = st.columns([1.4, 1])

    with col_map:
        st.subheader(f"Sensor Map  —  Predicted speed @ {horizon_step*5} min")
        m = folium.Map(location=[30.295, -97.76], zoom_start=12,
                       tiles="CartoDB positron")

        for i, sid in enumerate(sensor_order):
            row = sensor_locs[sensor_locs["int_id"] == sid]
            if row.empty:
                continue
            lat, lon = float(row["lat"].iloc[0]), float(row["lon"].iloc[0])
            pv = float(pred_speed[h_idx, i])
            av = float(actual_speed[h_idx, i])
            name = SENSOR_NAMES.get(sid, f"Sensor {sid}")

            folium.CircleMarker(
                location=[lat, lon], radius=14,
                color="#1e293b", weight=1.5,
                fill=True, fill_color=speed_to_hex(pv), fill_opacity=0.9,
                popup=folium.Popup(
                    f"<b>{name}</b><br>"
                    f"Predicted: <b>{pv:.1f} mph</b><br>"
                    f"Actual: {av:.1f} mph<br>"
                    f"Error: {abs(pv-av):.1f} mph",
                    max_width=200,
                ),
                tooltip=f"{name}: {pv:.1f} mph predicted",
            ).add_to(m)

        st_folium(m, width=520, height=360, returned_objects=[])

    with col_metrics:
        st.subheader(f"Network summary  @  {horizon_step*5} min")
        mae  = float(np.mean(np.abs(pred_speed[h_idx] - actual_speed[h_idx])))
        mape = float(np.mean(
            np.abs(pred_speed[h_idx] - actual_speed[h_idx]) /
            np.clip(actual_speed[h_idx], 1, None)) * 100)

        c1, c2 = st.columns(2)
        c1.metric("MAE",  f"{mae:.2f} mph")
        c2.metric("MAPE", f"{mape:.1f}%")
        c1.metric("Avg Predicted", f"{pred_speed[h_idx].mean():.1f} mph")
        c2.metric("Avg Actual",    f"{actual_speed[h_idx].mean():.1f} mph")

        st.divider()
        rows = []
        for i, sid in enumerate(sensor_order):
            p = float(pred_speed[h_idx, i])
            a = float(actual_speed[h_idx, i])
            rows.append({
                "Sensor": SENSOR_NAMES.get(sid, f"Sensor {sid}"),
                "Predicted (mph)": round(p, 1),
                "Actual (mph)":    round(a, 1),
                "Error (mph)":     round(abs(p - a), 1),
            })
        st.dataframe(
            pd.DataFrame(rows).sort_values("Error (mph)", ascending=False),
            use_container_width=True, hide_index=True,
        )

    # ── per-sensor forecast charts ────────────────────────────────────────────
    st.divider()
    st.subheader("Predicted vs Actual — per sensor")

    hist_x   = [-STEP_MIN * (T - j) for j in range(T)]
    future_x = [s * STEP_MIN for s in range(1, T + 1)]

    cols = st.columns(5)
    for i, sid in enumerate(sensor_order):
        name = SENSOR_NAMES.get(sid, f"Sensor {sid}").split("/")[-1].strip()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_x, y=input_speed[:, i],
                                 mode="lines", name="History",
                                 line=dict(color="#94A3B8", width=1.5)))
        fig.add_trace(go.Scatter(x=future_x, y=actual_speed[:, i],
                                 mode="lines+markers", name="Actual",
                                 line=dict(color="#2563EB", width=2),
                                 marker=dict(size=4)))
        fig.add_trace(go.Scatter(x=future_x, y=pred_speed[:, i],
                                 mode="lines+markers", name="Predicted",
                                 line=dict(color="#EA580C", width=2, dash="dash"),
                                 marker=dict(size=4)))
        fig.add_vline(x=0, line_dash="dot", line_color="#64748B", line_width=1)
        fig.update_layout(
            title=dict(text=name, font_size=11),
            margin=dict(l=10, r=10, t=28, b=20), height=200,
            showlegend=(i == 0),
            xaxis=dict(title="min", zeroline=False, tickfont_size=8),
            yaxis=dict(title="mph", tickfont_size=8),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.35, font_size=9),
        )
        cols[i % 5].plotly_chart(fig, use_container_width=True,
                                 config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.title("Model Comparison")

    # ── aggregate metrics table ───────────────────────────────────────────────
    st.subheader("Test Set Results — full evaluation")
    rows = []
    for label, path in RESULT_FILES.items():
        if not os.path.exists(path):
            continue
        with open(path) as f:
            res = json.load(f)
        m = res.get("test_metrics", {})
        rows.append({
            "Model":       label,
            "MAE @15min":  m.get("15min", {}).get("mae",  None),
            "MAE @30min":  m.get("30min", {}).get("mae",  None),
            "MAE @60min":  m.get("60min", {}).get("mae",  None),
            "RMSE @15min": m.get("15min", {}).get("rmse", None),
            "MAPE @15min": m.get("15min", {}).get("mape", None),
        })

    df_res = pd.DataFrame(rows)
    best_mae = df_res["MAE @15min"].min()

    st.dataframe(
        df_res.style.map(
            lambda v: "background-color:#DCFCE7; font-weight:bold" if v == best_mae else "",
            subset=["MAE @15min"],
        ),
        use_container_width=True, hide_index=True,
    )

    # ── MAE by horizon bar chart ───────────────────────────────────────────────
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

    # ── live actual vs predicted — side by side ───────────────────────────────
    st.divider()
    st.subheader(f"Actual vs Predicted — {window_time.strftime('%a %b %d %Y  %H:%M')}")
    st.caption("Network-average speed (mean of all 10 sensors) — live inference on selected window.")

    T = 12
    hist_x   = [-STEP_MIN * (T - j) for j in range(T)]
    future_x = [s * STEP_MIN for s in range(1, T + 1)]

    # Run both live models
    results_live = {}
    for mname in LIVE_MODELS:
        with st.spinner(f"Running {mname}..."):
            ps, ac, ins, _ = run_inference(mname, window_idx)
            results_live[mname] = (ps, ac, ins)

    # ── network-average chart: all models + actual on one plot ────────────────
    fig = go.Figure()

    # history (same for all models — use first)
    ins0 = list(results_live.values())[0][2]
    fig.add_trace(go.Scatter(
        x=hist_x, y=ins0.mean(axis=1),
        mode="lines", name="History",
        line=dict(color="#94A3B8", width=2),
    ))

    # actual (same for all models)
    ac0 = list(results_live.values())[0][1]
    fig.add_trace(go.Scatter(
        x=future_x, y=ac0.mean(axis=1),
        mode="lines+markers", name="Actual",
        line=dict(color="#1e293b", width=2.5),
        marker=dict(size=6, symbol="circle"),
    ))

    # predicted per model
    for mname, (ps, ac, ins) in results_live.items():
        mae_val = float(np.mean(np.abs(ps - ac)))
        fig.add_trace(go.Scatter(
            x=future_x, y=ps.mean(axis=1),
            mode="lines+markers",
            name=f"{mname} (MAE {mae_val:.2f})",
            line=dict(color=MODEL_COLORS[mname], width=2, dash="dash"),
            marker=dict(size=5),
        ))

    fig.add_vline(x=0, line_dash="dot", line_color="#64748B", line_width=1.2,
                  annotation_text="now", annotation_position="top right")
    fig.update_layout(
        height=360, xaxis_title="Minutes from window start",
        yaxis_title="Avg speed (mph)",
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── per-sensor actual vs predicted at each horizon ─────────────────────────
    st.subheader("Per-Sensor Actual vs Predicted @ Each Horizon")

    horizon_tab_labels = ["15 min", "30 min", "60 min"]
    horizon_steps      = [3, 6, 12]
    tabs = st.tabs(horizon_tab_labels)

    for tab, h_step in zip(tabs, horizon_steps):
        with tab:
            h_idx = h_step - 1
            sensor_cols = st.columns(5)
            for i, sid in enumerate(sensor_order):
                sname = SENSOR_NAMES.get(sid, f"Sensor {sid}").split("/")[-1].strip()
                actual_val = float(list(results_live.values())[0][1][h_idx, i])

                fig = go.Figure()
                # Actual value — horizontal reference line
                fig.add_hline(y=actual_val, line_color="#1e293b", line_width=2,
                              annotation_text=f"Actual {actual_val:.1f}",
                              annotation_position="bottom right",
                              annotation_font_size=9)

                for mname, (ps, ac, ins) in results_live.items():
                    fig.add_trace(go.Bar(
                        name=mname,
                        x=[mname.split("—")[-1].strip().split("+")[0].strip()],
                        y=[float(ps[h_idx, i])],
                        marker_color=MODEL_COLORS[mname],
                        text=[f"{float(ps[h_idx,i]):.1f}"],
                        textposition="outside",
                    ))

                fig.update_layout(
                    title=dict(text=sname, font_size=10),
                    margin=dict(l=5, r=5, t=28, b=5), height=200,
                    showlegend=False,
                    yaxis=dict(title="mph", range=[
                        min(actual_val, min(float(ps[h_idx, i]) for ps, _, _ in results_live.values())) - 5,
                        max(actual_val, max(float(ps[h_idx, i]) for ps, _, _ in results_live.values())) + 8,
                    ], tickfont_size=8),
                    xaxis=dict(tickfont_size=8),
                    plot_bgcolor="white", paper_bgcolor="white",
                    bargap=0.3,
                )
                sensor_cols[i % 5].plotly_chart(fig, use_container_width=True,
                                                config={"displayModeBar": False})

    # ── full horizon line: per sensor ──────────────────────────────────────────
    st.subheader("Full Forecast Trajectory — Per Sensor")
    sensor_cols2 = st.columns(5)
    for i, sid in enumerate(sensor_order):
        sname = SENSOR_NAMES.get(sid, f"Sensor {sid}").split("/")[-1].strip()
        ins0 = list(results_live.values())[0][2]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_x, y=ins0[:, i],
                                 mode="lines", name="History",
                                 line=dict(color="#94A3B8", width=1.5)))
        fig.add_trace(go.Scatter(x=future_x, y=list(results_live.values())[0][1][:, i],
                                 mode="lines+markers", name="Actual",
                                 line=dict(color="#1e293b", width=2),
                                 marker=dict(size=4)))
        for mname, (ps, _, _) in results_live.items():
            fig.add_trace(go.Scatter(x=future_x, y=ps[:, i],
                                     mode="lines+markers", name=mname.split("—")[-1].strip(),
                                     line=dict(color=MODEL_COLORS[mname], width=2, dash="dash"),
                                     marker=dict(size=4)))
        fig.add_vline(x=0, line_dash="dot", line_color="#64748B", line_width=1)
        fig.update_layout(
            title=dict(text=sname, font_size=10),
            margin=dict(l=5, r=5, t=28, b=20), height=200,
            showlegend=(i == 0),
            xaxis=dict(title="min", zeroline=False, tickfont_size=8),
            yaxis=dict(title="mph", tickfont_size=8),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.4, font_size=8),
        )
        sensor_cols2[i % 5].plotly_chart(fig, use_container_width=True,
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
Grey traces show individual sensors — some run faster (highway) and some slower (city intersections).
The bottom panel shows daily speed variability — higher variance in mid-2021 as COVID restrictions lifted.
        """)

    with tab2:
        st.image(os.path.join(FIG_DIR, "stl_decomposition.png"), use_container_width=True)
        st.markdown("""
**Components:**
- **Trend**: slow decline from ~47 mph (Apr 2020, COVID low traffic) to ~41 mph (mid-2021), then recovery.
- **Seasonal (weekly, period=7)**: ±4 mph weekly cycle — weekends faster, Mon–Fri show rush-hour congestion.
- **Residual**: anomalies around the expected pattern. Large spikes = unusual traffic days.
        """)

    with tab3:
        st.image(os.path.join(FIG_DIR, "daily_weekly_pattern.png"), use_container_width=True)
        st.markdown("""
**Key patterns:**
- **Weekday rush hours**: speeds dip to ~35 mph at 8am and 5–6pm.
- **Weekend**: smooth ~55 mph profile all day with no morning dip.
- **Heatmap**: darkest cells (slowest) are Mon–Fri 7–9am and 4–7pm.
        """)

    with tab4:
        st.image(os.path.join(FIG_DIR, "anomaly_periods.png"), use_container_width=True)
        st.markdown("""
| Cause | Why it appears |
|---|---|
| COVID lockdown (Apr 2020) | Speeds *above* trend — roads empty, no rush-hour congestion |
| Heavy rain | Volume drops → free-flow speeds, or drivers slow down in severe rain |
| Winter Storm Uri (Feb 15, 2021) | Most extreme day — ~15°F, icy roads, city shut down |
| Holidays | Break the weekly pattern (Thanksgiving, Christmas, MLK Day, Labor Day) |
| High wind (Nov 15, 2020) | 20+ mph gusts slowed traffic |
| Event + rain (Aug 2021) | ACE permitted events coinciding with heavy rain |
        """)
