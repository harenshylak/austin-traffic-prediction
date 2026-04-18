"""
eda_plots.py — Exploratory Data Analysis: STL Decomposition & Seasonality

Produces four publication-quality figures saved to docs/figures/:

  1. speed_overview.png     — Raw speed time series (all sensors, denormalized)
                              with train / val / test split bands
  2. stl_decomposition.png  — STL decomposition of network-average speed:
                              observed, trend, daily seasonal, weekly seasonal,
                              and residual components
  3. daily_weekly_pattern.png — Average speed profile by hour-of-day and
                                day-of-week (heatmap + line plots)
  4. anomaly_periods.png    — Residual magnitude over time; high-residual
                              windows flagged as anomaly periods

Usage:
    python -m src.analysis.eda_plots
"""

import json
import os
import pickle

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

# ── output directory ───────────────────────────────────────────────────────────
OUT_DIR = "docs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":     150,
})
BLUE   = "#2563EB"
ORANGE = "#EA580C"
GREEN  = "#16A34A"
GREY   = "#94A3B8"
RED    = "#DC2626"

SPLIT_COLORS = {
    "Train":      ("#DBEAFE", [pd.Timestamp("2020-04-01"), pd.Timestamp("2021-03-31")]),
    "Validation": ("#FEF3C7", [pd.Timestamp("2021-04-01"), pd.Timestamp("2021-06-30")]),
    "Test":       ("#DCFCE7", [pd.Timestamp("2021-07-01"), pd.Timestamp("2021-09-30")]),
}

# ── load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
traffic_norm = np.load("data/graph/traffic.npy", mmap_mode="r")   # (T, N, 3)
timestamps   = np.load("data/graph/timestamps.npy", allow_pickle=True)
with open("data/graph/scaler_traffic.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("data/graph/sensor_order.json") as f:
    sensor_ids = json.load(f)

# denormalize speed (feature 0) → mph
# scaler was fit on all 3 features; undo only feature 0 manually
speed_norm = np.array(traffic_norm[:, :, 0])                        # (T, N)
speed_mph  = speed_norm * scaler.scale_[0] + scaler.mean_[0]        # (T, N)

ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
N  = speed_mph.shape[1]

# network-average speed (mean across all 10 sensors)
avg_speed = pd.Series(speed_mph.mean(axis=1), index=ts)

print(f"  Timesteps: {len(ts):,}  |  Sensors: {N}  |  "
      f"Speed range: {speed_mph.min():.1f}–{speed_mph.max():.1f} mph")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Raw speed overview with split bands
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 1: speed overview...")

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})

ax = axes[0]
# individual sensors (thin, transparent)
for n in range(N):
    ax.plot(ts, speed_mph[:, n], color=GREY, alpha=0.25, linewidth=0.4)
# network average
ax.plot(ts, avg_speed, color=BLUE, linewidth=1.2, label="Network average", zorder=3)

for label, (color, (t0, t1)) in SPLIT_COLORS.items():
    ax.axvspan(t0, t1, color=color, alpha=0.55, label=label)

ax.set_ylabel("Speed (mph)")
ax.set_title("Austin Traffic Speed — Full Dataset (10 sensors, 5-min resolution)",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, ncol=4)
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

# daily std dev (variability)
ax2 = axes[1]
daily_std = avg_speed.resample("D").std()
ax2.bar(daily_std.index, daily_std.values, width=1, color=ORANGE, alpha=0.7)
for label, (color, (t0, t1)) in SPLIT_COLORS.items():
    ax2.axvspan(t0, t1, color=color, alpha=0.4)
ax2.set_ylabel("Daily Std Dev\n(mph)")
ax2.set_xlabel("")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
fig.autofmt_xdate(rotation=30)

plt.tight_layout()
path = os.path.join(OUT_DIR, "speed_overview.png")
plt.savefig(path, bbox_inches="tight")
plt.close()
print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — STL decomposition
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 2: STL decomposition (this may take ~30s)...")

# STL on daily-resampled data to keep it tractable (one year of daily means)
# Use weekly seasonality (period=7) on daily data
daily_avg = avg_speed.resample("D").mean().dropna()

stl = STL(daily_avg, period=7, robust=True)
result = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

components = [
    (daily_avg,          "Observed (network avg speed)",  BLUE),
    (result.trend,       "Trend",                         ORANGE),
    (result.seasonal,    "Seasonal (weekly)",             GREEN),
    (result.resid,       "Residual",                      RED),
]

for ax, (series, label, color) in zip(axes, components):
    ax.plot(series.index, series.values, color=color, linewidth=1.0)
    if label == "Residual":
        ax.fill_between(series.index, series.values, 0,
                        where=series.values > 0, color=RED,   alpha=0.4)
        ax.fill_between(series.index, series.values, 0,
                        where=series.values < 0, color=BLUE,  alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    for lbl, (bg, (t0, t1)) in SPLIT_COLORS.items():
        ax.axvspan(t0, t1, color=bg, alpha=0.35)
    ax.set_ylabel(label, fontsize=9)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
fig.autofmt_xdate(rotation=30)
fig.suptitle("STL Decomposition — Austin Network-Average Traffic Speed (Daily, period=7 days)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
path = os.path.join(OUT_DIR, "stl_decomposition.png")
plt.savefig(path, bbox_inches="tight")
plt.close()
print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Daily and weekly seasonality patterns
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 3: daily/weekly patterns...")

df = pd.DataFrame({"speed": avg_speed.values}, index=ts)
df["hour"]    = df.index.hour
df["minute"]  = df.index.minute
df["tod"]     = df["hour"] + df["minute"] / 60          # time-of-day in hours
df["dow"]     = df.index.dayofweek                      # 0=Mon … 6=Sun
df["weekday"] = df["dow"] < 5

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ── hourly profile (weekday vs weekend) ───────────────────────────────────────
ax = axes[0]
for is_wd, label, color in [(True, "Weekday", BLUE), (False, "Weekend", ORANGE)]:
    sub = df[df["weekday"] == is_wd].groupby("tod")["speed"]
    m   = sub.mean()
    s   = sub.std()
    ax.plot(m.index, m.values, color=color, linewidth=2, label=label)
    ax.fill_between(m.index, m - s, m + s, color=color, alpha=0.15)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Avg speed (mph)")
ax.set_title("Average Speed by Time of Day", fontweight="bold")
ax.set_xticks(range(0, 25, 3))
ax.set_xlim(0, 24)
ax.legend()

# ── day-of-week bar chart ──────────────────────────────────────────────────────
ax = axes[1]
dow_mean = df.groupby("dow")["speed"].mean()
dow_std  = df.groupby("dow")["speed"].std()
colors   = [BLUE if d < 5 else ORANGE for d in dow_mean.index]
bars = ax.bar(range(7), dow_mean.values, color=colors,
              yerr=dow_std.values, capsize=4, alpha=0.85,
              error_kw={"elinewidth": 1.2})
ax.set_xticks(range(7))
ax.set_xticklabels(DAYS)
ax.set_ylabel("Avg speed (mph)")
ax.set_title("Average Speed by Day of Week", fontweight="bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="Weekday"),
                   Patch(color=ORANGE, label="Weekend")], fontsize=9)

# ── heatmap: hour × day-of-week ───────────────────────────────────────────────
ax = axes[2]
pivot = df.groupby(["dow", "hour"])["speed"].mean().unstack(level=1)  # (7, 24)
im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
               vmin=pivot.values.min(), vmax=pivot.values.max())
ax.set_xticks(range(0, 24, 3))
ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)], fontsize=8)
ax.set_yticks(range(7))
ax.set_yticklabels(DAYS)
ax.set_xlabel("Hour of day")
ax.set_title("Speed Heatmap (Day × Hour)", fontweight="bold")
plt.colorbar(im, ax=ax, label="Avg speed (mph)", shrink=0.85)

plt.suptitle("Traffic Speed Seasonality — Austin Sensor Network",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
path = os.path.join(OUT_DIR, "daily_weekly_pattern.png")
plt.savefig(path, bbox_inches="tight")
plt.close()
print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Anomaly periods (high STL residuals)
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 4: anomaly periods...")

resid = result.resid.dropna()
threshold = resid.abs().quantile(0.95)          # top 5% residuals = anomalies
is_anomaly = resid.abs() > threshold

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})

ax = axes[0]
ax.plot(daily_avg.index, daily_avg.values, color=BLUE, linewidth=0.9, label="Observed speed")
ax.plot(result.trend.index, result.trend.values, color=ORANGE,
        linewidth=2, label="STL trend", zorder=3)
ax.scatter(daily_avg.index[is_anomaly], daily_avg.values[is_anomaly],
           color=RED, s=25, zorder=4, label=f"Anomaly day (|resid| > 95th pct)")
for label, (color, (t0, t1)) in SPLIT_COLORS.items():
    ax.axvspan(t0, t1, color=color, alpha=0.35)
ax.set_ylabel("Speed (mph)")
ax.set_title("Austin Traffic Anomaly Periods — STL Residual Analysis",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9)

ax2 = axes[1]
ax2.bar(resid.index, resid.abs().values, width=1,
        color=[RED if v else GREY for v in is_anomaly], alpha=0.8)
ax2.axhline(threshold, color=RED, linewidth=1.2, linestyle="--",
            label=f"95th pct threshold ({threshold:.2f} mph)")
for label, (color, (t0, t1)) in SPLIT_COLORS.items():
    ax2.axvspan(t0, t1, color=color, alpha=0.35)
ax2.set_ylabel("|Residual| (mph)")
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
fig.autofmt_xdate(rotation=30)

plt.tight_layout()
path = os.path.join(OUT_DIR, "anomaly_periods.png")
plt.savefig(path, bbox_inches="tight")
plt.close()
print(f"  Saved {path}")

print("\nAll figures saved to", OUT_DIR)
