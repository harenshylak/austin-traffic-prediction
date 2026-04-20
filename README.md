# Austin Traffic Speed Prediction

Machine learning system that predicts vehicle speed at 10 traffic sensor locations across Austin, Texas for up to 60 minutes into the future. Takes 60 minutes of historical traffic observations as input and outputs speed predictions at 15-minute, 30-minute, and 60-minute horizons.

---

## Models

| Model | Type | MAE @15min | MAE @30min | MAE @60min |
|---|---|---|---|---|
| ARIMA(2,1,2) | Classical baseline | 3.276 mph | 3.542 mph | 3.914 mph |
| Chronos-T5-Base | Foundation model (zero-shot) | 3.446 mph | 3.649 mph | 4.101 mph |
| LSTM — Sensor Only | Deep learning | **3.174 mph** | **3.372 mph** | **3.637 mph** |
| LSTM + Weather | Deep learning | 3.227 mph | 3.426 mph | 3.701 mph |
| LSTM + Weather + Events | Deep learning | 3.317 mph | 3.503 mph | 3.733 mph |

All results on the Jul–Sep 2021 test set. Speed in mph, de-normalized.

---

## Dataset

| Source | Details |
|---|---|
| Traffic sensors | City of Austin Open Data Portal — 10 radar sensors, Apr 2020–Sep 2021, 5-min resolution |
| Weather | Open-Meteo historical API — temperature, precipitation, wind, humidity, visibility |
| Events | City of Austin ACE Events API — 31 permitted public events with exact timestamps |

**Splits:** Train Apr 2020–Mar 2021 · Validation Apr–Jun 2021 · Test Jul–Sep 2021

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run the Dashboard

```bash
streamlit run app.py
```

Opens an interactive dashboard at `http://localhost:8501` with three pages:

- **What-If Simulator** — Pick a sensor and time window, adjust weather/event conditions, compare Sensor-Only vs +Weather & Events model predictions side by side
- **Model Comparison** — Aggregate results table, MAE-by-horizon bar chart, live forecast comparison under custom weather conditions
- **Data Analytics** — STL decomposition, daily/weekly seasonality patterns, annotated anomaly periods

---

## Training

```bash
# LSTM sensor only
python -m src.training.trainer --config configs/experiment/lstm_only.yaml

# LSTM + weather (K=6)
python -m src.training.trainer --config configs/experiment/sensor_weather.yaml

# LSTM + weather + events (K=17)
python -m src.training.trainer --config configs/experiment/sensor_weather_events.yaml
```

### Baselines

```bash
# ARIMA(2,1,2) — classical baseline
python -m src.models.arima_baseline

# Chronos-T5-Base — zero-shot foundation model
python -m src.models.chronos_baseline --model-id amazon/chronos-t5-base --batch-windows 4
```

---

## Data Pipeline

```bash
bash scripts/run_data_pipeline.sh
```

Downloads and preprocesses sensor, weather, and event data into model-ready arrays under `data/graph/`.

---

## Project Structure

```
austin-traffic-prediction/
├── app.py                        # Streamlit dashboard
├── configs/
│   ├── default.yaml              # Base hyperparameters
│   └── experiment/
│       ├── lstm_only.yaml        # Sensor-only ablation
│       ├── sensor_weather.yaml   # +Weather ablation (K=6)
│       └── sensor_weather_events.yaml  # +Weather+Events (K=17)
├── data/
│   ├── raw/                      # Downloaded CSVs / API responses
│   ├── processed/                # Cleaned tensors and sensor metadata
│   └── graph/                    # Normalized arrays, scalers, splits
├── docs/
│   └── figures/                  # EDA plots (STL, seasonality, anomalies)
├── results/                      # Evaluation JSON files per run
├── checkpoints/                  # Saved model weights
├── src/
│   ├── data/                     # Download + preprocessing scripts
│   ├── models/
│   │   ├── lstm_baseline.py      # Encoder-decoder LSTM (sensor only)
│   │   ├── lstm_context.py       # LSTM + weather/events context injection
│   │   ├── arima_baseline.py     # ARIMA(2,1,2) per-sensor baseline
│   │   └── chronos_baseline.py   # Chronos-T5 zero-shot evaluation
│   ├── training/
│   │   ├── trainer.py            # Training loop, checkpointing, evaluation
│   │   └── metrics.py            # MAE, RMSE, MAPE computation
│   └── analysis/
│       └── eda_plots.py          # STL decomposition and seasonality plots
└── scripts/                      # End-to-end shell scripts
```

---

## Key Findings

1. **LSTM beats ARIMA** by ~0.10 mph at 15 min; gap widens to ~0.28 mph at 60 min — deep learning captures longer-range temporal patterns.
2. **Chronos-T5-Base (zero-shot) matches ARIMA** — a 200M-parameter foundation model without fine-tuning cannot capture local Austin traffic patterns.
3. **Adding weather/events hurts LSTM performance** slightly — simple hidden-state context injection is too coarse; the model cannot selectively use context at individual decoder steps.
4. **Teacher forcing mismatch** — decoder trained with ratio=0.5 but evaluated fully autoregressively (ratio=0.0), causing compounding errors at longer horizons.
5. **Anomaly detection** — STL residual analysis identifies COVID lockdown (Apr 2020), Winter Storm Uri (Feb 2021), holidays, and heavy rain events as the primary traffic anomaly drivers.
