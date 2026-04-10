# Austin Traffic Flow Prediction — HM-STGN

Multimodal deep learning for Austin traffic speed prediction using sensor data, weather, road graphs, and event information.

## Setup

```bash
pip install -r requirements.txt
```

## Data Pipeline

```bash
bash scripts/run_data_pipeline.sh
```

This downloads sensor, weather, event, and graph data, then preprocesses everything into model-ready tensors under `data/processed/`.

## Training

```bash
# LSTM baseline (Layer 0)
bash scripts/run_experiment.sh configs/experiment/lstm_only.yaml

# GCN + LSTM (Layer 2)
bash scripts/run_experiment.sh configs/experiment/gcn_lstm.yaml

# Full HM-STGN
bash scripts/run_experiment.sh configs/experiment/full_hmstgn.yaml
```

## Project Structure

```
austin-traffic-prediction/
├── configs/            # Hyperparameter configs (YAML)
├── data/
│   ├── raw/            # Downloaded CSVs / API responses
│   ├── processed/      # Cleaned, aligned tensors (.pt)
│   └── graph/          # Adjacency matrices (.npy)
├── src/
│   ├── data/           # Download + preprocessing scripts
│   ├── models/         # All model modules
│   ├── training/       # Training loop + metrics
│   └── visualization/  # Plots and maps
├── notebooks/          # EDA and results notebooks
└── scripts/            # End-to-end shell scripts
```

## Model Layers (build incrementally)

| Layer | Model | Status |
|-------|-------|--------|
| 0 | LSTM baseline | TODO |
| 1 | LSTM + weather | TODO |
| 2 | GCN + LSTM (static graph) | TODO |
| 3 | Adaptive graph learning | TODO |
| 4 | Event-aware context encoder | TODO |
| 5 | Full HM-STGN | TODO |

## Evaluation

Primary metric: **MAE (mph)** at 15 / 30 / 60 min horizons.
Also reports RMSE and MAPE. All metrics on de-normalized predictions.
