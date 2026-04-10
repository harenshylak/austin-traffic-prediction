#!/usr/bin/env bash
# End-to-end: download all raw data then preprocess into model-ready tensors.
# Run from the project root: bash scripts/run_data_pipeline.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Downloading sensor data ==="
python src/data/download_sensors.py

echo "=== Downloading weather data ==="
python src/data/download_weather.py

echo "=== Downloading events data ==="
python src/data/download_events.py

echo "=== Building road graph ==="
python src/data/build_graph.py

echo "=== Preprocessing ==="
python src/data/preprocess.py

echo "=== Done. Processed tensors are in data/processed/ ==="
