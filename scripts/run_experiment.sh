#!/usr/bin/env bash
# Train and evaluate one model configuration.
# Usage: bash scripts/run_experiment.sh configs/experiment/full_hmstgn.yaml

set -e
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/experiment/full_hmstgn.yaml}
echo "=== Running experiment: $CONFIG ==="
python -m src.training.trainer --config "$CONFIG"
