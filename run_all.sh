#!/usr/bin/env bash
set -euo pipefail

# Run all experiments sequentially using the YAML configs.
# Usage:
#   bash run_all.sh
# Optional:
#   bash run_all.sh /path/to/python
# Example:
#   bash run_all.sh .venv/bin/python

PYTHON_BIN="${1:-python}"

echo "Using python: ${PYTHON_BIN}"
${PYTHON_BIN} -V

echo "----------------------------------------"
echo "Running: forward"
${PYTHON_BIN} -m src.forward_train --config configs/forward.yaml

echo "----------------------------------------"
echo "Running: inverse"
${PYTHON_BIN} -m src.inverse_train --config configs/inverse.yaml

echo "----------------------------------------"
echo "Running: forward_scaled"
${PYTHON_BIN} -m src.forward_scaled_train --config configs/forward_scaled.yaml

echo "----------------------------------------"
echo "Done. Results written under:"
echo "  results/forward/"
echo "  results/inverse/"
echo "  results/forward_scaled/"
