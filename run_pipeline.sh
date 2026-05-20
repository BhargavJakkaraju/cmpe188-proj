#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo "=== Step 1: EDA ==="
"$PYTHON" 01eda.py

echo "=== Step 2: Preprocessing ==="
"$PYTHON" 02_preprocessing.py

echo "=== Step 3: Model Training ==="
"$PYTHON" 03_model_training.py

echo "=== Step 4: Model Optimization ==="
"$PYTHON" 04_optimize_best_model.py

echo "=== Step 5: Synthetic Data + Batch Inference ==="
"$PYTHON" scripts/generate_synthetic.py --n 20

echo "=== Step 6: Demo Summary ==="
"$PYTHON" scripts/generate_demo_summary.py

echo "=== Pipeline complete ==="
echo "Run: streamlit run app/streamlit_app.py"
