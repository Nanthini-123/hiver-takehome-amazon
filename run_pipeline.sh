#!/usr/bin/env bash
set -e

# Detect Python binary
if [ -x "/usr/local/bin/python3" ]; then
  PYTHON_CMD="/usr/local/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
else
  PYTHON_CMD="python"
fi

echo "=========================================================="
echo "  Hiver SDE Take-Home: @AmazonHelp Production AI Pipeline "
echo "=========================================================="
echo "Using Python: $PYTHON_CMD"

echo "=== 1. Processing Kaggle Dataset for @AmazonHelp ==="
$PYTHON_CMD src/data_processor.py

echo "=== 2. Running Baselines ==="
$PYTHON_CMD src/baselines.py

echo "=== 3. Executing Agent Pipeline ==="
$PYTHON_CMD src/pipeline.py

echo "=== 4. Running Evaluation Harness & LLM-as-Judge ==="
$PYTHON_CMD src/evaluate.py

echo "=== 5. Running Human-Judge Agreement Verification ==="
$PYTHON_CMD src/human_agreement.py

echo "=== 6. Generating Confusion Matrix & Per-Intent Breakdown ==="
$PYTHON_CMD src/confusion_matrix.py

echo "=== 7. Evaluating Business Cost-Based Risk Optimization ==="
$PYTHON_CMD src/cost_analysis.py

echo "=========================================================="
echo "  Pipeline Completed Successfully in < 15 minutes!        "
echo "  To test live tweets, run: python3 src/demo_cli.py       "
echo "=========================================================="
