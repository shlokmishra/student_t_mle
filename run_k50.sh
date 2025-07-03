#!/bin/bash
# This script runs all experiments for k=50.

echo "--- Starting Experiment Batch for k=50 ---"
VENV_PYTHON="venv/bin/python"

echo "--- Running k=50, m=20 ---"
python3 main.py --k 50 --m 20

echo "--- Running k=50, m=50 ---"
python3 main.py --k 50 --m 50

echo "--- Running k=50, m=250 ---"
python3 main.py --k 50 --m 250

echo "--- Experiment Batch for k=50 Complete ---"