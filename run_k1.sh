#!/bin/bash
# This script runs all experiments for k=1.

echo "--- Starting Experiment Batch for k=1 ---"

# First, activate the virtual environment
source venv/bin/activate

# Run for m=20
echo "--- Running k=1, m=20 ---"
python main.py --k 1 --m 20

# Run for m=50
echo "--- Running k=1, m=50 ---"
python main.py --k 1 --m 50

# Run for m=250
echo "--- Running k=1, m=250 ---"
python main.py --k 1 --m 250

# Deactivate the environment
deactivate

echo "--- Experiment Batch for k=1 Complete ---"