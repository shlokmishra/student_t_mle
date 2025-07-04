#!/bin/bash
# This script runs all experiments for k=3.

echo "--- Starting Experiment Batch for k=3 ---"

# We call the python executable from the venv directly.
# This is more robust than activating the environment.

# Run for m=20
echo "--- Running k=3, m=20 ---"
python3 main.py --k 3 --m 20

# Run for m=50
echo "--- Running k=3, m=50 ---"
python3 main.py --k 3 --m 50

# Run for m=250
echo "--- Running k=3, m=250 ---"
python3 main.py --k 3 --m 250

echo "--- Experiment Batch for k=3 Complete ---"