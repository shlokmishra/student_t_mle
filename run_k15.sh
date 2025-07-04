#!/bin/bash
# This script runs all experiments for k=15.

echo "--- Starting Experiment Batch for k=15 ---"

echo "--- Running k=15, m=20 ---"
python3 main.py --k 15 --m 20

echo "--- Running k=15, m=50 ---"
python3 main.py --k 15 --m 50

echo "--- Running k=15, m=250 ---"
python3 main.py --k 15 --m 250

echo "--- Experiment Batch for k=15 Complete ---"