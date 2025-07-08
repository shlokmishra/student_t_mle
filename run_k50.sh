#!/bin/bash
# This script runs all experiments for k=50.

echo "--- Starting Experiment Batch for k=50 ---"

echo "--- Running k=50, m=20 ---"
python3 main_outlier_experiment.py --k 50 --m 20

echo "--- Running k=50, m=50 ---"
python3 main_outlier_experiment.py --k 50 --m 50

echo "--- Running k=50, m=250 ---"
python3 main_outlier_experiment.py --k 50 --m 250

echo "--- Experiment Batch for k=50 Complete ---"