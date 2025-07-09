#!/bin/bash
# This script runs all experiments for k=5.

echo "--- Starting Experiment Batch for k=5 ---"

echo "--- Running k=5, m=3 ---"
python3 main_outlier_experiment.py --k 5 --m 3

echo "--- Running k=5, m=5 ---"
python3 main_outlier_experiment.py --k 5 --m 5

# echo "--- Running k=5, m=10 ---"
# python3 main_outlier_experiment.py --k 5 --m 10

# echo "--- Running k=5, m=20 ---"
# python3 main_outlier_experiment.py --k 5 --m 20

echo "--- Experiment Batch for k=5 Complete ---"