#!/bin/bash
# Script 1 of 4 for running outlier analysis experiments.

echo "--- Starting Experiment Batch 1 ---"

# --- 1 Outlier ---
python3 main.py --num_outliers 1 -k 1 -m 3
python3 main.py --num_outliers 1 -k 1 -m 5
python3 main.py --num_outliers 1 -k 1 -m 10
python3 main.py --num_outliers 1 -k 1 -m 20

python3 main.py --num_outliers 1 -k 2 -m 3
python3 main.py --num_outliers 1 -k 2 -m 5
python3 main.py --num_outliers 1 -k 2 -m 10

# --- 2 Outliers ---
python3 main.py --num_outliers 2 -k 1 -m 5
python3 main.py --num_outliers 2 -k 1 -m 10
python3 main.py --num_outliers 2 -k 1 -m 20

# --- 3 Outliers ---
python3 main.py --num_outliers 3 -k 1 -m 10

echo "--- Experiment Batch 1 Finished ---"
