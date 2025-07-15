#!/bin/bash
# Script 4 of 4 for running outlier analysis experiments.

echo "--- Starting Experiment Batch 4 ---"

# --- 2 Outliers ---
python3 main.py --num_outliers 2 -k 5 -m 5
python3 main.py --num_outliers 2 -k 5 -m 10
python3 main.py --num_outliers 2 -k 5 -m 20

# --- 3 Outliers ---
python3 main.py --num_outliers 3 -k 5 -m 10
python3 main.py --num_outliers 3 -k 5 -m 20

# --- 4 Outliers ---
python3 main.py --num_outliers 4 -k 2 -m 10
python3 main.py --num_outliers 4 -k 2 -m 20
python3 main.py --num_outliers 4 -k 3 -m 10
python3 main.py --num_outliers 4 -k 3 -m 20
python3 main.py --num_outliers 4 -k 5 -m 10
python3 main.py --num_outliers 4 -k 5 -m 20

echo "--- Experiment Batch 4 Finished ---"
