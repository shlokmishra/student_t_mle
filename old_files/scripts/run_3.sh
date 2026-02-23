#!/bin/bash
# Script 3 of 4 for running outlier analysis experiments.

echo "--- Starting Experiment Batch 3 ---"

# --- 1 Outlier ---
python3 main.py --num_outliers 1 -k 5 -m 3
python3 main.py --num_outliers 1 -k 5 -m 5
python3 main.py --num_outliers 1 -k 5 -m 10
python3 main.py --num_outliers 1 -k 5 -m 20

# --- 2 Outliers ---
python3 main.py --num_outliers 2 -k 3 -m 5
python3 main.py --num_outliers 2 -k 3 -m 10
python3 main.py --num_outliers 2 -k 3 -m 20

# --- 3 Outliers ---
python3 main.py --num_outliers 3 -k 3 -m 10
python3 main.py --num_outliers 3 -k 3 -m 20

# --- 4 Outliers ---
python3 main.py --num_outliers 4 -k 1 -m 10
python3 main.py --num_outliers 4 -k 1 -m 20


echo "--- Experiment Batch 3 Finished ---"
