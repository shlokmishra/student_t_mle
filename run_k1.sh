#!/bin/bash
# This script runs all experiments for k=1.

echo "--- Starting Experiment Batch for k=1 ---"

echo "--- Running k=1, m=3 ---"
python3 main.py --k 1 --m 5

echo "--- Running k=1, m=5 ---"
python3 main.py --k 1 --m 10

echo "--- Running k=1, m=20 ---"
python3 main.py --k 1 --m 20

echo "--- Running k=1, m=20 ---"
python3 main.py --k 1 --m 50

echo "--- Experiment Batch for k=1 Complete ---"