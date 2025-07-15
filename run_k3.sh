#!/bin/bash
# This script runs all experiments for k=2.

echo "--- Starting Experiment Batch for k=2 ---"

echo "--- Running k=2, m=5 ---"
python3 main.py --k 3 --m 5

echo "--- Running k=2, m=10 ---"
python3 main.py --k 3 --m 10

echo "--- Running k=2, m=20 ---"
python3 main.py --k 3 --m 20

echo "--- Running k=2, m=50 ---"
python3 main.py --k 3 --m 50


echo "--- Experiment Batch for k=2 Complete ---"