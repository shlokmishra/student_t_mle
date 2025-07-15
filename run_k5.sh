#!/bin/bash
# This script runs all experiments for k=5.

echo "--- Starting Experiment Batch for k=5 ---"


echo "--- Running k=5, m=5 ---"
python3 main.py --k 5 --m 5

echo "--- Running k=5, m=10 ---"
python3 main.py --k 5 --m 10

echo "--- Running k=5, m=20 ---"
python3 main.py --k 5 --m 20

echo "--- Running k=5, m=50 ---"
python3 main.py --k 5 --m 50

echo "--- Experiment Batch for k=5 Complete ---"