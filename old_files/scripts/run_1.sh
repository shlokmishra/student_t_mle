#!/bin/bash
# Script 1 of 4 for running outlier analysis experiments.

echo "--- Starting Experiment Batch 1 ---"

python3 main.py -m 10 -k 1
python3 main.py -m 10 -k 2
python3 main.py -m 10 -k 3
python3 main.py -m 10 -k 5
echo "--- Experiment Batch 1 Finished ---"
