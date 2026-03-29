#!/bin/bash
echo "Starting FutureScope TMLR Benchmark Suite..."
python benchmark_suite.py
python run_ablation.py
echo "Benchmarks completed. Results in benchmarks.csv and figures/"
