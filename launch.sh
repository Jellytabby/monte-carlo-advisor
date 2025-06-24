#!/usr/bin/bash

set -euo pipefail

LIST_FILE="$1"
core=8
while IFS= read -r bench_path; do
  # skip empty lines or comments
  [[ -z "$bench_path" || "${bench_path:0:1}" == "#" ]] && continue
  echo "Running Monte Carlo for $bench_path on core $core"
  python3 src/monte_carlo_main.py -lua -ia -r 200 -c $core $bench_path -t 120 --loop-unroll-advisor-model ../model.tflite --plot-directory model_max_32_v3 &> $bench_path.txt &
  core=$((core+1))
done < "$LIST_FILE"

wait
echo "All benchmarks completed"
