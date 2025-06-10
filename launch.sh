#!/usr/bin/bash

set -euo pipefail


display_usage() {
  echo "Usage: $0 [-s|-c|-m] <bench-list-file>"
}

display_help() {
    display_usage
    echo ""
    echo "-s    Split benchmarks into <benchmark>_main.c and <benchmark>_module.[c|h] files"
    echo "-c    Clean created files"
}


LIST_FILE="$1"
core=0
while IFS= read -r bench_path; do
  # skip empty lines or comments
  [[ -z "$bench_path" || "${bench_path:0:1}" == "#" ]] && continue
  echo "Running Monte Carlo for $bench_path on core $core"
  taskset -c $core python3 monte_carlo_main.py -lua -ia -r 100 -c $((core+1)) $bench_path &> $bench_path.txt &
  core=$((core+2))
done < "$LIST_FILE"

wait
echo "All benchmarks completed"
