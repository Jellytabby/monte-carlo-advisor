#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <benchmark-list-file>"
  exit 1
fi

LIST_FILE="$1"

# collect only non-empty, non-comment lines
benchmarks=()
while IFS= read -r line; do
  [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
  benchmarks+=("$line")
done < "$LIST_FILE"

total=${#benchmarks[@]}
half=$(( (total + 1) / 2 ))   # first half gets the extra one if odd

RUNS=1000
# You can tweak this in one place to affect both halves:
CMD="python3 src/monte_carlo_main.py -lua -ia -r $RUNS -t 120 --plot-directory random_max_32_$RUNS"

core=8

echo "=== Running first half ($half benchmarks) ==="
for ((i=0; i<half; i++)); do
  bench="${benchmarks[i]}"
  echo "  [$((i+1))/$half] $bench → core $core"
  $CMD -c "$core" "$bench" &> "$bench.txt" &
  core=$((core+1))
done

wait
echo "=== First half complete ==="

echo "=== Running second half ($((total-half)) benchmarks) ==="
for ((i=half; i<total; i++)); do
  bench="${benchmarks[i]}"
  echo "  [$((i-half+1))/$((total-half))] $bench → core $core"
  $CMD -c "$core" "$bench" &> "$bench.txt" &
  core=$((core+1))
done

wait
echo "=== All benchmarks completed ==="

