#!/usr/bin/env bash
# usage: ./check_missing.sh <dir> <benchmark-list-file>

if [ $# -ne 2 ]; then
  echo "Usage: $0 <directory> <benchmark-list-file>"
  exit 1
fi

DIR="$1"
LIST="$2"

# 1) extract the “expected” names = basename of each path in your list
mapfile -t expected < <(
  awk -F/ '/./{print $NF}' "$LIST" | sort -u
)

# 2) extract the “actual” names = subdirectory names under $DIR
mapfile -t actual < <(
  find "$DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -u
)

# 3) compare and print those in expected but not in actual
echo "Missing benchmarks:"
comm -23 <(printf '%s\n' "${expected[@]}") \
         <(printf '%s\n' "${actual[@]}")

