#!/usr/bin/env bash
# Run probe.py at every commit in a range, then on the working tree, to find the first commit where
# a scenario's behaviour changes. Each commit is checked out as a sparse worktree (src/syn_grid
# only); the probe and the interpreter come from this checkout, so only the code under test varies.
# Commits that touch none of $PATHS are skipped; set PATHS="" to sweep every commit.
#
# Usage:   sweep.sh <start> [end] [scenario]      end defaults to HEAD, the start is swept too
# Example: bash scripts/replay/sweep.sh af98fd8 HEAD goal_single_tier_chain_spatial
set -uo pipefail

[ $# -ge 1 ] && [ $# -le 3 ] || { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
START="$1" END="${2:-HEAD}" SCENARIO="${3:-goal_single_tier_chain_spatial}"
OUT="$REPO/output/replay_diff"
PY="$REPO/.venv/bin/python"
PROBE="$REPO/scripts/replay/probe.py"
PATHS="${PATHS-src/syn_grid/core src/syn_grid/gymnasium src/syn_grid/scenario src/syn_grid/config}"
WT_ROOT="${TMPDIR:-/tmp}/syngrid-replay-wt"

for rev in "$START" "$END"; do
  git -C "$REPO" rev-parse --verify --quiet "$rev^{commit}" >/dev/null \
    || { echo "not a valid revision: '$rev'" >&2; exit 2; }
done

# Git's A..B leaves out A, which is the baseline row, so it is put back first
mapfile -t REFS < <(git -C "$REPO" log --format=%h --reverse "$START..$END" -- $PATHS)
REFS=("$(git -C "$REPO" rev-parse --short "$START")" "${REFS[@]}")

mkdir -p "$OUT" "$WT_ROOT"
trap 'for r in "${REFS[@]}"; do git -C "$REPO" worktree remove --force "$WT_ROOT/$r" 2>/dev/null; done; rm -rf "$WT_ROOT"' EXIT

probe() {  # probe <label> <src dir>
  local line
  line=$(PYTHONPATH="$2" "$PY" "$PROBE" --label "$1" --out "$OUT" --scenario "$SCENARIO" 2>&1 | tail -1)
  case "$line" in
    "$1":*) printf '%-10s %s\n' "$1" "${line#*: }" ;;
    *) printf '%-10s FAILED: %s\n' "$1" "${line:0:140}" ;;
  esac
}

echo "==> ${#REFS[@]} commits, $START..$END, PATHS='$PATHS'"
for ref in "${REFS[@]}"; do
  wt="$WT_ROOT/$ref"
  git -C "$REPO" worktree add --no-checkout -q "$wt" "$ref" 2>/dev/null || { echo "$ref: WORKTREE FAILED"; continue; }
  git -C "$wt" sparse-checkout set --cone src/syn_grid >/dev/null 2>&1
  git -C "$wt" checkout -q "$ref" 2>/dev/null
  probe "$ref" "$wt/src"
  printf '%-10s %s\n' "" "$(git -C "$REPO" log -1 --format='%ad %s' --date=short "$ref")"
done
probe WORKTREE "$REPO/src"
