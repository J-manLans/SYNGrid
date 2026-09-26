#!/usr/bin/env bash
# Replay a fixed action sequence at every commit in a range, to find the first
# commit where the environment's observable behaviour changes.
#
# The probe script and the interpreter always come from the current checkout, so
# the harness and the dependency set are held constant and only the code under test
# varies. Worktrees are sparse (src/syn_grid only) to keep this cheap despite the
# repository's large reproduction_package.
#
# Usage:
#   replay_sweep.sh <start> [end] [output_dir]
#   replay_sweep.sh <start..end> [output_dir]
#
#   <start>       required. Oldest commit to sweep FROM (inclusive).
#   [end]         newest commit TO (inclusive). Defaults to HEAD.
#   [output_dir]  defaults to output/replay_diff.
#
# One commit means "from there to now"; two mean an explicit range. Oldest
# first, matching git's own argument order. The start commit IS swept, which
# git's A..B excludes -- that row is the baseline the rest are compared against.
#
# Examples:
#   bash scripts/replay_sweep.sh 15c225f
#   bash scripts/replay_sweep.sh 15c225f d439509
#   bash scripts/replay_sweep.sh v0.3.0..v0.4.0
#   PATHS="" bash scripts/replay_sweep.sh main HEAD
set -uo pipefail

usage() {
  # Header comment minus the Usage block, which the caller already prints.
  sed -n '2,8p;14,26p' "${BASH_SOURCE[0]}" | sed 's/^#\{1,\} \{0,1\}//'
}

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
  echo "usage: $(basename "$0") <start> [end] [output_dir]" >&2
  echo "       $(basename "$0") <start..end> [output_dir]" >&2
  echo >&2
  usage >&2
  exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Accept either "<start> [end]" or a single git-native "<start>..end".
case "$1" in
  *..*)
    RANGE="$1"
    OUT="${2:-$REPO/output/replay_diff}"
    ;;
  *)
    RANGE="$1..${2:-HEAD}"
    OUT="${3:-$REPO/output/replay_diff}"
    ;;
esac

WT_ROOT="${TMPDIR:-/tmp}/syngrid-replay-wt"
PY="$REPO/.venv/bin/python"
PROBE="$REPO/scripts/replay_probe.py"

# Exactly one "..". Two would make both ends extract cleanly and then fail inside
# git log, printing "fatal: bad revision" and continuing with a partial list.
case "$RANGE" in
  *..*..*)
    echo "range must contain exactly one '..': '$RANGE'" >&2
    exit 2
    ;;
esac

# Reject an unresolvable range up front. Without this git prints "fatal: bad
# revision" to stderr and the sweep quietly proceeds with a partial commit list.
for rev in "${RANGE%%..*}" "${RANGE##*..}"; do
  if ! git -C "$REPO" rev-parse --verify --quiet "${rev}^{commit}" >/dev/null; then
    echo "not a valid revision: '$rev'  (range was '$RANGE')" >&2
    exit 2
  fi
done

# Path filter. Commits that cannot change environment behaviour are skipped, which
# is what keeps the output readable. Set PATHS="" to sweep every commit.
PATHS="${PATHS-src/syn_grid/core src/syn_grid/gymnasium}"

if [ -n "$PATHS" ]; then
  mapfile -t REFS < <(git -C "$REPO" log --format=%h --reverse "$RANGE" -- $PATHS)
else
  mapfile -t REFS < <(git -C "$REPO" log --format=%h --reverse "$RANGE")
fi

# Add the start commit back: git's A..B excludes it, and it is the row every
# other row is compared against.
BASE="$(git -C "$REPO" rev-parse --short "${RANGE%%..*}" 2>/dev/null)" || BASE=""
if [ -n "$BASE" ] && [ "${REFS[0]:-}" != "$BASE" ]; then
  REFS=("$BASE" "${REFS[@]}")
fi

if [ "${#REFS[@]}" -eq 0 ]; then
  echo "no commits matched RANGE='$RANGE' PATHS='$PATHS'" >&2
  exit 1
fi
echo "==> sweeping ${#REFS[@]} commits over RANGE='$RANGE' PATHS='$PATHS'"
echo

mkdir -p "$OUT" "$WT_ROOT"

cleanup() {
  for ref in "${REFS[@]}"; do
    git -C "$REPO" worktree remove --force "$WT_ROOT/$ref" 2>/dev/null || true
  done
  rm -rf "$WT_ROOT"
}
trap cleanup EXIT

for ref in "${REFS[@]}"; do
  wt="$WT_ROOT/$ref"
  rm -rf "$wt"
  if ! git -C "$REPO" worktree add --no-checkout -q "$wt" "$ref" 2>/dev/null; then
    echo "$ref: WORKTREE FAILED"
    continue
  fi
  git -C "$wt" sparse-checkout init --cone >/dev/null 2>&1
  git -C "$wt" sparse-checkout set src/syn_grid >/dev/null 2>&1
  git -C "$wt" checkout -q "$ref" 2>/dev/null

  subj=$(git -C "$REPO" log -1 --format="%ad %s" --date=short "$ref")
  for mode in sc cont; do
    if [ "$mode" = "sc" ]; then
      extra=(--single-chain true --max-steps 40)
    else
      extra=(--single-chain false --max-steps 40)
    fi
    line=$(PYTHONPATH="$wt/src" "$PY" "$PROBE" \
      --label "$ref-$mode" --out "$OUT" --episodes 300 "${extra[@]}" 2>&1 | tail -1)
    case "$line" in
      "$ref-$mode":*) printf '%-10s %-6s %s\n' "$ref" "$mode" "${line#*: }" ;;
      *) printf '%-10s %-6s FAILED: %s\n' "$ref" "$mode" "$(echo "$line" | tail -1 | cut -c1-140)" ;;
    esac
  done
  printf '%-10s %s\n' "" "$subj"
done

# The dirty working tree is the code as currently edited, which is not a commit.
for mode in sc cont; do
  if [ "$mode" = "sc" ]; then
    extra=(--single-chain true --max-steps 40)
  else
    extra=(--single-chain false --max-steps 40)
  fi
  line=$(PYTHONPATH="$REPO/src" "$PY" "$PROBE" \
    --label "WORKTREE-$mode" --out "$OUT" --episodes 300 "${extra[@]}" 2>&1 | tail -1)
  printf '%-10s %-6s %s\n' "WORKTREE" "$mode" "${line#*: }"
done
