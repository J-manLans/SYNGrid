#!/usr/bin/env bash
# Run with ./scripts/tidy.sh to fix and format, or ./scripts/tidy.sh --check to change nothing and fail on anything
# the plain run would fix or list. CI runs the --check form on a pull request into main and a push to it.
set -e

TARGETS="src/syn_grid tests"
CHECK=false
if [ "$1" = "--check" ]; then
  CHECK=true
fi

if $CHECK; then
  echo "==> Checking lint"
  ruff check $TARGETS
  echo "==> Checking formatting"
  ruff format --check $TARGETS
else
  echo "==> Fixing lint issues (unused imports, etc.)"
  ruff check --fix $TARGETS
  echo "==> Formatting"
  ruff format $TARGETS
fi

echo "==> Markdown lines past 120 characters (rewrap by hand)"
# Lists, never edits. Code blocks, table rows and lines holding a link are skipped, since they cannot or should not be
# broken. README.md keeps a paragraph to a line, and the archive is not rewrapped, so both are left out.
LONG=$(find docs AGENTS.md CONTRIBUTING.md -name '*.md' -not -path 'docs/dev/archive/*' -print0 |
  xargs -0 awk '
    FNR == 1          { fence = 0 }
    /^```/            { fence = !fence; next }
    fence             { next }
    /^\|/             { next }
    /https?:\/\//     { next }
    {
      t = $0
      gsub(/[\200-\277]/, "", t)     # count characters, not bytes: drop UTF-8 continuation bytes
      if (length(t) > 120) printf "%s:%d: %d chars\n", FILENAME, FNR, length(t)
    }
  ')
if [ -n "$LONG" ]; then
  echo "$LONG"
  if $CHECK; then
    exit 1
  fi
fi

echo "==> Done"
