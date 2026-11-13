# run with ./scripts/tidy.sh

#!/usr/bin/env bash
set -e

TARGETS="src/syn_grid tests"

echo "==> Fixing lint issues (unused imports, etc.)"
ruff check --fix $TARGETS

echo "==> Formatting"
ruff format $TARGETS

echo "==> Markdown lines past 120 characters (rewrap by hand)"
# Lists, never edits or fails. Code blocks, table rows and lines holding a link are skipped, since they cannot or
# should not be broken. README.md keeps a paragraph to a line, and the archive is not rewrapped, so both are left out.
find docs AGENTS.md CONTRIBUTING.md -name '*.md' -not -path 'docs/dev/archive/*' -print0 |
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
  '

echo "==> Done"