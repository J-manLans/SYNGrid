# run with ./scripts/tidy.sh

#!/usr/bin/env bash
set -e

TARGETS="src/syn_grid tests"

echo "==> Fixing lint issues (unused imports, etc.)"
ruff check --fix $TARGETS

echo "==> Formatting"
ruff format $TARGETS

echo "==> Done"