#!/usr/bin/env bash
# Run backend pytest suite from the repo root using the local venv.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d "$ROOT/.venv" ]]; then
  echo "No .venv yet — run scripts/run.sh once to bootstrap it." >&2
  exit 1
fi

# Install dev deps on demand (cheap if already present).
"$ROOT/.venv/bin/pip" install --quiet -r "$ROOT/backend/requirements-dev.txt"

# scan_schema is a runtime dependency but lives editable outside requirements
# (see run.sh). test.sh is an independent entry point, so ensure it here too.
"$ROOT/.venv/bin/python" -c 'import scan_schema' 2>/dev/null || \
  "$ROOT/.venv/bin/pip" install --quiet -e "$ROOT/../../scan-schema"

exec "$ROOT/.venv/bin/pytest" "$@"
