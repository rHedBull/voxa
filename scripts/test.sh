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

exec "$ROOT/.venv/bin/pytest" "$@"
