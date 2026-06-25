#!/usr/bin/env bash
# Run the Voxa server. The server serves both the API (/api/*) and the
# frontend static files at /, so a single port is enough.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"

PORT="${VOXA_PORT:-8765}"
HOST="${VOXA_HOST:-127.0.0.1}"

if [[ ! -d "$ROOT/.venv" ]]; then
  echo "Creating venv at $ROOT/.venv …"
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/pip" install --quiet --upgrade pip
  "$ROOT/.venv/bin/pip" install --quiet -r requirements.txt
fi

# scan_schema: shared schema package (frame/layout/fingerprint/validate),
# installed editable from its sibling checkout under tools/. Kept out of
# requirements.txt because that file is pip-installed from two different cwds
# (run.sh from backend/, test.sh from repo root) so a relative path can't suit
# both. Ensured here unconditionally (idempotent) so a venv created before
# scan_schema became a dependency still self-heals on the next run.
"$ROOT/.venv/bin/python" -c 'import scan_schema' 2>/dev/null || \
  "$ROOT/.venv/bin/pip" install --quiet -e "$ROOT/../../scan-schema"

export VOXA_DATA_DIR="${VOXA_DATA_DIR:-$ROOT/data}"
mkdir -p "$VOXA_DATA_DIR/scenes" "$VOXA_DATA_DIR/annotations"

echo "──────────────────────────────────────────────────────────"
echo "  Voxa  →  http://$HOST:$PORT"
echo "  data  →  $VOXA_DATA_DIR"
echo "──────────────────────────────────────────────────────────"

# --reload uses inotify, which can blow past the system watcher limit if many
# other projects are also watched. Opt in via VOXA_RELOAD=1 if you want it.
RELOAD_FLAG=""
if [[ "${VOXA_RELOAD:-0}" == "1" ]]; then
  RELOAD_FLAG="--reload"
fi

exec "$ROOT/.venv/bin/uvicorn" main:app --host "$HOST" --port "$PORT" $RELOAD_FLAG
