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

export VOXA_DATA_DIR="${VOXA_DATA_DIR:-$ROOT/data}"
mkdir -p "$VOXA_DATA_DIR/scenes" "$VOXA_DATA_DIR/annotations"

echo "──────────────────────────────────────────────────────────"
echo "  Voxa  →  http://$HOST:$PORT"
echo "  data  →  $VOXA_DATA_DIR"
echo "──────────────────────────────────────────────────────────"

exec "$ROOT/.venv/bin/uvicorn" main:app --host "$HOST" --port "$PORT" --reload
