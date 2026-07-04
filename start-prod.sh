#!/usr/bin/env bash
# Production start: build the frontend bundle, then serve API + static UI on
# one port via the backend. Override host/port with VOXA_HOST / VOXA_PORT.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "Building frontend …"
npm run build

exec bash "$ROOT/scripts/run.sh"
