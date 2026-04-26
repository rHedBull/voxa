#!/usr/bin/env bash
# Quick helper: copy a PLY/GLB into data/scenes/<name>/source.<ext>
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <name> <path-to-ply-or-glb>" >&2
  exit 1
fi

NAME="$1"
SRC="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="${VOXA_DATA_DIR:-$ROOT/data}/scenes/$NAME"
EXT="${SRC##*.}"
EXT_LC="$(echo "$EXT" | tr '[:upper:]' '[:lower:]')"

if [[ "$EXT_LC" != "ply" && "$EXT_LC" != "glb" ]]; then
  echo "Source must be .ply or .glb (got .$EXT)" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
cp "$SRC" "$DEST_DIR/source.$EXT_LC"
echo "Imported $SRC → $DEST_DIR/source.$EXT_LC"
