#!/usr/bin/env sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"

exec python3 -m uvicorn app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-9001}"
