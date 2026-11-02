#!/usr/bin/env bash
# Reproducible offline Health-track demo (no API key, no Qdrant server, no model download).
#
#   bash scripts/demo.sh                 # console walkthrough + self-check (exit 1 on any misbehaviour)
#   bash scripts/demo.sh --serve [PORT]  # same, then serve the web UI in demo mode (default port 8000)
#
# PowerShell equivalent: scripts/demo.ps1
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

"$PY" scripts/demo.py --check

if [[ "${1:-}" == "--serve" ]]; then
  PORT="${2:-8000}"
  echo
  echo "Serving the demo UI at http://localhost:${PORT}  (Ctrl+C to stop)"
  echo "  cited answer:       http://localhost:${PORT}/?q=How%20do%20I%20prepare%20ORS%20for%20a%20child%20with%20diarrhoea%3F"
  echo "  agent, infant fever: http://localhost:${PORT}/?agent=1&q=My%20child%20has%20fever%20since%20yesterday&age=1%20month"
  VOICE_RAG_DEMO=1 exec "$PY" -m uvicorn app.main:app --port "$PORT"
fi
