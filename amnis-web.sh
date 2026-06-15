#!/bin/bash
# Amnis Web UI — quick start script
# Usage: ./amnis-web.sh [port]
PORT=${1:-8799}
cd "$(dirname "$0")"
echo "☀️ Amnis Web UI starting on http://127.0.0.1:$PORT"
echo "   Press Ctrl+C to stop"
exec ~/amnis/env/bin/python -m amnis web
