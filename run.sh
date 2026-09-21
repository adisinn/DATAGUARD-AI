#!/bin/bash
# DataGuard AI — Launch Script
# Run from the project directory: bash run.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Ollama ──────────────────────────────────────────────────────────────────
OLLAMA=/Applications/Ollama.app/Contents/Resources/ollama

if ! $OLLAMA list &>/dev/null; then
    echo "▶  Starting Ollama server…"
    $OLLAMA serve &>/tmp/ollama.log &
    sleep 3
fi

MODEL=$($OLLAMA list 2>/dev/null | grep "dataguard-ai")
if [ -z "$MODEL" ]; then
    echo "▶  Creating dataguard-ai model…"
    $OLLAMA create dataguard-ai -f "$SCRIPT_DIR/Modelfile"
fi

echo "✅  AI model ready: dataguard-ai"

# ── Kill any old instance on port 8000 ──────────────────────────────────────
lsof -ti:8000 | xargs kill -9 &>/dev/null || true

# ── Launch FastAPI server ────────────────────────────────────────────────────
echo "🛡️  Launching DataGuard AI at http://localhost:8000"
echo ""

export PYTHONPATH="$PARENT_DIR"
exec python3 "$SCRIPT_DIR/server.py"
