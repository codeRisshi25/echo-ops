#!/bin/bash
# setup.sh — One-time setup for Echo-Ops
# Puts venv and model cache on /home/risshi/data to avoid filling /home partition

set -e   # exit on first error

VENV="/home/risshi/data/echo-ops-venv"
CACHE="/home/risshi/data/.fastembed_cache"
PROJECT="$(cd "$(dirname "$0")" && pwd)"

echo "⚡ Echo-Ops Setup"
echo "  Project : $PROJECT"
echo "  Venv    : $VENV"
echo "  Cache   : $CACHE"
echo ""

# 1. Create venv on /data
if [ ! -d "$VENV" ]; then
    echo "→ Creating venv at $VENV ..."
    python3 -m venv "$VENV"
else
    echo "→ Venv already exists, skipping."
fi

# 2. Install dependencies
echo "→ Installing dependencies..."
PIP_CACHE_DIR="$CACHE/pip" "$VENV/bin/pip" install -r "$PROJECT/requirements.txt" --quiet

# 3. Copy .env if not present
if [ ! -f "$PROJECT/.env" ]; then
    cp "$PROJECT/.env.example" "$PROJECT/.env"
    echo "→ Created .env from .env.example — please set OPENROUTER_API_KEY inside it."
else
    echo "→ .env already exists."
fi

echo ""
echo "✓ Setup complete!"
echo ""
echo "Run the demo:"
echo "  FASTEMBED_CACHE_PATH=$CACHE $VENV/bin/python $PROJECT/main.py --demo"
