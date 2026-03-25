#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  Satya Drishti — Backend + Cloudflare Tunnel Launcher
# ═══════════════════════════════════════════════════════════
# Starts the FastAPI server and creates a free Cloudflare tunnel.
# The tunnel URL changes each run — update VITE_API_BASE in Vercel.
#
# Usage:  bash start-backend.sh
# Stop:   Ctrl+C (kills both server and tunnel)

set -e

# Config
VENV_DIR="D:/satyadrishti/.venv"
PROJECT_DIR="D:/satyadrishti"
PORT=8000
CLOUDFLARED="C:/Program Files (x86)/cloudflared/cloudflared.exe"

# Suppress noisy ML logs
export TQDM_DISABLE=1
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo ""
echo "  ╔═══════════════════════════════════════════╗"
echo "  ║   SATYA DRISHTI — Backend Launcher        ║"
echo "  ╚═══════════════════════════════════════════╝"
echo ""

# Activate venv
source "$VENV_DIR/Scripts/activate"

# Start FastAPI server in background
echo "[1/2] Starting FastAPI server on port $PORT..."
cd "$PROJECT_DIR"
python -m server.app &
SERVER_PID=$!
sleep 4

# Verify server is up
if ! curl -s http://localhost:$PORT/docs > /dev/null 2>&1; then
    echo "ERROR: Server failed to start!"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi
echo "  ✓ Server running (PID $SERVER_PID)"

# Start Cloudflare tunnel
echo "[2/2] Creating Cloudflare tunnel..."
"$CLOUDFLARED" tunnel --url http://localhost:$PORT 2>&1 | while IFS= read -r line; do
    # Extract and display the tunnel URL
    if echo "$line" | grep -q "trycloudflare.com"; then
        URL=$(echo "$line" | grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com')
        if [ -n "$URL" ]; then
            echo ""
            echo "  ╔═══════════════════════════════════════════════════════════╗"
            echo "  ║  TUNNEL LIVE!                                             ║"
            echo "  ║                                                           ║"
            echo "  ║  Backend URL: $URL"
            echo "  ║                                                           ║"
            echo "  ║  → Set this as VITE_API_BASE in Vercel dashboard          ║"
            echo "  ║  → Then redeploy the frontend                             ║"
            echo "  ║                                                           ║"
            echo "  ║  API Docs: ${URL}/docs                                    ║"
            echo "  ║  Local:    http://localhost:${PORT}/docs                   ║"
            echo "  ╚═══════════════════════════════════════════════════════════╝"
            echo ""
        fi
    fi
done &
TUNNEL_PID=$!

# Cleanup on Ctrl+C
trap "echo ''; echo 'Shutting down...'; kill $SERVER_PID $TUNNEL_PID 2>/dev/null; exit 0" INT TERM

echo ""
echo "Press Ctrl+C to stop both server and tunnel."
echo ""

# Wait for either to exit
wait
