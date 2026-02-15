#!/usr/bin/env bash
# Start / stop / restart the REST API server.
#
# Usage:
#   bash server.sh start   [--port 8000] [--device auto] [--output-dir ./output]
#   bash server.sh stop
#   bash server.sh restart [...]
#   bash server.sh status

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_DIR/scripts/_activate.sh"

PID_FILE="$PROJECT_DIR/.server.pid"
LOG_FILE="$PROJECT_DIR/server.log"

# ── Helpers ─────────────────────────────────────────────────────

is_running() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

do_start() {
    if is_running; then
        echo "Server is already running (PID $(cat "$PID_FILE"))"
        return 1
    fi

    # Pass remaining args (e.g. --port 9000 --device cuda) through to serve
    echo "Starting server... (log: $LOG_FILE)"
    nohup uv run rzem-ai-inference-engine serve "$@" > "$LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"

    # Wait briefly and check it didn't die immediately
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
        echo "Server started (PID $pid)"
    else
        echo "Server failed to start — check $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

do_stop() {
    if ! is_running; then
        echo "Server is not running"
        rm -f "$PID_FILE"
        return 0
    fi

    local pid
    pid="$(cat "$PID_FILE")"
    echo "Stopping server (PID $pid)..."
    kill "$pid"

    # Wait up to 10s for graceful shutdown
    local i=0
    while kill -0 "$pid" 2>/dev/null && [ $i -lt 10 ]; do
        sleep 1
        i=$((i + 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
        echo "Force-killing server..."
        kill -9 "$pid"
    fi

    rm -f "$PID_FILE"
    echo "Server stopped"
}

do_status() {
    if is_running; then
        echo "Server is running (PID $(cat "$PID_FILE"))"
    else
        echo "Server is not running"
        rm -f "$PID_FILE"
    fi
}

# ── Main ────────────────────────────────────────────────────────

cmd="${1:-}"
shift || true

case "$cmd" in
    start)   do_start "$@" ;;
    stop)    do_stop ;;
    restart) do_stop; do_start "$@" ;;
    status)  do_status ;;
    *)
        echo "Usage: $0 {start|stop|restart|status} [serve options...]"
        echo ""
        echo "Examples:"
        echo "  $0 start"
        echo "  $0 start --port 9000 --device cuda"
        echo "  $0 start --host 0.0.0.0 --port 8000 --output-dir /data/images"
        echo "  $0 stop"
        echo "  $0 restart --port 8000"
        echo "  $0 status"
        exit 1
        ;;
esac
