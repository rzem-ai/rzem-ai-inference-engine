# Ensure the project environment is synced.
# Expects PROJECT_DIR to be set by the calling script.

cd "$PROJECT_DIR"

if [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo "Virtual environment not found — running first-time setup..."
    uv sync
fi
