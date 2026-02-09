# Activate the project venv, creating and installing if needed.
# Expects PROJECT_DIR to be set by the calling script.

if [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo "Virtual environment not found — running first-time setup..."
    python3 -m venv "$PROJECT_DIR/.venv"
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.venv/bin/activate"
    pip install -e "$PROJECT_DIR" --quiet
else
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.venv/bin/activate"
fi
