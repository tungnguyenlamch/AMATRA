#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required for environments/set_up.sh"
    exit 1
fi

if ! conda info --envs | grep -q "py11"; then
    echo "Creating conda environment from $ROOT_DIR/environments/py11.yml"
    conda env create -f "$ROOT_DIR/environments/py11.yml"
fi

if [[ -f "$ENV_FILE" ]]; then
    echo ".env already exists at $ENV_FILE"
    exit 0
fi

cat > "$ENV_FILE" <<'EOF'
HF_TOKEN=
GOOGLE_APPLICATION_CREDENTIALS=
EOF

echo "Created $ENV_FILE"
