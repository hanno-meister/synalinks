#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))

echo "Installing UV project manager..."

curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv