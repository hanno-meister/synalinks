#!/bin/bash
set -Eeuo pipefail

uv run pytest --cov-config=pyproject.toml