#!/bin/bash
set -Eeuo pipefail

uv run pytest --cov-config=pyproject.toml
uvx --from 'genbadge[coverage]' genbadge coverage -i coverage.xml