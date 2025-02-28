#!/bin/bash
set -Eeuo pipefail

# Cleanup the cache
uv cache clean
# Uninstall current version
uv pip uninstall .
# Install new version
uv pip install .