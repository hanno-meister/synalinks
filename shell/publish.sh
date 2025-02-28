#!/bin/bash
set -Eeuo pipefail

# Cleanup the cache
uv cache clean
# Build the project
uv build
# Publish on Pypi
uv publish