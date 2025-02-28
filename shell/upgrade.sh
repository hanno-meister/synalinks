#!/bin/bash
set -Eeuo pipefail

# Upgrade packages
uv lock --upgrade
# Sync lock
uv sync