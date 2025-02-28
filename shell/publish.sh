#!/bin/bash
set -Eeuo pipefail

uv build
uv publish