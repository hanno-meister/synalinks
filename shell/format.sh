#!/bin/bash
set -Euo pipefail

base_dir=$(dirname $(dirname $0))

uvx ruff check --config "${base_dir}/pyproject.toml" --fix .

uvx ruff format --config "${base_dir}/pyproject.toml" .
