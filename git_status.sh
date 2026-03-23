#!/bin/bash
# Show status of both the main repo and the paper submodule.
set -e

echo "=== Main repo ==="
git status --short
echo ""

echo "=== Paper submodule ==="
git -C paper status --short
