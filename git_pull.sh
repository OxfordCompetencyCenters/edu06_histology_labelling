#!/bin/bash
# Pull latest changes for both the main repo and the paper submodule.
set -e

echo "Pulling main repo..."
git pull

echo "Pulling paper submodule..."
git submodule update --init --recursive

echo "Done."
