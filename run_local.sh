#!/bin/bash
# Histology Cell Labelling App - Docker Runner
# This script builds and runs the application in a Docker container
#
# Usage:
#   ./run_local.sh /path/to/qa_sample_parent
#
# The data directory is mounted into the container at /data.
# In the app, enter '/data' as the directory path.

set -e  # Exit on error

IMAGE_NAME="histology-labelling-app"
CONTAINER_NAME="histology-labelling-app"

echo "Histology Cell Labelling App - Docker Setup"
echo "=============================================="
echo ""

# Check data directory argument
if [ $# -lt 1 ]; then
    echo "ERROR: No data directory supplied!"
    echo "Usage: $0 /path/to/qa_sample_parent"
    exit 1
fi

DATA_DIR="$(cd "$1" && pwd)"

# Check if container is running and stop it
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping running container..."
    docker stop $CONTAINER_NAME
fi

# Remove container if it exists
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Removing old container..."
    docker rm $CONTAINER_NAME
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building Docker image..."
docker build -t $IMAGE_NAME "$SCRIPT_DIR/labelling_app"

echo "Cleaning up dangling images..."
docker image prune -f

echo ""
echo "Running container..."
MSYS_NO_PATHCONV=1 docker run -d \
  --name $CONTAINER_NAME \
  -p 8000:8000 \
  -v "${DATA_DIR}:/data" \
  --restart unless-stopped \
  $IMAGE_NAME

echo ""
echo "Container started successfully!"
echo ""
echo "Access the app at: http://localhost:8000"
echo "In the app, enter '/data' as the directory path."
echo ""
echo "Useful commands:"
echo "  View logs:    docker logs -f $CONTAINER_NAME"
echo "  Stop:         docker stop $CONTAINER_NAME"
echo "  Remove:       docker rm $CONTAINER_NAME"
echo "  Restart:      docker restart $CONTAINER_NAME"
echo ""
