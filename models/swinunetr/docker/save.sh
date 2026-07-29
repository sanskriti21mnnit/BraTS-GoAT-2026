#!/usr/bin/env bash
# Export the image to a tarball for upload to Synapse.
# Usage: ./save.sh [image_name] [tag]
set -euo pipefail

IMAGE_NAME="${1:-brats-goat-swinunetr-ssl}"
TAG="${2:-latest}"

docker save "${IMAGE_NAME}:${TAG}" | gzip > "${IMAGE_NAME}_${TAG}.tar.gz"
echo "Wrote ${IMAGE_NAME}_${TAG}.tar.gz"
