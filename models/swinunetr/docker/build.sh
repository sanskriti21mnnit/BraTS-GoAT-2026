#!/usr/bin/env bash
# Build the BraTS-GoAT SwinUNETR-SSL submission image.
# Run from this directory. Expects the trained weights at ./model/model.pth
set -euo pipefail

IMAGE_NAME="${1:-brats-goat-swinunetr-ssl}"
TAG="${2:-latest}"

if [ ! -f "model/model.pth" ]; then
  echo "ERROR: model/model.pth not found."
  echo "Place your trained 3-region sigmoid checkpoint at ./model/model.pth first."
  exit 1
fi

docker build -t "${IMAGE_NAME}:${TAG}" .
echo "Built ${IMAGE_NAME}:${TAG}"
