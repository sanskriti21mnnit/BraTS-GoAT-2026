#!/usr/bin/env bash
# Local test run of the submission image.
# Usage: ./run.sh <input_dir> <output_dir> [image_name] [tag]
set -euo pipefail

INPUT_DIR="$(realpath "${1:?need input dir}")"
OUTPUT_DIR="$(realpath "${2:?need output dir}")"
IMAGE_NAME="${3:-brats-goat-swinunetr-ssl}"
TAG="${4:-latest}"

mkdir -p "${OUTPUT_DIR}"

# Synapse mounts /input read-only and /output read-write. Mirror that here.
docker run --rm --gpus all \
  -v "${INPUT_DIR}":/input:ro \
  -v "${OUTPUT_DIR}":/output:rw \
  "${IMAGE_NAME}:${TAG}"
