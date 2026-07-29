#!/usr/bin/env bash
# One-shot: initialise git and make the first commit.
# Run from the repo root:  bash scripts/init_git.sh
set -euo pipefail

git init -b main
git add .
git commit -m "Initial commit: BraTS-GoAT SwinUNETR pipeline + U-Net placeholder"

echo
echo "Done. Now create an empty repo on GitHub (no README/gitignore) and run:"
echo "  git remote add origin git@github.com:<username>/brats-goat.git"
echo "  git push -u origin main"
