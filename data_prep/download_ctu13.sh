#!/usr/bin/env bash
# Direct, no-registration download + extraction of CTU-13 (~1.9GB).
set -euo pipefail

TARGET_DIR="${1:-data/raw/ctu13}"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "Downloading CTU-13 into $TARGET_DIR ..."
curl -L -O https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2

echo "Extracting..."
tar -xjf CTU-13-Dataset.tar.bz2

echo "Done. Top-level contents:"
find . -maxdepth 2 -type d
