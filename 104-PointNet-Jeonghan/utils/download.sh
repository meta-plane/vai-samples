#!/bin/bash
# Download ModelNet40 dataset for PointNet testing
# This script downloads the dataset from Kaggle using their API

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
DATASET_DIR="${PROJECT_ROOT}/assets/datasets"
OUTPUT_FILE="${DATASET_DIR}/modelnet40.zip"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "╔════════════════════════════════════════════════════════╗"
echo "║         ModelNet40 Dataset Download Script            ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Create dataset directory
mkdir -p "${DATASET_DIR}"

# Check if already downloaded
if [ -d "${DATASET_DIR}/ModelNet40" ]; then
    echo -e "${YELLOW}⚠ ModelNet40 directory already exists${NC}"
    read -p "Do you want to re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}✓ Using existing dataset${NC}"
        exit 0
    fi
    echo "Removing existing directory..."
    rm -rf "${DATASET_DIR}/ModelNet40"
fi

# Download using curl from Kaggle API
echo "Downloading ModelNet40 dataset from Kaggle..."
echo "Dataset: balraj98/modelnet40-princeton-3d-object-dataset"
echo "Size: ~2GB (compressed)"
echo ""

curl -L -o "${OUTPUT_FILE}" \
    --progress-bar \
    "https://www.kaggle.com/api/v1/datasets/download/balraj98/modelnet40-princeton-3d-object-dataset"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Download failed${NC}"
    echo "Make sure you have internet connection"
    exit 1
fi

echo -e "${GREEN}✓ Download complete${NC}"
echo ""

# Extract
echo "Extracting archive..."
unzip -q "${OUTPUT_FILE}" -d "${DATASET_DIR}"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Extraction failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Extraction complete${NC}"
echo ""

# Show dataset structure
echo "Dataset structure:"
echo "  Location: ${DATASET_DIR}/ModelNet40/"
echo "  Categories:"
ls "${DATASET_DIR}/ModelNet40/" | head -10 | sed 's/^/    - /'
TOTAL_CATEGORIES=$(ls -1 "${DATASET_DIR}/ModelNet40/" | wc -l)
if [ ${TOTAL_CATEGORIES} -gt 10 ]; then
    echo "    ... and $((TOTAL_CATEGORIES - 10)) more categories"
fi
echo ""

# Optional: Remove zip file to save space
read -p "Remove downloaded zip file (saves 2GB)? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    rm "${OUTPUT_FILE}"
    echo -e "${GREEN}✓ Zip file removed${NC}"
fi

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║              Dataset Ready to Use!                    ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "Test the dataset with:"
echo "  ./build.sh"
echo "  cd /path/to/build && ./104-PointNet-Jeonghan/test_off_loader"
