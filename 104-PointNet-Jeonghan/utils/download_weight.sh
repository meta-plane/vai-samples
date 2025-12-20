#!/bin/bash
# Download PointNet2 Part Segmentation Weights
# Source: yanx27/Pointnet_Pointnet2_pytorch
# Model: PointNet2 MSG Part Segmentation (best_model.pth)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
WEIGHTS_DIR="${PROJECT_ROOT}/assets/weights"
OUTPUT_FILE="${WEIGHTS_DIR}/best_model.pth"

# GitHub raw URL for the pretrained model
GITHUB_URL="https://github.com/yanx27/Pointnet_Pointnet2_pytorch/raw/master/log/part_seg/pointnet2_part_seg_msg/checkpoints/best_model.pth"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "╔════════════════════════════════════════════════════════╗"
echo "║     PointNet2 Part Segmentation Weights Download      ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo -e "${BLUE}Model: yanx27/Pointnet_Pointnet2_pytorch${NC}"
echo -e "${BLUE}Type: Part Segmentation (Multi-Scale Grouping)${NC}"
echo -e "${BLUE}Size: ~80MB${NC}"
echo ""

# Create weights directory
mkdir -p "${WEIGHTS_DIR}"

# Check if already downloaded
if [ -f "${OUTPUT_FILE}" ]; then
    echo -e "${YELLOW}⚠ Weight file already exists at: ${OUTPUT_FILE}${NC}"
    read -p "Do you want to re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}✓ Using existing weights${NC}"
        exit 0
    fi
    echo "Removing existing file..."
    rm -f "${OUTPUT_FILE}"
fi

# Download using curl
echo "Downloading PointNet2 weights from GitHub..."
echo "URL: ${GITHUB_URL}"
echo ""

curl -L -o "${OUTPUT_FILE}" \
    --progress-bar \
    --max-time 300 \
    "${GITHUB_URL}"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Download failed${NC}"
    echo "Please check your internet connection or try again later"
    exit 1
fi

echo -e "${GREEN}✓ Download complete${NC}"
echo ""

# Verify file
if [ -f "${OUTPUT_FILE}" ]; then
    FILE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
    echo "File information:"
    echo "  Path: ${OUTPUT_FILE}"
    echo "  Size: ${FILE_SIZE}"
    echo ""
    
    # Check if it's a valid PyTorch file (starts with PK for zip format)
    if file "${OUTPUT_FILE}" | grep -q "Zip\|data"; then
        echo -e "${GREEN}✓ File appears to be a valid PyTorch checkpoint${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: File may not be a valid PyTorch checkpoint${NC}"
    fi
else
    echo -e "${RED}✗ Download verification failed${NC}"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║              Weights Ready to Use!                    ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Ask if user wants to convert to SafeTensors
echo -e "${BLUE}Convert to SafeTensors format?${NC}"
echo "SafeTensors is the recommended format for this project."
read -p "Convert now? (Y/n): " -n 1 -r
echo
echo ""

if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    SAFETENSORS_OUTPUT="${WEIGHTS_DIR}/pointnet2_part_seg.safetensors"
    
    echo "Converting to SafeTensors format..."
    echo "Output: ${SAFETENSORS_OUTPUT}"
    echo ""
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}✗ Python3 not found${NC}"
        echo "Please install Python3 to convert weights"
        exit 1
    fi
    
    # Run conversion script
    cd "${PROJECT_ROOT}"
    python3 utils/convert_pytorch_weights.py \
        --checkpoint assets/weights/best_model.pth \
        --output assets/weights/pointnet2_part_seg.safetensors
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Conversion complete!${NC}"
        CONVERTED_SIZE=$(du -h "${SAFETENSORS_OUTPUT}" | cut -f1)
        echo "  SafeTensors file: ${SAFETENSORS_OUTPUT}"
        echo "  Size: ${CONVERTED_SIZE}"
    else
        echo ""
        echo -e "${YELLOW}⚠ Conversion failed${NC}"
        echo "You can convert manually later using:"
        echo "  python utils/convert_pytorch_weights.py \\"
        echo "    --checkpoint assets/weights/best_model.pth \\"
        echo "    --output assets/weights/pointnet2_part_seg.safetensors"
    fi
else
    echo "Skipping conversion."
    echo "You can convert manually later using:"
    echo "  python utils/convert_pytorch_weights.py \\"
    echo "    --checkpoint assets/weights/best_model.pth \\"
    echo "    --output assets/weights/pointnet2_part_seg.safetensors"
fi

echo ""
echo "For more information, see:"
echo "  - WEIGHTS_README.md"
echo "  - https://github.com/yanx27/Pointnet_Pointnet2_pytorch"
echo ""

