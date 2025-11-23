#!/bin/bash
# 104-PointNet-Jeonghan incremental build script
# External 라이브러리를 다시 빌드하지 않고 이 프로젝트만 빌드합니다.

set -e  # Stop script on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
PROJECT_NAME="104-PointNet-Jeonghan"

echo "=========================================="
echo "Building ${PROJECT_NAME} (incremental)"
echo "=========================================="
echo "Project directory: ${SCRIPT_DIR}"
echo "Build directory: ${BUILD_DIR}"
echo ""

# Check if build directory exists
if [ ! -d "${BUILD_DIR}" ]; then
    echo "Error: Build directory not found: ${BUILD_DIR}"
    echo "Please run the root build.sh first to configure the project:"
    echo "  cd ${PROJECT_ROOT}"
    echo "  ./build.sh"
    exit 1
fi

# Check if CMake has been configured
if [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    echo "Error: CMake configuration not found."
    echo "Please run the root build.sh first to configure the project:"
    echo "  cd ${PROJECT_ROOT}"
    echo "  ./build.sh"
    exit 1
fi

# Change to build directory
cd "${BUILD_DIR}"

# Build only this target
echo "Building ${PROJECT_NAME}..."
make ${PROJECT_NAME} -j$(nproc)

echo ""
echo "=========================================="
echo "${PROJECT_NAME} build complete!"
echo "=========================================="
echo "Executable: ${BUILD_DIR}/bin/debug/${PROJECT_NAME}"
echo ""
echo "To run:"
echo "  cd ${BUILD_DIR}/bin/debug"
echo "  ./${PROJECT_NAME}"
echo ""

