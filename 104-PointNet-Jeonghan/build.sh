#!/bin/bash
# 104-PointNet-Jeonghan incremental build script
# External 라이브러리를 다시 빌드하지 않고 이 프로젝트만 빌드합니다.
#
# Usage:
#   ./build.sh              # Build main project only
#   ./build.sh --test       # Build main project + tests

set -e  # Stop script on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
PROJECT_NAME="104-PointNet-Jeonghan"

# Parse arguments
BUILD_TESTS=OFF
if [ "$1" == "--test" ] || [ "$1" == "-t" ]; then
    BUILD_TESTS=ON
fi

echo "=========================================="
echo "Building ${PROJECT_NAME} (incremental)"
if [ "$BUILD_TESTS" == "ON" ]; then
    echo "Mode: WITH TESTS"
else
    echo "Mode: MAIN PROJECT ONLY"
fi
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

# Reconfigure if BUILD_TESTING changed
echo "Configuring with BUILD_TESTING=${BUILD_TESTS}..."
cmake -DBUILD_TESTING=${BUILD_TESTS} "${PROJECT_ROOT}"

if [ "$BUILD_TESTS" == "ON" ]; then
    echo "Building ${PROJECT_NAME} and all test targets..."
    # 모든 test_ 타겟을 빌드 (CMake가 생성한 타겟 사용)
    # test 타겟을 빌드하면 CTest에 등록된 모든 테스트가 빌드됨
    make ${PROJECT_NAME} -j$(nproc)
    make all -j$(nproc)  # 테스트 포함 전체 빌드
else
    # 지정된 프로젝트 타깃만 빌드
    echo "Building ${PROJECT_NAME}..."
    make ${PROJECT_NAME} -j$(nproc)
fi

echo ""
echo "=========================================="
echo "${PROJECT_NAME} build complete!"
echo "=========================================="
echo "Executable: ${BUILD_DIR}/bin/debug/${PROJECT_NAME}"
if [ "$BUILD_TESTS" == "ON" ]; then
    echo "Tests built (BUILD_TESTING=ON)."
    echo "You can run tests with:"
    echo "  cd ${BUILD_DIR}"
    echo "  ctest --output-on-failure"
fi
echo ""
echo "To run main executable:"
echo "  cd ${BUILD_DIR}/bin/debug"
echo "  ./${PROJECT_NAME}"
echo ""

