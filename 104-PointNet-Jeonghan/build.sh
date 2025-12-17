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
BUILD_TYPE=Debug
for arg in "$@"; do
    case $arg in
        --test|-t)
            BUILD_TESTS=ON
            ;;
        --release|-r)
            BUILD_TYPE=Release
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--test|-t] [--release|-r]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Building ${PROJECT_NAME} (incremental)"
echo "Build type: ${BUILD_TYPE}"
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

# Reconfigure with build options
echo "Configuring with BUILD_TESTING=${BUILD_TESTS}, CMAKE_BUILD_TYPE=${BUILD_TYPE}..."
cmake -DBUILD_TESTING=${BUILD_TESTS} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} "${PROJECT_ROOT}"

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

# Determine binary path based on build type
if [ "${BUILD_TYPE}" == "Release" ]; then
    BIN_DIR="${PROJECT_ROOT}/bin/release"
else
    BIN_DIR="${PROJECT_ROOT}/bin/debug"
fi

echo "Main executable:"
echo "  ${BIN_DIR}/${PROJECT_NAME}"
if [ "$BUILD_TESTS" == "ON" ]; then
    echo ""
    echo "Test executables:"
    echo "  ${BIN_DIR}/test_*"
    echo ""
    echo "Run all tests:"
    echo "  cd ${BUILD_DIR} && ctest --output-on-failure"
    echo ""
    echo "Run individual test:"
    echo "  ${BIN_DIR}/test_mlp"
fi
echo ""
echo "Run executable:"
echo "  ${BIN_DIR}/${PROJECT_NAME}"
echo ""

