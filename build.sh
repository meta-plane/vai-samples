#!/bin/bash
# 루트 프로젝트 빌드 스크립트

set -e  # 에러 발생 시 스크립트 중단

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
GCC_13_PATH="/usr/bin/gcc-13"
GXX_13_PATH="/usr/bin/g++-13"

echo "=========================================="
echo "vai-samples 프로젝트 빌드 시작"
echo "=========================================="
echo "프로젝트 디렉토리: ${PROJECT_DIR}"
echo "빌드 디렉토리: ${BUILD_DIR}"
echo ""

# CMake 버전 확인
CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
echo "CMake 버전: ${CMAKE_VERSION}"

# CMake 3.22.1 이상 필요 (SPIRV-Tools 요구사항)
REQUIRED_CMAKE_VERSION="3.22.1"
if [ "$(printf '%s\n' "${REQUIRED_CMAKE_VERSION}" "${CMAKE_VERSION}" | sort -V | head -n1)" != "${REQUIRED_CMAKE_VERSION}" ]; then
    echo ""
    echo "경고: CMake ${REQUIRED_CMAKE_VERSION} 이상이 필요합니다."
    echo "현재 버전: ${CMAKE_VERSION}"
    echo ""
    exit 1
fi

# GCC 14 확인
if [ ! -f "${GCC_13_PATH}" ] || [ ! -f "${GXX_13_PATH}" ]; then
    echo "경고: GCC 13를 찾을 수 없습니다."
    echo "설치 방법: sudo apt-get install gcc-14 g++-14"
    echo "기본 컴파일러를 사용합니다."
    GCC_13_PATH=""
    GXX_13_PATH=""
fi

# 빌드 디렉토리 생성
if [ -d "${BUILD_DIR}" ]; then
    echo "기존 빌드 디렉토리 정리 중..."
    rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"

# CMake 설정
echo ""
echo "CMake 설정 중..."
cd "${BUILD_DIR}"

if [ -n "${GCC_13_PATH}" ] && [ -n "${GXX_13_PATH}" ]; then
    cmake -DCMAKE_C_COMPILER="${GCC_13_PATH}" \
          -DCMAKE_CXX_COMPILER="${GXX_13_PATH}" \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          "${PROJECT_DIR}"
else
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "${PROJECT_DIR}"
fi

# 빌드 실행
echo ""
echo "빌드 실행 중..."
make -j$(nproc)

echo ""
echo "=========================================="
echo "빌드 완료!"
echo "=========================================="
