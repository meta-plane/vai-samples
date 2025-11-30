cmake_minimum_required(VERSION 3.16)

project(101-EfficientNet-dongyeob)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(EXTERNAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../external")
set(SPIRV_HEADERS_DIR "${EXTERNAL_DIR}/SPIRV-Headers")

find_package(Vulkan REQUIRED)

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${EXTERNAL_DIR}/glfw ${CMAKE_BINARY_DIR}/external/glfw)

set(SPIRV_HEADERS_SOURCE_DIR "${SPIRV_HEADERS_DIR}" CACHE STRING "" FORCE)
set(SPIRV-Headers_SOURCE_DIR "${SPIRV_HEADERS_DIR}" CACHE STRING "" FORCE)
set(mimalloc_SOURCE_DIR "${EXTERNAL_DIR}/mimalloc" CACHE STRING "" FORCE)
add_subdirectory(${EXTERNAL_DIR}/SPIRV-Tools ${CMAKE_BINARY_DIR}/external/SPIRV-Tools)

set(SPIRV_REFLECT_STATIC_LIB ON CACHE BOOL "" FORCE)
add_subdirectory(${EXTERNAL_DIR}/SPIRV-Reflect ${CMAKE_BINARY_DIR}/external/SPIRV-Reflect)

set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
set(BUILD_EXTERNAL OFF CACHE BOOL "" FORCE)
set(ALLOW_EXTERNAL_SPIRV_TOOLS ON CACHE BOOL "" FORCE)
set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
set(ENABLE_RTTI ON CACHE BOOL "" FORCE)
set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "" FORCE)
set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
add_subdirectory(${EXTERNAL_DIR}/glslang ${CMAKE_BINARY_DIR}/external/glslang)

add_subdirectory(${EXTERNAL_DIR}/json ${CMAKE_BINARY_DIR}/external/json)

add_executable(${PROJECT_NAME} 
    main.cpp
    test.cpp
    library/vulkanApp.cpp
    library/neuralNodes.cpp
    networks/efficientNet.cpp
    library/spirvHelpers.cpp
    library/jsonParser.cpp
    library/safeTensorsParser.cpp
)

add_definitions(
    -DPROJECT_CURRENT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    -DGLFW_INCLUDE_VULKAN
)

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EXTERNAL_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    spirv-reflect-static
    ${Vulkan_LIBRARIES}
    glslang
    glslang-default-resource-limits
    SPIRV-Tools
    nlohmann_json
)
