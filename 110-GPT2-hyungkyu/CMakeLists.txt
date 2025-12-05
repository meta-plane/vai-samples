cmake_minimum_required(VERSION 3.16)

project(110-GPT2-hyungkyu)

# ============================================================================
# Common Source Files (shared across all executables)
# ============================================================================

set(COMMON_SOURCES
    # Core
    core/vulkanApp.cpp
    core/spirvHelpers.cpp
    core/globalContext.cpp

    # Tokenizer
    tokenizer/bpeTokenizer.cpp

    # Model components
    model/cache/kvCache.cpp
    model/embedding/embeddingNode.cpp
    model/attention/attentionNode.cpp
    model/transformerBlock/transformer.cpp
    model/lmHeadNode.cpp
)

# ============================================================================
# Executable 1: Inference CLI
# ============================================================================

add_executable(gpt2-inference
    main.cpp
    model/inference.cpp
    ${COMMON_SOURCES}
)

# ============================================================================
# Executable 2: Test Runner (Unit tests for GPT-2 layers)
# ============================================================================

add_executable(gpt2-unit-tests
    test/runTests.cpp
    test/graphTest.cpp
    test/jsonParser.cpp
    ${COMMON_SOURCES}
)

# ============================================================================
# Aggregate Target: Build All
# ============================================================================

# Create a target that builds all executables
add_custom_target(${PROJECT_NAME} ALL
    DEPENDS gpt2-inference gpt2-unit-tests
)

# ============================================================================
# Common Configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(
    -DPROJECT_CURRENT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

# Apply to all targets
foreach(target gpt2-inference gpt2-unit-tests)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${Vulkan_INCLUDE_DIRS}
    )

    target_link_libraries(${target} PRIVATE
        nlohmann_json
        glfw
        spirv-reflect-static
        ${Vulkan_LIBRARIES}
        glslang
        glslang-default-resource-limits
    )
endforeach()

# ============================================================================
# Installation (optional)
# ============================================================================

# Install executables
install(TARGETS gpt2-inference gpt2-unit-tests
    RUNTIME DESTINATION bin
)

# Print build information
message(STATUS "")
message(STATUS "═══════════════════════════════════════════════════════════")
message(STATUS "  GPT-2 Build Configuration")
message(STATUS "═══════════════════════════════════════════════════════════")
message(STATUS "  Build targets:")
message(STATUS "    ${PROJECT_NAME}         - Build all executables (default)")
message(STATUS "    gpt2-inference          - Inference CLI only")
message(STATUS "    gpt2-unit-tests         - Unit test runner")
message(STATUS "")
message(STATUS "  Executables:")
message(STATUS "    1. gpt2-inference       - Inference CLI (generate, compare, interactive)")
message(STATUS "    2. gpt2-unit-tests      - Unit tests for GPT-2 layers")
message(STATUS "═══════════════════════════════════════════════════════════")
message(STATUS "")
