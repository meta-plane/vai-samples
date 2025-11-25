cmake_minimum_required(VERSION 3.16)

project(110-GPT2-hyungkyu)

# ============================================================================
# Common Source Files (shared across all executables)
# ============================================================================

set(COMMON_SOURCES
    # Core
    core/vulkanApp.cpp
    core/spirvHelpers.cpp
    core/jsonParser.cpp
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
# Executable 2: Test Runner (unit tests)
# ============================================================================

add_executable(runAllTests
    test/runAllTests.cpp
    test/layerTests.cpp
    test/attentionTests.cpp
    ${COMMON_SOURCES}
)

# ============================================================================
# Aggregate Target: Build All
# ============================================================================

# Create a target that builds both executables
add_custom_target(${PROJECT_NAME} ALL
    DEPENDS gpt2-inference runAllTests
)

# ============================================================================
# Common Configuration
# ============================================================================

add_definitions(
    -DPROJECT_CURRENT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

# Apply to all targets
foreach(target gpt2-inference runAllTests)
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
install(TARGETS gpt2-inference runAllTests
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
message(STATUS "    runAllTests             - Unit test runner only")
message(STATUS "")
message(STATUS "  Executables:")
message(STATUS "    1. gpt2-inference       - Inference CLI (generate, compare, interactive)")
message(STATUS "    2. runAllTests          - Unit test runner")
message(STATUS "═══════════════════════════════════════════════════════════")
message(STATUS "")
