cmake_minimum_required(VERSION 3.16)

project(104-PointNet-Jeonghan)

# Testing option (default: OFF)
option(BUILD_TESTING "Build tests" OFF)

add_library(VAI_LIBRARY
    library/vulkanApp.cpp
    library/neuralNodes.cpp
    library/spirvHelpers.cpp
    library/stb_image.cpp
    library/jsonParser.cpp
    library/safeTensorsParser.cpp
)

# Set include directories and link libraries for VAI_LIBRARY
target_include_directories(VAI_LIBRARY PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/library
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(VAI_LIBRARY PUBLIC
    glfw
    spirv-reflect-static
    ${Vulkan_LIBRARIES}
    glslang
    glslang-default-resource-limits
    nlohmann_json
)

add_library(POINTNET_LIBRARY
    networks/src/pointnet.cpp
    networks/src/inference.cpp
    networks/src/weights.cpp
)

target_include_directories(POINTNET_LIBRARY PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/networks/include
    ${CMAKE_CURRENT_SOURCE_DIR}/library
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(POINTNET_LIBRARY PUBLIC
    VAI_LIBRARY
)

# Main executable
add_executable(${PROJECT_NAME} 
    main.cpp
)

add_definitions(
    -DPROJECT_CURRENT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/library
    ${CMAKE_CURRENT_SOURCE_DIR}/networks/include
)

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    VAI_LIBRARY
    POINTNET_LIBRARY
)

# Enable testing only if BUILD_TESTING is ON
if(BUILD_TESTING)
    enable_testing()

    set(TEST_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin/debug")
    file(MAKE_DIRECTORY "${TEST_OUTPUT_DIR}")

    # Test: BatchNorm1D
    add_executable(test_batchnorm
        test/batchnorm/test_batchnorm.cpp
    )

    target_include_directories(test_batchnorm PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/library
        ${Vulkan_INCLUDE_DIRS}
    )

    target_link_libraries(test_batchnorm PRIVATE
        VAI_LIBRARY
    )
    
    # Relative output directory (creates at ../../bin/debug/ from build/104-PointNet-Jeonghan)
    set_target_properties(test_batchnorm PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TEST_OUTPUT_DIR}"
    )
    
    # Add test - use TARGET_FILE generator expression to get the actual output path
    add_test(NAME BatchNorm1D COMMAND $<TARGET_FILE:test_batchnorm>)
    
    # Test: PointWiseMLP
    add_executable(test_mlp
        test/mlp/test_mlp.cpp
    )

    target_include_directories(test_mlp PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/library
        ${Vulkan_INCLUDE_DIRS}
    )

    target_link_libraries(test_mlp PRIVATE
        VAI_LIBRARY
    )
    
    set_target_properties(test_mlp PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TEST_OUTPUT_DIR}"
    )
    
    add_test(NAME PointWiseMLP COMMAND $<TARGET_FILE:test_mlp>)
    
    # Test: FullyConnected
    add_executable(test_fc
        test/fc/test_fc.cpp
    )

    target_include_directories(test_fc PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/library
        ${Vulkan_INCLUDE_DIRS}
    )

    target_link_libraries(test_fc PRIVATE
        VAI_LIBRARY
    )
    
    set_target_properties(test_fc PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TEST_OUTPUT_DIR}"
    )
    
    add_test(NAME FullyConnected COMMAND $<TARGET_FILE:test_fc>)
    
    # Test: FCSequence (chain of FC layers)
    add_executable(test_fcseq
        test/fcseq/test_fcseq.cpp
    )

    target_include_directories(test_fcseq PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/library
        ${CMAKE_CURRENT_SOURCE_DIR}/networks/include
        ${Vulkan_INCLUDE_DIRS}
    )

    target_link_libraries(test_fcseq PRIVATE
        VAI_LIBRARY
        POINTNET_LIBRARY
    )
    
    set_target_properties(test_fcseq PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TEST_OUTPUT_DIR}"
    )
    
    add_test(NAME FCSequence COMMAND $<TARGET_FILE:test_fcseq>)
    
    # MLPSequence test
    add_executable(test_mlpseq
        test/mlpseq/test_mlpseq.cpp
    )

    target_include_directories(test_mlpseq PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/library
        ${CMAKE_CURRENT_SOURCE_DIR}/networks/include
        ${Vulkan_INCLUDE_DIRS}
    )

    target_link_libraries(test_mlpseq PRIVATE
        VAI_LIBRARY
        POINTNET_LIBRARY
    )
    
    set_target_properties(test_mlpseq PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TEST_OUTPUT_DIR}"
    )
    
    add_test(NAME MLPSequence COMMAND $<TARGET_FILE:test_mlpseq>)
    
    # MatMul test
    add_executable(test_matmul
        test/matmul/test_matmul.cpp
    )

    target_include_directories(test_matmul PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/library
        ${Vulkan_INCLUDE_DIRS}
    )

    target_link_libraries(test_matmul PRIVATE
        VAI_LIBRARY
    )
    
    set_target_properties(test_matmul PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TEST_OUTPUT_DIR}"
    )
    
    add_test(NAME MatMul COMMAND $<TARGET_FILE:test_matmul>)
    
    # =========================================================================
    # Test 7: TNetBlock (Spatial Transformer Network)
    # =========================================================================
    add_executable(test_tnet test/tnet/test_tnet.cpp)
    
    target_include_directories(test_tnet PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/library
        ${CMAKE_CURRENT_SOURCE_DIR}/networks/include
        ${Vulkan_INCLUDE_DIRS}
    )

    target_link_libraries(test_tnet PRIVATE
        VAI_LIBRARY
        POINTNET_LIBRARY
    )
    
    set_target_properties(test_tnet PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TEST_OUTPUT_DIR}"
    )
    
    add_test(NAME TNetBlock COMMAND $<TARGET_FILE:test_tnet>)
    
    # =========================================================================
    # Test 8: MaxPooling1D
    # =========================================================================
    add_executable(test_maxpool test/maxpool/test_maxpool.cpp)
    
    target_include_directories(test_maxpool PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/library
        ${Vulkan_INCLUDE_DIRS}
    )

    target_link_libraries(test_maxpool PRIVATE
        VAI_LIBRARY
    )
    
    set_target_properties(test_maxpool PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TEST_OUTPUT_DIR}"
    )
    
    add_test(NAME MaxPooling1D COMMAND $<TARGET_FILE:test_maxpool>)
    
    # =========================================================================
    # Test 9: MLP + MaxPool (TNet first part)
    # =========================================================================
    add_executable(test_mlp_maxpool test/mlp_maxpool/test_mlp_maxpool.cpp)
    
    target_include_directories(test_mlp_maxpool PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/library
        ${CMAKE_CURRENT_SOURCE_DIR}/networks/include
        ${Vulkan_INCLUDE_DIRS}
    )

    target_link_libraries(test_mlp_maxpool PRIVATE
        VAI_LIBRARY
        POINTNET_LIBRARY
    )
    
    set_target_properties(test_mlp_maxpool PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TEST_OUTPUT_DIR}"
    )
    
    add_test(NAME MLPMaxPool COMMAND $<TARGET_FILE:test_mlp_maxpool>)
    
    # =========================================================================
    # Test 10: AddIdentityNode
    # =========================================================================
    add_executable(test_add_identity test/add_identity/test_add_identity.cpp)
    
    target_include_directories(test_add_identity PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/library
        ${Vulkan_INCLUDE_DIRS}
    )

    target_link_libraries(test_add_identity PRIVATE
        VAI_LIBRARY
    )
    
    set_target_properties(test_add_identity PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TEST_OUTPUT_DIR}"
    )
    
    add_test(NAME AddIdentity COMMAND $<TARGET_FILE:test_add_identity>)
    
    # FCBN test (FullyConnected + BatchNorm + ReLU)
    add_executable(test_fcbn
        test/fcbn/test_fcbn.cpp
    )

    target_include_directories(test_fcbn PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/library
        ${CMAKE_CURRENT_SOURCE_DIR}/networks/include
        ${Vulkan_INCLUDE_DIRS}
    )

    target_link_libraries(test_fcbn PRIVATE
        VAI_LIBRARY
        POINTNET_LIBRARY
    )
    
    set_target_properties(test_fcbn PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TEST_OUTPUT_DIR}"
    )
    
    add_test(NAME FCBNNode COMMAND $<TARGET_FILE:test_fcbn>)

    # FCBNSequence test
    add_executable(test_fcbn_seq
        test/fcbn_seq/test_fcbn_seq.cpp
    )

    target_include_directories(test_fcbn_seq PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/library
        ${CMAKE_CURRENT_SOURCE_DIR}/networks/include
        ${Vulkan_INCLUDE_DIRS}
    )

    target_link_libraries(test_fcbn_seq PRIVATE
        VAI_LIBRARY
        POINTNET_LIBRARY
    )
    
    set_target_properties(test_fcbn_seq PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TEST_OUTPUT_DIR}"
    )
    
    add_test(NAME FCBNSequence COMMAND $<TARGET_FILE:test_fcbn_seq>)
    
    # Test: MLP + MaxPool + FC (T-Net path without reshape/identity/matmul)
    add_executable(test_mlp_maxpool_fc
        test/mlp_maxpool_fc/test_mlp_maxpool_fc.cpp
    )

    target_include_directories(test_mlp_maxpool_fc PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/library
        ${CMAKE_CURRENT_SOURCE_DIR}/networks/include
        ${Vulkan_INCLUDE_DIRS}
    )

    target_link_libraries(test_mlp_maxpool_fc PRIVATE
        VAI_LIBRARY
        POINTNET_LIBRARY
    )
    
    set_target_properties(test_mlp_maxpool_fc PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TEST_OUTPUT_DIR}"
    )
    
    add_test(NAME MLPMaxPoolFC COMMAND $<TARGET_FILE:test_mlp_maxpool_fc>)
    
    # Test: Transform Path (MLP -> MaxPool -> FC -> Reshape -> AddIdentity)
    add_executable(test_transform_path
        test/tnet_transform_path/test_transform_path.cpp
    )

    target_include_directories(test_transform_path PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/library
        ${CMAKE_CURRENT_SOURCE_DIR}/networks/include
        ${Vulkan_INCLUDE_DIRS}
    )

    target_link_libraries(test_transform_path PRIVATE
        VAI_LIBRARY
        POINTNET_LIBRARY
    )
    
    set_target_properties(test_transform_path PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TEST_OUTPUT_DIR}"
    )
    
    add_test(NAME TransformPath COMMAND $<TARGET_FILE:test_transform_path>)
    
    # Test: PointNetEncoder (full encoder pipeline with intermediate checks)
    add_executable(test_encoder
        test/encoder/test_encoder.cpp
    )

    target_include_directories(test_encoder PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/library
        ${CMAKE_CURRENT_SOURCE_DIR}/networks/include
        ${Vulkan_INCLUDE_DIRS}
    )

    target_link_libraries(test_encoder PRIVATE
        VAI_LIBRARY
        POINTNET_LIBRARY
    )
    
    set_target_properties(test_encoder PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TEST_OUTPUT_DIR}"
    )
    
    add_test(NAME PointNetEncoder COMMAND $<TARGET_FILE:test_encoder>)
    
    # Test: Accuracy comparison (Vulkan vs PyTorch)
    add_executable(test_accuracy
        test/validation/test_accuracy.cpp
    )

    target_include_directories(test_accuracy PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/library
        ${CMAKE_CURRENT_SOURCE_DIR}/networks/include
        ${Vulkan_INCLUDE_DIRS}
    )

    target_link_libraries(test_accuracy PRIVATE
        VAI_LIBRARY
        POINTNET_LIBRARY
    )
    
    set_target_properties(test_accuracy PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TEST_OUTPUT_DIR}"
    )
    
    add_test(NAME AccuracyComparison COMMAND $<TARGET_FILE:test_accuracy>)
    
else()
    message(STATUS "✗ Tests disabled")
    message(STATUS "  Enable with: cmake -DBUILD_TESTING=ON ..")
endif()


