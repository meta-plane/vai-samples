cmake_minimum_required(VERSION 3.16)

project(104-PointNet-Jeonghan)

add_executable(${PROJECT_NAME} 
    main.cpp
    library/vulkanApp.cpp
    library/neuralNodes.cpp
    library/spirvHelpers.cpp
    library/stb_image.cpp
    library/jsonParser.cpp
    library/safeTensorsParser.cpp
    networks/src/pointnet.cpp
    networks/src/inference.cpp
    networks/src/weights.cpp
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
    glfw
    spirv-reflect-static
    ${Vulkan_LIBRARIES}
    glslang
    glslang-default-resource-limits
    nlohmann_json
)
