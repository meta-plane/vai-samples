cmake_minimum_required(VERSION 3.16)

project(103-MobileNetV2-SangIn)

add_executable(${PROJECT_NAME} 
    main.cpp
    test.cpp    
    models/MobileNetV2.cpp
    library/vulkanApp.cpp
    library/neuralNodes.cpp
    library/spirvHelpers.cpp
    library/jsonParser.cpp
    library/safeTensorsParser.cpp
)

add_definitions(
    -DPROJECT_CURRENT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
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