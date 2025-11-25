cmake_minimum_required(VERSION 3.16)

project(100-UNet-seungwoo)

add_executable(${PROJECT_NAME} 
    main.cpp
    library/vulkanApp.cpp
    library/neuralNodes.cpp
    library/spirvHelpers.cpp
    library/stb_image.cpp
    library/jsonParser.cpp

    networks/src/unet.cpp
 "networks/src/unet.cpp")

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