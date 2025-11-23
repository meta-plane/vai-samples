cmake_minimum_required(VERSION 3.16)

project(vulkan-raytracing-lectures)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin/debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin/release)

# Set this source directory as the tree root for project.
set(ae2f_ProjRoot ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "Tree root for project")
set(ae2f_submod  .submod CACHE STRING "Relative path to submodule (subdirectory)")

# Directory to inject corescript
set(ae2f_CoreScript_Wh ./cmake/corescript)

# Execute git to fetch CoreScript when repository does not exist
if(NOT EXISTS ${ae2f_CoreScript_Wh})
	execute_process(
		COMMAND git clone https://codeberg.org/ae2f/CoreScript
		${ae2f_CoreScript_Wh}
		)
endif()

# add subdirectory
add_subdirectory(${ae2f_CoreScript_Wh})

ae2f_CoreLibFetchX_NS(KhronosGroup KhronosGroup SPIRV-Tools	main)
ae2f_CoreLibFetchX_NS(KhronosGroup KhronosGroup SPIRV-Cross	main)
ae2f_CoreLibFetchX_NS(KhronosGroup KhronosGroup glslang	main)
ae2f_CoreLibFetchX_NS(glfw glfw glfw	master)

set(SPIRV_REFLECT_EXECUTABLE OFF CACHE BOOL "Build spirv-reflect executable")
set(SPIRV_REFLECT_STATIC_LIB ON CACHE BOOL "Build a SPIRV-Reflect static library")

# Execute git to fetch CoreScript when repository does not exist
if(NOT EXISTS external/Reflect)
	execute_process(
		COMMAND git clone 
		https://github.com/KhronosGroup/SPIRV-Reflect
		external/Reflect
		)
endif()

include_directories(external/Reflect)
# add subdirectory
add_subdirectory(external/Reflect)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Debug)
endif()

add_definitions(
	-DPROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
	-DGLFW_INCLUDE_VULKAN=1
	)

include_directories(
	${CMAKE_SOURCE_DIR}/external
	)

option(REQUIRE_BACKEND_VULKAN "Require Vulkan backend" ON)
option(REQUIRE_BACKEND_OPENCL "Require OpenCL backend" OFF)
option(REQUIRE_BACKEND_CUDA "Require CUDA backend" OFF)
option(USE_KOMPUTE "Use Kompute library" OFF)
option(USE_CTRE "Use ctre library" OFF)


################################################################################
# OpenCL
################################################################################
if(REQUIRE_BACKEND_OPENCL)
	message(STATUS "[OpenCL] Configuring OpenCL SDK...")
	set(BUILD_TESTING OFF CACHE BOOL "Disable building tests for OpenCL SDK")
	set(BUILD_DOCS OFF CACHE BOOL "Disable building documentation for OpenCL SDK")
	set(BUILD_EXAMPLES OFF CACHE BOOL "Disable building examples for OpenCL SDK")
	set(OPENCL_SDK_BUILD_SAMPLES OFF CACHE BOOL "Disable building OpenCL SDK samples")
	add_subdirectory(external/OpenCL-SDK)
	message(STATUS "[OpenCL] OpenCL SDK configuration completed.")
endif()


################################################################################
# CUDA
################################################################################
if (REQUIRE_BACKEND_CUDA)
	message(STATUS "[CUDA] Configuring CUDA Toolkit...")

	find_package(CUDAToolkit)
	if (NOT CUDAToolkit_FOUND)
		message(FATAL_ERROR "CUDA Toolkit not found. Please install the CUDA Toolkit.")
	endif()
	include_directories(${CUDAToolkit_INCLUDE_DIRS})

	message(STATUS "[CUDA] CUDA Toolkit configuration completed.")
endif()


################################################################################
# Vulkan
################################################################################
if (REQUIRE_BACKEND_VULKAN)
	message(STATUS "[Vulkan] Configuring Vulkan SDK...")

	find_package(Vulkan)
	if (NOT Vulkan_FOUND)
		message(FATAL_ERROR "Vulkan not found. Please install the Vulkan SDK.")
	endif()

	message(STATUS "[Vulkan] Vulkan SDK configuration completed.")
endif()

################################################################################
# ctre
################################################################################
if (USE_CTRE)
	message(STATUS "[ctre] Configuring ctre...")
	add_subdirectory(external/ctre)
	message(STATUS "[ctre] ctre configuration completed.")
endif()


################################################################################
# kompute
################################################################################
if (Vulkan_FOUND AND USE_KOMPUTE)
	message(STATUS "[Kompute] Configuring Kompute...")
	add_subdirectory(external/kompute)
	message(STATUS "[Kompute] Kompute configuration completed.")
endif()


################################################################################
# json
################################################################################
message(STATUS "[json] Configuring json...")
add_subdirectory(external/json)
message(STATUS "[json] json configuration completed.")


################################################################################
# main-project
################################################################################

add_subdirectory(10-mnist)
add_subdirectory(11-mnist-refactor)

add_subdirectory(100-UNet-seungwoo)
# add_subdirectory(114-MobileNetV2-hwPark)
add_subdirectory(110-GPT2-hyungkyu)
add_subdirectory(117-yolo-dalmurii)
