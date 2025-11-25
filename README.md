# vai-samples

Vulkan AI Sample Project

## Required Dependencies

### 1. CMake 3.22.1 or Higher

```bash
# Check version
cmake --version

# Install (Ubuntu/Debian)
#TODO
```

### 2. GCC 13 or Higher

```bash
# Install
sudo apt update
sudo apt install gcc-13 g++-13
```

### 3. Vulkan SDK (1.3 or Higher Required)

**Important**: This project uses Vulkan 1.3 API.

#### Method 1: Download LunarG Vulkan SDK tar.gz (Recommended, Ubuntu 20.04 Compatible)

```bash
# 1. Remove existing Vulkan packages (prevent conflicts)
sudo apt remove --purge -y libvulkan-dev vulkan-tools libvulkan1 vulkan-validationlayers vulkan-validationlayers-dev

# 2. Download latest Vulkan SDK (1.3.290.0)
wget https://sdk.lunarg.com/sdk/download/1.3.290.0/linux/vulkansdk-linux-x86_64-1.3.290.0.tar.xz

# 3. Extract
mkdir -p $HOME/vulkan_sdk
tar -xf vulkansdk-linux-x86_64-1.3.290.0.tar.xz -C $HOME/vulkan_sdk

# 4. Load environment variables
cd $HOME/vulkan_sdk/1.3.290.0
source setup-env.sh

# 5. Make permanent (add to .bashrc)
echo "source $HOME/vulkan_sdk/1.3.290.0/setup-env.sh" >> ~/.bashrc
source ~/.bashrc
```

#### Verify Vulkan 1.3 Support

```bash
# Check Vulkan API version
vulkaninfo | grep "apiVersion"
# Example output: apiVersion = 1.3.xxx (must be 1.3 or higher)

# Or check header file
grep VK_API_VERSION_1_3 $VULKAN_SDK/include/vulkan/vulkan_core.h
# If there's output, Vulkan 1.3 is supported
```

**Note**: This project includes `vulkansdk-linux-x86_64-1.3.290.0.tar.xz`. `build.sh` is configured to automatically use the SDK from the `VulkanSDK/1.3.290.0` directory.

### 4. Other Dependencies

```bash
# GLFW3 and other required libraries
sudo apt install libglfw3-dev
```

## Build Instructions

```bash
# Use Vulkan SDK included in the project (Recommended)
# Must be extracted in VulkanSDK/1.3.290.0 directory
./build.sh

```