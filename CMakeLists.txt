# 3.18 works, but setting 3.20 ensures robust C++20 defaults for newer CUDA toolchains
cmake_minimum_required(VERSION 3.20)

# Declare project and request modern first-class C++ and CUDA support
project(ModernCudaProject LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Lock down the compiler standard to C++20 (matching GCC 16 defaults)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable separate compilation if kernels are split across files
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Source configuration
add_executable(cuda_app main.cpp kernel.cu src/tensor.cu)

target_include_directories(cuda_app PRIVATE src)
# Let CMake 4.4 automatically discover and optimize compilation 
# matching your specific local graphics card architecture
set_target_properties(cuda_app PROPERTIES
    CUDA_ARCHITECTURES native
)
