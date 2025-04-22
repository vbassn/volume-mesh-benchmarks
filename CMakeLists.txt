cmake_minimum_required(VERSION 3.10)
project(test_solver LANGUAGES CXX CUDA)
#find_package(CUDAToolkit REQUIRED)

#Run cmake with cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc for compute node

cmake_policy(SET CMP0104 NEW)

# --- CPU solver ------------------------------------------------------
# Compile with normal C++ flags (like your original gcc command).

add_executable(test_solver_CPU test_solver_CPU.cpp)
target_compile_options(test_solver_CPU PRIVATE -O3)


# Include paths (-I):
target_include_directories(test_solver_CPU PRIVATE
    ${CMAKE_SOURCE_DIR}/../../external/amgcl
    ${CMAKE_SOURCE_DIR}
)

# Link the math library (as from -lm). The standard C++ library is usually automatic:
target_link_libraries(test_solver_CPU PRIVATE m)

# --- GPU solver ------------------------------------------------------
# We'll create a CUDA executable corresponding to your nvcc command.

set(CMAKE_CUDA_ARCHITECTURES 70)
add_executable(test_solver_GPU test_solver_GPU.cu)

# We want C++14 for the CUDA code too:
set_target_properties(test_solver_GPU PROPERTIES
    CXX_STANDARD 14
    CUDA_STANDARD 14
)

# Include paths (-I):
target_include_directories(test_solver_GPU PRIVATE
    ${CMAKE_SOURCE_DIR}/../../external/amgcl
    ${CMAKE_SOURCE_DIR}
)

# We can specify the GPU architecture (sm_70 for V100 or certain Turing GPUs).
# If you'd like a more modern approach, you can set: set(CMAKE_CUDA_ARCHITECTURES 70)
# But here's a direct compile option:
set_target_properties(test_solver_GPU PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(test_solver_GPU
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_70>
)

# Link libraries (-lcudart -lcusparse). 
# If you have a modern CMake + CUDA, you might do `find_package(CUDAToolkit REQUIRED)`
# and link with `CUDA::cudart` / `CUDA::cusparse`.
# For a simpler direct approach:
target_link_libraries(test_solver_GPU PRIVATE 
    # Usually nvcc handles libcudart automatically, but we can be explicit:
    "-L/usr/local/cuda/lib64"
    "-lcudart"
    "-lcusparse"
)

