cmake_minimum_required(VERSION 3.18)
project(CuTropicalGEMM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) 
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -O3 -use_fast_math -Xptxas -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -march=native")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

include_directories(include)
include_directories(${CUDA_INCLUDE_DIRS})

add_library(tropicalgemm SHARED
    src/tropicalgemm.cu
)

set_target_properties(tropicalgemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES "70;75;80;86"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

target_compile_features(tropicalgemm PUBLIC cxx_std_17)

target_link_libraries(tropicalgemm 
    CUDA::cudart
    CUDA::cublas
)

add_executable(test_tropicalgemm test/test_tropicalgemm.cu)
set_target_properties(test_tropicalgemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)
target_compile_features(test_tropicalgemm PUBLIC cxx_std_17)
target_link_libraries(test_tropicalgemm tropicalgemm CUDA::cudart CUDA::cublas)

add_executable(benchmark_tropicalgemm test/benchmark_tropicalgemm.cu)
set_target_properties(benchmark_tropicalgemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)
target_compile_features(benchmark_tropicalgemm PUBLIC cxx_std_17)
target_link_libraries(benchmark_tropicalgemm tropicalgemm CUDA::cudart CUDA::cublas)
    
add_executable(test_shared_memory
    test/test_shared_memory.cu
)

set_target_properties(test_shared_memory PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

target_compile_features(test_shared_memory PUBLIC cxx_std_17)
target_link_libraries(test_shared_memory CUDA::cudart CUDA::cublas)

enable_testing()
add_test(NAME TropicalGEMM_BasicTest COMMAND test_tropicalgemm)
add_test(NAME TropicalGEMM_SharedMemoryTest COMMAND test_shared_memory)

install(TARGETS tropicalgemm LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(FILES include/tropicalgemm.h DESTINATION include)