cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cppgcn CXX CUDA)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_BUILD_TYPE release)
add_executable(gcn gcn.cu)
target_link_libraries(gcn "${TORCH_LIBRARIES}" cusparse)
set_target_properties(gcn PROPERTIES 
    CXX_STANDARD 17
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 70
    CUDA_SEPARABLE_COMPILATION ON
)