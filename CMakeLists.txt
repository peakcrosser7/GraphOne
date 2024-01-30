cmake_minimum_required(VERSION 3.12)

project(GraphGenlX CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# # 库输出路径
# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
# # 可执行文件输出路径
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

#set(CMAKE_BUILD_TYPE DEBUG)
#set(CMAKE_BUILD_TYPE RELEASE)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    # set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
message(STATUS "BuildType: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall")

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
    message(STATUS "Using default GPU Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "GPU Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} --debug --device-debug")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")

# add_library(graph_genlx INTERFACE)
# set_target_properties(graph_genlx PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} # Set required architecture.
# )
# target_include_directories(graph_genlx
#     INTERFACE "include"
#     INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
# )

include_directories(third_party/CLI11)
add_subdirectory(third_party/nvtx3)

include_directories(include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

function (add_cuda_exec exec_name exec_file)
    # GPU代码
    add_executable(${exec_name} ${exec_file})
    # target_link_libraries(${exec_name} PRIVATE graph_genlx)
    set_target_properties(${exec_name} PROPERTIES 
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(${exec_name} PRIVATE nvtx3-cpp)
endfunction (add_cuda_exec)

add_subdirectory(apps)
add_subdirectory(test)


