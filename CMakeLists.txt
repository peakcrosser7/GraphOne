cmake_minimum_required(VERSION 3.12)

project(GraphGenlX CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# # 库输出路径
# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
# # 可执行文件输出路径
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

#set(CMAKE_BUILD_TYPE DEBUG)
#set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall")

include_directories(include)

macro (add_cuda_exec exec_name exec_file)
    # GPU代码
    add_executable(${exec_name} ${exec_file})
    set_target_properties("${exec_name}" PROPERTIES 
        CUDA_ARCHITECTURES "70"
        CUDA_SEPARABLE_COMPILATION ON
    )
endmacro (add_cuda_exec)

add_subdirectory(apps)
add_subdirectory(test)


