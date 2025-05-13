cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(GraphOne CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_BUILD_TYPE DEBUG)
#set(CMAKE_BUILD_TYPE RELEASE)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    # set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
message(STATUS "BuildType: ${CMAKE_BUILD_TYPE}")


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


include_directories(${TORCH_INCLUDE_DIRS})
include_directories("third_party")
include_directories("include")


function(add_cuda_exec exec_name exec_file)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs LIB)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_executable(${exec_name} ${exec_file})
    set_target_properties(${exec_name} PROPERTIES 
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        CUDA_SEPARABLE_COMPILATION ON
    )

    # common libraries
    target_link_libraries(${exec_name} PRIVATE ${TORCH_LIBRARIES} cusparse)

    if(ARG_LIB)
        message(STATUS "Linking ${exec_name} with libraries: ${ARG_LIB}")
        target_link_libraries(${exec_name} PRIVATE ${ARG_LIB})
    endif()
endfunction()


add_subdirectory(apps)
