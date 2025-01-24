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

add_subdirectory(apps)
