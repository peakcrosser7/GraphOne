#pragma once

#include <cub/util_allocator.cuh>

#define CUDA_CHECK(call) do {                                              \
    cudaError err = call;                                                 \
    if (cudaSuccess != err) {                                             \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
                  __FILE__, __LINE__, cudaGetErrorString(err));           \
      exit(EXIT_FAILURE);                                                 \
      } } while (0)

namespace graph_one::cuda::utils {

cub::CachingDeviceAllocator& get_allocator() {
    static cub::CachingDeviceAllocator allocator;
    return allocator;
}

}