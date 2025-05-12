#pragma once


#define CUDA_CHECK(call) do {                                              \
    cudaError err = call;                                                 \
    if (cudaSuccess != err) {                                             \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
                  __FILE__, __LINE__, cudaGetErrorString(err));           \
      exit(EXIT_FAILURE);                                                 \
      } } while (0)
