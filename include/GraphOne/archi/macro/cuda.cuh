#pragma once

#include <cuda.h>

#define __ONE_CUDA_KERNEL__   __global__
#define __ONE_CUDA__          __device__
#define __ONE_CUDA_INL__      __device__ __forceinline__
#define __ONE_CPU_CUDA__      __host__ __device__
#define __ONE_CPU_CUDA_INL__  __host__ __device__ __forceinline__