#pragma once

#include <cuda.h>

#define __GENLX_CUDA__          __device__
#define __GENLX_CPU_CUDA__      __host__ __device__
#define __GENLX_CPU_CUDA_INL__  __host__ __device__ __forceinline__