#pragma once

#include "GraphGenlX/type.hpp"
#include "GraphGenlX/archi/macro/cuda.cuh"

#define __GENLX_DEV__       __GENLX_CUDA__
#define __GENLX_DEV_INL__   __GENLX_CUDA_INL__
#define __GENLX_ARCH__      __GENLX_CPU_CUDA__
#define __GENLX_ARCH_INL__  __GENLX_CPU_CUDA_INL__