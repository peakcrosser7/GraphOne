#pragma once

#include "GraphOne/type.hpp"
#include "GraphOne/archi/macro/cuda.cuh"

#define __ONE_DEV__       __ONE_CUDA__
#define __ONE_DEV_INL__   __ONE_CUDA_INL__
#define __ONE_ARCH__      __ONE_CPU_CUDA__
#define __ONE_ARCH_INL__  __ONE_CPU_CUDA_INL__