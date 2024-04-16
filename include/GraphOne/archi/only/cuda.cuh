#pragma once

#include <cuda.h>

#include "GraphOne/debug/debug.hpp"
#include "GraphOne/archi/macro/cuda.cuh"

namespace graph_one::archi::cuda {

constexpr unsigned kWarpFullMask = 0xffffffff;

__ONE_CUDA__
inline char* strncat(char* dest, const char* src, unsigned n) {
    int i = 0;
    while (dest[i] != 0) {
        ++i;
    }
    for (int j = 0; j < n; ++j, ++i) {
        dest[i] = src[j];
        if (src[j] == 0) {
            break;
        }
    }
    return dest;
}

template <typename ... Ts>
__ONE_CUDA_INL__
void print(const char* fmt, Ts&&... args) {
#if defined(DEBUG_KERNEL)
    char buf[256] = "[DEBUG-KERNEL] ";
    archi::cuda::strncat(buf, fmt, sizeof(buf) - 16);
    printf(buf, args...);
#endif
}

template <typename IT, typename T>
__ONE_CUDA_INL__
IT UpperBound(IT begin, IT end, const T& target) {
    IT mid;
    while (begin < end) {
        mid = begin + ((end - begin) >> 1);
        if (*mid <= target) {
            begin = mid + 1;
        } else {
            end = mid;
        }
    }
    return begin;
}

template <typename T>
__ONE_CUDA_INL__
T AtomicMin(T* address, T value) {
    return ::atomicMin(address, value);
}

template <>
__ONE_CUDA_INL__
float AtomicMin<float>(float* address, float value) {
  int* addr_as_int = reinterpret_cast<int*>(address);
  int old = *addr_as_int;
  int expected;
  do {
    expected = old;
    old = ::atomicCAS(addr_as_int, expected,
                      __float_as_int(::fminf(value, __int_as_float(expected))));
  } while (expected != old);
  return __int_as_float(old);
}

template <typename T>
__ONE_CUDA_INL__
T WarpReduceSum(T value) {
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        value += __shfl_down_sync(kWarpFullMask, value, s);
    }
    return value;
}

template <typename T>
__ONE_CUDA_INL__
T BlockReduceSum(T val, T* shared) {
    const int tid = threadIdx.x;
    const int lid = tid % 32;
    const int wid = tid / 32;
    val = WarpReduceSum(val);
    __syncthreads();
    if (lid == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    val = (tid < blockDim.x / 32) ? shared[lid] : T(0);
    if (wid == 0) {
        val = WarpReduceSum(val);
    }
    return val;
}

} // namespace graph_one