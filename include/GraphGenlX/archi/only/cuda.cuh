#pragma once

#include <cuda.h>


namespace graph_genlx::archi::cuda {

__device__
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
__device__
void print(const char* fmt, Ts&&... args) {
#if defined(DEBUG_CUDA) && defined(DEBUG_LOG)
    char buf[256] = "[DEBUG-KERNEL] ";
    archi::cuda::strncat(buf, fmt, sizeof(buf) - 16);
    printf(buf, args...);
#endif
}

template <typename T>
__device__ 
T atomicMin(T* address, T value) {
    return ::atomicMin(address, value);
}

template <>
__device__ 
float atomicMin<float>(float* address, float value) {
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
    
} // namespace graph_genlx