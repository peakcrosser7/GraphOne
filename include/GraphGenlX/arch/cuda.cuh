#pragma once

#include <cuda.h>

#include "GraphGenlX/arch/arch.hpp"

namespace graph_genlx::archi {

template <typename T>
void CheckCuda(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) CheckCuda((val), #val, __FILE__, __LINE__)

template<>
struct memalloc<arch_t::cuda> {
    template <typename T>
    static T* call(size_t size) {
        T* ptr;
        checkCudaErrors(cudaMalloc(&ptr, sizeof(T) * size));
        return ptr;
    }
};

template<>
struct memfree<arch_t::cuda> {
    template<typename T>
    static void call(T* ptr) {
        checkCudaErrors(cudaFree(ptr));
    }
};

template<>
struct memcpy<arch_t::cpu, arch_t::cuda> {
    template <typename T>
    static void call(T* dst, const T* src, size_t size) {
        checkCudaErrors(cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyDeviceToHost));
    }
};

template<>
struct memcpy<arch_t::cuda, arch_t::cpu> {
    template <typename T>
    static void call(T* dst, const T* src, size_t size) {
        checkCudaErrors(cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyHostToDevice));
    }
};

template<>
struct memcpy<arch_t::cuda, arch_t::cuda> {
    template <typename T>
    static void call(T* dst, const T* src, size_t size) {
        checkCudaErrors(cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyDeviceToDevice));
    }
};

    
} // namespace graph_genlx::archi