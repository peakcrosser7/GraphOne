#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "GraphGenlX/archi/check/cuda.cuh"
#include "GraphGenlX/archi/mem/def.hpp"

namespace graph_genlx::archi {

template<>
struct memalloc_t<arch_t::cuda> {
    template <typename T>
    static T* call(size_t size) {
        T* ptr;
        checkCudaErrors(cudaMalloc(&ptr, sizeof(T) * size));
        return ptr;
    }
};

template<>
struct memfree_t<arch_t::cuda> {
    template<typename T>
    static void call(T* ptr) {
        checkCudaErrors(cudaFree(ptr));
    }
};

template<>
struct memset_t<arch_t::cuda> {
    template <typename T>
    static void call(T* ptr, int value, size_t size) {
        checkCudaErrors(cudaMemset(ptr, value, sizeof(T) * size));
    }
};

template<>
struct memcpy_t<arch_t::cpu, arch_t::cuda> {
    template <typename T>
    static void call(T* dst, const T* src, size_t size) {
        checkCudaErrors(cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyDeviceToHost));
    }
};

template<>
struct memcpy_t<arch_t::cuda, arch_t::cpu> {
    template <typename T>
    static void call(T* dst, const T* src, size_t size) {
        checkCudaErrors(cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyHostToDevice));
    }
};

template<>
struct memcpy_t<arch_t::cuda, arch_t::cuda> {
    template <typename T>
    static void call(T* dst, const T* src, size_t size) {
        checkCudaErrors(cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyDeviceToDevice));
    }
};
    
} // namespace graph_genlx::archi