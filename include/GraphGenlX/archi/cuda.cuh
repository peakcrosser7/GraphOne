#pragma once

#include <cuda.h>

#include <thrust/device_vector.h>

#include "GraphGenlX/archi/archi.hpp"
#include "GraphGenlX/debug/cuda.cuh"

#define CODE_CPU_CUDA __host__ __device__

namespace graph_genlx::archi {

template<>
struct Vector_t<arch_t::cuda> {
    template<typename value_t>
    using type = thrust::device_vector<value_t>;
};

template<>
struct ExecPolicy<arch_t::cuda> {
    static constexpr decltype(thrust::device) value{};
};

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
struct memset<arch_t::cuda> {
    template <typename T>
    static void call(T* ptr, int value, size_t size) {
        checkCudaErrors(cudaMemset(ptr, value, sizeof(T) * size));
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