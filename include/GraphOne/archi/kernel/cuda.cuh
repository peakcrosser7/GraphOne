#pragma once

#include <cuda.h>

#include "GraphOne/type.hpp"
#include "GraphOne/archi/macro/cuda.cuh"
#include "GraphOne/archi/kernel/def.hpp"

namespace graph_one::archi {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kNumWaves = 32;

uint32_t GetNumBlock(int64_t n) {
    return  std::max<int>(1, (n + kBlockSize - 1) / kBlockSize);
}

template <>
struct LaunchTparams<arch_t::cuda> {
    constexpr static uint32_t block_size = kBlockSize;
    constexpr static uint32_t warp_size = 32;
    
    __ONE_CUDA_INL__ 
    static uint32_t grid_dim() {
        return gridDim.x;
    }

    __ONE_CUDA_INL__ 
    static uint32_t block_id() {
        return blockIdx.x;
    }

    __ONE_CUDA_INL__ 
    constexpr static uint32_t block_dim() {
        return block_size;
    }

    __ONE_CUDA_INL__ 
    static uint32_t thread_id() {
        return threadIdx.x;
    }

    __ONE_CUDA_INL__ 
    static uint32_t global_tid() {
        return block_id() * block_dim() + thread_id();
    }

    __ONE_CUDA_INL__ 
    static uint32_t warp_id() {
        return (global_tid() >> 5);
    }

    __ONE_CUDA_INL__ 
    static uint32_t lane_id () {
        return (thread_id() & 0x1f);
    }
};

template <>
struct LaunchParams<arch_t::cuda> {

    LaunchParams(int64_t n, size_t smem_bytes = 0, cudaStream_t stream = 0) 
        : grid_dim(GetNumBlock(n)), smem_bytes(smem_bytes), stream(stream) {}

    dim3 grid_dim;
    size_t smem_bytes;
    cudaStream_t stream;   
};

template<>
struct Launcher<arch_t::cuda> {

    constexpr static arch_t arch_value = arch_t::cuda;

    using err_t = cudaError_t;

    template <typename tparams, typename func_t, typename... args_t>
    static err_t launch(const LaunchParams<arch_value> &params, const func_t &func,
                 args_t &&...args) {
        func<<<params.grid_dim, tparams::block_size, params.smem_bytes,
                 params.stream>>>(std::forward<args_t>(args)...);
        return cudaGetLastError();
    }

    static err_t sync(cudaStream_t stream = 0) {
        return stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
    }
};

} // namespace graph_one::archi