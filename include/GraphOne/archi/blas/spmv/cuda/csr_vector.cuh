#pragma once

#include <cuda.h>

#define FULL_MASK 0xffffffff

namespace graph_one::blas {

template <unsigned size, typename functor_t, typename T>
__device__ __forceinline__
T WarpReduce(T sum) {
    if constexpr (size >= 32) sum = functor_t::reduce(sum, __shfl_down_sync(FULL_MASK, sum, 16)); // 0-16, 1-17, 2-18, etc.
    if constexpr (size >= 16) sum = functor_t::reduce(sum, __shfl_down_sync(FULL_MASK, sum, 8));  // 0-8, 1-9, 2-10, etc.
    if constexpr (size >= 8)  sum = functor_t::reduce(sum, __shfl_down_sync(FULL_MASK, sum, 4));  // 0-4, 1-5, 2-6, etc.
    if constexpr (size >= 4)  sum = functor_t::reduce(sum, __shfl_down_sync(FULL_MASK, sum, 2));  // 0-2, 1-3, 4-6, 5-7, etc.
    if constexpr (size >= 2)  sum = functor_t::reduce(sum, __shfl_down_sync(FULL_MASK, sum, 1));  // 0-1, 2-3, 4-5, etc.
    return sum;
}

template <unsigned VECTORS_PER_BLOCK, unsigned THREADS_PER_VECTOR,
          typename functor_t,
          typename index_t, typename offset_t, typename mat_value_t,
          typename vecx_value_t, typename vecy_value_t>
__global__ void SpMV_cuda_csr_vector_kernel(index_t n_rows,
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vecx_value_t *x, vecy_value_t *y) {

    constexpr index_t THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;
    constexpr index_t thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    constexpr index_t thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    constexpr index_t row_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    
    if (row_id < n_rows) {
        offset_t row_start = Ap[row_id];
        offset_t row_end = Ap[row_id + 1];

        vecy_value_t sum = functor_t::initialize();

        // accumulate local sums
        for(offset_t jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR) {
            sum = reduce(sum, combine(Ax[jj], x[Aj[jj]]));
        }

        sum = WarpReduce<THREADS_PER_VECTOR, functor_t>(sum);
        if (thread_lane == 0) {
            y[row_id] = sum;
        }
    }
}

template <unsigned THREADS_PER_VECTOR,
          typename functor_t,
          typename index_t, typename offset_t, typename mat_value_t,
          typename vecx_value_t, typename vecy_value_t>
cudaError_t SpMV_cuda_csr_vector(index_t n_rows,
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vecx_value_t *x, vecy_value_t *y) {

    constexpr unsigned THREADS_PER_BLOCK = 128;
    constexpr unsigned VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
    const unsigned num_blocks = std::max<int>(1, (n_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    SpMV_cuda_csr_vector_kernel<VECTORS_PER_BLOCK, THREADS_PER_VECTOR, functor_t>
        <<<num_blocks, THREADS_PER_BLOCK>>>(n_rows, Ap, Aj, Ax, x, y);
    cudaDeviceSynchronize();
    
    return cudaGetLastError();
}
    
} // namespace graph_one::blas
