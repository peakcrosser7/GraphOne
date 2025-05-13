#pragma once

#include <cuda.h>

#include <cub/device/device_scan.cuh>

#include "graph_one/log.hpp"
#include "graph_one/allocator.h"
#include "graph_one/arch/cuda/utils.cuh"


namespace graph_one::cuda {

namespace {

template <typename T>
__device__ T AtomicAdd(T* address, T val) {
    if constexpr (std::is_same_v<T, int64_t>) {
        return atomicAdd(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(val));
    } else {
        return atomicAdd(address, val);
    }
}

}

template <typename offset_t, typename index_t>
__global__ void TrilCSRIndicesAndMaskKernel(
    offset_t n_rows, offset_t nnz, 
    const offset_t* row_offsets, const index_t* col_indices, 
    offset_t* row_indices, bool* col_mask, offset_t* counts) {
    
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) {
        return;
    }

    offset_t low = 0;
    offset_t high = n_rows;
    while (low < high) {
        offset_t mid = low + (high - low) / 2;
        if (row_offsets[mid] > i)  {
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    offset_t row = low - 1;
    index_t col = col_indices[i];
    bool is_lower = (col <= row);
    col_mask[i] = is_lower;
    row_indices[i] = row;

    if (is_lower) {
        AtomicAdd(&counts[row], offset_t(1));
    }
}

template <typename offset_t, typename index_t, typename value_t>
__global__ void CopySelectedElementsKernel(
    offset_t nnz,
    const index_t* col_indices, const value_t* values, 
    const index_t* row_indices, const bool* mask, 
    index_t* col_indices_out, value_t* values_out, offset_t* current_row_counter) {
    
    unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz || !mask[k]) {
        return;
    }

    index_t row = row_indices[k];
    offset_t pos = AtomicAdd(&current_row_counter[row], offset_t(1));
    col_indices_out[pos] = col_indices[k];
    values_out[pos] = values[k];
}

template <typename offset_t, typename index_t, typename value_t>
void TrilCSR(index_t n_rows, index_t n_cols, offset_t nnz,
            const offset_t* row_offsets, const index_t* col_indices, const value_t* csr_values,
            offset_t* row_offsets_out, index_t** col_indices_out, value_t** csr_values_out) {

    auto& allocator = MemAllocator::Get();
    unsigned n_threads = 256;
    unsigned n_blocks = (nnz + n_threads - 1) / n_threads;

    // Count the number of non-zero elements in each row which is in the lower triangular part
    offset_t* counts = allocator.CudaAllocate<offset_t>(n_rows + 1);
    index_t* row_indices = allocator.CudaAllocate<index_t>(nnz);
    bool* col_mask = allocator.CudaAllocate<bool>(nnz);
    CUDA_CHECK(cudaMemsetAsync(counts, 0, (n_rows + 1) * sizeof(offset_t)));
    TrilCSRIndicesAndMaskKernel<<<n_blocks, n_threads>>>(n_rows, nnz, row_offsets, col_indices, 
                                                         row_indices, col_mask, counts + 1);
    CUDA_CHECK(cudaGetLastError());

    // Exclusive scan to get the new row offsets
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes,
                                  counts, row_offsets_out,
                                  n_rows + 1));

    void* temp_storage = allocator.CudaAllocate(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes,
                                  counts, row_offsets_out,
                                  n_rows + 1));
    allocator.Free(temp_storage);

    offset_t new_nnz;
    CUDA_CHECK(cudaMemcpyAsync(&new_nnz, row_offsets_out + n_rows, sizeof(offset_t),
                               cudaMemcpyDeviceToHost));
    *col_indices_out = allocator.CudaAllocate<index_t>(new_nnz);
    *csr_values_out = allocator.CudaAllocate<value_t>(new_nnz);
    
    // reuse the counts array to store the current row counter
    CUDA_CHECK(cudaMemcpyAsync(counts, row_offsets_out,
                                n_rows * sizeof(offset_t), cudaMemcpyDeviceToDevice));
    
    CopySelectedElementsKernel<<<n_blocks, n_threads>>>(nnz, col_indices, csr_values,
        row_indices, col_mask, *col_indices_out, *csr_values_out, counts);
    CUDA_CHECK(cudaGetLastError());
    
    allocator.Free(row_indices);
    allocator.Free(col_mask);
    allocator.Free(counts);
}

} // namespace graph_one::cuda
