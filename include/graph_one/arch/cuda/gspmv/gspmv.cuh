/**
 * base on cub v1.15.1
 * 
*/

#pragma once

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

// use texture memory for vector X (get worse performance, not use in cub)
// #define MERGE_USE_TEXTURE

#include "graph_one/arch/cuda/utils.cuh"

#include "./device_spmv.cuh"

namespace graph_one::cuda {

/// cub merge-based Generalized CsrMV with independent code
template <typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t,
          typename combine_t, typename reduce_t>
void GSpMV_CSR_merge_based(
    index_t n_rows, index_t n_cols, offset_t nnz,
    const offset_t* row_offsets, const index_t* col_indices, const mat_value_t* csr_values, 
    const vec_x_value_t* x, vec_y_value_t* y,
    const combine_t& combine_op, const reduce_t& reduce_op) {

    // Caching allocator for device memory
    cub::CachingDeviceAllocator& allocator = utils::get_allocator();      

    // Get amount of temporary storage needed
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceSpmv::CsrMV(nullptr, temp_storage_bytes, 
                                   const_cast<mat_value_t *>(csr_values), 
                                   const_cast<offset_t *>(row_offsets), 
                                   const_cast<index_t *>(col_indices), 
                                   const_cast<vec_x_value_t *>(x), 
                                   y, n_rows, n_cols, nnz,
                                   combine_op, reduce_op,
                                   (cudaStream_t)0, false));

    // Allocate
    void *d_temp_storage = nullptr;
    CUDA_CHECK(allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    CUDA_CHECK(DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, 
                                   const_cast<mat_value_t *>(csr_values), 
                                   const_cast<offset_t *>(row_offsets), 
                                   const_cast<index_t *>(col_indices), 
                                   const_cast<vec_x_value_t *>(x), 
                                   y, n_rows, n_cols, nnz,
                                   combine_op, reduce_op,
                                   (cudaStream_t)0, false));

    CUDA_CHECK(allocator.DeviceFree(d_temp_storage));
}

} // namespace graph_one::blas