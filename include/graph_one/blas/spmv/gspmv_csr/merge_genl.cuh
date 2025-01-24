/**
 * base on cub v1.15.1
 * 
*/

#pragma once

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

// use texture memory for vector X (get worse performance, not use in cub)
// #define MERGE_USE_TEXTURE

#include <cub/util_allocator.cuh>

#include "./device_spmv.cuh"

namespace graph_one::blas {

/// cub merge-based Generalized CsrMV with independent code
template <typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t,
          typename combine_t, typename reduce_t>
void GSpMV_CSR_merge_based(
    index_t n_rows, index_t n_cols, offset_t nnz,
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vec_x_value_t *x, vec_y_value_t *y,
    const combine_t& combine_op, const reduce_t& reduce_op) {

    // Allocate temporary storage
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;

    // Caching allocator for device memory
    cub::CachingDeviceAllocator  allocator(true);      

    // Get amount of temporary storage needed
    CubDebugExit(DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, 
                                   const_cast<mat_value_t *>(Ax), 
                                   const_cast<offset_t *>(Ap), 
                                   const_cast<index_t *>(Aj), 
                                   const_cast<vec_x_value_t *>(x), 
                                   y, n_rows, n_cols, nnz,
                                   combine_op, reduce_op,
                                   (cudaStream_t)0, false));

    // Allocate
    CubDebugExit(allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    CubDebugExit(DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, 
                                   const_cast<mat_value_t *>(Ax), 
                                   const_cast<offset_t *>(Ap), 
                                   const_cast<index_t *>(Aj), 
                                   const_cast<vec_x_value_t *>(x), 
                                   y, n_rows, n_cols, nnz,
                                   combine_op, reduce_op,
                                   (cudaStream_t)0, false));
}

} // namespace graph_one::blas