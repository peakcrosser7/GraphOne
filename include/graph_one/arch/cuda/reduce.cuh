#pragma once

#include <cuda.h>
#include <cub/device/device_segmented_reduce.cuh>

namespace graph_one::cuda {

template <typename index_t, typename offset_t, typename value_t, typename reduce_t>
void ReduceCSR(index_t n_rows, index_t n_cols, offset_t nnz,
               const offset_t* row_offsets, const index_t* col_indices, const value_t* csr_values, 
               value_t* output,
               const reduce_t& op) {

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(nullptr, temp_storage_bytes,
        csr_values, output, n_rows, row_offsets, row_offsets + 1, op, op.template identity<value_t>()));

    // Caching allocator for device memory
    cub::CachingDeviceAllocator& allocator = utils::get_allocator();  
    void* d_temp_storage = nullptr;
    CUDA_CHECK(allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes,
        csr_values, output, n_rows, row_offsets, row_offsets + 1, op, op.template identity<value_t>()));
    
    CUDA_CHECK(allocator.DeviceFree(d_temp_storage));
}


}