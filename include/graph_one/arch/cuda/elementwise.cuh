#pragma once

#include <cuda.h>

#include "graph_one/arch/cuda/utils.cuh"

namespace graph_one::cuda {

namespace {

template <typename index_t, typename offset_t, 
          typename edge_value_t, typename vertex_value_t,
          typename output_value_t, typename binary_t>
__global__ void RowWiseCSRKernel(const binary_t& binary_op,
                                     index_t n_rows,
                                     const offset_t* row_offsets,
                                     const edge_value_t* edge_input,
                                     const vertex_value_t* vertex_input,
                                     output_value_t* output) {
  unsigned thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned warp_id   = thread_id / 32;
  unsigned lane_id   = thread_id & (32 - 1);
  if (warp_id < n_rows) {
    offset_t row_start = row_offsets[warp_id];
    offset_t row_end   = row_offsets[warp_id + 1];
    vertex_value_t b = vertex_input[warp_id];

    offset_t ind = row_start + lane_id;
    for (offset_t ind_start = row_start; ind_start < row_end; ind_start += 32) {
      if (ind < row_end) {
        edge_value_t a = edge_input[ind];
        output_value_t c = binary_op(a, b);
        output[ind] = c;
      }
      ind += 32;
    }
  }
}

template <typename index_t, typename offset_t, 
          typename edge_value_t, typename vertex_value_t,
          typename output_value_t, typename binary_t>
__global__ void ColumnWiseCSRKernel(const binary_t& binary_op,
                                     index_t n_rows,
                                     const offset_t* row_offsets,
                                     const index_t* col_indices,
                                     const edge_value_t* edge_input,
                                     const vertex_value_t* vertex_input,
                                     output_value_t* output) {
  unsigned thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned warp_id   = thread_id / 32;
  unsigned lane_id   = thread_id & (32 - 1);
  if (warp_id < n_rows) {
    offset_t row_start = row_offsets[warp_id];
    offset_t row_end   = row_offsets[warp_id + 1];

    offset_t ind = row_start + lane_id;
    for (offset_t ind_start = row_start; ind_start < row_end; ind_start += 32) {
      if (ind < row_end) {
        index_t B_ind = col_indices[ind];
        vertex_value_t b = vertex_input[B_ind];

        edge_value_t a = edge_input[ind];
        output_value_t c = binary_op(a, b);
        output[ind] = c;
      }
      ind += 32;
    }
  }
}

} // namespace 


template <typename index_t, typename offset_t, 
          typename edge_value_t, typename vertex_value_t,
          typename output_value_t, typename binary_t>
void ElementWiseCSR(index_t n_rows, index_t n_cols, offset_t nnz,
                    const offset_t* row_offsets, const index_t* col_indices, 
                    const edge_value_t* edge_input, const vertex_value_t* vertex_input, 
                    output_value_t* output,
                    bool use_rows,
                    const binary_t& binary_op) {
    
    constexpr int kNumRowsPerBlock = 8;
    constexpr int kNumThreads = 32 * kNumRowsPerBlock;
    int num_blocks = (n_rows + kNumRowsPerBlock - 1) / kNumRowsPerBlock;

    if (use_rows) {
      RowWiseCSRKernel<<<num_blocks, kNumThreads>>>(
          binary_op, n_rows, row_offsets, edge_input, vertex_input, output);
    } else {
      ColumnWiseCSRKernel<<<num_blocks, kNumThreads>>>(
          binary_op, n_rows, row_offsets, col_indices, edge_input, vertex_input, output);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace graph_one::cuda
