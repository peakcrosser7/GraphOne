#pragma once

#include <cuda.h>

#include "graph_one/arch/cuda/utils.cuh"

namespace graph_one::cuda {

namespace {

template <typename index_t>
__device__ index_t BinarySearch(const index_t* array,
                              index_t target,
                              index_t begin,
                              index_t end) {
    while (begin < end) {
        int mid = begin + (end - begin) / 2;
        int item = array[mid];
        if (item == target) return mid;
        bool larger = (item > target);
        if (larger) {
            end = mid;
        } else {
            begin = mid + 1;
        }
    }
    return -1;
}

} // namespace 


// Reference:
// https://github.com/gunrock/graphblast/blob/master/graphblas/backend/cuda/kernels/spgemm.hpp
// Sparse matrix-Sparse matrix multiplication with sparse matrix mask
// Strategy:
// 1) Loop through mask using 1 warp/row
// 2) For each nonzero (row, col) of mask:
//    i)   initialize each thread to identity
//    ii)  compute dot-product A(row, :) x B(:, col)
//    iii) use warp on each nonzero at mat_a_value_t time
//    iv)  tally up accumulated sum using warp reduction
//    v)   write to global memory c_values
template <typename index_t, typename offset_t,
          typename mat_a_value_t, typename mat_b_value_t, 
          typename mat_c_value_t, typename mask_value_t,
          typename combine_t, typename reduce_t>
__global__ void GSpGEMM_Masked_CSRxCSRwCSR_graphblast_Kernel(
    index_t a_nrows, 
    const offset_t* a_row_offsets, const index_t* a_col_indices, const mat_a_value_t* a_values, 
    const offset_t* b_row_offsets, const index_t* b_col_indices, const mat_b_value_t* b_values,
    const offset_t* mask_row_offsets, const index_t* mask_col_indices, mask_value_t* mask_values,
    mat_c_value_t* c_values, 
    combine_t mul_op, reduce_t add_op) {
    
    index_t g_tid = blockIdx.x * blockDim.x + threadIdx.x;
    index_t warp_id = g_tid / 32;
    index_t lane_id = g_tid & (32 - 1);
    if (warp_id < a_nrows) {
        index_t row_start = mask_row_offsets[warp_id];
        index_t row_end = mask_row_offsets[warp_id + 1];

        // Entire warp works together on each nonzero
        for (index_t edge = row_start; edge < row_end; ++edge) {
            mask_value_t mask_val = mask_values[edge];
            mat_c_value_t accumulator =
                add_op.template identity<mat_c_value_t>();
            if (mask_val) {
                // Load B bounds on which we must do binary search
                index_t B_ind = mask_col_indices[edge];
                index_t B_col_start = b_row_offsets[B_ind];
                index_t B_col_end = b_row_offsets[B_ind + 1];

                // Each thread iterates along row
                // Does binary search on B_row to try to find A_col
                // Adds result to accumulator if found
                index_t ind = row_start + lane_id;
                for (index_t ind_start = row_start; ind_start < row_end;
                     ind_start += 32) {
                    if (ind < row_end) {
                        index_t A_col = a_col_indices[ind];
                        index_t B_row = BinarySearch(b_col_indices, A_col,
                                                     B_col_start, B_col_end);

                        if (B_row != -1) {
                            mat_a_value_t A_t = a_values[ind];
                            mat_b_value_t B_t = b_values[B_row];
                            mat_c_value_t C_t = mul_op(A_t, B_t);
                            accumulator = add_op(C_t, accumulator);
                        }
                    }
                    ind += 32;
                }

                // Warp reduce for each edge
                for (int i = 1; i < 32; i *= 2) {
                    accumulator = add_op(__shfl_xor_sync(0xFFFFFFFF, accumulator, i),
                                         accumulator);
                }
            }
            // Write to output
            if (lane_id == 0) {
                c_values[edge] = accumulator;
            }
        }
    }
}

template <typename index_t, typename offset_t, 
          typename mat_a_value_t, typename mat_b_value_t, 
          typename mat_c_value_t, typename mask_value_t,
          typename combine_t, typename reduce_t>
void GSpGEMM_Masked_CSRxCSRwCSR_graphblast(
    index_t m, index_t n, index_t k,
    offset_t a_nnz, offset_t b_nnz, offset_t mask_nnz,
    const offset_t* a_row_offsets, const index_t* a_col_indices, const mat_a_value_t* a_csr_values,
    const offset_t* b_row_offsets, const index_t* b_col_indices, const mat_b_value_t* b_csr_values,
    const offset_t* mask_row_offsets, const index_t* mask_col_indices, const mask_value_t* mask_csr_values,
    offset_t* c_row_offsets, index_t* c_col_indices, mat_c_value_t* c_csr_values,
    const combine_t& mul_op, const reduce_t& add_op) {

    constexpr unsigned kNumRowsPerBlock = 8;
    constexpr unsigned kNumThreads = 32 * kNumRowsPerBlock;
    unsigned num_blocks = (m + kNumRowsPerBlock - 1) / kNumRowsPerBlock;
    
    GSpGEMM_Masked_CSRxCSRwCSR_graphblast_Kernel<<<num_blocks, kNumThreads>>>(
        m, 
        a_row_offsets, a_col_indices, a_csr_values,
        b_row_offsets, b_col_indices, b_csr_values,
        mask_row_offsets, mask_col_indices, mask_csr_values,
        c_csr_values, mul_op, add_op);
    
    CUDA_CHECK(cudaGetLastError());
}        


}  // namespace graph_one::cuda
