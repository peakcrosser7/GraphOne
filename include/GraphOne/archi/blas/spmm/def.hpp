#pragma once

#include "GraphOne/type.hpp"

namespace graph_one::blas {

template <
    typename index_t,
    typename offset_t,
    typename spmat_value_t,
    bool matB_row_major,
    typename matB_value_t,
    bool matC_row_major,
    typename matC_value_t,
    typename compute_value_t = matC_value_t>
struct SpmmCsrParams {
    index_t n_rows_A;
    index_t n_cols_A;
    offset_t nnz_A;
    index_t n_cols_B;
    const offset_t* row_offsets;
    const index_t* col_indices;
    const spmat_value_t* csr_values;
    const matB_value_t* mat_B;
    matC_value_t* mat_C;
};

template <typename compute_value_t>
struct DefaultSpmmFunctor {
    static constexpr bool use_cusparse = true;

    __host__ __device__ __forceinline__ 
    static compute_value_t
    combine(const compute_value_t &nonezero, const compute_value_t &x) {
        return nonezero * x;
    }

    __host__ __device__ __forceinline__ 
    static compute_value_t
    reduce(const compute_value_t &lhs, const compute_value_t &rhs) {
        return lhs + rhs;
    }
};

template <typename kind, typename functor_t,
          typename index_t, typename offset_t, typename spmat_value_t,
          bool matB_row_major, typename matB_value_t, 
          bool matC_row_major, typename matC_value_t, typename compute_value_t = matC_value_t>
struct SpmmDispatcher {};


template <arch_t arch, typename kind, typename functor_t,
          bool matB_row_major, bool matC_row_major,
          typename index_t, typename offset_t, typename spmat_value_t,
          typename matB_value_t, typename matC_value_t, typename compute_value_t = matC_value_t>
SpmmDispatcher<kind, functor_t, index_t, offset_t, spmat_value_t, matB_row_major, matB_value_t, 
               matC_row_major, matC_value_t, compute_value_t>
MakeCsrSpMM(index_t n_rows_A, index_t n_cols_A, offset_t nnz_A, index_t n_cols_B,
            const offset_t* row_offsets, const index_t* col_indices, const spmat_value_t* csr_values,
            const matB_value_t* mat_B, matC_value_t* mat_C) {
    static_assert(arch == kind::arch_type);

    SpmmCsrParams<index_t, offset_t, spmat_value_t, 
                  matB_row_major, matB_value_t, 
                  matC_row_major, matC_value_t, compute_value_t> params;
    params.n_rows_A = n_rows_A;
    params.n_cols_A = n_cols_A;
    params.nnz_A = nnz_A;
    params.n_cols_B = n_cols_B;
    params.row_offsets = row_offsets;
    params.col_indices = col_indices;
    params.csr_values = csr_values;
    params.mat_B = mat_B;
    params.mat_C = mat_C;

    return SpmmDispatcher<kind, functor_t, index_t, offset_t, 
                          spmat_value_t, matB_row_major, matB_value_t,
                          matC_row_major, matC_value_t, compute_value_t>(params);
}


    
} // namespace graph_one::blas