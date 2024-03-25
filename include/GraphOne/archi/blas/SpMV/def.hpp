#pragma once 

#include "GraphOne/type.hpp"
#include "GraphOne/archi/macro/macro.h"

namespace graph_one::blas {

template <
    typename        index_t,
    typename        offset_t,
    typename        mat_value_t,
    typename        vec_x_value_t,
    typename        vec_y_value_t>
struct SpmvParams {
    index_t              n_rows;            ///< Number of rows of matrix <b>A</b>.
    index_t              n_cols;            ///< Number of columns of matrix <b>A</b>.
    offset_t             nnz;               ///< Number of nonzero elements of matrix <b>A</b>.
    offset_t*            row_offsets;       ///< Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
    index_t*             col_indices;       ///< Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
    mat_value_t*         values;            ///< Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
    vec_x_value_t*       vector_x;          ///< Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    vec_y_value_t*       vector_y;          ///< Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
};

template <typename mat_value_t, 
          typename vec_x_value_t,
          typename vec_y_value_t>
struct DefaultSpmvFunctor {
    __host__ __device__ __forceinline__ 
    static vec_y_value_t initialize() {
        return vec_y_value_t{0};
    }

    __host__ __device__ __forceinline__ 
    static vec_y_value_t
    combine(const mat_value_t &nonezero, const vec_x_value_t &x) {
        return (nonezero < x) ? nonezero : x;
    }

    __host__ __device__ __forceinline__ 
    static vec_y_value_t
    reduce(const vec_y_value_t &lhs, const vec_y_value_t &rhs) {
        return lhs + rhs;
    }
};

template<typename kind, typename functor_t,
         typename index_t, typename offset_t, typename mat_value_t,
         typename vec_x_value_t, typename vec_y_value_t>
struct SpmvDispatcher {};

template <arch_t arch, typename kind, typename functor_t,
          typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t>
SpmvDispatcher<kind, functor_t, index_t, offset_t, mat_value_t, vec_x_value_t, vec_y_value_t> 
MakeSpMV(index_t n_rows, index_t n_cols, offset_t nnz,
         const offset_t *row_offsets, const index_t *col_indices, const mat_value_t *values,
         const vec_x_value_t *vector_x, vec_y_value_t *vector_y) {
    static_assert(arch == kind::arch_type);

    SpmvParams<index_t, offset_t, mat_value_t, vec_x_value_t, vec_y_value_t> params;
    params.n_rows      = n_rows;
    params.n_cols      = n_cols;
    params.nnz         = nnz;
    params.row_offsets = const_cast<offset_t *>(row_offsets);
    params.col_indices = const_cast<index_t *>(col_indices);
    params.values      = const_cast<mat_value_t *>(values);
    params.vector_x    = const_cast<vec_x_value_t *>(vector_x);
    params.vector_y    = vector_y;

    return SpmvDispatcher<kind, functor_t, index_t, offset_t, mat_value_t, vec_x_value_t, vec_y_value_t>(
        params);
}



} // namespace graph_one::blas 
