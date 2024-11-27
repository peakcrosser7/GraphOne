#pragma once

#include "GraphOne/archi/check/cuda.cuh"
#include "GraphOne/archi/blas/spmv/def.hpp"
#include "GraphOne/archi/blas/spmv/cuda/csr_vector.cuh"
#include "GraphOne/archi/blas/spmv/cuda/merge_based/merge_based_spmv.cuh"

namespace graph_one::blas {

struct SpmvCudaCsrVector {
    constexpr static arch_t arch_type = arch_t::cuda;
};

template <typename functor_t,
          typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t>
struct SpmvDispatcher<SpmvCudaCsrVector, functor_t, 
                      index_t, offset_t, mat_value_t,
                      vec_x_value_t, vec_y_value_t> {

    using spmv_params_t =
        SpmvCsrParams<index_t, offset_t, mat_value_t, vec_x_value_t, vec_y_value_t>;
    
    SpmvDispatcher(spmv_params_t &spmv_params)
    : params(spmv_params),
      nnz_per_row(spmv_params.nnz / spmv_params.n_rows) {}

    void operator() (bool sync = true) {
        cudaError_t error;
        if (nnz_per_row <= 2) {
            error = SpMV_cuda_csr_vector<2, functor_t>(
                params.n_rows, params.row_offsets, params.col_indices,
                params.values, params.vector_x, params.vector_y);
            return;
        }
        if (nnz_per_row <= 4) {
            error = SpMV_cuda_csr_vector<4, functor_t>(
                params.n_rows, params.row_offsets, params.col_indices,
                params.values, params.vector_x, params.vector_y);
            return;
        }
        if (nnz_per_row <= 8) {
            error = SpMV_cuda_csr_vector<8, functor_t>(
                params.n_rows, params.row_offsets, params.col_indices,
                params.values, params.vector_x, params.vector_y);
            return;
        }
        error = SpMV_cuda_csr_vector<16, functor_t>(
            params.n_rows, params.row_offsets, params.col_indices,
            params.values, params.vector_x, params.vector_y);
        checkCudaErrors(error);

        if(sync) {
            checkCudaErrors(cudaDeviceSynchronize());
        }
    }

    spmv_params_t params;

    offset_t nnz_per_row;
};


struct SpmvCudaCsrMergeBased {
    constexpr static arch_t arch_type = arch_t::cuda;
};

template <typename functor_t,
          typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t>
struct SpmvDispatcher<SpmvCudaCsrMergeBased, functor_t, 
                      index_t, offset_t, mat_value_t,
                      vec_x_value_t, vec_y_value_t> {

    SpmvDispatcher(SpmvCsrParams<index_t, offset_t, mat_value_t, 
                                 vec_x_value_t, vec_y_value_t> &spmv_params)
    : spmv_obj(), 
      temp_storage_bytes(0), 
      d_temp_storage(nullptr) {
        params.d_values = const_cast<mat_value_t*>(spmv_params.values);
        params.d_row_end_offsets = const_cast<offset_t*>(spmv_params.row_offsets) + 1;
        params.d_column_indices = const_cast<index_t*>(spmv_params.col_indices);
        params.d_vector_x = const_cast<vec_x_value_t*>(spmv_params.vector_x);
        params.d_vector_y = spmv_params.vector_y;
        params.num_rows = spmv_params.n_rows;
        params.num_cols = spmv_params.n_cols;
        params.num_nonzeros = spmv_params.nnz;

        checkCudaErrors(spmv_obj.GetBufferSize(d_temp_storage, temp_storage_bytes, params));
        checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        checkCudaErrors(spmv_obj.SegmentSearch(d_temp_storage, temp_storage_bytes, params));
    }

    void operator() (bool sync = true) {
        checkCudaErrors(spmv_obj.SpmvMerge(d_temp_storage, temp_storage_bytes, params));
        if (sync) {
            checkCudaErrors(cudaDeviceSynchronize());
        }
    }

    ~SpmvDispatcher() {
        checkCudaErrors(cudaFree(d_temp_storage));
    }


    merge::SpmvCsrParams<index_t, offset_t, mat_value_t, vec_x_value_t,
                      vec_y_value_t>
        params;
    merge::MergeBasedSpmv<index_t, offset_t, mat_value_t, vec_x_value_t,
                          vec_y_value_t, functor_t>
        spmv_obj;
    size_t temp_storage_bytes;
    void *d_temp_storage;
};

} // namespace graph_one::blas