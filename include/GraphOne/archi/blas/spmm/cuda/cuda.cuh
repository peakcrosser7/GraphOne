#pragma once

#include <cusparse.h>

#include "GraphOne/type.hpp"
#include "GraphOne/archi/check/cuda.cuh"
#include "GraphOne/archi/blas/spmm/def.hpp"

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        abort();                                                               \
    }                                                                          \
}

namespace graph_one::blas {

struct SpmmCudaCsrCusparse {
    constexpr static arch_t arch_type = arch_t::cuda;
};

namespace {

template <typename T>
cusparseIndexType_t get_cusparse_index_t();

template <>
cusparseIndexType_t get_cusparse_index_t<int32_t>() {
    return CUSPARSE_INDEX_32I;
}

template <>
cusparseIndexType_t get_cusparse_index_t<uint32_t>() {
    return CUSPARSE_INDEX_32I;
}

template <>
cusparseIndexType_t get_cusparse_index_t<int64_t>() {
    return CUSPARSE_INDEX_64I;
}

template <>
cusparseIndexType_t get_cusparse_index_t<uint64_t>() {
    return CUSPARSE_INDEX_64I;
}

template <typename T>
cudaDataType_t get_cuda_data_t();

template <>
cudaDataType_t get_cuda_data_t<float>() {
    return CUDA_R_32F;
}

template <>
cudaDataType_t get_cuda_data_t<double>() {
    return CUDA_R_64F;
}

} // namespace 

template<typename functor_t,
         typename index_t, typename offset_t, typename spmat_value_t,
         typename matB_value_t, typename matC_value_t, typename compute_value_t>
struct SpmmDispatcher<SpmmCudaCsrCusparse, functor_t,
                      index_t, offset_t, spmat_value_t, 
                      true, matB_value_t, true, matC_value_t, compute_value_t> {
    constexpr static bool matB_row_major = true;
    constexpr static bool matC_row_major = true;

    SpmmDispatcher(SpmmCsrParams<index_t, offset_t, spmat_value_t, 
                                 matB_row_major, matB_value_t, 
                                 matC_row_major, matC_value_t>& spmm_params) {
        CHECK_CUSPARSE(cusparseCreate(&handle));
        CHECK_CUSPARSE(cusparseCreateConstCsr(&spmat_descr, 
                                              spmm_params.n_rows_A, spmm_params.n_cols_A, spmm_params.nnz_A,
                                              spmm_params.row_offsets, spmm_params.col_indices, spmm_params.csr_values,
                                              get_cusparse_index_t<index_t>(), get_cusparse_index_t<offset_t>(), CUSPARSE_INDEX_BASE_ZERO,
                                              get_cuda_data_t<spmat_value_t>()));
        CHECK_CUSPARSE(cusparseCreateConstDnMat(&matB_descr, spmm_params.n_cols_A, spmm_params.n_cols_B, spmm_params.n_cols_B, 
                                                spmm_params.mat_B, get_cuda_data_t<matB_value_t>(), CUSPARSE_ORDER_ROW));
        CHECK_CUSPARSE(cusparseCreateDnMat(&matC_descr, spmm_params.n_rows_A, spmm_params.n_cols_B, spmm_params.n_cols_B, 
                                            spmm_params.mat_C, get_cuda_data_t<matC_value_t>(), CUSPARSE_ORDER_ROW));
        
        compute_value_t alpha = 1.0, beta = 0.0;
        CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha, spmat_descr, matB_descr, &beta, matC_descr, 
                                               get_cuda_data_t<compute_value_t>(), CUSPARSE_SPMM_CSR_ALG2, &temp_storage_bytes));
        checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    }

    void operator() () {
        compute_value_t alpha = 1.0, beta = 0.0;
        CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, spmat_descr, matB_descr, &beta, matC_descr,
                                    get_cuda_data_t<compute_value_t>(), CUSPARSE_SPMM_CSR_ALG2, d_temp_storage));
    }

    ~SpmmDispatcher() {
        checkCudaErrors(cudaFree(d_temp_storage));
        CHECK_CUSPARSE(cusparseDestroySpMat(spmat_descr));
        CHECK_CUSPARSE(cusparseDestroyDnMat(matB_descr));
        CHECK_CUSPARSE(cusparseDestroyDnMat(matC_descr));
        CHECK_CUSPARSE(cusparseDestroy(handle));
    }

    cusparseHandle_t handle;
    cusparseConstSpMatDescr_t spmat_descr;
    cusparseConstDnMatDescr_t matB_descr;
    cusparseDnMatDescr_t matC_descr;
    size_t temp_storage_bytes;
    void *d_temp_storage;
};

    
} // namespace graph_one::blas
