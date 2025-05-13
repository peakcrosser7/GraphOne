#pragma once

#include <cusparse.h>

#include "graph_one/allocator.h"
#include "graph_one/arch/cuda/utils.cuh"


namespace graph_one::cuda {


void SpGEMM_CSRxCSR_cusparse(
    int64_t m, int64_t n, int64_t k,
    int64_t a_nnz, int64_t b_nnz,
    const int32_t* a_row_offsets, const int32_t* a_col_indices, const float* a_csr_values,
    const int32_t* b_row_offsets, const int32_t* b_col_indices, const float* b_csr_values,
    int32_t* c_row_offsets, int32_t** c_col_indices, float** c_csr_values) {

    auto& allocator = MemAllocator::Get();
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = nullptr;
    cusparseConstSpMatDescr_t matA, matB;
    cusparseSpMatDescr_t matC;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    float                alpha       = 1.0f;
    float                beta        = 0.0f;
    cusparseIndexType_t  offset_type = CUSPARSE_INDEX_32I;
    cusparseIndexType_t  index_type  = CUSPARSE_INDEX_32I;
    cudaDataType         computeType = CUDA_R_32F;
    cusparseOperation_t  opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t  opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;

    CUSPARSE_CHECK( cusparseCreate(&handle) );
    // Create sparse matrix A in CSR format
    CUSPARSE_CHECK( cusparseCreateConstCsr(&matA, m, k, a_nnz,
                                           a_row_offsets, a_col_indices, a_csr_values,
                                           offset_type, index_type,
                                           CUSPARSE_INDEX_BASE_ZERO, computeType) );
    CUSPARSE_CHECK( cusparseCreateConstCsr(&matB, k, n, b_nnz,
                                           b_row_offsets, b_col_indices, b_csr_values,
                                           offset_type, index_type,
                                           CUSPARSE_INDEX_BASE_ZERO, computeType) );
    CUSPARSE_CHECK( cusparseCreateCsr(&matC, m, n, 0,
                                      c_row_offsets, nullptr, nullptr,
                                      offset_type, index_type,
                                      CUSPARSE_INDEX_BASE_ZERO, computeType) );
    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CUSPARSE_CHECK( cusparseSpGEMM_createDescr(&spgemmDesc) );

    // ask bufferSize1 bytes for external memory
    CUSPARSE_CHECK(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, nullptr) );
    void* dBuffer1 = allocator.CudaAllocate(bufferSize1);
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CUSPARSE_CHECK(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, dBuffer1) );

    // ask bufferSize2 bytes for external memory
    CUSPARSE_CHECK(
        cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, matA, matB, &beta, matC,
                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, nullptr) );
    void* dBuffer2 = allocator.CudaAllocate(bufferSize2);

    // compute the intermediate product of A * B
    CUSPARSE_CHECK( cusparseSpGEMM_compute(handle, opA, opB,
                                           &alpha, matA, matB, &beta, matC,
                                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc, &bufferSize2, dBuffer2) );
    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    CUSPARSE_CHECK( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                         &C_nnz1) );
    // allocate matrix C
    *c_col_indices = allocator.CudaAllocate<int32_t>(C_nnz1);
    *c_csr_values = allocator.CudaAllocate<float>(C_nnz1);

    // NOTE: if 'beta' != 0, the values of C must be update after the allocation
    //       of dC_values, and before the call of cusparseSpGEMM_copy

    // update matC with the new pointers
    CUSPARSE_CHECK(
        cusparseCsrSetPointers(matC, c_row_offsets, *c_col_indices, *c_csr_values) );

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

    // copy the final products to the matrix C
    CUSPARSE_CHECK(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) );

    // destroy matrix/vector descriptors
    CUSPARSE_CHECK( cusparseSpGEMM_destroyDescr(spgemmDesc) );
    CUSPARSE_CHECK( cusparseDestroySpMat(matA) );
    CUSPARSE_CHECK( cusparseDestroySpMat(matB) );
    CUSPARSE_CHECK( cusparseDestroySpMat(matC) );
    CUSPARSE_CHECK( cusparseDestroy(handle) );

    // free temporary buffers
    allocator.Free(dBuffer1);
    allocator.Free(dBuffer2);
}


} // namespace graph_one::cuda
