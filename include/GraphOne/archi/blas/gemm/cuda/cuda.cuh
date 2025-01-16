#pragma once
#include <type_traits>

#include <cuda.h>

#include "cutlass/gemm/device/gemm.h"

#include "GraphOne/type.hpp"
#include "GraphOne/archi/blas/gemm/def.hpp"

namespace graph_one::blas {

struct GemmCutlass {
    constexpr static arch_t arch_type = arch_t::cuda;
};

template <bool matA_row_major, typename matA_value_t,
          bool matB_row_major, typename matB_value_t,
          bool matC_row_major, typename matC_value_t, 
          typename compute_value_t, typename index_t>
struct GemmDispatcher<GemmCutlass, 
                      matA_row_major, matA_value_t,
                      matB_row_major, matB_value_t,
                      matC_row_major, matC_value_t, 
                      compute_value_t, index_t> {

    template <bool row_major>
    using layout_t = std::conditional_t<row_major, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;

    using CutlassGemm = cutlass::gemm::device::Gemm<matA_value_t,        // Data-type of A matrix
                                                layout_t<matA_row_major>,  // Layout of A matrix
                                                matB_value_t,        // Data-type of B matrix
                                                layout_t<matB_row_major>,  // Layout of B matrix
                                                matC_value_t,        // Data-type of C matrix
                                                layout_t<matC_row_major>,   // Layout of C matrix
                                                compute_value_t>;                         
    using CutlassArgs = typename CutlassGemm::Arguments;
    
    GemmDispatcher(GemmParams<matA_row_major, matA_value_t,
                              matB_row_major, matB_value_t,
                              matC_row_major, matC_value_t, 
                              compute_value_t, index_t>& params) {
        index_t lda = matA_row_major ? params.k : params.m;
        index_t ldb = matB_row_major ? params.n : params.k;
        index_t ldc = matC_row_major ? params.n : params.m;

        args = CutlassArgs({static_cast<int>(params.m), static_cast<int>(params.n), static_cast<int>(params.k)},  // Gemm Problem dimensions
                            {params.mat_A, lda},    // Tensor-ref for source matrix A
                            {params.mat_B, ldb},    // Tensor-ref for source matrix B
                            {params.mat_C, ldc},    // Tensor-ref for source matrix C
                            {params.mat_C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                            {1.0, 0.0});  // alpha, beta
    }

    void operator() () {
        CutlassGemm gemm_operator;
        cutlass::Status status = gemm_operator(args);

        if (status != cutlass::Status::kSuccess) {
            printf("CUTLASS GEMM failed at line %d with error: %s (%d)\n",
               __LINE__, cutlass::cutlassGetStatusString(status), status);
            abort();  
        }
    }


    CutlassArgs args;
};


} // namespace graph_one::blas
