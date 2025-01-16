#pragma once

#include "GraphOne/type.hpp"

namespace graph_one::blas {

template <bool matA_row_major, typename matA_value_t,
          bool matB_row_major, typename matB_value_t,
          bool matC_row_major, typename matC_value_t, 
          typename compute_value_t = matC_value_t, typename index_t = int>
struct GemmParams {
    index_t m;
    index_t n;
    index_t k;
    const matA_value_t* mat_A;
    const matB_value_t* mat_B;
    matC_value_t* mat_C;
};


template <typename kind,
    bool matA_row_major,
    typename matA_value_t,
    bool matB_row_major,
    typename matB_value_t,
    bool matC_row_major,
    typename matC_value_t,
    typename compute_value_t = matC_value_t,
    typename index_t = int>
struct GemmDispatcher {};


template <arch_t arch, typename kind,
    bool matA_row_major,
    bool matB_row_major,
    bool matC_row_major,
    typename matA_value_t,
    typename matB_value_t,
    typename matC_value_t,
    typename compute_value_t = matC_value_t,
    typename index_t = int>
GemmDispatcher<kind, matA_row_major, matA_value_t,
              matB_row_major, matB_value_t, 
              matC_row_major, matC_value_t, 
              compute_value_t, index_t>
MakeGemm(index_t m, index_t n, index_t k,
         const matA_value_t* mat_A, const matB_value_t* mat_B, 
         matC_value_t* mat_C) {
    static_assert(arch == kind::arch_type);

    GemmParams<matA_row_major, matA_value_t, 
               matB_row_major, matB_value_t,
               matC_row_major, matC_value_t, 
               compute_value_t, index_t> params;
    params.m = m;
    params.n = n;
    params.k = k;
    params.mat_A = mat_A;
    params.mat_B = mat_B;
    params.mat_C = mat_C;

    return GemmDispatcher<kind, matA_row_major, matA_value_t,
                          matB_row_major, matB_value_t,
                          matC_row_major, matC_value_t, 
                          compute_value_t, index_t>(params);
}
    
} // namespace graph_one::blas
