#pragma once 

#include "GraphGenlX/archi/blas/SpMV/def.hpp"
#include "GraphGenlX/archi/blas/SpMV/cpu/naive.hpp"

namespace graph_genlx::blas {

struct SpmvCpuNaive {
    constexpr static arch_t arch_type = arch_t::cpu;
};

template <typename functor_t,
          typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t>
struct SpmvDispatcher<SpmvCpuNaive, functor_t, 
                      index_t, offset_t, mat_value_t,
                      vec_x_value_t, vec_y_value_t> {

    using spmv_params_t =
        SpmvParams<index_t, offset_t, mat_value_t, vec_x_value_t, vec_y_value_t>;

    SpmvDispatcher(spmv_params_t &spmv_params)
    : params(spmv_params) {}

    void operator()() {
        SpMV_cpu_navie<functor_t>(params.n_rows, 
            params.row_offsets, params.col_indices, 
            params.values, params.vector_x, params.vector_y);
    }

    spmv_params_t params;
};


} // namespace graph_genlx::blas 