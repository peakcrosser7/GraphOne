#pragma once

namespace graph_genlx::blas {

template <typename functor_t,
          typename index_t, typename offset_t, typename mat_value_t,
          typename vecx_value_t, typename vecy_value_t>
void SpMV_cpu_navie(index_t n_rows,
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vecx_value_t *x, vecy_value_t *y) {
    
    for (index_t row = 0; row < n_rows; ++row) {
        vecy_value_t sum = functor_t::initialize();

        for (index_t col_off = Ap[row]; col_off < Ap[row+1]; ++col_off) {
            sum = functor_t::reduce(sum, functor_t::combine(Ax[col_off], x[Aj[col_off]]));
        }
        y[row] = sum;
    }
}

} // namespace graph_genlx::blas 