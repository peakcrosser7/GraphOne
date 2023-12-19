#pragma once

#include <vector>

#include "GraphGenlX/type.hpp"
#include "GraphGenlX/utils.h"
#include "GraphGenlX/archi.h"

namespace graph_genlx {

template <arch_t arch,
          typename value_t, 
          typename index_t = uint32_t,
          typename nnz_t = uint64_t>
struct CooMat {
    CooMat() = default;

    CooMat(index_t n_rows, index_t n_cols)
        : n_rows(n_rows), n_cols(n_cols) {}
    
    CooMat(index_t n_rows, index_t n_cols,
        thrust_vec<arch, index_t>&& row_indices, 
        thrust_vec<arch, index_t>&& col_indices,
        thrust_vec<arch, value_t>&& values)
        : n_rows(n_rows), n_cols(n_cols), nnz(row_indices.size()),
          row_indices(std::move(row_indices)), 
          col_indices(std::move(col_indices)), 
          values(std::move(values)) {}

    std::string ToString() const {
        return "CooMat{ "
            "n_rows:" + utils::NumToString(n_rows) + ", " +
            "n_cols:" + utils::NumToString(n_cols) + ", " +
            "nnz:" + utils::NumToString(nnz) + ", " +
            "row_indices:" + utils::VecToString(row_indices) + ", " +
            "col_indices:" + utils::VecToString(col_indices) + ", " +
            "values:" + utils::VecToString(values) +
            " }";
    }

    index_t n_rows{0};
    index_t n_cols{0};
    index_t nnz{0};

    thrust_vec<arch, index_t> row_indices{};
    thrust_vec<arch, index_t> col_indices{};
    thrust_vec<arch, value_t> values{};
};

} // namespace graph_genlx