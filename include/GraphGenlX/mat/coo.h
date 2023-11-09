#pragma once

#include <vector>

#include "GraphGenlX/type.hpp"
#include "GraphGenlX/utils.h"
#include "GraphGenlX/vec/vector.cuh"

namespace graph_genlx {

template <arch_t arch,
          typename value_t, 
          typename index_t = uint32_t,
          typename nnz_t = uint32_t>
struct CooMat {
    CooMat() = default;

    CooMat(index_t num_rows, index_t num_cols)
        : n_rows(n_rows), n_cols(num_cols) {}
    
    CooMat(index_t num_rows, index_t num_cols,
        std::vector<index_t>&& rows, std::vector<index_t>&& cols,
        std::vector<value_t>&& weights)
        : n_rows(num_rows), n_cols(num_cols), nnz(rows.size()),
          row_indices(rows), col_indices(cols), values(weights) {}

    std::string ToString() const {
        return "CooMat{ "
            "n_rows:" + utils::ToString(n_rows) + ", " +
            "n_cols:" + utils::ToString(n_cols) + ", " +
            "nnz:" + utils::ToString(nnz) + ", " +
            "row_indices:" + utils::VecToString(row_indices) + ", " +
            "col_indices:" + utils::VecToString(col_indices) + ", " +
            "values:" + utils::VecToString(values) +
            " }";
    }

    index_t n_rows{0};
    index_t n_cols{0};
    index_t nnz{0};

    vector_t<index_t> row_indices{};
    vector_t<index_t> col_indices{};
    vector_t<value_t> values{};
};

} // namespace graph_genlx