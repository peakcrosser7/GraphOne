#pragma once

#include <string>

#include "GraphOne/type.hpp"
#include "GraphOne/utils.h"
#include "GraphOne/base/buffer.h"

namespace graph_one {

template <arch_t arch,
          typename value_t,
          typename index_t = uint32_t,
          typename offset_t = uint64_t,
          vstart_t v_start = vstart_t::FROM_0_TO_0>
struct CsrMat {
    constexpr static arch_t arch_value = arch;
    constexpr static vstart_t vstart_value = v_start;

    using value_type = value_t;
    using index_type = index_t;
    using offset_type = offset_t;

    CsrMat() = default;

    CsrMat(index_t n_rows, index_t n_cols, offset_t nnz,
           Buffer<arch, offset_t, index_t> &&row_offsets,
           Buffer<arch, index_t, offset_t> &&col_indices,
           Buffer<arch, value_t, offset_t> &&values)
        : n_rows(n_rows), n_cols(n_cols), nnz(nnz),
          row_offsets(std::move(row_offsets)),
          col_indices(std::move(col_indices)), 
          values(std::move(values)) {}

    CsrMat(index_t n_rows, index_t n_cols, offset_t nnz,
           const Buffer<arch, offset_t, index_t> &row_offsets,
           const Buffer<arch, index_t, offset_t> &col_indices,
           const Buffer<arch, value_t, offset_t> &values)
        : n_rows(n_rows), n_cols(n_cols), nnz(nnz),
          row_offsets(row_offsets), 
          col_indices(col_indices),
          values(values) {}

    std::string ToString() const {
        return "CsrMat{ "
            "arch_value:" + utils::ToString(arch_value) + ", " +
            "n_rows:" + utils::NumToString(n_rows) + ", " +
            "n_cols:" + utils::NumToString(n_cols) + ", " +
            "nnz:" + utils::NumToString(nnz) + ", " +
            "row_offsets:" + row_offsets.ToString() + ", " +
            "col_indices:" + col_indices.ToString() + ", " +
            "values:" + values.ToString() +
            " }";
    }

    index_t n_rows;
    index_t n_cols;
    offset_t nnz;

    Buffer<arch, offset_t, index_t> row_offsets;
    Buffer<arch, index_t, offset_t> col_indices;
    Buffer<arch, value_t, offset_t> values;

};


} // namespace graph_one