#pragma once

#include <string>

#include "GraphGenlX/type.hpp"
#include "GraphGenlX/utils.h"
#include "GraphGenlX/base/buffer.h"

namespace graph_genlx {

template <arch_t arch,
          typename value_t,
          typename index_t = uint32_t,
          typename offset_t = uint64_t>
struct CsrMat {
    constexpr static arch_t arch_type = arch;

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
            "arch_type:" + utils::ToString(arch_type) + ", " +
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


} // namespace graph_genlx