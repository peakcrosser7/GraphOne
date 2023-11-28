#pragma once

#include "GraphGenlX/type.hpp"
#include "GraphGenlX/utils.h"
#include "GraphGenlX/base/buffer.h"

namespace graph_genlx {

template <arch_t arch,
          typename value_t,
          typename index_t = uint32_t,
          typename offset_t = uint64_t>
struct CscMat {
    constexpr static arch_t arch_type = arch;

    CscMat() = default;

    CscMat(index_t n_rows, index_t n_cols, offset_t nnz,
           Buffer<arch, offset_t, index_t> &&col_offsets,
           Buffer<arch, index_t, offset_t> &&row_indices,
           Buffer<arch, value_t, offset_t> &&values)
        : n_rows(n_rows), n_cols(n_cols), nnz(nnz),
          col_offsets(std::move(col_offsets)),
          row_indices(std::move(row_indices)), 
          values(std::move(values)) {}

    CscMat(index_t n_rows, index_t n_cols, offset_t nnz,
           const Buffer<arch, offset_t, index_t> &col_offsets,
           const Buffer<arch, index_t, offset_t> &row_indices,
           const Buffer<arch, value_t, offset_t> &values)
        : n_rows(n_rows), n_cols(n_cols), nnz(nnz),
          col_offsets(col_offsets), 
          row_indices(row_indices),
          values(values) {}

    std::string ToString() const {
        return "CscMat{ "
            "arch_type:" + utils::ToString(arch_type) + ", " +
            "n_rows:" + utils::NumToString(n_rows) + ", " +
            "n_cols:" + utils::NumToString(n_cols) + ", " +
            "nnz:" + utils::NumToString(nnz) + ", " +
            "col_offsets:" + col_offsets.ToString() + ", " +
            "row_indices:" + row_indices.ToString() + ", " +
            "values:" + values.ToString() +
            " }";
    }

    index_t n_rows;
    index_t n_cols;
    offset_t nnz;

    Buffer<arch, offset_t, index_t> col_offsets;
    Buffer<arch, index_t, offset_t> row_indices;
    Buffer<arch, value_t, offset_t> values;

};

} // namespace graph_genlx