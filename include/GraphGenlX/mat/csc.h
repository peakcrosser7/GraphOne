#pragma once

#include "GraphGenlX/type.hpp"

namespace graph_genlx {

template <arch_t arch,
          typename value_t,
          typename index_t = uint32_t,
          typename offset_t = uint64_t>
class CscMat {
public:
    CscMat() = default;

    CscMat(index_t n_rows, index_t n_cols, offset_t nnz,
           Buffer<offset_t, arch, index_t> &&col_offsets,
           Buffer<index_t, arch, offset_t> &&row_indices,
           Buffer<value_t, arch, offset_t> &&values)
        : n_rows(n_rows), n_cols(n_cols), nnz(nnz),
          col_offsets(std::move(col_offsets)),
          row_indices(std::move(row_indices)), 
          values(std::move(values)) {}

    CscMat(index_t n_rows, index_t n_cols, offset_t nnz,
           const Buffer<offset_t, arch, index_t> &col_offsets,
           const Buffer<index_t, arch, offset_t> &row_indices,
           const Buffer<value_t, arch, offset_t> &values)
        : n_rows(n_rows), n_cols(n_cols), nnz(nnz),
          col_offsets(col_offsets), 
          row_indices(row_indices),
          values(values) {}

    std::string ToString() const {
        return "CsrMat{ "
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

    Buffer<offset_t, arch, index_t> col_offsets;
    Buffer<index_t, arch, offset_t> row_indices;
    Buffer<value_t, arch, offset_t> values;

};

} // namespace graph_genlx