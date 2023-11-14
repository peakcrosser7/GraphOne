#pragma once

#include <string>

#include "GraphGenlX/type.hpp"
#include "GraphGenlX/utils.h"
#include "GraphGenlX/buffer.h"
#include "GraphGenlX/mat/coo.h"

namespace graph_genlx {

struct MatBuilder;

template <arch_t arch,
          typename value_t,
          typename index_t = uint32_t,
          typename offset_t = uint64_t>
class CsrMat {
public:
    // static CsrMat FromCoo(const CooMat<arch_t::cpu, value_t, index_t, offset_t>& coo) {
    //     index_t n_rows = coo.n_rows;
    //     index_t n_cols = coo.n_cols;
    //     offset_t nnz = coo.nnz;


    //     Buffer<offset_t, arch_t::cpu, index_t> row_offsets(coo.n_rows + 1);
    //     Buffer<index_t, arch_t::cpu> col_indices(coo.nnz);
    //     Buffer<value_t, arch_t::cpu> values(coo.nnz);

    //     // compute number of non-zero entries per row
    //     for (offset_t i = 0; i < nnz; ++i) {
    //         ++row_offsets[coo.row_indices[i]];
    //     }
        
    //     // cumulative sum the nnz per row to get row_offsets[]
    //     for (index_t r = 0, total = 0, r <= n_rows; ++r) {
    //         index_t tmp = row_offsets[r];
    //         row_offsets[r] = total;
    //         total += tmp;
    //     }
    //     row_offsets[n_rows] = nnz;

    //     for (offset_t i = 0; i < nnz; ++i) {
    //         index_t row = coo.row_indices[i];
    //         index_t row_off = row_offsets[row];
    //         col_indices[row_off] = coo.col_indices[i];
    //         values[row_off] = coo.values[i];
    //         ++row_offsets[row];
    //     }

    //     for (index_t r = 0, pre = 0; r <= n_rows; ++r) {
    //         index_t tmp = row_offsets[r];
    //         row_offsets[r] = pre;
    //         pre = tmp;
    //     }

    //     CsrMat<arch, value_t, index_t, offset_t> csr;
    //     csr.n_rows_ = n_rows;
    //     csr.n_cols_ = n_cols;
    //     csr.nnz_ = nnz;
    //     csr.row_offsets_ = row_offsets;
    //     csr.col_indices_ = col_indices;
    //     csr.values_ = values;

    //     return csr;
    // }

    std::string ToString() const {
        return "CsrMat{ "
            "n_rows_:" + utils::NumToString(n_rows_) + ", " +
            "n_cols_:" + utils::NumToString(n_cols_) + ", " +
            "nnz_:" + utils::NumToString(nnz_) + ", " +
            "row_offsets_:" + row_offsets_.ToString() + ", " +
            "col_indices_:" + col_indices_.ToString() + ", " +
            "values_:" + values_.ToString() +
            " }";
    }

protected:

    CsrMat() = default;

    friend class MatBuilder;

    index_t n_rows_;
    index_t n_cols_;
    offset_t nnz_;

    Buffer<offset_t, arch, index_t> row_offsets_;
    Buffer<index_t, arch, offset_t> col_indices_;
    Buffer<value_t, arch, offset_t> values_;

};


} // namespace graph_genlx