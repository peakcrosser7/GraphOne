#pragma once

#include <cassert>

#include <torch/torch.h>

#include "graph_one/blas/spmv/gspmv_csr/merge_genl.cuh"

namespace graph_one::blas {

template <typename construct_t, typename gather_t>
torch::Tensor GSpMV(torch::Tensor spmat, torch::Tensor vec, 
                    const construct_t& construct_op, const gather_t& gather_op) {
    assert(spmat.layout() != torch::kStrided);
    assert(spmat.dim() == 2);
    assert(vec.layout() == torch::kStrided);
    assert(vec.dim() == 1);
    assert(spmat.dtype() == vec.dtype());

    torch::Tensor output = torch::empty({spmat.size(0)}, vec.options());
    if (spmat.layout() == torch::kSparseCsr) {
        AT_DISPATCH_ALL_TYPES(spmat.scalar_type(), "gspmv_csr_merge_based", [&] {
            using IndexType = int64_t;
            using ValueType = scalar_t;
            using OutputType = scalar_t;

            blas::GSpMV_CSR_merge_based(
                spmat.size(0), spmat.size(1), spmat._nnz(),
                spmat.crow_indices().data_ptr<IndexType>(), 
                spmat.col_indices().data_ptr<IndexType>(),
                spmat.values().data_ptr<ValueType>(), 
                vec.data_ptr<ValueType>(), 
                output.data_ptr<OutputType>(),
                construct_op, gather_op);
        });
    } else {
        assert(false && "other sparse formats of spmat are not supported yet");
    }
    return output;

}

} // namespace graph_one