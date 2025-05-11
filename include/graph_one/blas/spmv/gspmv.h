#pragma once

#include <torch/torch.h>

#include "graph_one/arch/cuda/gspmv/gspmv.cuh"

namespace graph_one::blas {

namespace {

template <typename construct_t, typename gather_t>
torch::Tensor GSpMV_CSR(torch::Tensor spmat, torch::Tensor vec, 
                        const construct_t& construct_op, const gather_t& gather_op) {

    torch::Tensor output = torch::empty({spmat.size(0)}, vec.options());

    if (spmat.device().is_cuda()) {
        AT_DISPATCH_ALL_TYPES(spmat.scalar_type(), "gspmv_csr_merge_based", [&] {
            using IndexType = int64_t;
            using ValueType = scalar_t;
            using OutputType = scalar_t;

            cuda::GSpMV_CSR_merge_based(
                spmat.size(0), spmat.size(1), spmat._nnz(),
                spmat.crow_indices().data_ptr<IndexType>(), 
                spmat.col_indices().data_ptr<IndexType>(),
                spmat.values().data_ptr<ValueType>(), 
                vec.data_ptr<ValueType>(), 
                output.data_ptr<OutputType>(),
                construct_op, gather_op);
        });
    } else {
        TORCH_CHECK(false, "GSpMV_CSR only supports CUDA device");
    }
    
    return output;
}

}

template <typename construct_t, typename gather_t>
torch::Tensor GSpMV(torch::Tensor spmat, torch::Tensor vec, 
                    const construct_t& construct_op, const gather_t& gather_op) {
    TORCH_CHECK(spmat.layout() != torch::kStrided, "GSpMV only supports sparse tensor for spmat");
    TORCH_CHECK(spmat.dim() == 2, "GSpMV only supports 2D sparse tensor for spmat");
    TORCH_CHECK(vec.layout() == torch::kStrided, "GSpMV only supports strided tensor for vec");
    TORCH_CHECK(vec.dim() == 1, "GSpMV only supports 1D strided tensor for vec");
    TORCH_CHECK(spmat.dtype() == vec.dtype(), "GSpMV only supports same dtype for spmat and vec");
    TORCH_CHECK(spmat.device() == vec.device(), "GSpMV only supports same device for spmat and vec");

    if (spmat.layout() == torch::kSparseCsr) {
        return GSpMV_CSR(spmat, vec, construct_op, gather_op);
    } else {
        TORCH_CHECK(false, "other sparse formats of spmat are not supported yet in GSpMV");
    }
}

} // namespace graph_one