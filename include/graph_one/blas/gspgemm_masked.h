#pragma once

#include <torch/torch.h>

#include "graph_one/torch_utils.hpp"
#include "graph_one/arch/cuda/gspgemm_masked.cuh"

namespace graph_one::blas {

namespace {

template <typename construct_t, typename gather_t>
torch::Tensor GSpGEMM_Masked_CSRxCSRmCSR(
    torch::Tensor spmat_a, torch::Tensor spmat_b, torch::Tensor mask,
    const construct_t& construct_op, const gather_t& gather_op) {
    
    torch::Tensor rowoffsets_out = mask.crow_indices().clone();
    torch::Tensor colindices_out;
    torch::Tensor values_out;

    if (spmat_a.is_cuda()) {
        TORCH_CHECK(spmat_a.dtype() == spmat_b.dtype(), "SpGEMM only supports same dtype for spmat_a and spmat_b");

        colindices_out = mask.col_indices().clone();
        values_out = torch::empty_like(mask.values(), spmat_a.dtype());

        GRAPH_ONE_DISPATCH(spmat_a.scalar_type(), "spgemm_masked_csrxcsrwcsr", [&] {
            using index_t = int64_t;
            using value_t = scalar_t;

            graph_one::cuda::GSpGEMM_Masked_CSRxCSRwCSR_graphblast(
                spmat_a.size(0), spmat_b.size(1), spmat_a.size(1),
                spmat_a._nnz(), spmat_b._nnz(), mask._nnz(),
                spmat_a.crow_indices().data_ptr<index_t>(),
                spmat_a.col_indices().data_ptr<index_t>(),
                spmat_a.values().data_ptr<value_t>(),
                spmat_b.crow_indices().data_ptr<index_t>(),
                spmat_b.col_indices().data_ptr<index_t>(),
                spmat_b.values().data_ptr<value_t>(),
                mask.crow_indices().data_ptr<index_t>(),
                mask.col_indices().data_ptr<index_t>(),
                mask.values().data_ptr<value_t>(),
                rowoffsets_out.data_ptr<index_t>(),
                colindices_out.data_ptr<index_t>(),
                values_out.data_ptr<value_t>(),
                construct_op, gather_op
            );
        });
    } else {
        TORCH_CHECK(false, "SpGEMM_Masked_CSRxCSRmCSR only supports CUDA device");
    }

    return torch::sparse_csr_tensor(
        rowoffsets_out,
        colindices_out,
        values_out,
        {spmat_a.size(0), spmat_b.size(1)},
        spmat_a.options()
    );
}


} // namespace 

template <typename construct_t, typename gather_t>
torch::Tensor GSpGEMM_Masked(torch::Tensor spmat_a, torch::Tensor spmat_b, torch::Tensor mask,
                             const construct_t& construct_op, const gather_t& gather_op) {
    TORCH_CHECK(spmat_a.layout() != torch::kStrided, "SpGEMM only supports sparse tensor for spmat_a");
    TORCH_CHECK(spmat_a.dim() == 2, "SpGEMM only supports 2D sparse tensor for spmat_a");
    TORCH_CHECK(spmat_b.layout() != torch::kStrided, "SpGEMM only supports sparse tensor for spmat_b");
    TORCH_CHECK(spmat_b.dim() == 2, "SpGEMM only supports 2D sparse tensor for spmat_b");
    TORCH_CHECK(spmat_a.dtype() == spmat_b.dtype(), "SpGEMM only supports same dtype for spmat_a and spmat_b");
    TORCH_CHECK(spmat_a.device() == spmat_b.device(), "SpGEMM only supports same device for spmat_a and spmat_b");
    TORCH_CHECK(spmat_a.size(1) == spmat_b.size(0), "SpGEMM only supports compatible dimensions for spmat_a and spmat_b");
    TORCH_CHECK(mask.dim() == 2, "SpGEMM only supports 2D mask tensor for mask");
    TORCH_CHECK(mask.device() == spmat_a.device(), "SpGEMM only supports same device for mask and spmat_a");
    TORCH_CHECK(mask.size(0) == spmat_a.size(0), "SpGEMM only supports same number of rows for mask and spmat_a");
    TORCH_CHECK(mask.size(1) == spmat_b.size(1), "SpGEMM only supports same number of columns for mask and spmat_b");

    if (spmat_a.layout() == torch::kSparseCsr 
        && spmat_b.layout() == torch::kSparseCsr
        && mask.layout() == torch::kSparseCsr) {
        return GSpGEMM_Masked_CSRxCSRmCSR(spmat_a, spmat_b, mask, construct_op, gather_op);
    } else {
        TORCH_CHECK(false, "other sparse formats of spmat_a, spmat_b and mask are not supported yet in SpGEMM");
    }
}


} // namespace graph_one::blas
