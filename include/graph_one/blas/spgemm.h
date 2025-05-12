#pragma once

#include <torch/torch.h>


namespace graph_one::blas {

namespace {

torch::Tensor SpGEMM_CSRxCSR(torch::Tensor spmat_a, torch::Tensor spmat_b) {
    
}


} // namespace 


torch::Tensor SpGEMM(torch::Tensor spmat_a, torch::Tensor spmat_b) {
    TORCH_CHECK(spmat_a.layout() != torch::kStrided, "SpGEMM only supports sparse tensor for spmat_a");
    TORCH_CHECK(spmat_a.dim() == 2, "SpGEMM only supports 2D sparse tensor for spmat_a");
    TORCH_CHECK(spmat_b.layout() != torch::kStrided, "SpGEMM only supports sparse tensor for spmat_b");
    TORCH_CHECK(spmat_b.dim() == 2, "SpGEMM only supports 2D sparse tensor for spmat_b");
    TORCH_CHECK(spmat_a.dtype() == spmat_b.dtype(), "SpGEMM only supports same dtype for spmat_a and spmat_b");
    TORCH_CHECK(spmat_a.device() == spmat_b.device(), "SpGEMM only supports same device for spmat_a and spmat_b");
    TORCH_CHECK(spmat_a.size(1) == spmat_b.size(0), "SpGEMM only supports compatible dimensions for spmat_a and spmat_b");

    if (spmat_a.layout() == torch::kSparseCsr && spmat_b.layout() == torch::kSparseCsr) {

    } else {
        TORCH_CHECK(false, "other sparse formats of spmat_a and spmat_b are not supported yet in SpGEMM");
    }
}


} // namespace graph_one::blas
