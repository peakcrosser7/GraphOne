#pragma once

#include <torch/torch.h>

#include "graph_one/allocator.h"
#include "graph_one/arch/cuda/spgemm.cuh"


namespace graph_one::blas {

namespace {

torch::Tensor SpGEMM_CSRxCSR(torch::Tensor spmat_a, torch::Tensor spmat_b) {
    torch::Tensor rowoffsets_out = torch::empty({spmat_a.size(0) + 1}, spmat_a.crow_indices().options());
    torch::Tensor colindices_out;
    torch::Tensor values_out;

    if (spmat_a.device().is_cuda()) {
        TORCH_CHECK(spmat_a.dtype() == spmat_b.dtype(), "SpGEMM only supports same dtype for spmat_a and spmat_b");

        if (spmat_a.dtype() == torch::kFloat32) {
            // cusparse only supports int32_t for index_t/offset_t and float for value_t
            using index_t = int32_t;
            using value_t = float;

            index_t* d_colindices_out;
            value_t* d_values_out;

            torch::ScalarType index_dtype = torch::kInt32;
            torch::Tensor a_rowoffsets = spmat_a.crow_indices().to(index_dtype);
            torch::Tensor a_colindices = spmat_a.col_indices().to(index_dtype);
            torch::Tensor b_rowoffsets = spmat_b.crow_indices().to(index_dtype);
            torch::Tensor b_colindices = spmat_b.col_indices().to(index_dtype);
            torch::Tensor c_rowoffsets = torch::empty_like(rowoffsets_out, index_dtype);

            graph_one::cuda::SpGEMM_CSRxCSR_cusparse(
                spmat_a.size(0), spmat_b.size(1), spmat_a.size(1),
                spmat_a._nnz(), spmat_b._nnz(),
                a_rowoffsets.data_ptr<index_t>(),
                a_colindices.data_ptr<index_t>(),
                spmat_a.values().data_ptr<value_t>(),
                b_rowoffsets.data_ptr<index_t>(),
                b_colindices.data_ptr<index_t>(),
                spmat_b.values().data_ptr<value_t>(),
                c_rowoffsets.data_ptr<index_t>(),
                &d_colindices_out, &d_values_out
            );

            auto& allocator = MemAllocator::Get();
            colindices_out = allocator.PopTensor(d_colindices_out);
            values_out = allocator.PopTensor(d_values_out);

            TORCH_CHECK(colindices_out.device() == spmat_a.device(), "colindices_out device mismatch");
            TORCH_CHECK(values_out.device() == spmat_a.device(), "values_out device mismatch");

            rowoffsets_out = c_rowoffsets.to(spmat_a.crow_indices().options());
            colindices_out = colindices_out.to(spmat_a.col_indices().options());
        } else {
            TORCH_CHECK(false, "SpGEMM only supports float32 for spmat_a and spmat_b");
        }

    } else {
        TORCH_CHECK(false, "SpGEMM_CSRxCSR only supports CUDA device");
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


torch::Tensor SpGEMM(torch::Tensor spmat_a, torch::Tensor spmat_b) {
    TORCH_CHECK(spmat_a.layout() != torch::kStrided, "SpGEMM only supports sparse tensor for spmat_a");
    TORCH_CHECK(spmat_a.dim() == 2, "SpGEMM only supports 2D sparse tensor for spmat_a");
    TORCH_CHECK(spmat_b.layout() != torch::kStrided, "SpGEMM only supports sparse tensor for spmat_b");
    TORCH_CHECK(spmat_b.dim() == 2, "SpGEMM only supports 2D sparse tensor for spmat_b");
    TORCH_CHECK(spmat_a.dtype() == spmat_b.dtype(), "SpGEMM only supports same dtype for spmat_a and spmat_b");
    TORCH_CHECK(spmat_a.device() == spmat_b.device(), "SpGEMM only supports same device for spmat_a and spmat_b");
    TORCH_CHECK(spmat_a.size(1) == spmat_b.size(0), "SpGEMM only supports compatible dimensions for spmat_a and spmat_b");

    if (spmat_a.layout() == torch::kSparseCsr && spmat_b.layout() == torch::kSparseCsr) {
        return SpGEMM_CSRxCSR(spmat_a, spmat_b);
    } else {
        TORCH_CHECK(false, "other sparse formats of spmat_a and spmat_b are not supported yet in SpGEMM");
    }
}


} // namespace graph_one::blas
