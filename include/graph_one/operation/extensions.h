#pragma once

#include <torch/torch.h>

#include "graph_one/graph.hpp"
#include "graph_one/torch_utils.hpp"
#include "graph_one/arch/cuda/tril.cuh"


namespace graph_one {

torch::Tensor TrilCSR(torch::Tensor spmat) {
    TORCH_CHECK(spmat.layout() == torch::kSparseCsr, "spmat must be in sparse_csr format");

    torch::Tensor rowoffsets_out = torch::empty_like(spmat.crow_indices());
    torch::Tensor colindices_out;
    torch::Tensor values_out;

    if (spmat.is_cuda()) {
        GRAPH_ONE_DISPATCH(spmat.scalar_type(), "tril_csr", [&] {
            using index_t = int64_t;
            using value_t = scalar_t;

            index_t* d_colindices_out;
            value_t* d_values_out;

            graph_one::cuda::TrilCSR(
                spmat.size(0), spmat.size(1), spmat._nnz(),
                spmat.crow_indices().data_ptr<index_t>(),
                spmat.col_indices().data_ptr<index_t>(),
                spmat.values().data_ptr<value_t>(),
                rowoffsets_out.data_ptr<index_t>(),
                &d_colindices_out, &d_values_out
            );

            auto& allocator = MemAllocator::Get();
            colindices_out = allocator.PopTensor(d_colindices_out);
            values_out = allocator.PopTensor(d_values_out);

            TORCH_CHECK(colindices_out.device() == spmat.device(), "colindices_out device mismatch");
            TORCH_CHECK(values_out.device() == spmat.device(), "values_out device mismatch");
        });
    } else {
        TORCH_CHECK(false, "TrilCSR only supports CUDA device now");
    }

    return torch::sparse_csr_tensor(
        rowoffsets_out, colindices_out, values_out,
        spmat.sizes(), spmat.options()
    );
}

GraphX Tril(GraphX& g) {
    torch::Tensor adj = g.adj();
    torch::Tensor adj_trans = g.adj_trans();

    torch::Tensor adj_tril = TrilCSR(adj);
    torch::Tensor csc_tril = adj_tril.to_sparse_csc();
    torch::Tensor adj_trans_tril = torch::sparse_csr_tensor(
        csc_tril.ccol_indices(),
        csc_tril.row_indices(),
        csc_tril.values(),
        csc_tril.sizes(),
        adj_tril.options()
    );

    return GraphX(adj_tril, adj_trans_tril);
}

} // namespace graph_one
