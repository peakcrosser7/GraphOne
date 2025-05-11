#pragma once

#include <torch/torch.h>

#include "graph_one/graph.hpp"
#include "graph_one/arch/cuda/reduce.cuh"

namespace graph_one {

namespace {

template <typename reduce_t>
torch::Tensor ReduceCSR(const reduce_t& reduce_op, torch::Tensor spmat, torch::Tensor edge_input) {

    torch::Tensor output = torch::empty({spmat.size(0)}, edge_input.options());
    if (spmat.is_cuda()) {
        AT_DISPATCH_ALL_TYPES(edge_input.scalar_type(), "reduce_csr", [&] {
            using IndexType = int64_t;
            using ValueType = scalar_t;

            LOG_DEBUG("ReduceCSR");
            graph_one::cuda::ReduceCSR(
                spmat.size(0), spmat.size(1), spmat._nnz(),
                spmat.crow_indices().data_ptr<IndexType>(),
                spmat.col_indices().data_ptr<IndexType>(),
                edge_input.data_ptr<ValueType>(),
                output.data_ptr<ValueType>(),
                reduce_op);
        });
    } else {
        TORCH_CHECK(false, "ReduceCSR only supports CUDA device");
    }

    return output;
}

}

struct ReduceOpts {
    bool use_out_edges_ = true;

    ReduceOpts() = default;

    ReduceOpts use_out_edges() {
        use_out_edges_ = true;
        return *this;
    }

    ReduceOpts use_in_edges() {
        use_out_edges_ = false;
        return *this;
    }
};
    
template <typename reduce_t>
torch::Tensor GraphReduce(const reduce_t& reduce_op, GraphX& g, 
                          torch::Tensor edge_input, const ReduceOpts& opts = {}) {
    torch::Tensor spmat;
    if (opts.use_out_edges_) {
        spmat = g.adj();
    } else {
        spmat = g.adj_trans();
    }
    
    TORCH_CHECK(spmat.layout() != torch::kStrided, "spmat must be Sparse tensor");
    TORCH_CHECK(spmat.device() == edge_input.device(), "spmat and edge_input must be the same device");
    TORCH_CHECK(edge_input.dim() == 1, "GSpMV only supports 1D tensor for edge_input");
    TORCH_CHECK(spmat._nnz() == edge_input.size(0), "edge_input must have the same size as the nonzeros in the spmat");


    if (spmat.layout() == torch::kSparseCsr) {
        TORCH_CHECK(spmat.dim() == 2, "spmat must be 2D tensor");
        return ReduceCSR(reduce_op, spmat, edge_input);
    } else {
        TORCH_CHECK(false, "other sparse formats of spmat are not supported yet in Reduce");
    }
    return torch::Tensor{};
}
    
} // namespace graph_one