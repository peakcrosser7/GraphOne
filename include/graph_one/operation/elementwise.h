#pragma once

#include <torch/torch.h>

#include "graph_one/graph.hpp"

#include "graph_one/arch/cuda/elementwise.cuh"

namespace graph_one {

namespace {

template <typename binary_t>
torch::Tensor ElementWiseCSR(const binary_t& binary_op, torch::Tensor spmat,
                             torch::Tensor edge_input, torch::Tensor vertex_input,
                             bool use_rows) {
    
    torch::Tensor output = torch::empty({edge_input.size(0)}, edge_input.options());
    if (spmat.is_cuda()) {
        AT_DISPATCH_ALL_TYPES(edge_input.scalar_type(), "elementwise_csr", [&] {
            using IndexType = int64_t;
            using ValueType = scalar_t;

            graph_one::cuda::ElementWiseCSR(
                spmat.size(0), spmat.size(1), spmat._nnz(),
                spmat.crow_indices().data_ptr<IndexType>(),
                spmat.col_indices().data_ptr<IndexType>(),
                edge_input.data_ptr<ValueType>(),
                vertex_input.data_ptr<ValueType>(),
                output.data_ptr<ValueType>(),
                use_rows,
                binary_op);
        });
    } else {
        TORCH_CHECK(false, "ElementWiseCSR only supports CUDA device");
    }

    return output;

}

} // namespace 



struct ElementWiseOpts {
    bool use_out_edges_ = true;
    bool use_src_vertex_ = true;

    ElementWiseOpts() = default;
    
    ElementWiseOpts use_out_edges() {
        use_out_edges_ = true;
        return *this;
    }

    ElementWiseOpts use_in_edges() {
        use_out_edges_ = false;
        return *this;
    }

    ElementWiseOpts use_src_vertex() {
        use_src_vertex_ = true;
        return *this;
    }

    ElementWiseOpts use_dst_vertex() {
        use_src_vertex_ = false;
        return *this;
    }

};

template <typename binary_t>
torch::Tensor GraphWise(const binary_t& binary_op, GraphX& g,
                        torch::Tensor edge_input, torch::Tensor vertex_input, 
                        const ElementWiseOpts& opts = {}) {
    torch::Tensor spmat;
    if (opts.use_out_edges_) {
        spmat = g.adj();
    } else {
        spmat = g.adj_t();
    }

    TORCH_CHECK(spmat.layout() != torch::kStrided, "spmat must be Sparse tensor");
    TORCH_CHECK(spmat.device() == edge_input.device(), "spmat and edge_input must be the same device");
    TORCH_CHECK(spmat.device() == vertex_input.device(), "spmat and vertex_input must be the same device");
    TORCH_CHECK(edge_input.dim() == 1, "GSpMV only supports 1D tensor for edge_input");
    TORCH_CHECK(vertex_input.dim() == 1, "GSpMV only supports 1D tensor for vertex_input");
    TORCH_CHECK(spmat._nnz() == edge_input.size(0), "edge_input must have the same size as the nonzeros in the spmat");
    TORCH_CHECK(spmat.size(0) == vertex_input.size(0), "vertex_input must have the same size as the spmat");

    bool use_rows = opts.use_out_edges_ == opts.use_src_vertex_;

    if (spmat.layout() == torch::kSparseCsr) {
        TORCH_CHECK(spmat.dim() == 2, "spmat must be 2D tensor");
        return ElementWiseCSR(binary_op, spmat, edge_input, vertex_input, use_rows);
    } else {
        TORCH_CHECK(false, "other sparse formats of spmat are not supported yet in Reduce");
    }
    return torch::Tensor{};  
}

} // namespace graph_one
