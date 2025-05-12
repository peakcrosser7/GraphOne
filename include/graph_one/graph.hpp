#pragma once

#include <cassert>

#include <torch/torch.h>

// #define DEBUG_LOG
#include "graph_one/types.hpp"
#include "graph_one/log.hpp"

namespace graph_one {

class GraphX {
public:

    GraphX(torch::Tensor adj, torch::Tensor adj_trans, torch::Device device) 
        : device_(device) {
        assert(adj.layout() == torch::kSparseCsr);
        assert(!adj_trans.defined() || adj_trans.layout() == torch::kSparseCsr);

        num_vertices_ = adj.size(0);
        num_edges_ = adj._nnz();
        LOG_DEBUG("GraphX: num_vertices_=", num_vertices_, ", num_edges_=", num_edges_, 
            ", device_=", device_);

        adj_ = adj.to(device);
        adj_trans_ = adj_trans.to(device);
    }

    GraphX(torch::Tensor adj, torch::Tensor adj_trans) : GraphX(adj, adj_trans, adj.device()) {}

    vid_t num_vertices() const {
        return num_vertices_;
    }

    eid_t num_edges() const {
        return num_edges_;
    }

    torch::Device device() const {
        return device_;
    }
    
    GraphX to(torch::Device device) {
        return GraphX(adj_, adj_trans_, device);
    }

    torch::Tensor adj() const {
        return adj_;
    }

    torch::Tensor adj_trans() const {
        return adj_trans_;
    }

    torch::Tensor outedge_weights() const {
        return adj_.values();
    }

    torch::Tensor inedge_weights() const {
        return adj_trans_.values();
    }

    void set_outedge_weights(torch::Tensor outedge_weights, bool both = false) {
        TORCH_CHECK(outedge_weights.layout() == torch::kStrided, 
            "edge_weights must be a strided tensor");
        TORCH_CHECK(outedge_weights.size(0) == num_edges_, 
            "edge_weights must have the same size as the number of edges in the graph");
        adj_ = torch::sparse_csr_tensor(
                adj_.crow_indices(),
                adj_.col_indices(),
                outedge_weights,
                adj_.sizes(),
                adj_.options()
            );
        if (both) {
            torch::Tensor csc = adj_.to_sparse_csc();
            adj_trans_ = torch::sparse_csr_tensor(
                csc.ccol_indices(),
                csc.row_indices(),
                csc.values(),
                csc.sizes(),
                csc.options()
            );
        }
    }

    void set_inedge_weights(torch::Tensor inedge_weights, bool both = false) {
        TORCH_CHECK(inedge_weights.layout() == torch::kStrided, 
            "edge_weights must be a strided tensor");
        TORCH_CHECK(inedge_weights.size(0) == num_edges_, 
            "edge_weights must have the same size as the number of edges in the graph");
        adj_trans_ = torch::sparse_csr_tensor(
                adj_trans_.crow_indices(),
                adj_trans_.col_indices(),
                inedge_weights,
                adj_trans_.sizes(),
                adj_trans_.options()
            );
        if (both) {
            torch::Tensor csc = adj_trans_.to_sparse_csc();
            adj_ = torch::sparse_csr_tensor(
                csc.ccol_indices(),
                csc.row_indices(),
                csc.values(),
                csc.sizes(),
                csc.options()
            );
        }
    }

private:
    vid_t num_vertices_;
    eid_t num_edges_;

    torch::Device device_;

    torch::Tensor adj_;
    torch::Tensor adj_trans_;
};


} // namespace graph_one
