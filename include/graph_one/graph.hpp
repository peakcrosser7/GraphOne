#pragma once

#include <cassert>

#include <torch/torch.h>

// #define DEBUG_LOG
#include "graph_one/types.hpp"
#include "graph_one/log.hpp"

namespace graph_one {

class GraphX {
public:

    GraphX(torch::Tensor adj, torch::Tensor adj_t, torch::Device device) 
        : device_(device) {
        assert(adj.layout() == torch::kSparseCsr);
        assert(!adj_t.defined() || adj_t.layout() == torch::kSparseCsr);

        num_vertices_ = adj.size(0);
        num_edges_ = adj._nnz();
        LOG_DEBUG("GraphX: num_vertices_=", num_vertices_, ", num_edges_=", num_edges_, 
            ", device_=", device_);

        adj_ = adj.to(device);
        adj_t_ = adj_t.to(device);
    }

    GraphX(torch::Tensor adj, torch::Tensor adj_t) : GraphX(adj, adj_t, adj.device()) {}

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
        return GraphX(adj_, adj_t_, device);
    }

    torch::Tensor adj() const {
        return adj_;
    }

    torch::Tensor adj_t() const {
        return adj_t_;
    }

    torch::Tensor edge_weights() const {
        return adj_.values();
    }

    torch::Tensor edge_weights_t() const {
        return adj_t_.values();
    }

private:
    vid_t num_vertices_;
    eid_t num_edges_;

    torch::Device device_;

    torch::Tensor adj_;
    torch::Tensor adj_t_;
};


} // namespace graph_one
