#pragma once

#include <vector>
#include <string>

#include <torch/torch.h>

#include "graph_loader/loader.hpp"

#include "graph_one/types.hpp"
#include "graph_one/graph.hpp"
#include "graph_one/log.hpp"

namespace graph_one {

using LoaderOpts = graph_loader::LoaderOpts;

template <typename value_t>
GraphX LoadGraphFromTxt(const std::string& filepath, LoaderOpts& opts, torch::Device device = torch::kCPU) {

    int64_t num_vertices;
    std::vector<int64_t> row_indices_vec;
    std::vector<int64_t> col_indices_vec;
    std::vector<value_t> values_vec;

    auto pre_load_func = [&](int64_t num_v, eid_t num_e) {
        num_vertices = num_v;
        row_indices_vec.reserve(num_e);
        col_indices_vec.reserve(num_e);
        values_vec.reserve(num_e);
    };

    auto edge_load_func = [&](eid_t eidx, int64_t& src, int64_t& dst, value_t& val) -> bool {
        row_indices_vec.push_back(src);
        col_indices_vec.push_back(dst);
        values_vec.push_back(val);

        if (opts.undirected() && src != dst) {
            row_indices_vec.push_back(dst);
            col_indices_vec.push_back(src);
            values_vec.push_back(val);
        }
        return true;
    };

    graph_loader::CoreLoader<int64_t, eid_t, value_t>::Load(filepath, opts, edge_load_func, pre_load_func);

    torch::Tensor row_indices = torch::from_blob(row_indices_vec.data(), 
        {static_cast<int64_t>(row_indices_vec.size())}, torch::kInt64);
    torch::Tensor col_indices = torch::from_blob(col_indices_vec.data(), 
        {static_cast<int64_t>(col_indices_vec.size())}, torch::kInt64);
    torch::Tensor coo_indices = torch::stack({row_indices, col_indices});
    torch::Tensor coo_values = torch::from_blob(values_vec.data(), 
        {static_cast<int64_t>(values_vec.size())}, 
        torch::CppTypeToScalarType<value_t>::value);
    torch::Tensor coo = torch::sparse_coo_tensor(coo_indices, coo_values, 
        {num_vertices, num_vertices});

    torch::Tensor adj = coo.to_sparse_csr();
    torch::Tensor csc = coo.to_sparse_csc();

    torch::Tensor adj_t = torch::sparse_csr_tensor(csc.ccol_indices(), csc.row_indices(), csc.values(), csc.sizes(),
        csc.dtype());

    return GraphX(adj, adj_t, device);
}

template <typename value_t = float>
GraphX load_graph(const std::string& filepath, torch::Device device = torch::kCPU) {
    LoaderOpts opts;
    if (graph_loader::utils::StrEndWith(filepath, ".mtx")) {
        opts = graph_loader::OptsFactory::MatrixMarket();
    }
    return LoadGraphFromTxt<value_t>(filepath, opts, device);
}

} // namespace graph_one
