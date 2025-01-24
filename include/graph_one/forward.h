#pragma once

#include <cassert>
#include <type_traits>

#include <torch/torch.h>

#include "graph_one/graph.hpp"
#include "graph_one/log.hpp"
#include "graph_one/blas/spmv/gspmv.h"

namespace graph_one {


namespace {

template <typename T>
using raw_type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

}

struct ForwardOpts {
    bool src_to_dst = true;
};

template <typename functor_t>
torch::Tensor GraphForward(GraphX& g, 
                           torch::Tensor vertex_feat, torch::Tensor edge_feat,
                           const functor_t& functor, const ForwardOpts& opts = {}) {

    assert(vertex_feat.size(0) == g.num_vertices());
    assert(!edge_feat.defined() || edge_feat.size(0) == g.num_edges());

    torch::Tensor spmat;
    if (opts.src_to_dst) {
        spmat = g.adj_t();
    } else {
        spmat = g.adj();
    }

    // replace values of spmat with edge_feat
    if (edge_feat.defined() && edge_feat.dim() == 1) {
        if (spmat.layout() == torch::kSparseCsr) {
            spmat = torch::sparse_csr_tensor(spmat.crow_indices(), spmat.col_indices(), 
                edge_feat, spmat.sizes(), spmat.options());
        } else {
            assert(false && "other spmat formats of spmat are not supported yet");
        }

        // change edge_feat to empty tensor
        edge_feat = torch::Tensor{};
    }
    
    auto& construct_op = functor.construct_op;
    auto& gather_op = functor.gather_op;
    auto& apply_func = functor.apply_func;

    torch::Tensor output;
    if (vertex_feat.layout() == torch::kStrided) {  // dense vertex_feat
        if (!edge_feat.defined()) { // standard SpMV / SpMM
            if (std::is_same_v<raw_type<decltype(construct_op)>, op::Mult>
                && std::is_same_v<raw_type<decltype(gather_op)>, op::Add>) {
                if (vertex_feat.dim() == 1) {
                    LOG_DEBUG("use torch::mv");
                    // output = torch::mv(spmat, vertex_feat);
                    output = blas::GSpMV(spmat, vertex_feat, construct_op, gather_op);
                } else {
                    LOG_DEBUG("use torch::mm");
                    output = torch::mm(spmat, vertex_feat);
                }
            } else {    // generalized SpMV / SpMM
                if (vertex_feat.dim() == 1) {
                    LOG_DEBUG("use GSpMV");
                    output = blas::GSpMV(spmat, vertex_feat, construct_op, gather_op);
                } else {
                    // TODO
                }
            }
        } else {    // has vertex_feat
            // TODO
        }
    } else {    // sparse vertex_feat
        // TODO SpMSpV/SpGEMM

    }

    output = apply_func(output);

    return output;
}
    
} // namespace graph_one
