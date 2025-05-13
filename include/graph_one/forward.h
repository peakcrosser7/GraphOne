#pragma once

#include <cassert>
#include <type_traits>

#include <torch/torch.h>

#include "graph_one/graph.hpp"
#include "graph_one/log.hpp"
#include "graph_one/blas/gspmv.h"
#include "graph_one/blas/spgemm.h"
#include "graph_one/blas/gspgemm_masked.h"

namespace graph_one {


namespace {

template <typename T>
using raw_type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

}

struct ForwardOpts {
    bool use_inedge_weights_ = true;

    ForwardOpts() = default;

    ForwardOpts use_inedge_weights() {
        use_inedge_weights_ = true;
        return *this;
    }

    ForwardOpts use_outedge_weights() {
        use_inedge_weights_ = false;
        return *this;
    }
};

template <typename functor_t>
torch::Tensor GraphForward(const functor_t& functor, GraphX& g, 
                           torch::Tensor vertex_feat, torch::Tensor edge_feat = {},
                           const ForwardOpts& opts = {}) {

    TORCH_CHECK(vertex_feat.size(0) == g.num_vertices(), "vertex_feat must have the same size as the number of vertices in the graph");
    TORCH_CHECK(!edge_feat.defined() || edge_feat.size(0) == g.num_edges(), "edge_feat must have the same size as the number of edges in the graph");

    torch::Tensor spmat;
    if (opts.use_inedge_weights_) {
        spmat = g.adj_trans();
    } else {
        spmat = g.adj();
    }

    // replace values of spmat with edge_feat
    if (edge_feat.defined() && edge_feat.dim() == 1) {
        if (spmat.layout() == torch::kSparseCsr) {
            spmat = torch::sparse_csr_tensor(spmat.crow_indices(), spmat.col_indices(), 
                edge_feat, spmat.sizes(), spmat.options());
        } else {
            TORCH_CHECK(false, "other spmat formats of spmat are not supported yet");
        }

        // change edge_feat to empty tensor
        edge_feat = torch::Tensor{};
    }
    
    auto& construct_op = functor.construct_op;
    auto& gather_op = functor.gather_op;
    auto& apply_func = functor.apply_func;

    torch::Tensor output;
    bool do_apply_func = true;
    if (vertex_feat.layout() == torch::kStrided) {  // dense vertex_feat
        if (!edge_feat.defined()) {
            if (vertex_feat.dim() == 1) {   // (G)SpMV
                if constexpr (std::is_same_v<raw_type<decltype(construct_op)>, op::Mult>
                              && std::is_same_v<raw_type<decltype(gather_op)>, op::Add>) {    // standard SpMV
                    LOG_DEBUG("use torch::mv");
                    output = torch::mv(spmat, vertex_feat);
                } else {    // generalized SpMV
                    LOG_DEBUG("use blas::GSpMV");
                    output = blas::GSpMV(spmat, vertex_feat, construct_op, gather_op);
                }
            } else {    // (G)SpMM
                if constexpr (std::is_same_v<raw_type<decltype(construct_op)>, op::Mult>
                    && std::is_same_v<raw_type<decltype(gather_op)>, op::Add>) {    // standard SpMM
                    LOG_DEBUG("use torch::mm");
                    output = torch::mm(spmat, vertex_feat);
                } else {    // generalized SpMM
                    // TODO
                    TORCH_CHECK(false, "generalized SpMM is not supported yet");
                }
            }
        } else {    // has edge_feat (dim >= 2)
            // TODO
            TORCH_CHECK(false, "edge_feat is not supported yet");
        }
    } else {    // sparse vertex_feat
        // TODO SpMSpV/SpGEMM
        if (!edge_feat.defined()) { 
            if (vertex_feat.dim() == 1) {
                // SpMSpV
                TORCH_CHECK(false, "SpMSpV is not supported yet");
            } else if (vertex_feat.dim() == 2) {
                if constexpr (std::is_same_v<raw_type<decltype(construct_op)>, op::Mult>
                              && std::is_same_v<raw_type<decltype(gather_op)>, op::Add>
                              && std::is_same_v<raw_type<decltype(apply_func)>, DummyApplier>) {    // standard SpGEMM
                    LOG_DEBUG("use SpGEMM");
                    output = blas::SpGEMM(spmat, vertex_feat);
                } else if constexpr (std::is_same_v<raw_type<decltype(apply_func)>, MaskApplier>) {
                    LOG_DEBUG("use Masked-GSpGEMM");
                    torch::Tensor mask = apply_func.mask();
                    output = blas::GSpGEMM_Masked(spmat, vertex_feat, mask, 
                        construct_op, gather_op);
                    do_apply_func = false;
                } else {
                    TORCH_CHECK(false, "generalized SpGEMM is not supported yet");
                }
            } else {
                TORCH_CHECK(false, "vertex_feat must be 1D or 2D sparse-tensor");
            }
        } else {    // has edge_feat (dim >= 2)
            // TODO
            TORCH_CHECK(false, "edge_feat is not supported yet");
        }
    }

    if (do_apply_func) {
        output = apply_func(output);
    }

    return output;
}
    
} // namespace graph_one
