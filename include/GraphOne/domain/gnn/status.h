#pragma once

#include "GraphOne/type.hpp"
#include "GraphOne/mat/dense.h"

namespace graph_one::gnn {

template <arch_t arch, typename feat_t, typename index_t = uint32_t>
struct status_t {
    using dim_type = index_t;

    dim_type feat_dim() const {
        return ipt_vertex_emb.n_cols;
    }


    gnn::tensor_t<arch, feat_t, index_t> ipt_vertex_emb;
    gnn::tensor_t<arch, feat_t, index_t> opt_vertex_emb;
};
    
} // namespace graph_one::gnn