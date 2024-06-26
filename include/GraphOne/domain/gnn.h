#pragma once

#include <string>

#include "GraphOne/mat/dense.h"
#include "GraphOne/vec/dense.h"

namespace graph_one::gnn {

template <arch_t arch,
          typename feat_t = double, 
          typename label_t = int32_t,
          typename index_t = uint32_t>
struct vprop_t {
    constexpr static arch_t arch_value = arch;

    vprop_t() = default;
    vprop_t(index_t num_vertices, index_t feat_dim) 
        : features(num_vertices, feat_dim), labels(num_vertices) {}

    std::string ToString() const {
        return "gnn::vprop_t{ "
            "features:" + features.ToString() + ", " +
            "labels:" + labels.ToString() + 
            " }";
    }

    DenseMat<arch, feat_t, index_t> features;
    DenseVec<arch, label_t, index_t> labels;
};
    
} // namespace graph_one::gnn