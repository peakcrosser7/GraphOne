#pragma once

#include <string>
#include <vector>

#include "GraphOne/utils/log.hpp"
#include "GraphOne/mat/dense.h"
#include "GraphOne/vec/dense.h"

namespace graph_one::gnn {

template <arch_t arch,
          typename feat_t = float, 
          typename label_t = int32_t,
          typename index_t = uint32_t>
struct vprop_t {
    constexpr static arch_t arch_value = arch;
    
    using dim_type = index_t;

    index_t num_vertices;
    index_t feat_dim;
    DenseMat<arch, feat_t, index_t> features;
    DenseVec<arch, label_t, index_t> labels;


    vprop_t() = default;
    vprop_t(index_t num_v, index_t feat_d) 
        : num_vertices(num_v), feat_dim(feat_d), features(num_v, feat_d), labels(num_v) {}

    vprop_t(const std::vector<std::vector<feat_t>>& feats, const std::vector<label_t>& labs) 
        : num_vertices(feats.size()), feat_dim(feats[0].size()), features(feats), labels(labs) {
        if (features.n_rows != labels.size()) {
            LOG_ERROR("features and labels should have the same vertices, but features.n_rows=", 
                      features.n_rows, " labels.size()=", labels.size());
        }
    }

    vprop_t(const DenseMat<arch, feat_t, index_t>& feats, const DenseVec<arch, label_t, index_t>& labs)
        : num_vertices(feats.n_rows), feat_dim(feats.n_cols), features(feats), labels(labs) {
        if (features.n_rows != labels.size()) {
            LOG_ERROR("features and labels should have the same vertices, but features.n_rows=", 
                      features.n_rows, " labels.size()=", labels.size());
        }
    }

    std::string ToString() const {
        return "gnn::vprop_t{ "
            "num_vertices:" + utils::NumToString(num_vertices) + ", " +
            "feat_dim:" + utils::NumToString(feat_dim) + ", " +
            "features:" + features.ToString() + ", " +
            "labels:" + labels.ToString() + 
            " }";
    }
};

    
} // namespace graph_one::gnn