#pragma once

#include <memory>

#include "GraphOne/mat/dense.h"

namespace graph_one::gnn {

template <arch_t arch, typename value_t=float>
using tensor_t = DenseMat<arch, value_t>;

template <arch_t arch, typename value_t=float>
using param_t = std::shared_ptr<tensor_t<arch, value_t>>;
    
} // namespace graph_one