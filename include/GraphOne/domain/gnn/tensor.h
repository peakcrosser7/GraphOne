#pragma once

#include <memory>

#include "GraphOne/mat/dense.h"

namespace graph_one::gnn {

template <arch_t arch, typename value_t = float, typename index_t = uint32_t>
using TensorBase = DenseMat<arch, value_t, index_t>;

template <arch_t arch, typename value_t = float, typename index_t = uint32_t>
using tensor_t = std::shared_ptr<TensorBase<arch, value_t>>;

template <arch_t arch, typename value_t = float, typename index_t = uint32_t>
using param_t = std::shared_ptr<TensorBase<arch, value_t>>;


template <arch_t arch, typename value_t = float, typename index_t = uint32_t, typename... args_t>
tensor_t<arch, value_t, index_t> tensor(args_t&&... args) {
    return std::make_shared<TensorBase<arch, value_t, index_t>>(std::forward<args_t>(args)...);
}

    
} // namespace graph_one