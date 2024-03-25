#pragma once

#include "GraphOne/type.hpp"

namespace graph_one {

template <arch_t arch,
          typename vertex_t,
          typename index_t = vertex_t>
class DenseFrontier : public BaseFrontier<DENSE_BASED> {
public:
    using vertex_type = vertex_t;
    using index_type = index_t;

    DenseFrontier() = default;

    
}


} // namespace graph_one
