#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "GraphGenlX/type.hpp"

namespace graph_genlx {
    
template <typename type_t, arch_t arch = arch_t::cpu>
using vector_t =
    std::conditional_t<arch == arch_t::cpu,  // condition
                       thrust::host_vector<type_t>,    // host_type
                       thrust::device_vector<type_t>   // device_type
                       >;

} // namespace graph_genlx