#pragma once 

#include <cuda.h>
#include <thrust/device_vector.h>

#include "GraphGenlX/archi/thrust/def.hpp"

namespace graph_genlx::archi {

template<>
struct ThrustVec<arch_t::cuda> {
    template<typename value_t>
    using type = thrust::device_vector<value_t>;
};

template<>
struct ExecPolicy<arch_t::cuda> {
    static constexpr decltype(thrust::device) value{};
};
    
} // namespace graph_genlx::archi
