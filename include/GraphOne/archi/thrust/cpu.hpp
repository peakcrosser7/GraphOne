#pragma once 

#include <thrust/host_vector.h>

#include "GraphOne/archi/thrust/def.hpp"

namespace graph_one::archi {

template<>
struct ThrustVec<arch_t::cpu> {
    template<typename value_t>
    using type = thrust::host_vector<value_t>;
};

template<>
struct ExecPolicy<arch_t::cpu> {
    static constexpr decltype(thrust::host) value{};
};
    
} // namespace graph_one::archi
