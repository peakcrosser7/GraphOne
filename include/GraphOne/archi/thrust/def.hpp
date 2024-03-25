#pragma once 

#include <thrust/execution_policy.h>

#include "GraphOne/type.hpp"

namespace graph_one {

namespace archi {

template <arch_t arch>
struct ThrustVec {
    template<typename value_t>
    using type = std::vector<value_t>;    
};
template <arch_t arch, typename value_t>
using thrust_vec = typename archi::ThrustVec<arch>::template type<value_t>;

template <arch_t arch>
struct ExecPolicy {};
template <arch_t arch>
static const auto exec_policy = archi::ExecPolicy<arch>::value;

} // namespace archi

template <arch_t arch, typename value_t>
using thrust_vec = archi::thrust_vec<arch, value_t>;
    
} // namespace graph_one
