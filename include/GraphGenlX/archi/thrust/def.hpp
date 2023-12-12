#pragma once 

#include <thrust/execution_policy.h>

#include "GraphGenlX/type.hpp"

namespace graph_genlx {

namespace archi {

template <arch_t arch>
struct Vector_t {
    template<typename value_t>
    using type = std::vector<value_t>;    
};
template <arch_t arch, typename value_t>
using vector_t = typename archi::Vector_t<arch>::template type<value_t>;

template <arch_t arch>
struct ExecPolicy {};
template <arch_t arch>
static const __GENLX_DEV__ auto exec_policy = archi::ExecPolicy<arch>::value;

} // namespace archi

template <arch_t arch, typename value_t>
using vector_t = archi::vector_t<arch, value_t>;
    
} // namespace graph_genlx
