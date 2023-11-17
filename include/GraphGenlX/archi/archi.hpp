#pragma once

#include <vector>

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
inline constexpr auto exec_policy = archi::ExecPolicy<arch>::value;

template <arch_t arch>
struct memalloc {
    template <typename T>
    static T* call(size_t size) { return nullptr; }
};

template <arch_t arch>
struct memfree {
    template <typename T>
    static void call(T* ptr) {}
};

template<arch_t arch>
struct memset {
    template <typename T>
    static void call(T* ptr, int value, size_t size) {}
};

template <arch_t dst_arch, arch_t src_arch>
struct memcpy {
    template <typename T>
    static void call(T* dst, const T* src, size_t size) {}
};

}   // namespace archi

template <arch_t arch, typename value_t>
using vector_t = archi::vector_t<arch, value_t>;

} // namespace graph_genlx