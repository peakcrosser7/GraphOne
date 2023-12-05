#pragma once

#include <vector>

#include "GraphGenlX/type.hpp"

namespace graph_genlx::archi {
    
template <arch_t arch>
struct Vector_t {
    template<typename value_t>
    using type = std::vector<value_t>;    
};

template <arch_t arch>
struct ExecPolicy {};
template <arch_t arch>
constexpr auto exec_policy = archi::ExecPolicy<arch>::value;

template <arch_t arch>
struct memalloc_t {
    template <typename T>
    static T* call(size_t size) { return nullptr; }
};

template <arch_t arch>
struct memfree_t {
    template <typename T>
    static void call(T* ptr) {}
};

template<arch_t arch>
struct memset_t {
    template <typename T>
    static void call(T* ptr, int value, size_t size) {}
};

template <arch_t dst_arch, arch_t src_arch>
struct memcpy_t {
    template <typename T>
    static void call(T* dst, const T* src, size_t size) {}
};

template <arch_t arch>
struct memfill_t {
    template <typename T>
    static void call(T* ptr, size_t size, const T& value) {}
};

} // namespace graph_genlx::archi