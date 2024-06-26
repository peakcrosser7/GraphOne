#pragma once

#include <vector>

#include "GraphOne/type.hpp"

namespace graph_one::archi {

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

template<arch_t arch, typename value_t>
constexpr auto memalloc = archi::memalloc_t<arch>::template call<value_t>;

template<arch_t arch, typename value_t>
constexpr auto memfree = archi::memfree_t<arch>::template call<value_t>;

template<arch_t arch, typename value_t>
constexpr auto memset = archi::memset_t<arch>::template call<value_t>;

template<arch_t to_arch, arch_t from_arch, typename value_t>
constexpr auto memcpy = archi::memcpy_t<to_arch, from_arch>::template call<value_t>;

} // namespace graph_one::archi