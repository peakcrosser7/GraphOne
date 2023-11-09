#pragma once

#include "GraphGenlX/type.hpp"

namespace graph_genlx::archi {

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

template <arch_t dst_arch, arch_t src_arch>
struct memcpy {
    template <typename T>
    static void call(T* dst, const T* src, size_t size) {}
};

} // namespace graph_genlx::archi