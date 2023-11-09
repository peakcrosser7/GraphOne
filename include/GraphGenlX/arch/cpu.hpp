#pragma once

#include <algorithm>

#include "GraphGenlX/arch/arch.hpp"

namespace graph_genlx::archi {

template<>
struct memalloc<arch_t::cpu> {
    template <typename T>
    static T* call(size_t size) {
        return new T[size];
    }
};

template<>
struct memfree<arch_t::cpu> {
    template <typename T>
    static void call(T* ptr) {
        delete[] ptr;
    }
};

template<>
struct memcpy<arch_t::cpu, arch_t::cpu> {
    template <typename T>
    static void call(T* dst, const T* src, size_t size) {
        std::copy(src, src + size, dst);
    }
};

} // namespace graph_genlx::archi