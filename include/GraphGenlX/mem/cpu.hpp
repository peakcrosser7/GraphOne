#pragma once

#include <cstring>

#include "GraphGenlX/mem/mem.hpp"

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
struct memset<arch_t::cpu> {
    template <typename T>
    static void call(T* ptr, int value, size_t size) {
        std::memset(ptr, value, sizeof(T) * size);
    }
};

template<>
struct memcpy<arch_t::cpu, arch_t::cpu> {
    template <typename T>
    static void call(T* dst, const T* src, size_t size) {
        std::memcpy(dst, src, sizeof(T) * size);
    }
};

} // namespace graph_genlx::archi