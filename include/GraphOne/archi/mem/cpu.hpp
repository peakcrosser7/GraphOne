#pragma once

#include <cstring>

#include "GraphOne/archi/mem/def.hpp"

namespace graph_one::archi {

template<>
struct memalloc_t<arch_t::cpu> {
    template <typename T>
    static T* call(size_t size) {
        return new T[size];
    }
};

template<>
struct memfree_t<arch_t::cpu> {
    template <typename T>
    static void call(T* ptr) {
        delete[] ptr;
    }
};

template<>
struct memset_t<arch_t::cpu> {
    template <typename T>
    static void call(T* ptr, int value, size_t size) {
        std::memset(ptr, value, sizeof(T) * size);
    }
};

template<>
struct memcpy_t<arch_t::cpu, arch_t::cpu> {
    template <typename T>
    static void call(T* dst, const T* src, size_t size) {
        std::memcpy(dst, src, sizeof(T) * size);
    }
};

} // namespace graph_one::archi