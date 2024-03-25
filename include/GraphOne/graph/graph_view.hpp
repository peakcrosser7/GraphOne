#pragma once

#include <cstdint>

#include "GraphOne/type.hpp"
#include "GraphOne/mat/csr.h"

namespace graph_one {

enum class graph_view_t : uint8_t {
    normal    = 1 << 0,
    transpose = 1 << 1,
    coo       = 1 << 2,
    csr       = 1 << 3
};

constexpr inline graph_view_t operator|(graph_view_t lhs, graph_view_t rhs) {
    return static_cast<graph_view_t>(static_cast<uint8_t>(lhs) |
                                static_cast<uint8_t>(rhs));
}

constexpr inline graph_view_t set_view(graph_view_t lhs, graph_view_t rhs) {
    return static_cast<graph_view_t>(static_cast<uint8_t>(lhs) |
                                static_cast<uint8_t>(rhs));
}

constexpr inline graph_view_t unset_view(graph_view_t lhs, graph_view_t rhs) {
    return static_cast<graph_view_t>(static_cast<uint8_t>(lhs) &
                                ~static_cast<uint8_t>(rhs));
}

constexpr inline bool has_view(graph_view_t lhs, graph_view_t rhs) {
    return (static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs)) ==
            static_cast<uint8_t>(rhs);
}

template <typename... args_t>
constexpr inline bool has_all_views(graph_view_t target, args_t... masks) {
    return ((has_view(target, masks)) && ...);
}

template <typename... args_t>
constexpr inline bool has_some_views(graph_view_t target, args_t... masks) {
    return ((has_view(target, masks)) || ...);
}

template<uint8_t N>
struct highest_bit {
    static constexpr uint8_t value = highest_bit<N/2>::value + 1;
};

template<>
struct highest_bit<1> {
    static constexpr uint8_t value = 1;
};

template<>
struct highest_bit<0> {
    static constexpr uint8_t value = 0;
};

template<uint8_t N>
constexpr uint8_t get_highest_bit_value() {
    static_assert(N > 0);
    return 1 << (highest_bit<N>::value - 1);
}

template<graph_view_t views>
constexpr graph_view_t get_first_view() {
    return static_cast<graph_view_t>(
        get_highest_bit_value<static_cast<uint8_t>(views)>());
}

template <graph_view_t view,
          arch_t arch,
          typename value_t,
          typename index_t = uint32_t,
          typename offset_t = uint64_t,
          vstart_t v_start = vstart_t::FROM_0_TO_0>
struct graph_mat;

template <arch_t arch,
          typename value_t,
          typename index_t,
          typename offset_t,
          vstart_t v_start>
struct graph_mat<graph_view_t::csr, arch, value_t, index_t, offset_t, v_start> {
    using type = CsrMat<arch, value_t, index_t, offset_t, v_start>;
};


} // namespace graph_one