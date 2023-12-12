#pragma once

#include <cstdint>

namespace graph_genlx {

enum class graph_view_t : uint8_t {
    none    = 1 << 0,
    csr     = 1 << 1,
    csc     = 1 << 2,
    coo     = 1 << 3
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

} // namespace graph_genlx