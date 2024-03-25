#pragma once

#include <type_traits>
#include <limits>

#include "GraphOne/type.hpp"

namespace graph_one::utils {

template <typename type_t, typename enable_t = void>
struct numeric_limits : std::numeric_limits<type_t> {};

/// Numeric Limits (additional support) for invalid() values.
/// 算术限制-有符号整型特化
/// @tparam type_t 数据类型
template <typename type_t>
struct numeric_limits<
    type_t,
    typename std::enable_if_t<std::is_integral<type_t>::value &&
                              std::is_signed<type_t>::value>>
    : std::numeric_limits<type_t> {
  /// @brief type_t数据的无效值
  /// @return type_t(-1)
  constexpr static type_t invalid() {
    return std::integral_constant<type_t, -1>::value;
  }
};

/// 算术限制-无符号整型特化
template <typename type_t>
struct numeric_limits<
    type_t,
    typename std::enable_if_t<std::is_integral<type_t>::value &&
                              std::is_unsigned<type_t>::value>>
    : std::numeric_limits<type_t> {
  constexpr static type_t invalid() {
    return std::integral_constant<type_t,
                                  std::numeric_limits<type_t>::max()>::value;
  }
};

/// 算术限制-浮点数特化
template <typename type_t>
struct numeric_limits<
    type_t,
    typename std::enable_if_t<std::is_floating_point<type_t>::value>>
    : std::numeric_limits<type_t> {
  constexpr static type_t invalid() {
    // 浮点数NaN值
    return std::numeric_limits<type_t>::quiet_NaN();
  }
};

template <vstart_t v_start, // cannot deduce
          typename vertex_t>
constexpr bool is_vertex_valid(vertex_t vid) {
    static_assert(std::is_integral_v<vertex_t>, 
                  "vertex_t must be an intergral type");
    if constexpr (v_start == vstart_t::FROM_0_TO_0) {
        return (vid != utils::numeric_limits<vertex_t>::invalid());
    } else {
        return (vid != 0);
    }
}

template <vstart_t v_start, typename vertex_t>
constexpr vertex_t invalid_vertex() {
    if constexpr (v_start == vstart_t::FROM_0_TO_0) {
        return utils::numeric_limits<vertex_t>::invalid();
    } else {
        return 0;
    }
}
    
} // namespace graph_one::utils