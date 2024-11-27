#pragma once

#include <cstdint>

namespace graph_one {

enum FrontierKind : uint8_t {
    SPARSE_BASED = 1 << 1,
    DENSE_BASED  = 1 << 2,
    ALL_ACTIVE   = (1 << 1) | (1 << 2)
};

constexpr inline FrontierKind operator|(FrontierKind lhs, FrontierKind rhs) {
  return static_cast<FrontierKind>(static_cast<uint8_t>(lhs) |
                             static_cast<uint8_t>(rhs));
}

constexpr inline bool check_frontier(FrontierKind target, FrontierKind base) {
  return (static_cast<uint8_t>(target) & static_cast<uint8_t>(base)) ==
         static_cast<uint8_t>(base);
}

template <FrontierKind kind_, bool has_output_>
struct BaseFrontier {
    constexpr static FrontierKind kind = kind_;
    constexpr static bool has_output = has_output_;

    virtual void BeforeEngine() {}

    virtual void AfterEngine() {}

    /// @brief converge according to the frontier
    /// @return default return false when not according to the frontier
    virtual bool IsConvergent() = 0;
};

    
} // namespace graph_one