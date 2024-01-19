#pragma once

#include <cstdint>

namespace graph_genlx {

enum FrontierKind : uint8_t {
    SPARSE_BASED = 1 << 1,
    DENSE_BASED  = 1 << 2
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

    virtual bool IsConvergent() = 0;
};

    
} // namespace graph_genlx