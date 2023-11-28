#pragma once

#include <cstdint>
#include <string>

namespace graph_genlx {

using vid_t = uint32_t;
using eid_t = uint32_t;

enum class arch_t {
    cpu,
    cuda
};

struct empty_t {
    operator std::string() const {
        return "";
    }
};


}   // namespace graph_genlx 