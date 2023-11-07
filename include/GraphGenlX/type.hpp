#pragma once
#include <cstdint>

namespace graph_genlx {

using vid_t = uint32_t;
using eid_t = uint64_t;

struct empty_t {
    operator std::string() const {
        return "";
    }
};


}   // namespace graph_genlx 