#pragma once

#include <string>

#include "GraphOne/type.hpp"
#include "GraphOne/frontier/base.h"
#include "GraphOne/utils/string.hpp"


namespace graph_one {

template <arch_t arch,
          typename vertex_t,
          typename index_t = vertex_t>
class AllActiveFrontier : public BaseFrontier<ALL_ACTIVE, false> {
public:
    using vertex_type = vertex_t;
    using index_type = index_t;

    AllActiveFrontier(index_t size, std::initializer_list<vertex_t> vids) 
        : size_(size) {}

    index_t input_size() const {
        return size_;
    }

    index_t output_size() const {
        return size_;
    }

    index_t size() const {
        return size_;
    }

    bool IsConvergent() override {
        return false;
    }

    std::string ToString() const {
        std::string str;
        str += "AllActiveFrontier{ ";
        str += "size_: " + utils::NumToString(size_);
        str += " }";
        return str;
    }

protected:
    index_t size_;
};


} // namespace graph_one