#pragma once

#include <vector>

#include "GraphOne/type.hpp"
#include "GraphOne/base/buffer.h"

namespace graph_one {

template <arch_t arch,
          typename vertex_t,
          typename index_t = vertex_t>
class DenDblFrontier : public BaseFrontier<DENSE_BASED, true> {
public:
    using vertex_type = vertex_t;
    using index_type = index_t;

    DenDblFrontier(index_t size, std::initializer_list<vertex_t> vids)
    : in_size_(vids.size()), out_size_(0), init_buf_(vids.size()) {

        init_buf_.copy_from(vids);
    }

    index_t input_size() const {
        return in_size_;
    }

    index_t& output_size() {
        return out_size_;
    }

    void swap_inout() {
        std::swap(in_size_, out_size_);
    }

    Buffer<arch, vertex_t, index_t>& init_buf() {
        return init_buf_;
    }

    std::string ToString() const {
        std::string str;
        str += "DenDblFrontier{ ";
        str += "init_buf_:" + init_buf_.ToString() + ", ";
        str += "in_size_:" + utils::NumToString(in_size_) + ", ";
        str += "out_size_:" + utils::NumToString(out_size_);
        str += " }";
        return str;
    }

    void AfterEngine() override {
        swap_inout();
    }

    bool IsConvergent() override {
        return in_size_ == 0;
    }
    
protected:
    index_t in_size_;
    index_t out_size_;
    Buffer<arch, vertex_t, index_t> init_buf_;
    
};


} // namespace graph_one
