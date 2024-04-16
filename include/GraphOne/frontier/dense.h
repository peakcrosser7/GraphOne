#pragma once

#include <vector>

#include "GraphOne/type.hpp"
#include "GraphOne/base/buffer.h"

namespace graph_one {

template <arch_t arch,
          typename value_t,
          typename index_t>
class DenDblFrontier : public BaseFrontier<DENSE_BASED, true> {
public:
    using value_type = value_t;
    using index_type = index_t;

    DenDblFrontier(index_t size, std::initializer_list<index_t> vids)
    : input_selector_(0), size_(size), init_buf_(vids.size()), nnz_(), frontiers_() {
        frontiers_[0].resize(size);
        frontiers_[1].resize(size);

        init_buf_.copy_from(vids);
    }

    index_t input_size() const {
        return size_;
    }

    index_t output_size() const {
        return size_;
    }

    index_t& input_nnz() {
        return nnz_[input_selector_];
    }

    index_t& output_nnz() {
        return nnz_[input_selector_^1];
    }

    Buffer<arch, value_t, index_t>& input() {
        return frontiers_[0];
    }

    Buffer<arch, value_t, index_t>& output() {
        return frontiers_[1];
    }

    void swap_inout() {
        input_selector_ ^= 1;
    }

    Buffer<arch, index_t, index_t>& init_buf() {
        return init_buf_;
    }

    std::string ToString() const {
        std::string str;
        str += "DenDblFrontier{ ";
        str += "input_selector_:" + utils::NumToString(input_selector_) + ", ";
        str += "size_:" + utils::NumToString(size_) + ", ";
        str += "init_buf_:" + init_buf_.ToString() + ", ";
        str += "(input)nnz_[" + utils::NumToString(input_selector_) + "]:" 
                + utils::NumToString(nnz_[input_selector_]) + ", ";
        str += "(output)nnz_[" + utils::NumToString(input_selector_^1) + "]:"
                + utils::NumToString(nnz_[input_selector_^1]);
        str += " }";
        return str;
    }

    void AfterEngine() override {
        swap_inout();
    }

    bool IsConvergent() override {
        return init_buf_.empty() && nnz_[input_selector_] == 0;
    }
    
protected:
    int input_selector_;
    index_t size_;
    Buffer<arch, index_t, index_t> init_buf_;
    index_t nnz_[2];
    Buffer<arch, value_t, index_t> frontiers_[2];
    
};


} // namespace graph_one
