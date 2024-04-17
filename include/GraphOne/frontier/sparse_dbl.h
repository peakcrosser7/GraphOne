#pragma once

#include <string>
#include <initializer_list>

#include "GraphOne/type.hpp"
#include "GraphOne/utils.h"
#include "GraphOne/base/buffer.h"
#include "GraphOne/archi/macro/macro.h"
#include "GraphOne/archi/thrust/thrust.h"
#include "GraphOne/frontier/base.h"

namespace graph_one {

template <arch_t arch,
          typename vertex_t,
          typename index_t = vertex_t>
class SpDblFrontier : public BaseFrontier<SPARSE_BASED, true> {
public:
    using vertex_type = vertex_t;
    using index_type = index_t;

    struct arch_ref_t {
        constexpr static bool has_output = SpDblFrontier::has_output;
        using vertex_type = vertex_t;
        using index_type = index_t;

        __ONE_DEV_INL__
        vertex_t get(index_t i) const {
            return frontiers[input_selector][i];
        }

        __ONE_DEV_INL__
        void set(index_t i, vertex_t v) {
            frontiers[input_selector^1][i] = v;
        }

        __ONE_DEV_INL__
        const vertex_t* input() const {
            return frontiers[input_selector];
        }

        __ONE_DEV_INL__
        vertex_t* ouput() {
            return frontiers[input_selector^1];
        }

        int input_selector;
        index_t input_size;
        index_t output_size;
        vertex_t* frontiers[2];
    };


    SpDblFrontier(index_t size, std::initializer_list<vertex_t> vids)
    : input_selector_(0), in_size_(0), out_size_(0), frontiers_() {
        index_t sz = vids.size();
        index_t max_sz = std::max(size, sz);
        reserve_input(max_sz);
        reserve_output(max_sz);

        reset_input(sz);

        frontiers_[input_selector_].copy_from(vids);
    }

    void reserve_input(index_t size) {
        if (size > frontiers_[input_selector_].size()) {
            frontiers_[input_selector_].resize(size);
        }
    }

    void reserve_output(index_t size) {
        if (size > frontiers_[input_selector_^1].size()) {
            frontiers_[input_selector_^1].resize(size);
        }
    }

    void reset_input(index_t size) {
        if (size > frontiers_[input_selector_].size()) {
            frontiers_[input_selector_].resize(size * 2);
        }
        in_size_ = size;
    }

    void reset_output(index_t size) {
        if (size > frontiers_[input_selector_^1].size()) {
            frontiers_[input_selector_^1].resize(size * 2);
        }
        out_size_ = size;
    }

    index_t input_size() const {
        return in_size_;
    }

    index_t output_size() const {
        return out_size_;
    }

    void swap_inout() {
        input_selector_ ^= 1;
        std::swap(in_size_, out_size_);
    }

    const Buffer<arch, vertex_t, index_t>& input() const {
        return frontiers_[input_selector_];
    }

    Buffer<arch, vertex_t, index_t>& output() {
        return frontiers_[input_selector_^1];
    }

    arch_ref_t ToArch() {
        arch_ref_t arch_ref;
        arch_ref.input_selector = input_selector_;
        arch_ref.input_size = in_size_;
        arch_ref.output_size = out_size_;
        arch_ref.frontiers[0] = frontiers_[0].data();
        arch_ref.frontiers[1] = frontiers_[1].data();
        
        return arch_ref;
    }

    std::string ToString() const {
        std::string str;
        str += "SpDblFrontier{ ";
        str += "input_selector_:" + utils::NumToString(input_selector_) + ", ";
        str += "in_size_:" + utils::NumToString(in_size_) + ", ";
        str += "out_size_:" + utils::NumToString(out_size_) + ", ";
        str += "(input)frontiers_[" + utils::NumToString(input_selector_) + "]:" 
                + BufferToString_(frontiers_[input_selector_], in_size_) + ", ";
        str += "(output)frontiers_[" + utils::NumToString(input_selector_^1) + "]:"
                + BufferToString_(frontiers_[input_selector_^1], out_size_);
        str += " }";
        return str;
    }


    void AfterEngine() override {
        swap_inout();
    }

    bool IsConvergent() override {
        return in_size_ == 0;
    }

private:
    static std::string 
    BufferToString_(const Buffer<arch, vertex_t, index_t>& frontier, index_t size) {
        size = std::min(size, frontier.size());
        std::string str("[");
        if constexpr (arch != arch_t::cpu) {
            vertex_t* h_data = archi::memalloc<arch_t::cpu, vertex_t>(size);
            archi::memcpy<arch_t::cpu, arch, vertex_t>(h_data, frontier.data(), size);
            for (index_t i = 0; i < size; ++i) {
                str += utils::NumToString(h_data[i]);
                if (i < size - 1) {
                    str += ",";
                }
            }
            archi::memfree<arch_t::cpu, vertex_t>(h_data);
        } else {
            const vertex_t* h_data = frontier.data();
            for (index_t i = 0; i < size; ++i) {
                str += utils::NumToString(h_data[i]);
                if (i < size - 1) {
                    str += ",";
                }
            }
        }
        str += "]";
        return str;
    }

protected:
    int input_selector_;
    index_t in_size_;
    index_t out_size_;
    Buffer<arch, vertex_t, index_t> frontiers_[2];
};
    
} // namespace graph_one