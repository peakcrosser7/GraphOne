#pragma once

#include <vector>

#include "GraphGenlX/type.hpp"
#include "GraphGenlX/utils.h"
#include "GraphGenlX/base/buffer.h"
#include "GraphGenlX/archi/macro/macro.h"
#include "GraphGenlX/archi/thrust/thrust.h"
#include "GraphGenlX/frontier/base.h"

namespace graph_genlx {

template <arch_t arch,
          typename vertex_t,
          typename index_t = vertex_t>
class DblBufFrontier : public BaseFrontier<SPARSE_BASED, true> {
public:
    using vertex_type = vertex_t;
    using index_type = index_t;

    struct arch_ref_t {
        constexpr static bool has_output = DblBufFrontier::has_output;
        using vertex_type = vertex_t;
        using index_type = index_t;

        __GENLX_DEV_INL__
        vertex_t get(index_t i) const {
            return frontiers[input_selector][i];
        }

        __GENLX_DEV_INL__
        void set(index_t i, vertex_t v) {
            frontiers[!input_selector][i] = v;
        }

        __GENLX_DEV_INL__
        const vertex_t* input() const {
            return frontiers[input_selector];
        }

        __GENLX_DEV_INL__
        vertex_t* ouput() {
            return frontiers[!input_selector];
        }

        bool input_selector;
        index_t input_size;
        index_t output_size;
        vertex_t* frontiers[2];
    };


    template <typename ...vids_t>
    DblBufFrontier(index_t size, vids_t... vids) {
        index_t sz = sizeof...(vids);
        index_t max_sz = std::max(size, sz);
        reserve_input(max_sz);
        reserve_output(max_sz);

        reset_input(sz);
        int i = 0;
        if constexpr (arch == arch_t::cpu) {
            ((frontiers_[input_selector_][i++] = vids), ...);
        } else {
            Buffer<arch_t::cpu, vertex_t, index_t> h_buf(sz);
            ((h_buf[i++] = vids), ...);
            frontiers_[input_selector_] = h_buf;
        }  
    }

    void reserve_input(index_t size) {
        if (size > frontiers_[input_selector_].size()) {
            frontiers_[input_selector_].reset(size);
        }
    }

    void reserve_output(index_t size) {
        if (size > frontiers_[!input_selector_].size()) {
            frontiers_[!input_selector_].reset(size);
        }
    }

    void reset_input(index_t size) {
        if (size > frontiers_[input_selector_].size()) {
            frontiers_[input_selector_].reset(size * 2);
        }
        in_size_ = size;
    }

    void reset_output(index_t size) {
        // LOG_DEBUG("reset_output  ouput_size=", out_size_, 
        //           " output buf size=", frontiers_[!input_selector_].size(),
        //           " param size=", size);
        if (size > frontiers_[!input_selector_].size()) {
            frontiers_[!input_selector_].reset(size * 2);
        }
        out_size_ = size;
        // LOG_DEBUG("reset_output end, new output_size=", out_size_,
        //           " output buf size=", frontiers_[!input_selector_].size());
    }

    // void reset(index_t size) {
    //     reset_output(size);
    // }

    index_t input_size() const {
        return in_size_;
    }

    index_t output_size() const {
        return out_size_;
    }

    void swap_inout() {
        input_selector_ = !input_selector_;
        std::swap(in_size_, out_size_);
    }

    // bool empty() {
    //     return in_size_ == 0;
    // }

    const Buffer<arch, vertex_t, index_t>& input() const {
        return frontiers_[input_selector_];
    }

    Buffer<arch, vertex_t, index_t>& output() {
        return frontiers_[!input_selector_];
    }

    // todo: move to adv engine, it's not general.
    // template <typename graph_t>
    // typename graph_t::edge_type CaclOuputSize(const graph_t& g) {
    //     using edge_t = typename graph_t::edge_type;
    //     auto graph_ref = g.ToArch();
    //     auto frontier_ref = ToArch();
    //     auto v_degree_func = [=] __GENLX_ARCH__ (const vertex_t &i) {
    //             auto vid = frontier_ref.get(i);
    //             // archi::cuda::print("vid=%u, degree=%u\n", vid, graph_ref.get_degree(vid));
    //             return (utils::is_vertex_valid<graph_t::vstart_value>(vid)
    //                         ? graph_ref.get_degree(vid) : 0);
    //     };

    //     return archi::transform_reduce<arch>(
    //         thrust::make_counting_iterator<vertex_t>(0),
    //         thrust::make_counting_iterator<vertex_t>(in_size_),
    //         v_degree_func,
    //         edge_t(0),
    //         thrust::plus<edge_t>()
    //     );
    // }

    // void Init(index_t size, const std::vector<vertex_t>& vids) {
    //     index_t sz = vids.size();
    //     index_t max_sz = std::max(size, sz);
    //     reserve_input(max_sz);
    //     reserve_output(max_sz);
    //     reset_input(sz);
    //     int i = 0;
    //     if constexpr (arch == arch_t::cpu) {
    //         for(vertex_t vid: vids) {
    //             frontiers_[input_selector_][i++] = vid;
    //         }
    //     } else {
    //         Buffer<arch_t::cpu, vertex_t, index_t> h_buf(sz);
    //         for (vertex_t vid: vids) {
    //             h_buf[i++] = vid;
    //         }
    //         frontiers_[input_selector_] = h_buf;
    //     }  
    // }

    // template <typename graph_t>
    // void BeforeEngine(const graph_t& g) {
    //     auto sz = CaclOuputSize(g);
    //     reset_output(sz);
    // }

    arch_ref_t ToArch() {
        arch_ref_t arch_ref;
        arch_ref.input_selector = input_selector_;
        arch_ref.input_size = in_size_;
        arch_ref.output_size = out_size_;
        arch_ref.frontiers[0] = frontiers_[0].data();
        arch_ref.frontiers[1] = frontiers_[1].data();
        
        return arch_ref;
    }

    void AfterEngine() override {
        swap_inout();
    }

    bool IsConvergent() override {
        return in_size_ == 0;
    }

protected:
    bool input_selector_{0};
    index_t in_size_;
    index_t out_size_;
    Buffer<arch, vertex_t, index_t> frontiers_[2];
};
    
} // namespace graph_genlx