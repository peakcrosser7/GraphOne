#pragma once

#include "GraphGenlX/type.hpp"
#include "GraphGenlX/utils.h"
#include "GraphGenlX/base/buffer.h"
#include "GraphGenlX/archi/macro/macro.h"
#include "GraphGenlX/archi/thrust/thrust.h"

namespace graph_genlx {

template <arch_t arch,
          typename vertex_t,
          typename index_t = vertex_t>
class ActiveFrontier {
public:
    using vertex_type = vertex_t;
    constexpr static bool has_output = true;

    struct arch_ref_t {
        constexpr static bool has_output = ActiveFrontier::has_output;
        
        __GENLX_DEV_INL__
        vertex_t get(index_t i) const {
            return frontiers[input_selector][i];
        }

        __GENLX_DEV_INL__
        void set(index_t i, vertex_t v) {
            frontiers[!input_selector][i] = v;
        }

        bool input_selector;
        vertex_t input_size;
        vertex_t output_size;
        vertex_t* frontiers[2];
    };


    ActiveFrontier() = default;

    void reset_input(vertex_t size) {
        if (size > frontiers_[input_selector_].size()) {
            frontiers_[input_selector_].reset(size);
        }
        in_size_ = size;
    }

    void reset_output(vertex_t size) {
        if (size > frontiers_[!input_selector_].size()) {
            frontiers_[!input_selector_].reset(size);
        }
        out_size_ = size;
    }

    void reset(vertex_t size) {
        reset_output(size);
    }

    vertex_t input_size() const {
        return in_size_;
    }

    vertex_t output_size() const {
        return out_size_;
    }

    void swap_inout() {
        input_selector_ = !input_selector_;
    }

    bool empty() {
        return in_size_ == 0;
    }

    template <typename graph_t>
    typename graph_t::edge_type CaclOuputSize(const graph_t& g) const {
        using edge_t = typename graph_t::edge_type;
        auto graph_ref = g.ToArch();
        auto v_degree_func = [=] __GENLX_ARCH__(const vertex_t &vid) {
                return (utils::is_vertex_valid<graph_t::vstart_value>(vid)
                            ? graph_ref.get_degree(vid) : 0);
        };

        return archi::transform_reduce<arch>(
            thrust::make_counting_iterator<vertex_t>(0),
            thrust::make_counting_iterator<vertex_t>(in_size_),
            v_degree_func,
            edge_t(0),
            thrust::plus<edge_t>()
        );
    }

    template <typename ...vids_t>
    void Init(vids_t... vids) {
        index_t sz = sizeof...(vids);
        reset_input(sz);
        int i = 0;
        ((frontiers_[input_selector_][i++] = vids), ...);
    }

    template <typename graph_t>
    void BeforeEngine(const graph_t& g) {
        auto sz = CaclOuputSize(g);
        reset_output(sz);
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

protected:
    bool input_selector_{0};
    vertex_t in_size_;
    vertex_t out_size_;
    Buffer<arch, vertex_t, index_t> frontiers_[2];
};
    
} // namespace graph_genlx