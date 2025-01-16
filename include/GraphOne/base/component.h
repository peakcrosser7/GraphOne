#pragma once

#include "GraphOne/type.hpp"

namespace graph_one {

template <typename graph_t,
        typename hstatus_t,
        typename dstatus_t>
struct ComponentX {
    using graph_type = graph_t;
    using hstatus_type = hstatus_t;
    using dstatus_type = dstatus_t;
    // using frontier_type = frontier_t;
    using vertex_type = typename graph_t::vertex_type;
    using edge_type = typename graph_t::edge_type;

    using dstatus_ref_t = std::conditional_t<std::is_same_v<dstatus_t, empty_t>, empty_t, dstatus_t&>;

    ComponentX(const graph_t& graph_, hstatus_t& h_status_, dstatus_ref_t d_status_)
        : graph(graph_), h_status(h_status_), d_status(d_status_) {}

    virtual void Init() {}

    /// @brief converge according to the algorithm
    /// @return default return false when not according to the algorithm
    virtual bool IsConvergent() {
        return false;
    }

    virtual void BeforeEngine() {}

    virtual void AfterEngine() {}

    virtual void Final() {}

    const graph_t& graph;
    hstatus_t& h_status;
    dstatus_ref_t d_status;
};


template <template<typename graph_t>class comp_t,
          typename graph_t,
          typename hstatus_t,
          typename dstatus_t>
comp_t<graph_t> make_component(const graph_t& graph_, hstatus_t& h_status_, dstatus_t& d_status_) {
    return comp_t<graph_t>{graph_, h_status_, d_status_};
}

template <template<typename graph_t>class comp_t,
          typename graph_t,
          typename hstatus_t>
comp_t<graph_t> make_component(const graph_t& graph_, hstatus_t& h_status_) {
    return comp_t<graph_t>{graph_, h_status_};
}


} // namespace graph_one