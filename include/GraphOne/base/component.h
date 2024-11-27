#pragma once 

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

    ComponentX(const graph_t& graph_, hstatus_t& h_status_, dstatus_t& d_status_)
        : graph(graph_), h_status(h_status_), d_status(d_status_) {}

    virtual void Init() {}

    /// @brief converge according to the algorithm
    /// @return default return false when not according to the algorithm
    virtual bool IsConvergent() {
        return false;
    }

    virtual void BeforeEngine() {}

    virtual void AfterEngine() {}

    const graph_t& graph;
    hstatus_t& h_status;
    dstatus_t& d_status;
};

} // namespace graph_one