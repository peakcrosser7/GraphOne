#pragma once

namespace graph_one::engine {

template <typename comp_t, typename frontier_t>
class BaseEngine {
protected:
    using graph_type = typename comp_t::graph_type;
    using hstatus_type = typename comp_t::hstatus_type;
    using dstatus_type = typename comp_t::dstatus_type;

    const graph_type& graph_;
    hstatus_type& h_status_;
    dstatus_type& d_status_;
    frontier_t& frontier_;

public:
    BaseEngine(comp_t &comp, frontier_t& frontier)
      : graph_(comp.graph), h_status_(comp.h_status), d_status_(comp.d_status),
        frontier_(frontier) {}


    virtual void Forward() = 0;
};
    
} // namespace graph_one