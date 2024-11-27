#pragma once

namespace graph_one::engine {

template <typename comp_t>
class BaseEngine {
protected:
    using graph_type = typename comp_t::graph_type;
    using hstatus_type = typename comp_t::hstatus_type;
    using dstatus_type = typename comp_t::dstatus_type;

    const graph_type& graph_;
    hstatus_type& h_status_;
    dstatus_type& d_status_;

public:
    BaseEngine(comp_t &comp) 
        : graph_(comp.graph), h_status_(comp.h_status), d_status_(comp.d_status) {}

    virtual void Forward() = 0;

};


template <typename comp_t, typename frontier_t>
class GcEngine: public BaseEngine<comp_t> {
protected:
    using graph_type = typename comp_t::graph_type;
    using hstatus_type = typename comp_t::hstatus_type;
    using dstatus_type = typename comp_t::dstatus_type;
    using bast_t = BaseEngine<comp_t>;

    frontier_t& frontier_;

public:
    GcEngine(comp_t &comp, frontier_t& frontier)
      : bast_t(comp), frontier_(frontier) {}
};
};
    
} // namespace graph_one