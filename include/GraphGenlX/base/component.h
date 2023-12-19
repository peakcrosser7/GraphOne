#pragma once 

#include <functional>

#include "GraphGenlX/archi/macro/macro.h"

namespace graph_genlx {

template <typename graph_t,
        typename hstatus_t,
        typename dstatus_t,
        typename frontier_t>
struct ComponentX {
    
    void Init() {}

    bool IsConvergent() {
        return frontier.empty();
    }

    void BeforeEngine() {
        frontier.BeforeEngine(graph);
    }

    void AfterEngine() {
        frontier.AfterEngine();
    }

    const graph_t& graph;
    hstatus_t& h_status;
    // use reference will occur an
    dstatus_t& d_status;
    frontier_t& frontier;
};

namespace compx {

template <template <typename, typename, typename, typename> class comp_t,
          typename graph_t, 
          typename hstatus_t, 
          typename dstatus_t,
          typename frontier_t>
comp_t<graph_t, hstatus_t, dstatus_t, frontier_t>
build(const graph_t &graph, hstatus_t &h_status, dstatus_t& d_status, frontier_t &frontier) {
    return comp_t<graph_t, hstatus_t, dstatus_t, frontier_t>{
        graph, h_status, d_status, frontier};
}

} // namespace comp

} // namespace graph_genlx