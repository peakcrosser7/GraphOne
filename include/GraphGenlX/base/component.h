#pragma once 

#include <functional>

#include "GraphGenlX/archi/macro/macro.h"

namespace graph_genlx {

template <typename graph_t,
        typename status_t,
        typename frontier_t>
struct ComponentX {
    
    void Init() {}

    bool IsConvergent() {
        return frontier.empty();
    }

    void BeforeEngine() {
        frontier.BeforeEngine(graph);
    }

    void Filter() {

    }

    const graph_t& graph;
    status_t& status;
    frontier_t& frontier;
};

namespace compx {

template <template <typename, typename, typename> class comp_t, 
          typename graph_t, 
          typename status_t,
          typename frontier_t>
comp_t<graph_t, status_t, frontier_t> 
build(const graph_t &graph, status_t &status, frontier_t &frontier) {
    return comp_t<graph_t, status_t, frontier_t>{graph, status, frontier};
}

} // namespace comp

} // namespace graph_genlx