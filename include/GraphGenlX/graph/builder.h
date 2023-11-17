#pragma once

#include "GraphGenlX/graph/graph.h"
#include "GraphGenlX/mat/csr.h"
#include "GraphGenlX/mat/csc.h"

namespace graph_genlx::graph {

template <graph_view_t views = graph_view_t::none, arch_t arch = arch_t::cpu, 
        //   typename g_prop_t, typename v_prop_t, typename e_prop_t, 
          typename vertex_t = vid_t, typename edge_t = eid_t, typename weight_t = double>
Graph<arch, views, /*g_prop_t, v_prop_t, e_prop_t,*/ vertex_t, edge_t, weight_t>
FromCsr(const CsrMat<arch, weight_t, vertex_t, edge_t>& csr) {
    return Graph<arch, views | graph_view_t::csr, vertex_t, edge_t, weight_t>(csr);
}

} // namespace graph_genlx