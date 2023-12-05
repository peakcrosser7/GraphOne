#pragma once

#include "GraphGenlX/graph/graph.h"
#include "GraphGenlX/mat/csr.h"
#include "GraphGenlX/mat/csc.h"

namespace graph_genlx::graph {

template <graph_view_t views = graph_view_t::none, 
          arch_t arch = arch_t::cpu, vstart_t v_start = vstart_t::FROM_0_TO_0, 
          typename vprop_t = empty_t, 
          typename vertex_t = vid_t, typename edge_t = eid_t, typename weight_t = double>
Graph<arch, views | graph_view_t::csr, v_start, vprop_t, vertex_t, edge_t, weight_t>
FromCsr(const CsrMat<arch, weight_t, vertex_t, edge_t, v_start>& csr, const vprop_t& v_props = empty_t()) {
    if constexpr (!std::is_same_v<vprop_t, empty_t>) {
        static_assert(csr.arch_value == v_props.arch_value);
    }
    return Graph<arch, views | graph_view_t::csr, v_start, vprop_t, vertex_t, edge_t, weight_t>(
      csr, v_props);
}

template <graph_view_t views = graph_view_t::none, 
          arch_t arch = arch_t::cpu, vstart_t v_start = vstart_t::FROM_0_TO_0, 
          typename vprop_t = empty_t, 
          typename vertex_t = vid_t, typename edge_t = eid_t, typename weight_t = double>
Graph<arch, views | graph_view_t::csr, v_start, vprop_t, vertex_t, edge_t, weight_t>
FromCsr(CsrMat<arch, weight_t, vertex_t, edge_t, v_start>&& csr, vprop_t&& v_props = empty_t()) {
    if constexpr (!std::is_same_v<vprop_t, empty_t>) {
        static_assert(csr.arch_value == v_props.arch_value);
    }
    return Graph<arch, views | graph_view_t::csr, v_start, vprop_t, vertex_t, edge_t, weight_t>(
      std::move(csr), std::move(v_props));
}

} // namespace graph_genlx