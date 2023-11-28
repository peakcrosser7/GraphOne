#pragma once

#include "GraphGenlX/graph/graph.h"
#include "GraphGenlX/mat/csr.h"
#include "GraphGenlX/mat/csc.h"

namespace graph_genlx::graph {

template <graph_view_t views = graph_view_t::none, arch_t arch = arch_t::cpu, 
          typename vprop_t = empty_t, /* typename e_prop_t, typename g_prop_t,*/
          typename vertex_t = vid_t, typename edge_t = eid_t, typename weight_t = double>
Graph<arch, views | graph_view_t::csr, vprop_t, /*g_prop_t, v_prop_t, e_prop_t,*/ vertex_t, edge_t, weight_t>
FromCsr(const CsrMat<arch, weight_t, vertex_t, edge_t>& csr, const vprop_t& v_props = empty_t()) {
    if constexpr (!std::is_same_v<vprop_t, empty_t>) {
        static_assert(csr.arch_type == v_props.arch_type);
    }
    return Graph<arch, views | graph_view_t::csr, vprop_t, vertex_t, edge_t, weight_t>(
      csr, v_props);
}

template <graph_view_t views = graph_view_t::none, arch_t arch = arch_t::cpu, 
          typename vprop_t = empty_t, /* typename e_prop_t, typename g_prop_t,*/
          typename vertex_t = vid_t, typename edge_t = eid_t, typename weight_t = double>
Graph<arch, views | graph_view_t::csr, vprop_t, /*g_prop_t, v_prop_t, e_prop_t,*/ vertex_t, edge_t, weight_t>
FromCsr(CsrMat<arch, weight_t, vertex_t, edge_t>&& csr, vprop_t&& v_props = empty_t()) {
    if constexpr (!std::is_same_v<vprop_t, empty_t>) {
        static_assert(csr.arch_type == v_props.arch_type);
    }
    return Graph<arch, views | graph_view_t::csr, vprop_t, vertex_t, edge_t, weight_t>(
      std::move(csr), std::move(v_props));
}

} // namespace graph_genlx