#pragma once

#include "GraphGenlX/loader/edge_cache.h"
#include "GraphGenlX/graph/graph.h"
#include "GraphGenlX/mat/csr.h"

namespace graph_genlx::graph {

template <arch_t arch, graph_view_t views, vstart_t v_start, typename vertex_t,
          typename edge_t, typename weight_t>
typename std::enable_if_t<
    has_all_views(views, graph_view_t::normal, graph_view_t::transpose) &&
        get_first_view<views>() == graph_view_t::csr,
    Graph<arch, views, v_start, empty_t, vertex_t, edge_t, weight_t>>
build(EdgeCache<v_start, weight_t, vertex_t, edge_t> &cache) {
    auto csr = cache.template ToCsr<arch, false>();
    auto csc = cache.template ToCsr<arch, true>();
    return Graph<arch, views, v_start, empty_t, vertex_t, edge_t, weight_t>(
        std::move(csr), std::move(csc), empty_t{});
}

template <arch_t arch, graph_view_t views, vstart_t v_start, typename vertex_t,
          typename edge_t, typename weight_t>
typename std::enable_if_t<
    (has_view(views, graph_view_t::normal) &&
     !has_view(views, graph_view_t::transpose) &&
     get_first_view<views>() == graph_view_t::csr),
    Graph<arch, views, v_start, empty_t, vertex_t, edge_t, weight_t>>
build(EdgeCache<v_start, weight_t, vertex_t, edge_t> &cache) {
    auto csr = cache.template ToCsr<arch, false>();
    return Graph<arch, views, v_start, empty_t, vertex_t, edge_t, weight_t>(
        std::move(csr), empty_t{});
}

template <arch_t arch, graph_view_t views, vstart_t v_start, typename vertex_t,
          typename edge_t, typename weight_t>
typename std::enable_if_t<
    (!has_view(views, graph_view_t::normal) &&
     has_view(views, graph_view_t::transpose) &&
     get_first_view<views>() == graph_view_t::csr),
    Graph<arch, views, v_start, empty_t, vertex_t, edge_t, weight_t>>
build(EdgeCache<v_start, weight_t, vertex_t, edge_t> &cache) {
    auto csc = cache.template ToCsr<arch, true>();
    return Graph<arch, views, v_start, empty_t, vertex_t, edge_t, weight_t>(
        std::move(csc), empty_t{});
}


} // namespace graph_genlx