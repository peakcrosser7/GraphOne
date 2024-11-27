#pragma once

#include <type_traits>

#include "GraphOne/type.hpp"
#include "GraphOne/graph/graph_view.hpp"
#include "GraphOne/archi/macro/macro.h"

namespace graph_one {

template <graph_view_t views, arch_t arch, 
          vstart_t v_start,
          typename vprop_t, 
          typename vertex_t, typename edge_t, typename weight_t,
          typename = void>
struct GraphArchRef;


template <graph_view_t views, arch_t arch, 
          vstart_t v_start,
          typename vprop_t, 
          typename vertex_t, typename edge_t, typename weight_t>
struct GraphArchRef<views, 
                    arch, v_start, vprop_t, 
                    vertex_t, edge_t, weight_t,
                    std::enable_if_t<has_view(views, graph_view_t::csr)>> {
    using vprop_type = vprop_t;
    using vertex_type = vertex_t;
    using edge_type = edge_t;
    using weight_type = weight_t;

    constexpr static arch_t arch_value = arch;
    constexpr static vstart_t vstart_value = v_start;
    constexpr static bool both_mat = has_all_views(views, graph_view_t::normal, graph_view_t::transpose);

    template <graph_view_t Views = views>
    __ONE_DEV_INL__
    std::enable_if_t<has_view(Views, graph_view_t::normal), vertex_t>
    get_out_degree(vertex_t vid) const {
        return row_offsets[vid + 1] - row_offsets[vid];
    }

    template <graph_view_t Views = views>
    __ONE_DEV_INL__
    std::enable_if_t<!has_view(Views, graph_view_t::normal), vertex_t>
    get_out_degree(vertex_t vid) = delete;


    template <graph_view_t Views = views>
    __ONE_DEV_INL__
    std::enable_if_t<has_view(Views, graph_view_t::transpose), vertex_t>
    get_in_degree(vertex_t vid) const {
        if constexpr (both_mat) {
            return trans_row_offsets[vid + 1] - trans_row_offsets[vid];
        } else {
            return row_offsets[vid + 1] - row_offsets[vid];
        }
    }

    template <graph_view_t Views = views>
    __ONE_DEV_INL__
    std::enable_if_t<!has_view(Views, graph_view_t::transpose), vertex_t>
    get_in_degree(vertex_t vid) = delete;


    template <graph_view_t Views = views>
    __ONE_DEV_INL__
    std::enable_if_t<has_view(Views, graph_view_t::normal),edge_t>
    get_starting_out_edge(vertex_t vid) const {
        return row_offsets[vid];
    }

    template <graph_view_t Views = views>
    __ONE_DEV_INL__
    std::enable_if_t<!has_view(Views, graph_view_t::normal), edge_t>
    get_starting_out_edge(vertex_t vid) = delete;


    template <graph_view_t Views = views>
    __ONE_DEV_INL__
    std::enable_if_t<has_view(Views, graph_view_t::normal), vertex_t>
    get_dst_vertex_by_out_edge(edge_t eid) const {
        return col_indices[eid];
    }

    template <graph_view_t Views = views>
    __ONE_DEV_INL__
    std::enable_if_t<!has_view(Views, graph_view_t::normal), vertex_t>
    get_dst_vertex_by_out_edge(edge_t eid) = delete;


    template <graph_view_t Views = views>
    __ONE_DEV_INL__
    std::enable_if_t<has_view(Views, graph_view_t::normal), weight_t>
    get_out_edge_weight(edge_t eid) const {
        return values[eid];
    }

    template <graph_view_t Views = views>
    __ONE_DEV_INL__
    std::enable_if_t<!has_view(Views, graph_view_t::normal), weight_t>
    get_out_edge_weight(edge_t eid) = delete;


    vertex_t num_vertices;
    edge_t num_edges;

    const edge_t* row_offsets;
    const vertex_t* col_indices;
    const weight_t* values;

    const edge_t* trans_row_offsets;
    const vertex_t* trans_col_indices;
    const weight_t* trans_values;
    vprop_t vprops{};    
};
    
} // namespace graph_one
