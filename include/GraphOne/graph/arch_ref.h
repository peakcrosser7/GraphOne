#pragma once

#include "GraphOne/type.hpp"
#include "GraphOne/archi/macro/macro.h"

namespace graph_one {

template <graph_view_t views, arch_t arch, 
          vstart_t v_start,
          typename vprop_t, 
          typename vertex_t, typename edge_t, typename weight_t>
struct GraphArchRef;

// NOT SUPPORT both mat and transpose mat now
template <arch_t arch, 
          vstart_t v_start,
          typename vprop_t, 
          typename vertex_t, typename edge_t, typename weight_t>
struct GraphArchRef<graph_view_t::csr, 
                    arch, v_start, vprop_t, 
                    vertex_t, edge_t, weight_t> {
    using vprop_type = vprop_t;
    using vertex_type = vertex_t;
    using edge_type = edge_t;
    using weight_type = weight_t;

    constexpr static arch_t arch_value = arch;
    constexpr static vstart_t vstart_value = v_start;
    
    __ONE_DEV_INL__ 
    edge_t get_degree(vertex_t vid) const {
        return row_offsets[vid + 1] - row_offsets[vid];
    }

    __ONE_DEV_INL__ 
    edge_t get_starting_edge(vertex_t vid) const {
        return row_offsets[vid];
    }

    __ONE_DEV_INL__
    vertex_t get_dst_vertex(edge_t eid) const {
        return col_indices[eid];
    }

    __ONE_DEV_INL__
    weight_t get_edge_weight(edge_t eid) const {
        return values[eid];
    }

    vertex_t num_vertices;
    edge_t num_edges;

    const edge_t* row_offsets;
    const vertex_t* col_indices;
    const weight_t* values;

    vprop_t vprops{};    
};
    
} // namespace graph_one
