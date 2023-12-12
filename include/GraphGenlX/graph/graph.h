#pragma once

#include "GraphGenlX/type.hpp"
#include "GraphGenlX/archi.h"
#include "GraphGenlX/graph/graph_view.hpp"
#include "GraphGenlX/mat/csr.h"
#include "GraphGenlX/mat/csc.h"
#include "GraphGenlX/mat/convert.h"

namespace graph_genlx {

template <arch_t arch, graph_view_t views,
          vstart_t v_start,
          typename vprop_t, 
          typename vertex_t, typename edge_t, typename weight_t>
class Graph {
public:
    using vprop_type = vprop_t;
    using vertex_type = vertex_t;
    using edge_type = edge_t;
    using weight_type = weight_t;

    constexpr static arch_t arch_value = arch;
    constexpr static vstart_t vstart_value = v_start;

protected:
    constexpr static bool has_csr_ = has_view(views, graph_view_t::csr);
    constexpr static bool has_csc_ = has_view(views, graph_view_t::csc);

    using csr_t = CsrMat<arch, weight_t, vertex_t, edge_t>;
    using csc_t = CscMat<arch, weight_t, vertex_t, edge_t>;

    using csr_or_ept_t = std::conditional_t<has_csr_, csr_t, empty_t>;
    using csc_or_ept_t = std::conditional_t<has_csc_, csc_t, empty_t>;

public:
    struct arch_ref_t {
        using vprop_type = vprop_t;
        using vertex_type = vertex_t;
        using edge_type = edge_t;
        using weight_type = weight_t;

        constexpr static arch_t arch_value = arch;
        constexpr static vstart_t vstart_value = v_start;
        
        __GENLX_DEV_INL__ 
        edge_t get_degree(vertex_t vid) const {
            if constexpr (has_csr_) {
                return row_offsets[vid + 1] - row_offsets[vid];
            } 
            if constexpr (has_csc_) {
                return col_offsets[vid + 1] - col_offsets[vid];
            }
            return 0;
        }

        __GENLX_DEV_INL__ 
        edge_t get_starting_edge(vertex_t vid) const {
            if constexpr (has_csr_) {
                return row_offsets[vid];
            } 
            if constexpr (has_csc_) {
                return col_offsets[vid];
            }
            return 0;
        }

        __GENLX_DEV_INL__
        vertex_t get_dst_vertex(edge_t eid) const {
            if constexpr (has_csr_) {
                return col_indices[eid];
            }
            if constexpr (has_csc_) {
                return row_indices[eid];
            }
            return 0;  
        }

        __GENLX_DEV_INL__
        weight_t get_edge_weight(edge_t eid) const {
            if constexpr (has_csr_) {
                return csr_values[eid];
            } 
            if constexpr (has_csc_) {
                return csc_values[eid];
            }
            return weight_t(0);
        }

        vertex_t num_vertices;
        edge_t num_edges;

        const edge_t* row_offsets;
        const vertex_t* col_indices;
        const weight_t* csr_values;

        const edge_t* col_offsets;
        const vertex_t* row_indices;
        const weight_t* csc_values;

        vprop_t vprops{};
    };

public:
    Graph(const csr_t& csr, const vprop_t& vprops)
    : csr_(csr), csc_(), vprops_(vprops) {
        if constexpr (has_csc_) {
            csc_ = mat::ToCsc(csr);
        }
    }

    Graph(csr_t&& csr, vprop_t&& vprops)
    : csr_(), csc_(), vprops_(std::move(vprops)) {
        if constexpr (has_csc_) {
            csc_ = mat::ToCsc(csr);
        }
        csr_ = std::move(csr);
    }

    const vprop_t& vprops() const {
        return vprops_;
    }

    vertex_t num_vertices() const {
        if constexpr (has_csr_) {
            return csr_.n_rows;
        }
        if constexpr (has_csc_) {
            return csc_.n_rows;
        }
        return 0;
    }

    edge_t num_edges() const {
        if constexpr (has_csr_) {
            return csr_.nnz;
        }
        if constexpr (has_csc_) {
            return csc_.nnz;
        }
        return 0;
    }

    edge_t get_degree(vertex_t vid) const {
        if constexpr (has_csr_) {
            return csr_.row_offsets[vid + 1] - csr_.row_offsets[vid];
        } 
        if constexpr (has_csc_) {
            return csc_.col_offsets[vid + 1] - csc_.col_offsets[vid];
        }
        return 0;
    }

    arch_ref_t ToArch() const {
        arch_ref_t arch_ref;
        if constexpr (has_csr_) {
            arch_ref.num_vertices = csr_.n_rows;
            arch_ref.num_edges = csr_.nnz;
            arch_ref.row_offsets = csr_.row_offsets.data();
            arch_ref.col_indices = csr_.col_indices.data();
            arch_ref.csr_values = csr_.values.data();
        }
        if constexpr (has_csc_) {
            arch_ref.num_vertices = csc_.n_rows;
            arch_ref.num_edges = csc_.nnz;
            arch_ref.col_offsets = csc_.col_offsets.data();
            arch_ref.row_indices = csc_.row_indices.data();
            arch_ref.csc_values = csc_.values.data();           
        }
        if constexpr (!std::is_same_v<vprop_t, empty_t>) {
            arch_ref.vprops = vprops_.ToArch();
        }

        return arch_ref;
    }

    std::string ToString() const {
        std::string str;
        str += "Graph{ ";
        str += "arch_value:" + utils::ToString(arch_value) + ", ";
        if constexpr (has_csr_) {
            str += "csr_:" + csr_.ToString() + ", ";
        }
        if constexpr (has_csc_) {
            str += "csc_:" + csc_.ToString() + ", ";
        }
        if constexpr (!std::is_same_v<vprop_t, empty_t>) {
            str += "vprops_:";
            if constexpr (utils::HasToStrMethod<vprop_t>::value) {
                str += vprops_.ToString() + ", ";
            } else {
                str += "..., ";
            }
        }
        str += " }";
        return str;
    }

protected:
    csr_or_ept_t csr_;
    csc_or_ept_t csc_;

    vprop_t vprops_;
};

} // namespace graph_genlx