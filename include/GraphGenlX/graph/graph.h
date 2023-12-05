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

protected:
    constexpr static arch_t arch_value_ = arch;
    constexpr static vstart_t vstart_value = v_start;

    constexpr static bool has_csr_ = has_view(views, graph_view_t::csr);
    constexpr static bool has_csc_ = has_view(views, graph_view_t::csc);

    using csr_t = CsrMat<arch, weight_t, vertex_t, edge_t>;
    using csc_t = CscMat<arch, weight_t, vertex_t, edge_t>;

    using csr_or_ept_t = std::conditional_t<has_csr_, csr_t, empty_t>;
    using csc_or_ept_t = std::conditional_t<has_csc_, csc_t, empty_t>;

public:
    template <typename = std::enable_if_t<has_csr_>>
    Graph(const csr_t& csr, const vprop_t& vprops)
    : csr_(csr), csc_(), vprops_(vprops) {
        if constexpr (has_csc_) {
            csc_ = mat::ToCsc(csr);
        }
    }

    template <typename = std::enable_if_t<has_csr_>>
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
        return csc_.n_rows;
    }

    edge_t num_edges() const {
        if constexpr (has_csr_) {
            return csr_.nnz;
        }
        return csc_.nnz;
    }

    GENLX_ARCH_INL
    typename std::enable_if_t<has_csr_, edge_t>
    get_out_degree(vertex_t vid) const {
        return csr_.row_offsets[vid + 1] - csr_.row_offsets[vid];
    }

    GENLX_ARCH_INL
    typename std::enable_if_t<has_csc_, edge_t>
    get_in_degree(vertex_t vid) const {
        return csc_.col_offsets[vid + 1] - csc_.col_offsets[vid];
    }

    GENLX_ARCH_INL edge_t get_degree(vertex_t vid) const {
        if constexpr (has_csr_) {
            return get_out_degree(vid);
        } 
        if constexpr (has_csc_) {
            return get_in_degree(vid);
        }
        return 0;
    }

    std::string ToString() const {
        std::string str;
        str += "Graph{ ";
        str += "arch_value_:" + utils::ToString(arch_value_) + ", ";
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