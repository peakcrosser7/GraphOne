#pragma once

#include "GraphOne/type.hpp"
#include "GraphOne/archi.h"
#include "GraphOne/graph/graph_view.hpp"
#include "GraphOne/graph/graph_arch_ref.h"
#include "GraphOne/mat/csr.h"
#include "GraphOne/mat/csc.h"
#include "GraphOne/mat/convert.h"

namespace graph_one {

// now Graph just only supports one format of matrix, i.e. csr
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

    constexpr static graph_view_t mat_view = get_first_view<views>();
    constexpr static bool both_mat = has_all_views(views, graph_view_t::normal, graph_view_t::transpose);

    using graph_mat_t = typename graph_mat<mat_view, arch, weight_t, vertex_t,
                                           edge_t, v_start>::type;
    using graph_tmat_t = std::conditional_t<both_mat, graph_mat_t, empty_t>;
    using arch_ref_t = GraphArchRef<views, arch, v_start, vprop_t, vertex_t, edge_t, weight_t>;

protected:
    static_assert(has_some_views(views, graph_view_t::normal, graph_view_t::transpose));

    constexpr static bool has_csr_ = has_view(views, graph_view_t::csr);

public:

    Graph(graph_mat_t&& mat, vprop_t&& vprops)
    : mat_(std::move(mat)), transpose_mat_(), vprops_(std::move(vprops)) {}

    Graph(graph_mat_t&& mat, graph_tmat_t&& transpose_mat, vprop_t&& vprops)
    : mat_(std::move(mat)), transpose_mat_(std::move(transpose_mat)),
      vprops_(std::move(vprops)) {}

    const vprop_t& vprops() const {
        return vprops_;
    }

    vertex_t num_vertices() const {
        return mat_.n_rows;
    }

    edge_t num_edges() const {
        return mat_.nnz;
    }

    template <graph_view_t Views = views>
    std::enable_if_t<has_csr_ && has_view(Views, graph_view_t::normal), vertex_t>
    get_out_degree(vertex_t vid) const {
        return mat_.row_offsets[vid + 1] - mat_.row_offsets[vid];
    }

    template <graph_view_t Views = views>
    std::enable_if_t<!has_csr_ || !has_view(Views, graph_view_t::normal), vertex_t>
    get_out_degree(vertex_t vid) = delete;

    template <graph_view_t Views = views>
    std::enable_if_t<has_csr_ && has_view(Views, graph_view_t::transpose), vertex_t>
    get_in_degree(vertex_t vid) const {
        if constexpr (both_mat) {
            return transpose_mat_.row_offsets[vid + 1] - transpose_mat_.row_offsets[vid];
        } else {
            return mat_.row_offsets[vid + 1] - mat_.row_offsets[vid];
        }
    }

    template <graph_view_t Views = views>
    std::enable_if_t<!has_csr_ || !has_view(Views, graph_view_t::transpose), vertex_t>
    get_in_degree(vertex_t vid) = delete;

    const graph_mat_t& graph_mat() const {
        return mat_;
    }

    const graph_tmat_t& graph_transpose_mat() const {
        return transpose_mat_;
    }

    arch_ref_t ToArch() const {
        arch_ref_t arch_ref;
        if constexpr (has_csr_) {
            arch_ref.num_vertices = mat_.n_rows;
            arch_ref.num_edges = mat_.nnz;
            arch_ref.row_offsets = mat_.row_offsets.data();
            arch_ref.col_indices = mat_.col_indices.data();
            arch_ref.values = mat_.values.data();
            if constexpr (both_mat) {
                arch_ref.trans_row_offsets = transpose_mat_.row_offsets.data();
                arch_ref.trans_col_indices = transpose_mat_.col_indices.data();
                arch_ref.trans_values = transpose_mat_.values.data();
            }
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
        str += "mat_:" + mat_.ToString() + ", ";
        if constexpr (both_mat) {
            str += "transpose_mat_:" + transpose_mat_.ToString() + ", ";
        }
        if constexpr (!std::is_same_v<vprop_t, empty_t>) {
            str += "vprops_:";
            if constexpr (utils::HasToStrMethod<vprop_t>::value) {
                str += vprops_.ToString();
            } else {
                str += "...";
            }
        }
        str += " }";
        return str;
    }

protected:
    graph_mat_t  mat_;
    graph_tmat_t transpose_mat_;

    vprop_t vprops_;
};

} // namespace graph_one