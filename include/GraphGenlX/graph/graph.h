#pragma once

#include "GraphGenlX/type.hpp"
#include "GraphGenlX/graph/graph_view.hpp"
#include "GraphGenlX/mat/csr.h"
#include "GraphGenlX/mat/csc.h"
#include "GraphGenlX/mat/builder.h"

namespace graph_genlx {

template <arch_t arch, graph_view_t views,
        //   typename v_prop_t, typename e_prop_t, typename g_prop_t,
          typename vertex_t, typename edge_t, typename weight_t>
class Graph {
protected:
    constexpr static arch_t arch_type_ = arch;

    constexpr static bool has_csr_ = has_view(views, graph_view_t::csr);
    constexpr static bool has_csc_ = has_view(views, graph_view_t::csc);

    using csr_t = CsrMat<arch, weight_t, vertex_t, edge_t>;
    using csc_t = CscMat<arch, weight_t, vertex_t, edge_t>;

    using csr_or_ept_t = std::conditional_t<has_csr_, csr_t, empty_t>;
    using csc_or_ept_t = std::conditional_t<has_csc_, csc_t, empty_t>;

public:
    template <typename = std::enable_if_t<has_csr_>>
    Graph(const CsrMat<arch, weight_t, vid_t, edge_t>& csr)
    : csr_(csr), csc_() {
        if constexpr (has_csc_) {
            csc_ = mat::ToCsc(csr);
        }
    }

    std::string ToString() const {
        std::string str;
        str += "Graph{ ";
        str += "arch_type_:" + utils::ArchToString(arch_type_) + ", ";
        if constexpr (has_csr_) {
            str += "csr_:" + csr_.ToString() + ", ";
        }
        if constexpr (has_csc_) {
            str += "csc_:" + csc_.ToString() + ", ";
        }
        str += " }";
        return str;
    }

protected:
    csr_or_ept_t csr_;
    csc_or_ept_t csc_;
};

} // namespace graph_genlx