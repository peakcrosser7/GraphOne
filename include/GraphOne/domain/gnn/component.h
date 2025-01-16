#pragma once 

#include "GraphOne/type.hpp"
#include "GraphOne/base/component.h"

namespace graph_one::gnn {

template <typename graph_t,
          typename hstatus_t,
          typename dstatus_t = empty_t>
struct Componment;

template <typename graph_t,
          typename hstatus_t,
          typename dstatus_t>
struct Componment : ComponentX<graph_t, hstatus_t, dstatus_t>{
    Componment(graph_t& g, hstatus_t& h_status, dstatus_t& d_status) 
            : ComponentX<graph_t, hstatus_t, dstatus_t>(g, h_status, d_status) {}

    using dim_type = typename hstatus_t::dim_type;

    dim_type feat_dim() const {
        return this->h_status.feat_dim();
    }

};

template <typename graph_t, typename hstatus_t>
struct Componment<graph_t, hstatus_t, empty_t> : ComponentX<graph_t, hstatus_t, empty_t>{
    Componment(graph_t& g, hstatus_t& h_status) 
            : ComponentX<graph_t, hstatus_t, empty_t>(g, h_status, empty_t{}) {}

    using dim_type = typename hstatus_t::dim_type;

    dim_type feat_dim() const {
        return this->h_status.feat_dim();
    }

};

}   // namespace graph_one::gnn