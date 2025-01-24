#pragma once

#include "graph_one/applier.hpp"

namespace graph_one {
    
template <typename construct_t, 
          typename gather_t,
          typename apply_t>
struct GraphFunctor {
    construct_t construct_op;
    gather_t gather_op;
    apply_t apply_func;
};


template <typename construct_t, 
          typename gather_t,
          typename apply_t = DummyApplier>
GraphFunctor<construct_t, gather_t, apply_t>
make_functor(construct_t construct_op, gather_t gather_op, apply_t apply_func = DummyApplier{}) {
    return GraphFunctor<construct_t, gather_t, apply_t>{construct_op, gather_op, apply_func};
}

} // namespace graph_one
