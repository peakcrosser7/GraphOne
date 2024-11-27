#pragma once

#include "GraphOne/utils/log.hpp"

namespace graph_one {

template <typename functor_t, typename comp_t, typename frontier_t>
void Run(comp_t& comp, frontier_t& frontier) {
    using engine_t =
        typename functor_t::engine_type<functor_t, comp_t, frontier_t>;

    LOG_DEBUG(">>>run App start...");
    engine_t engine(comp, frontier);

    comp.Init();
    LOG_DEBUG(">>>comp initialized");
    LOG_DEBUG("init ", frontier);

    while (comp.IsConvergent() == false && frontier.IsConvergent() == false) {

        comp.BeforeEngine();
        frontier.BeforeEngine();
        LOG_DEBUG(">>>before engine proc done");

        engine.Forward();
        LOG_DEBUG(">>>Forward done");
        LOG_DEBUG(frontier);

        frontier.AfterEngine();
        comp.AfterEngine();
        LOG_DEBUG(">>>after engine proc done\n");
    }

    LOG_DEBUG(">>>after convergent");
}

} // namespace graph_one