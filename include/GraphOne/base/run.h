#pragma once

#include "GraphOne/utils/log.hpp"
#include "GraphOne/domain/gnn/module_holder.h"

namespace graph_one {

template <typename functor_t, typename comp_t, typename frontier_t>
std::enable_if_t<!gnn::is_module_holder<frontier_t>::value>
Run(comp_t& comp, frontier_t& frontier) {
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
    comp.Final();
    
}


template <typename comp_t, typename module_t>
std::enable_if_t<gnn::is_module_holder<module_t>::value>
Run(comp_t& comp, module_t& module) {
    comp.Init();
    LOG_DEBUG(">>>comp initialized");

    module->forward(comp);
    LOG_DEBUG(">>>Forward done");

    comp.Final();
    LOG_DEBUG(">>>after Final");
}

} // namespace graph_one