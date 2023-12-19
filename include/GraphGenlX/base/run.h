#pragma once

#include "GraphGenlX/utils/log.hpp"

namespace graph_genlx {

template <typename factor_t, typename comp_t>
void Run(comp_t& comp) {
    using engine_t = typename factor_t::engine_type;

    LOG_DEBUG(">>>run App start...");
    comp.Init();
    LOG_DEBUG(">>>comp initialized");
    while (!comp.IsConvergent()) {
        LOG_DEBUG(">>>iter:", comp.d_status.iter);
        comp.BeforeEngine();
        LOG_DEBUG(">>>comp before engine proc done");
        engine_t::template Forward<factor_t>(comp);
        LOG_DEBUG(">>>comp Forward done");
        comp.AfterEngine();
        LOG_DEBUG(">>>comp after engine proc done\n");
    }
}

} // namespace graph_genlx