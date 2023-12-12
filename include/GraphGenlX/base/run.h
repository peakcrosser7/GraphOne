#pragma once 

namespace graph_genlx {

template <typename comp_t, typename factor_t>
void Run(comp_t& comp, factor_t& factor) {
    using engine_t = typename factor_t::engine_type;

    comp.Init();
    while (!comp.IsConvergent()) {
        comp.BeforeEngine();
        engine_t::Forward(comp, factor);
    }
}

} // namespace graph_genlx