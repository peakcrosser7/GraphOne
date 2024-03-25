#pragma once

#include "GraphOne/type.hpp"

namespace graph_one::archi {

template <arch_t arch>
struct LaunchTparams {};

template <arch_t arch>
struct LaunchParams {};

template <arch_t arch>
struct Launcher {};

template <arch_t arch, typename tparams, typename func_t, typename... args_t>
typename Launcher<arch>::err_t LaunchKernel(const LaunchParams<arch> &params,
                                            func_t &f, args_t &&...args) {
    using tparam_spec_t = std::conditional_t<std::is_same_v<tparams, empty_t>,
                                             LaunchTparams<arch>, tparams>;
    return Launcher<arch>::template launch<tparam_spec_t>(
        params, f, std::forward<args_t>(args)...);
}

template <arch_t arch, typename... args_t>
typename Launcher<arch>::err_t LaunchSync(args_t &&...args) {
    return Launcher<arch>::sync(std::forward<args_t>(args)...);
}

} // namespace graph_one::archi