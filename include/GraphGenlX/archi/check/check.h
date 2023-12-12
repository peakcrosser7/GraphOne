#pragma once

#include "GraphGenlX/type.hpp"
#include "GraphGenlX/archi/check/cuda.cuh"

#define ARCH_DEBUG

namespace graph_genlx {

template <arch_t arch, typename T>
void checkArchErr(T val, char const *const func, const char *const file,
           int const line) {
#ifdef ARCH_DEBUG
    if constexpr (arch == arch_t::cuda) {
        CheckCudaErr(val, func, file, line);
    }
#endif
}

#define checkArchErrors(arch, val) checkArchErr<arch>(val, #val, __FILE__, __LINE__)

} // namespace graph_genlx