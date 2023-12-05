#pragma once

#include <vector>

#include "GraphGenlX/type.hpp"
#include "GraphGenlX/archi/def.hpp"
#include "GraphGenlX/archi/cpu.hpp"
#include "GraphGenlX/archi/cuda.cuh"

#define GENLX_DEV GENLX_CUDA
#define GENLX_ARCH GENLX_CPU_CUDA
#define GENLX_ARCH_INL GENLX_CPU_CUDA_INL

namespace graph_genlx {

namespace archi {

template <arch_t arch, typename value_t>
using vector_t = typename archi::Vector_t<arch>::template type<value_t>;

template<arch_t arch, typename value_t>
constexpr auto memalloc = archi::memalloc_t<arch>::template call<value_t>;

template<arch_t arch, typename value_t>
constexpr auto memfree = archi::memfree_t<arch>::template call<value_t>;

template<arch_t arch, typename value_t>
constexpr auto memset = archi::memset_t<arch>::template call<value_t>;

template<arch_t to_arch, arch_t from_arch, typename value_t>
constexpr auto memcpy = archi::memcpy_t<to_arch, from_arch>::template call<value_t>;

template<arch_t arch, typename value_t>
constexpr auto memfill = archi::memfill_t<arch>::template call<value_t>;

}   // namespace archi

template <arch_t arch, typename value_t>
using vector_t = archi::vector_t<arch, value_t>;

} // namespace graph_genlx