#pragma once 

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>

#include "GraphGenlX/type.hpp"

namespace graph_genlx {

namespace archi {

template <arch_t arch>
struct Vector_t {
    template<typename value_t>
    using type = std::vector<value_t>;    
};
template <arch_t arch, typename value_t>
using vector_t = typename archi::Vector_t<arch>::template type<value_t>;

template <arch_t arch>
struct ExecPolicy {};
template <arch_t arch>
constexpr auto exec_policy = archi::ExecPolicy<arch>::value;

template <typename Policy, typename ForwardIterator, typename T>
void fill(const Policy &exec, ForwardIterator first, ForwardIterator last,
          const T &value) {
    thrust::fill(exec, first, last, value);
}

template <typename Policy, typename InputIterator1, typename InputIterator2,
          typename InputIterator3, typename RandomAccessIterator>
void scatter_if(const Policy &exec, InputIterator1 first, InputIterator1 last,
                InputIterator2 map, InputIterator3 stencil,
                RandomAccessIterator output) {
    thrust::scatter_if(exec, first, last, map, stencil, output);
}

template <typename Policy, typename InputIterator, typename OutputIterator,
          typename AssociativeOperator>
OutputIterator inclusive_scan(const Policy &exec, InputIterator first,
                              InputIterator last, OutputIterator result,
                              AssociativeOperator binary_op) {
    return thrust::inclusive_scan(exec, first, last, result, binary_op);
}

template <typename Policy, typename ForwardIterator, typename InputIterator,
          typename OutputIterator>
OutputIterator lower_bound(const Policy &exec, ForwardIterator first,
                           ForwardIterator last, InputIterator values_first,
                           InputIterator values_last, OutputIterator output) {
    return thrust::lower_bound(exec, first, last, values_first, values_last,
                               output);
}

template <typename Policy, typename RandomAccessIterator1,
          typename RandomAccessIterator2>
void sort_by_key(const Policy &exec, RandomAccessIterator1 keys_first,
                 RandomAccessIterator1 keys_last,
                 RandomAccessIterator2 values_first) {
    thrust::sort_by_key(exec, keys_first, keys_last, values_first);
}

} // namespace archi

template <arch_t arch, typename value_t>
using vector_t = archi::vector_t<arch, value_t>;
    
} // namespace graph_genlx
