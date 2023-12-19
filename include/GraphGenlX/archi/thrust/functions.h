#pragma once 

#include <thrust/fill.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include "GraphGenlX/archi/macro/macro.h"
#include "GraphGenlX/archi/thrust/def.hpp"

namespace graph_genlx::archi {

template <arch_t arch, typename ForwardIterator, typename T>
void fill(ForwardIterator first, ForwardIterator last, const T &value) {
    thrust::fill(exec_policy<arch>, first, last, value);
}

template <arch_t arch, typename InputIterator1, typename InputIterator2,
          typename InputIterator3, typename RandomAccessIterator>
void scatter_if(InputIterator1 first, InputIterator1 last, InputIterator2 map,
                InputIterator3 stencil, RandomAccessIterator output) {
    thrust::scatter_if(exec_policy<arch>, first, last, map, stencil, output);
}

template <arch_t arch, typename InputIterator, typename OutputIterator,
          typename AssociativeOperator>
OutputIterator inclusive_scan(InputIterator first, InputIterator last,
                              OutputIterator result,
                              AssociativeOperator binary_op) {
    return thrust::inclusive_scan(exec_policy<arch>, first, last, result,
                                  binary_op);
}

template <arch_t arch, typename ForwardIterator, typename InputIterator,
          typename OutputIterator>
OutputIterator lower_bound(ForwardIterator first, ForwardIterator last,
                           InputIterator values_first,
                           InputIterator values_last, OutputIterator output) {
    return thrust::lower_bound(exec_policy<arch>, first, last, values_first,
                               values_last, output);
}

template <arch_t arch, typename RandomAccessIterator1,
          typename RandomAccessIterator2>
void sort_by_key(RandomAccessIterator1 keys_first,
                 RandomAccessIterator1 keys_last,
                 RandomAccessIterator2 values_first) {
    thrust::sort_by_key(exec_policy<arch>, keys_first, keys_last, values_first);
}

template <arch_t arch, typename InputIterator, typename UnaryFunction,
          typename OutputType, typename BinaryFunction>
OutputType transform_reduce(InputIterator first, InputIterator last,
                            UnaryFunction unary_op, OutputType init,
                            BinaryFunction binary_op) {
    return thrust::transform_reduce(exec_policy<arch>, first, last, unary_op,
                                    init, binary_op);
}

template <arch_t arch, typename InputIterator, typename OutputIterator,
          typename UnaryFunction>
OutputIterator transform(InputIterator first, InputIterator last,
                         OutputIterator result, UnaryFunction op) {
    return thrust::transform(exec_policy<arch>, first, last, result, op);
}

} // namespace graph_genlx::archi