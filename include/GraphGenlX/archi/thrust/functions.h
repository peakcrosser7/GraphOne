#pragma once

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include "GraphGenlX/archi/macro/macro.h"
#include "GraphGenlX/archi/check/thrust.hpp"
#include "GraphGenlX/archi/thrust/def.hpp"

namespace graph_genlx::archi {

template <arch_t arch, typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename Predicate>
OutputIterator copy_if(InputIterator1 first, InputIterator1 last,
                       InputIterator2 stencil, OutputIterator result,
                       Predicate pred) {
    return checkThrustErrors_with_ret(
        thrust::copy_if(exec_policy<arch>, first, last, stencil, result, pred));
}

template <arch_t arch, typename ForwardIterator, typename T>
void fill(ForwardIterator first, ForwardIterator last, const T &value) {
    checkThrustErrors_no_ret(
        thrust::fill(exec_policy<arch>, first, last, value));
}

template <arch_t arch, typename InputIterator, typename OutputIterator,
          typename AssociativeOperator>
OutputIterator inclusive_scan(InputIterator first, InputIterator last,
                              OutputIterator result,
                              AssociativeOperator binary_op) {
    return checkThrustErrors_with_ret(thrust::inclusive_scan(
        exec_policy<arch>, first, last, result, binary_op));
}

template <arch_t arch, typename ForwardIterator, typename InputIterator,
          typename OutputIterator>
OutputIterator lower_bound(ForwardIterator first, ForwardIterator last,
                           InputIterator values_first,
                           InputIterator values_last, OutputIterator output) {
    return checkThrustErrors_with_ret(thrust::lower_bound(
        exec_policy<arch>, first, last, values_first, values_last, output));
}

template <arch_t arch, typename ItemsIt, typename MapIt, typename ResultIt>
void scatter(ItemsIt first, ItemsIt last, MapIt map, ResultIt result) {
    checkThrustErrors_no_ret(
        thrust::scatter(exec_policy<arch>, first, last, map, result));
}

template <arch_t arch, typename InputIterator1, typename InputIterator2,
          typename InputIterator3, typename RandomAccessIterator>
void scatter_if(InputIterator1 first, InputIterator1 last, InputIterator2 map,
                InputIterator3 stencil, RandomAccessIterator output) {
    checkThrustErrors_no_ret(thrust::scatter_if(exec_policy<arch>, first, last,
                                                map, stencil, output));
}

template <arch_t arch, typename RandomAccessIterator1,
          typename RandomAccessIterator2>
void sort_by_key(RandomAccessIterator1 keys_first,
                 RandomAccessIterator1 keys_last,
                 RandomAccessIterator2 values_first) {
    checkThrustErrors_no_ret(thrust::sort_by_key(exec_policy<arch>, keys_first,
                                                 keys_last, values_first));
}

template <arch_t arch, typename InputIterator, typename OutputIterator,
          typename UnaryFunction>
OutputIterator transform(InputIterator first, InputIterator last,
                         OutputIterator result, UnaryFunction op) {
    return checkThrustErrors_with_ret(
        thrust::transform(exec_policy<arch>, first, last, result, op));
}

template <arch_t arch, typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename BinaryFunction>
OutputIterator transform(InputIterator1 first1, InputIterator1 last1,
                         InputIterator2 first2, OutputIterator result,
                         BinaryFunction op) {
    return checkThrustErrors_with_ret(thrust::transform(
        exec_policy<arch>, first1, last1, first2, result, op));
}

template <arch_t arch, typename InputIterator, typename UnaryFunction,
          typename OutputType, typename BinaryFunction>
OutputType transform_reduce(InputIterator first, InputIterator last,
                            UnaryFunction unary_op, OutputType init,
                            BinaryFunction binary_op) {
    return checkThrustErrors_with_ret(thrust::transform_reduce(
        exec_policy<arch>, first, last, unary_op, init, binary_op));
}


} // namespace graph_genlx::archi