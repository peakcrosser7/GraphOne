#pragma once

#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/binary_search.h>

#include "GraphGenlX/type.hpp"
#include "GraphGenlX/archi.h"
#include "GraphGenlX/buffer.h"

namespace graph_genlx::mat {

template <arch_t arch, typename index_t, typename offset_t>
Buffer<index_t, arch, offset_t> OffsetsToIndices(
    const Buffer<offset_t, arch, index_t>& offsets,
    offset_t indices_size
) {
    indices_size = max(indices_size, offset_t(offsets.size() - 1));
    Buffer<index_t, arch, offset_t> indices(indices_size);

    // 将偏移值散列到索引的最高位置
    // 如:offsets[0, 2, 2, 3, 5, 5, 5, 7, 8]得到indices[0, 0, 2, 3, 0, 6, 0, 7]
    thrust::scatter_if(
        archi::exec_policy<arch>,                                  // execution policy
        thrust::counting_iterator<offset_t>(0),       // begin iterator
        thrust::counting_iterator<offset_t>(indices_size - 1), // end iterator
        offsets.begin(),                              // where to scatter
        thrust::make_transform_iterator( // 用于判断相邻两个元素是否相等
            thrust::make_zip_iterator( // 用于遍历两个相邻元素
                thrust::make_tuple(offsets.begin(), offsets.begin() + 1)),
            [] CODE_CPU_CUDA (
                const thrust::tuple<offset_t, offset_t> &t) {
                thrust::not_equal_to<offset_t> comp;
                return comp(thrust::get<0>(t), thrust::get<1>(t));
            }),
        indices.begin());

    // 前缀最大值运算,将上述操作散列的一个索引值分散成多个,从而转换成索引
    // 如:indices[0, 0, 2, 3, 0, 6, 0, 7]得到indices[0, 0, 2, 3, 3, 6, 6, 7]
    thrust::inclusive_scan(archi::exec_policy<arch>, indices.begin(), indices.end(),
                           indices.begin(), thrust::maximum<offset_t>());
    
    return indices;
}

template <arch_t arch, typename index_t, typename offset_t>
Buffer<offset_t, arch, index_t> IndicesToOffsets(
    const Buffer<index_t, arch, offset_t>& indices,
    index_t offsets_size
) {
    Buffer<offset_t, arch, index_t> offsets(offsets_size);
    // convert uncompressed indices into compressed offsets
    // 如:indices[0, 0, 2, 3, 3, 6, 6, 7]得到offsets[0, 2, 2, 3, 5, 5, 5, 7, 8]
    thrust::lower_bound(archi::exec_policy<arch>, indices.begin(), indices.end(),
                        thrust::counting_iterator<offset_t>(0),
                        thrust::counting_iterator<offset_t>(offsets_size),
                        offsets.begin());
    return offsets;
}
    
} // namespace graph_genlx