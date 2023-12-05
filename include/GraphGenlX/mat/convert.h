#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "GraphGenlX/type.hpp"
#include "GraphGenlX/archi.h"
#include "GraphGenlX/utils.h"
#include "GraphGenlX/base/buffer.h"
#include "GraphGenlX/mat/csr.h"
#include "GraphGenlX/mat/csc.h"

namespace graph_genlx::mat {

template <arch_t arch, typename index_t, typename offset_t>
Buffer<arch, index_t, offset_t> OffsetsToIndices(
    const Buffer<arch, offset_t, index_t>& offsets,
    offset_t indices_size
) {
    indices_size = max(indices_size, offset_t(offsets.size() - 1));
    Buffer<arch, index_t, offset_t> indices(indices_size);

    // 将偏移值散列到索引的最高位置
    // 如:offsets[0, 2, 2, 3, 5, 5, 5, 7, 8]得到indices[0, 0, 2, 3, 0, 6, 0, 7]
    archi::scatter_if<arch>(
        thrust::counting_iterator<offset_t>(0),       // begin iterator
        thrust::counting_iterator<offset_t>(indices_size - 1), // end iterator
        offsets.begin(),                              // where to scatter
        thrust::make_transform_iterator( // 用于判断相邻两个元素是否相等
            thrust::make_zip_iterator( // 用于遍历两个相邻元素
                thrust::make_tuple(offsets.begin(), offsets.begin() + 1)),
            [] __GENLX_ARCH__ (
                const thrust::tuple<offset_t, offset_t> &t) {
                thrust::not_equal_to<offset_t> comp;
                return comp(thrust::get<0>(t), thrust::get<1>(t));
            }),
        indices.begin());

    // 前缀最大值运算,将上述操作散列的一个索引值分散成多个,从而转换成索引
    // 如:indices[0, 0, 2, 3, 0, 6, 0, 7]得到indices[0, 0, 2, 3, 3, 6, 6, 7]
    archi::inclusive_scan<arch>(indices.begin(), indices.end(), indices.begin(),
                                thrust::maximum<offset_t>());

    return indices;
}

template <arch_t arch, typename index_t, typename offset_t>
Buffer<arch, offset_t, index_t> IndicesToOffsets(
    const Buffer<arch, index_t, offset_t>& indices,
    index_t offsets_size
) {
    Buffer<arch, offset_t, index_t> offsets(offsets_size);
    // convert uncompressed indices into compressed offsets
    // 如:indices[0, 0, 2, 3, 3, 6, 6, 7]得到offsets[0, 2, 2, 3, 5, 5, 5, 7, 8]
    archi::lower_bound<arch>(
        indices.begin(), indices.end(), thrust::counting_iterator<offset_t>(0),
        thrust::counting_iterator<offset_t>(offsets_size), offsets.begin());
    return offsets;
}

template <arch_t arch,
        typename value_t,
        typename index_t,
        typename offset_t>
CscMat<arch, value_t, index_t, offset_t>
ToCsc(const CsrMat<arch, value_t, index_t, offset_t>& csr) {
    Buffer<arch, index_t, offset_t> row_indices = 
        mat::OffsetsToIndices(csr.row_offsets, csr.nnz);
    Buffer<arch, index_t, offset_t> col_indices = csr.col_indices;
    Buffer<arch, value_t, offset_t> values = csr.values;

    auto zip_it = thrust::make_zip_iterator(
        thrust::make_tuple(row_indices.begin(), values.begin())
    );
    archi::sort_by_key<arch>(col_indices.begin(), col_indices.end(), zip_it);

    Buffer<arch, offset_t, index_t> col_offsets = 
        mat::IndicesToOffsets(col_indices, csr.n_cols + 1);

    return CscMat<arch, value_t, index_t, offset_t>(
        csr.n_rows, csr.n_cols, csr.nnz,
        std::move(col_offsets), 
        std::move(row_indices),
        std::move(values)
    );
}
    
} // namespace graph_genlx