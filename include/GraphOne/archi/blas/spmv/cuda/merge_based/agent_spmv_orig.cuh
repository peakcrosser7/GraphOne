/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * cub::AgentSpmv implements a stateful abstraction of CUDA thread blocks for participating in device-wide SpMV.
 */

#pragma once

#include <iterator>

#include <cub/util_type.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_exchange.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/tex_ref_input_iterator.cuh>

#include "thread_search.cuh"


namespace graph_one::blas::merge {

// /// Optional outer namespace(s)
// CUB_NS_PREFIX

// /// CUB namespace
// namespace cub {


/******************************************************************************
 * Tuning policy
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for AgentSpmv
 */
template <
    int                             _BLOCK_THREADS,                         ///< Threads per thread block
    int                             _ITEMS_PER_THREAD,                      ///< Items per thread (per tile of input)
    cub::CacheLoadModifier          _ROW_OFFSETS_SEARCH_LOAD_MODIFIER,      ///< Cache load modifier for reading CSR row-offsets during search
    cub::CacheLoadModifier          _ROW_OFFSETS_LOAD_MODIFIER,             ///< Cache load modifier for reading CSR row-offsets
    cub::CacheLoadModifier          _COLUMN_INDICES_LOAD_MODIFIER,          ///< Cache load modifier for reading CSR column-indices
    cub::CacheLoadModifier          _VALUES_LOAD_MODIFIER,                  ///< Cache load modifier for reading CSR values
    cub::CacheLoadModifier          _VECTOR_VALUES_LOAD_MODIFIER,           ///< Cache load modifier for reading vector values
    bool                            _DIRECT_LOAD_NONZEROS,                  ///< Whether to load nonzeros directly from global during sequential merging (vs. pre-staged through shared memory)
    cub::BlockScanAlgorithm         _SCAN_ALGORITHM>                        ///< The BlockScan algorithm to use
struct AgentSpmvPolicy {
    enum 
    {
        BLOCK_THREADS                                                   = _BLOCK_THREADS,                       ///< Threads per thread block
        ITEMS_PER_THREAD                                                = _ITEMS_PER_THREAD,                    ///< Items per thread (per tile of input)
        DIRECT_LOAD_NONZEROS                                            = _DIRECT_LOAD_NONZEROS,                ///< Whether to load nonzeros directly from global during sequential merging (pre-staged through shared memory)
    };

    static const cub::CacheLoadModifier  ROW_OFFSETS_SEARCH_LOAD_MODIFIER    = _ROW_OFFSETS_SEARCH_LOAD_MODIFIER;    ///< Cache load modifier for reading CSR row-offsets
    static const cub::CacheLoadModifier  ROW_OFFSETS_LOAD_MODIFIER           = _ROW_OFFSETS_LOAD_MODIFIER;           ///< Cache load modifier for reading CSR row-offsets
    static const cub::CacheLoadModifier  COLUMN_INDICES_LOAD_MODIFIER        = _COLUMN_INDICES_LOAD_MODIFIER;        ///< Cache load modifier for reading CSR column-indices
    static const cub::CacheLoadModifier  VALUES_LOAD_MODIFIER                = _VALUES_LOAD_MODIFIER;                ///< Cache load modifier for reading CSR values
    static const cub::CacheLoadModifier  VECTOR_VALUES_LOAD_MODIFIER         = _VECTOR_VALUES_LOAD_MODIFIER;         ///< Cache load modifier for reading vector values
    static const cub::BlockScanAlgorithm SCAN_ALGORITHM                      = _SCAN_ALGORITHM;                      ///< The BlockScan algorithm to use

};


/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

template <
    typename        index_t,              ///< Matrix and vector value type
    typename        offset_t,             ///< Signed integer type for sequence offsets
    typename        mat_value_t,
    typename        vec_x_value_t,
    typename        vec_y_value_t>
struct SpmvCsrParams {
    mat_value_t*         d_values;            ///< Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
    offset_t*            d_row_end_offsets;   ///< Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
    index_t*             d_column_indices;    ///< Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
    vec_x_value_t*       d_vector_x;          ///< Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    vec_y_value_t*       d_vector_y;          ///< Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
    index_t              num_rows;            ///< Number of rows of matrix <b>A</b>.
    index_t              num_cols;            ///< Number of columns of matrix <b>A</b>.
    offset_t             num_nonzeros;        ///< Number of nonzero elements of matrix <b>A</b>.
};

template <typename functor_t, typename T>
struct ReduceWrapper {
    __host__ __device__ __forceinline__ 
    T operator()(const T &a, const T &b) const {
        return functor_t::reduce(a, b);
    }
};

/**
 * \brief AgentSpmv implements a stateful abstraction of CUDA thread blocks for participating in device-wide SpMV.
 */
template <
    typename    AgentSpmvPolicyT,           ///< Parameterized AgentSpmvPolicy tuning policy type
    typename        index_t,
    typename        offset_t,
    typename        mat_value_t,
    typename        vec_x_value_t,
    typename        vec_y_value_t,
    typename        functor_t,
    int             PTX_ARCH = CUB_PTX_ARCH>    ///< PTX compute capability
struct AgentSpmv
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    /// Constants
    enum
    {
        BLOCK_THREADS           = AgentSpmvPolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD        = AgentSpmvPolicyT::ITEMS_PER_THREAD,
        TILE_ITEMS              = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    /// 2D merge path coordinate type
    using CoordinateT = typename cub::CubVector<offset_t, 2>::Type;

    /// Input iterator wrapper types (for applying cache modifiers)

    using RowOffsetsSearchIteratorT = cub::CacheModifiedInputIterator<
            AgentSpmvPolicyT::ROW_OFFSETS_SEARCH_LOAD_MODIFIER,
            offset_t,
            offset_t>;

    using RowOffsetsIteratorT = cub::CacheModifiedInputIterator<
            AgentSpmvPolicyT::ROW_OFFSETS_LOAD_MODIFIER,
            offset_t,
            offset_t>;

    using ColumnIndicesIteratorT = cub::CacheModifiedInputIterator<
            AgentSpmvPolicyT::COLUMN_INDICES_LOAD_MODIFIER,
            index_t,
            index_t>;

    using MatValueIteratorT = cub::CacheModifiedInputIterator<
            AgentSpmvPolicyT::VALUES_LOAD_MODIFIER,
            mat_value_t,
            offset_t>;

    using VectorXValueIteratorT = cub::CacheModifiedInputIterator<
            AgentSpmvPolicyT::VECTOR_VALUES_LOAD_MODIFIER,
            vec_x_value_t,
            index_t>;

    using VectorYValueIteratorT = cub::CacheModifiedInputIterator<
            AgentSpmvPolicyT::VECTOR_VALUES_LOAD_MODIFIER,
            vec_y_value_t,
            index_t>;

    // Tuple type for scanning (pairs accumulated segment-value with segment-index)
    using KeyValuePairT = cub::KeyValuePair<offset_t, vec_y_value_t>;

    // Reduce-value-by-segment scan operator
    using ReduceBySegmentOpT = cub::ReduceByKeyOp<ReduceWrapper<functor_t, vec_y_value_t>>;

    // BlockReduce specialization
    using BlockReduceT = cub::BlockReduce<
            vec_y_value_t,
            BLOCK_THREADS,
            cub::BLOCK_REDUCE_WARP_REDUCTIONS>;

    // BlockScan specialization
    using BlockScanT = cub::BlockScan<
            KeyValuePairT,
            BLOCK_THREADS,
            AgentSpmvPolicyT::SCAN_ALGORITHM>;

    // BlockScan specialization
    using BlockPrefixSumT = cub::BlockScan<
            vec_y_value_t,
            BLOCK_THREADS,
            AgentSpmvPolicyT::SCAN_ALGORITHM>;

    // BlockExchange specialization
    using BlockExchangeT = cub::BlockExchange<
            vec_y_value_t,
            BLOCK_THREADS,
            ITEMS_PER_THREAD>;

    /// Merge item type (either a non-zero value or a row-end offset)
    union MergeItem
    {
        // Value type to pair with index type OffsetT (NullType if loading values directly during merge)
        typedef typename std::conditional_t<AgentSpmvPolicyT::DIRECT_LOAD_NONZEROS, cub::NullType, vec_y_value_t> MergeValueT;

        offset_t     row_end_offset;
        MergeValueT  nonzero;
    };

    /// Shared memory type required by this thread block
    struct _TempStorage
    {
        CoordinateT tile_coords[2];

        union
        {
            // Smem needed for tile of merge items
            MergeItem merge_items[ITEMS_PER_THREAD + TILE_ITEMS + 1];

            // Smem needed for block exchange
            typename BlockExchangeT::TempStorage exchange;

            // Smem needed for block-wide reduction
            typename BlockReduceT::TempStorage reduce;

            // Smem needed for tile scanning
            typename BlockScanT::TempStorage scan;

            // Smem needed for tile prefix sum
            typename BlockPrefixSumT::TempStorage prefix_sum;
        };
    };

    /// Temporary storage type (unionable)
    struct TempStorage : cub::Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------


    _TempStorage&                   temp_storage;         /// Reference to temp_storage

    SpmvCsrParams<index_t, 
               offset_t, 
               mat_value_t, 
               vec_x_value_t,
               vec_y_value_t>&          spmv_params;

    MatValueIteratorT               wd_values;            ///< Wrapped pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
    RowOffsetsIteratorT             wd_row_end_offsets;   ///< Wrapped Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
    ColumnIndicesIteratorT          wd_column_indices;    ///< Wrapped Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
    VectorXValueIteratorT           wd_vector_x;          ///< Wrapped Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    VectorYValueIteratorT           wd_vector_y;          ///< Wrapped Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>


    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__ AgentSpmv(
        TempStorage&                    temp_storage,           ///< Reference to temp_storage
        SpmvCsrParams<index_t, 
               offset_t, 
               mat_value_t, 
               vec_x_value_t,
               vec_y_value_t>&              spmv_params)            ///< SpMV input parameter bundle
    :
        temp_storage(temp_storage.Alias()),
        spmv_params(spmv_params),
        wd_values(spmv_params.d_values),
        wd_row_end_offsets(spmv_params.d_row_end_offsets),
        wd_column_indices(spmv_params.d_column_indices),
        wd_vector_x(spmv_params.d_vector_x),
        wd_vector_y(spmv_params.d_vector_y)
    {}




    /**
     * Consume a merge tile, specialized for direct-load of nonzeros
     * 
     * 消耗合并路径的线程块分片. 直接从全局内存加载非零元
     */
    __device__ __forceinline__ KeyValuePairT ConsumeTile(
        offset_t             tile_idx,
        CoordinateT          tile_start_coord,
        CoordinateT          tile_end_coord,
        cub::Int2Type<true>  is_direct_load)     ///< Marker type indicating whether to load nonzeros directly during path-discovery or beforehand in batch
    {
        // 线程块分片包含的行数
        offset_t         tile_num_rows           = tile_end_coord.x - tile_start_coord.x;
        // 线程块分片包含的非零元数
        offset_t         tile_num_nonzeros       = tile_end_coord.y - tile_start_coord.y;
        // 记录线程块分片每行的终止行偏移的共享内存指针
        offset_t*        s_tile_row_end_offsets  = &temp_storage.merge_items[0].row_end_offset;

        // Gather the row end-offsets for the merge tile into shared memory
        // 将线程块分片每行的终止偏移写入共享内存
        for (int item = threadIdx.x; item < tile_num_rows + ITEMS_PER_THREAD; item += BLOCK_THREADS) {
            const offset_t offset =
              (cub::min)(static_cast<offset_t>(tile_start_coord.x + item),
                         static_cast<offset_t>(spmv_params.num_rows - 1));
            s_tile_row_end_offsets[item] = wd_row_end_offsets[offset];
        }

        __syncthreads();

        // Search for the thread's starting coordinate within the merge tile
        cub::CountingInputIterator<offset_t>  tile_nonzero_indices(tile_start_coord.y);
        CoordinateT                           thread_start_coord;

        // 搜索每个线程分片合并路径的起始坐标
        SearchMergePath(
            offset_t(threadIdx.x * ITEMS_PER_THREAD),    // Diagonal
            s_tile_row_end_offsets,                     // List A
            tile_nonzero_indices,                       // List B
            tile_num_rows,
            tile_num_nonzeros,
            thread_start_coord);

        __syncthreads();            // Perf-sync

        // Compute the thread's merge path segment
        // 线程合并路径的当前坐标
        CoordinateT             thread_current_coord = thread_start_coord;
        // 线程分片每个合并路径元素的(行偏移,点乘结果)键值对
        KeyValuePairT           scan_segment[ITEMS_PER_THREAD];
        // 非零元点乘结果的累加值
        vec_y_value_t           running_total = functor_t::initialize();

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
            // 非零元索引
            offset_t       nonzero_idx         = CUB_MIN(tile_nonzero_indices[thread_current_coord.y], spmv_params.num_nonzeros - 1);
            // 非零元列索引
            index_t        column_idx          = wd_column_indices[nonzero_idx];
            // 非零元值
            mat_value_t    value               = wd_values[nonzero_idx];

            // 向量X对应的元素
            vec_x_value_t  vector_value        = wd_vector_x[column_idx];

            // 非零元点乘结果
            vec_y_value_t  nonzero             = functor_t::combine(value, vector_value);

            offset_t       row_end_offset      = s_tile_row_end_offsets[thread_current_coord.x];

            if (tile_nonzero_indices[thread_current_coord.y] < row_end_offset) {
                // Move down (accumulate)
                // 列索引更小,消耗列索引数组(y轴)

                // 累加矩阵该行的非零元点乘结果
                running_total = functor_t::reduce(running_total, nonzero);
                // 合并路径元素的value为非零元累加结果
                scan_segment[ITEM].value    = running_total;
                // 合并路径元素的key为线程块分片行数
                scan_segment[ITEM].key      = tile_num_rows;
                ++thread_current_coord.y;
            } else {
                // Move right (reset)
                // 行偏移更小,消耗行偏移数组(x轴)

                // 合并路径元素的value为非零元累加结果
                scan_segment[ITEM].value    = running_total;
                // 合并路径元素的key为行偏移(矩阵该行最后一个非零元)
                scan_segment[ITEM].key      = thread_current_coord.x;
                // 重置矩阵该行的累加结果
                running_total               = functor_t::initialize();
                ++thread_current_coord.x;
            }
        }

        __syncthreads();

        // Block-wide reduce-value-by-segment

        // 线程块分片最后一行的累加结果
        KeyValuePairT       tile_carry;
        // 前缀和操作符
        ReduceBySegmentOpT  scan_op;
        // 当前线程分片的行偏移和累加结果 --求前缀和--> 上一线程分片最后一行的行偏移和累加结果
        KeyValuePairT       scan_item;

        // 赋值为线程分片最后一行的累加结果和行偏移
        scan_item.value = running_total;
        scan_item.key   = thread_current_coord.x;

        // 线程块级别求非包含前缀和(对每个线程分片的矩阵最后一行求非包含前缀和)
        // 每个线程对应的前缀和(上一线程分片最后一行的累加结果)写入scan_item
        // 线程块的总和(线程块分片最后一行的累加结果)写入tile_carry
        BlockScanT(temp_storage.scan).ExclusiveScan(scan_item, scan_item, scan_op, tile_carry);

        // 线程块分片包含多行
        if (tile_num_rows > 0) {
            // 线程0的线程分片行偏移置为-1
            if (threadIdx.x == 0) {
                scan_item.key = static_cast<index_t>(-1);
            }

            // Direct scatter
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
                // 表明元素对应矩阵该行的最后一个元素
                if (scan_segment[ITEM].key < tile_num_rows) {
                    // 上一线程的线程分片的最后一行与该线程的线程分片的首行相同(有跨线程分片的矩阵行)
                    if (scan_item.key == scan_segment[ITEM].key) {
                        // 将上一分片该行的累加结果记录到当前元素中
                        // 当前元素一定是该行的最后一个元素,因此只会累加一次
                        scan_segment[ITEM].value = functor_t::reduce(scan_item.value, scan_segment[ITEM].value);
                    }

                    // if (HAS_ALPHA) {
                    //     scan_segment[ITEM].value *= spmv_params.alpha;
                    // }

                    // if (HAS_BETA) {
                    //     // Update the output vector element
                    //     ValueT addend = spmv_params.beta * wd_vector_y[tile_start_coord.x + scan_segment[ITEM].key];
                    //     scan_segment[ITEM].value += addend;
                    // }

                    // Set the output vector element
                    // 将结果直接写回vector Y的全局内存中
                    spmv_params.d_vector_y[tile_start_coord.x + scan_segment[ITEM].key] = scan_segment[ITEM].value;
                }
            }
        }

        // Return the tile's running carry-out
        // 线程块分片最后一行的行偏移和累加结果(可能跨线程块分片的矩阵行的累加结果)
        return tile_carry;
    }



    /**
     * Consume a merge tile, specialized for indirect load of nonzeros
     * 
     * 消耗合并路径的线程块分片. 不直接从全局内存(从共享内存)加载非零元
     */
    __device__ __forceinline__ KeyValuePairT ConsumeTile(
        offset_t             tile_idx,
        CoordinateT          tile_start_coord,
        CoordinateT          tile_end_coord,
        cub::Int2Type<false> is_direct_load)     ///< Marker type indicating whether to load nonzeros directly during path-discovery or beforehand in batch
    {
        // 线程块分片包含的行数
        offset_t         tile_num_rows           = tile_end_coord.x - tile_start_coord.x;
        // 线程块分片包含的非零元数
        offset_t         tile_num_nonzeros       = tile_end_coord.y - tile_start_coord.y;

        // printf("tile_num_rows=%u, tile_num_nonzeros=%u \n", tile_num_rows, tile_num_nonzeros);
#if (CUB_PTX_ARCH >= 520)

        // 记录线程块分片每行的终止行偏移的共享内存指针
        offset_t*       s_tile_row_end_offsets  = &temp_storage.merge_items[0].row_end_offset;
        // 记录非零元相乘的中间结果的共享内存指针
        vec_y_value_t*  s_tile_nonzeros         = &temp_storage.merge_items[tile_num_rows + ITEMS_PER_THREAD].nonzero;

        // Gather the nonzeros for the merge tile into shared memory
        // 每个线程处理ITEMS_PER_THREAD个非零元
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
            // 当前处理的非零元相对当前线程块分片的起始非零元的索引
            int nonzero_idx = threadIdx.x + (ITEM * BLOCK_THREADS);

            // 使用CacheModifiedInputIterator迭代器以使用PTX缓存加载修饰符优化访存
            // 非零元值的迭代器
            MatValueIteratorT a                = wd_values + tile_start_coord.y + nonzero_idx;
            // 非零元列索引的迭代器
            ColumnIndicesIteratorT ci          = wd_column_indices + tile_start_coord.y + nonzero_idx;
            // 记录非零元点乘结果的迭代器
            vec_y_value_t* s                   = s_tile_nonzeros + nonzero_idx;

            if (nonzero_idx < tile_num_nonzeros) {

                index_t        column_idx              = *ci;  // 非零元的列索引
                mat_value_t    value                   = *a;   // 非零元的值

                // 向量X对应的元素
                vec_x_value_t  vector_value            = wd_vector_x[column_idx];
                // 计算SpMV的元素相乘
                vec_y_value_t  nonzero                 = functor_t::combine(value, vector_value);

                // 记录非零元点乘结果到共享内存
                *s    = nonzero;
            }
        }

#else

        offset_t*          s_tile_row_end_offsets  = &temp_storage.merge_items[0].row_end_offset;
        vec_y_value_t*     s_tile_nonzeros         = &temp_storage.merge_items[tile_num_rows + ITEMS_PER_THREAD].nonzero;

        // Gather the nonzeros for the merge tile into shared memory
        if (tile_num_nonzeros > 0) {
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
                int     nonzero_idx             = threadIdx.x + (ITEM * BLOCK_THREADS);
                // 相当于 `if (nonzero_idx < tile_num_nonzeros)`
                nonzero_idx                          = CUB_MIN(nonzero_idx, tile_num_nonzeros - 1);

                index_t        column_idx            = wd_column_indices[tile_start_coord.y + nonzero_idx];
                mat_value_t    value                 = wd_values[tile_start_coord.y + nonzero_idx];

                vec_x_value_t  vector_value          = wd_vector_x[column_idx];

                // 计算SpMV非零元相乘
                vec_y_value_t  nonzero               = functor_t::combine(value, vector_value);

                // 记录非零元点乘结果到共享内存
                s_tile_nonzeros[nonzero_idx]         = nonzero;
            }
        }

#endif

        // Gather the row end-offsets for the merge tile into shared memory
        // 将线程块分片每行的终止偏移写入共享内存
        #pragma unroll 1    // 使用1表示循环展开1次,即不展开
        for (int item = threadIdx.x; item < tile_num_rows + ITEMS_PER_THREAD; item += BLOCK_THREADS) {
            const offset_t offset =
              (cub::min)(static_cast<offset_t>(tile_start_coord.x + item),
                         static_cast<offset_t>(spmv_params.num_rows - 1));
            s_tile_row_end_offsets[item] = wd_row_end_offsets[offset];
        }

        __syncthreads();

        // Search for the thread's starting coordinate within the merge tile
        cub::CountingInputIterator<offset_t>  tile_nonzero_indices(tile_start_coord.y);
        CoordinateT                           thread_start_coord;

        // 搜索每个线程分片合并路径的起始坐标
        SearchMergePath(
            offset_t(threadIdx.x * ITEMS_PER_THREAD),    // Diagonal
            s_tile_row_end_offsets,                      // List A
            tile_nonzero_indices,                        // List B
            offset_t(tile_num_rows),
            tile_num_nonzeros,
            thread_start_coord);

        __syncthreads();            // Perf-sync

        // Compute the thread's merge path segment
        // 线程合并路径的当前坐标
        CoordinateT     thread_current_coord = thread_start_coord;
        // 线程分片每个合并路径元素的(行偏移,点乘结果)键值对
        KeyValuePairT   scan_segment[ITEMS_PER_THREAD];
        // 非零元点乘结果的累加值
        vec_y_value_t   running_total = functor_t::initialize();

        // 线程合并路径当前坐标的终止行偏移
        offset_t       row_end_offset  = s_tile_row_end_offsets[thread_current_coord.x];
        // 线程合并路径当前坐标的非零元点乘结果
        vec_y_value_t  nonzero         = s_tile_nonzeros[thread_current_coord.y];

        // 每个线程遍历线程分片连续的ITEMS_PER_THREAD个合并路径的元素(行偏移或非零元)
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
            if (tile_nonzero_indices[thread_current_coord.y] < row_end_offset) {
                // Move down (accumulate)
                // 列索引更小,消耗列索引数组(y轴)

                // 合并路径元素的value为非零元点乘结果
                scan_segment[ITEM].value    = nonzero;
                // 累加矩阵该行的非零元点乘结果
                running_total               = functor_t::reduce(running_total, nonzero);
                ++thread_current_coord.y;
                // 更新非零元点乘结果
                nonzero                     = s_tile_nonzeros[thread_current_coord.y];
            } else {
                // Move right (reset)
                // 行偏移更小,消耗行偏移数组(x轴)

                // 合并路径元素的value为0
                scan_segment[ITEM].value    = functor_t::initialize();
                // 重置矩阵该行的累加结果
                running_total               = functor_t::initialize();
                ++thread_current_coord.x;
                // 更新终止行偏移
                row_end_offset              = s_tile_row_end_offsets[thread_current_coord.x];
            }

            // 合并路径元素的key为行偏移
            scan_segment[ITEM].key = thread_current_coord.x;
        }

        __syncthreads();

        // Block-wide reduce-value-by-segment

        // 线程块分片最后一行的累加结果
        KeyValuePairT       tile_carry;
        // 前缀和操作符
        ReduceBySegmentOpT  scan_op;
        // 当前线程分片的行偏移和累加结果 --求前缀和--> 上一线程分片最后一行的行偏移和累加结果
        KeyValuePairT       scan_item;

        // 赋值为线程分片最后一行的累加结果和行偏移
        scan_item.value = running_total;
        scan_item.key   = thread_current_coord.x;

        // 线程块级别求非包含前缀和(对每个线程分片的矩阵最后一行求非包含前缀和)
        // 每个线程对应的前缀和(上一线程分片最后一行的累加结果)写入scan_item
        // 线程块的总和(线程块分片最后一行的累加结果)写入tile_carry
        BlockScanT(temp_storage.scan).ExclusiveScan(scan_item, scan_item, scan_op, tile_carry);

        // 线程0的前缀和(上一线程分片)行偏移和累加结果重置
        if (threadIdx.x == 0) {
            scan_item.key = thread_start_coord.x;
            scan_item.value = functor_t::initialize();
        }

        // 线程块分片包含多行,记录完整行的累加结果
        if (tile_num_rows > 0) {

            __syncthreads();

            // Scan downsweep and scatter
            // 记录矩阵每行累加结果的共享内存指针
            vec_y_value_t* s_partials = &temp_storage.merge_items[0].nonzero;

            // 上一线程的线程分片的最后一行与该线程的线程分片的第一行不同
            if (scan_item.key != scan_segment[0].key) {
                // 将上一线程分片最后一行的累加结果记录到共享内存的对应行偏移的位置上
                s_partials[scan_item.key] = scan_item.value;
            } else {    // 上一线程的线程分片的最后一行与该线程的线程分片的第一行相同(有跨线程分片的矩阵行)
                // 将上一分片该行的累加结果记录到本行首元素中 - 相当于线程分片的fixup
                // 因为前面使用的是非包含前缀和ExclusiveScan,不会包含本线程的线程分片
                scan_segment[0].value = functor_t::reduce(scan_segment[0].value, scan_item.value);
            }

            #pragma unroll
            for (int ITEM = 1; ITEM < ITEMS_PER_THREAD; ++ITEM) {
                if (scan_segment[ITEM - 1].key != scan_segment[ITEM].key) { // 若两个元素对应矩阵不同行
                    // 将上一行的累加结果写入共享内存
                    s_partials[scan_segment[ITEM - 1].key] = scan_segment[ITEM - 1].value;
                } else {    // 两个元素对应矩阵同一行
                    // 将线程分片同一行的结果累加
                    scan_segment[ITEM].value = functor_t::reduce(scan_segment[ITEM].value, scan_segment[ITEM - 1].value);
                }
            }

            __syncthreads();

            // 将累加结果从共享内存写回vector Y的全局内存中
            #pragma unroll 1
            for (int item = threadIdx.x; item < tile_num_rows; item += BLOCK_THREADS) {
                spmv_params.d_vector_y[tile_start_coord.x + item] = s_partials[item];
                // printf("write y[%u]=%d\n", tile_start_coord.x + item, s_partials[item]);
            }
        }

        // Return the tile's running carry-out
        // 线程块分片最后一行的行偏移和累加结果(可能跨线程块分片的矩阵行的累加结果)
        return tile_carry;
    }


    /**
     * Consume input tile
     */
    __device__ __forceinline__ void ConsumeTile(
        CoordinateT*    d_tile_coordinates,     ///< [in] Pointer to the temporary array of tile starting coordinates
        KeyValuePairT*  d_tile_carry_pairs,     ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
        offset_t             num_merge_tiles)        ///< [in] Number of merge tiles
    {
        offset_t tile_idx = (blockIdx.x * gridDim.y) + blockIdx.y;    // Current tile index

        if (tile_idx >= num_merge_tiles)
            return;

        // Read our starting coordinates
        // 两个线程分别对应写入线程块分片合并路径的起始坐标和终止坐标
        if (threadIdx.x < 2) {
            // printf("tile_idx=%u tile_idx+threadIdx.x=%u \n", tile_idx, tile_idx + threadIdx.x);
            temp_storage.tile_coords[threadIdx.x] = d_tile_coordinates[tile_idx + threadIdx.x];
        }

        __syncthreads();

        CoordinateT tile_start_coord     = temp_storage.tile_coords[0];
        CoordinateT tile_end_coord       = temp_storage.tile_coords[1];
        // printf("tile_start_coord.x=%u, tile_start_coord.y=%u\n", tile_start_coord.x, tile_start_coord.y);
        // printf("tile_end_coord.x=%u, tile_end_coord.y=%u\n", tile_end_coord.x, tile_end_coord.y);

        // Consume multi-segment tile
        // 消耗合并路径的线程块分,返回线程块分片最后一行的行偏移和累加结果
        KeyValuePairT tile_carry = ConsumeTile(
            tile_idx,
            tile_start_coord,
            tile_end_coord,
            cub::Int2Type<AgentSpmvPolicyT::DIRECT_LOAD_NONZEROS>() // 是否直接从全局内存读取非零元值数组
        );   

        // Output the tile's carry-out
        if (threadIdx.x == 0) {
            // 行偏移加上线程块分片的起始行偏移,得到实际的行偏移
            tile_carry.key += tile_start_coord.x;
            // 避免行偏移越界
            if (tile_carry.key >= spmv_params.num_rows) {
                // FIXME: This works around an invalid memory access in the
                // fixup kernel. The underlying issue needs to be debugged and
                // properly fixed, but this hack prevents writes to
                // out-of-bounds addresses. It doesn't appear to have an effect
                // on the validity of the results, since this only affects the
                // carry-over from last tile in the input.
                tile_carry.key = spmv_params.num_rows - 1;
                tile_carry.value = functor_t::initialize() ;
            };
            // 记录该线程块分片的最后一行的行偏移和累加结果
            d_tile_carry_pairs[tile_idx]    = tile_carry;
        }
    }


};

}   // namespace graph_one::blas::merge

// }               // CUB namespace
// CUB_NS_POSTFIX  // Optional outer namespace(s)

