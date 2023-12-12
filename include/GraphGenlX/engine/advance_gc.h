#pragma once 

#include <functional>

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>

#include "GraphGenlX/archi/macro/macro.h"
#include "GraphGenlX/archi/check/check.h"
#include "GraphGenlX/archi/kernel/kernel.h"
#include "GraphGenlX/utils/limits.hpp"

namespace graph_genlx {

namespace engine {

template <typename tparams,    // should be archi::LaunchTparams<arch_t::cuda>
          typename graph_t, 
          typename frontier_t, 
          typename factor_t> 
__global__ static void advance_engine_kernel(
    const graph_t graph, frontier_t frontier, factor_t fators, typename graph_t::vertex_type* output_size) {
    static_assert(graph_t::arch_value == arch_t::cuda);

    using vertex_t = typename graph_t::vertex_type;
    using edge_t = typename graph_t::edge_type;
    using block_scan_t = cub::BlockScan<edge_t, tparams::block_size>;
    constexpr auto vstart = graph_t::vstart_value;

    uint32_t g_tid = tparams::global_tid();
    uint32_t tid = tparams::thread_id();
    
        /// 1. Load input data to shared/register memory.
    __shared__ vertex_t vertices[tparams::block_size];  // 结点
    __shared__ edge_t degrees[tparams::block_size]; // 结点度数前缀和
    __shared__ edge_t sedges[tparams::block_size];  // 起始边偏移
    edge_t th_deg;  // 结点度数(前缀和)

    vertex_t input_size = frontier.input_size;
    if (g_tid < input_size) {  // 有效线程
        // 当前处理的前沿结点
        vertex_t v = frontier.get(g_tid); // 输入前沿结点ID
        vertices[tid] = v;
        if (utils::is_vertex_valid<vstart>(v)) {
            // 起始边偏移
            sedges[tid] = graph.get_starting_edge(v);
            th_deg = graph.get_degree(v);
        } else {
            th_deg = 0;
        }
    } else {
        // 记录为无效结点
        vertices[tid] = utils::invalid_vertex<vstart, vertex_t>();
        th_deg = 0;
    }
    __syncthreads();

    /// 2. Exclusive sum of degrees to find total work items per block.
    __shared__ typename block_scan_t::TempStorage scan;
    edge_t aggregate_degree_per_block;  // 每个线程块内结点总度数(聚合归约结果)
    // 对每个结点的度数进行线程块级前缀扫描
    block_scan_t(scan).ExclusiveSum(th_deg, th_deg, aggregate_degree_per_block);
    __syncthreads();

    // Store back to shared memory (to later use in the binary search).
    degrees[tid] = th_deg;
    

    /// 3. Compute block offsets if there's an output frontier.
    __shared__ vertex_t offset;
    if constexpr (frontier_t::has_output) {
        // Accumulate the output size to global memory, only done once per block by
        // threadIdx.x == 0, and retrieve the previously stored value from the
        // global memory. The previous value is now current block's starting offset.
        // All writes from this block will be after this offset. Note: this does not
        // guarantee that the data written will be in any specific order.
        if (tid == 0) {
        // 原子计算得到当前线程块在输出前沿的偏移
        offset = atomicAdd(
            output_size, (vertex_t)aggregate_degree_per_block);
        }
    }
    __syncthreads();

    // 每个线程块全局的长度范围
    // (threadIdx.x + blockDim.x * blockIdx.x) - threadIdx.x + blockDim.x
    // blockIdx<128> 0   1   2   3
    // length      128 216 384 512  
    auto length = g_tid - tid + tparams::block_size;

    // 确保均为有效范围
    if (input_size < length) {
        length = input_size;
    }

    // 线程块内的有效长度范围
    length -= g_tid - tid; // length = blockDim.x

    /// 4. Compute. Using binary search, find the source vertex each thread is
    /// processing, and the corresponding edge, neighbor and weight tuple. Passed
    /// to the user-defined lambda operator to process. If there's an output, the
    /// resultant neighbor or invalid vertex is written to the output frontier.
    // 每个线程负责一个目标邻结点(一个度数)
    for (edge_t i = tid;            // threadIdx.x
        i < aggregate_degree_per_block;  // total degree to process
        i += tparams::block_size     // increment by blockDim.x
    ) {
        // Binary search to find which vertex id to work on.
        // 确定当前线程负责的源结点索引
        int id = archi::upper_bound<arch_t::cuda>(degrees + 0, degrees + length, i) - 1 - degrees;

        // If the id is greater than the width of the block or the input size, we
        // exit.
        if (id >= length) {
            continue;
        }

        // Fetch the vertex corresponding to the id.
        vertex_t v = vertices[id];
        if (!utils::is_vertex_valid<vstart>(v)) {
            continue;
        }

        // If the vertex is valid, get its edge, neighbor and edge weight.
        // 边索引 (i-degrees[id])表示当前线程在结点v的邻接表中的偏移
        auto e = sedges[id] + i - degrees[id];
        auto n = graph.get_dst_vertex(e);   // 目标结点
        auto w = graph.get_edge_weight(e);          // 边权重

        // Use-defined advance condition.
        // advance操作 判断边是否满足条件
        bool cond = fators.advance(v, n, e, w);

        // Store [neighbor] into the output frontier.
        if constexpr (frontier_t::has_output) {
            // 满足条件则记录目标结点
            frontier.set(offset + i, cond ? n : utils::invalid_vertex<vstart, vertex_t>());
        }
    }
}

struct AdvanceGC {

    template <typename graph_t, typename frontier_t, typename factor_t>
    static void
    advance_engine(const graph_t& graph, frontier_t& frontier, factor_t& factors) {   
        constexpr arch_t arch = graph_t::arch_value;
        using tparams = archi::LaunchTparams<arch>;
        constexpr auto kernel = advance_engine_kernel<tparams, typename graph_t::arch_ref_t,
                                                      typename frontier_t::arch_ref_t, factor_t>;

        typename graph_t::vertex_type output_size;
        checkArchErrors(arch, (archi::LaunchKernel<arch, tparams>(
            {frontier.input_size()}, 
            kernel, 
            graph.ToArch(), 
            frontier.ToArch(), 
            factors, 
            &output_size
        )));
        frontier.reset_output(output_size);
    }

    template <typename comp_t, typename factor_t>
    static void Forward(comp_t& comp, factor_t& factors) {
        advance_engine(comp.graph, comp.frontier, factors); 
    }

};

} // namespce engine

template <typename vertex_t, typename edge_t,
          typename weight_t, typename vprop_t = empty_t>
struct AdvanceFactor {
    using engine_type = engine::AdvanceGC;

    bool advance(const vertex_t &src, const vertex_t &dst, const edge_t &edge,
                 const weight_t &weight) {
        return false;
    }
};
    
} // namespace graph_genlx