#pragma once 

#include <type_traits>

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
// #include <nvtx3/nvtx3.hpp>

#include "GraphGenlX/engine/base.h"
#include "GraphGenlX/frontier/base.h"
#include "GraphGenlX/archi/macro/macro.h"
#include "GraphGenlX/archi/check/check.h"
#include "GraphGenlX/archi/thrust/thrust.h"
#include "GraphGenlX/archi/kernel/kernel.h"
#include "GraphGenlX/archi/only/cuda.cuh"
#include "GraphGenlX/utils/limits.hpp"

namespace graph_genlx {

namespace engine {

template <typename tparams,    // should be archi::LaunchTparams<arch_t::cuda>
          typename functor_t,
          typename graph_t, 
          typename dstatus_t, 
          typename frontier_t> 
__GENLX_CUDA_KERNEL__ 
static void advance_engine_kernel(
    const graph_t graph, dstatus_t d_status, frontier_t frontier, typename frontier_t::index_type* output_size) {
    static_assert(graph_t::arch_value == arch_t::cuda);

    using index_t = typename frontier_t::index_type;
    using vertex_t = typename graph_t::vertex_type;
    using edge_t = typename graph_t::edge_type;
    using block_scan_t = cub::BlockScan<edge_t, tparams::block_size>;
    constexpr auto vstart = graph_t::vstart_value;

    uint32_t g_tid = tparams::global_tid();
    uint32_t tid = tparams::thread_id();
  
    /// 0. reset output_size to 0
    if (g_tid == 0) {
        *output_size = 0;
    }

    /// 1. Load input data to shared/register memory.
    __shared__ vertex_t vertices[tparams::block_size];  // 结点
    __shared__ edge_t degrees[tparams::block_size]; // 结点度数前缀和
    __shared__ edge_t sedges[tparams::block_size];  // 起始边偏移
    edge_t th_deg;  // 结点度数(前缀和)

    index_t input_size = frontier.input_size;
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
        archi::cuda::print("gtid=%d v=%u sedges[tid]=%d th_deg=%d\n", g_tid, v, sedges[tid], th_deg);
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
    __shared__ index_t offset;
    if constexpr (frontier_t::has_output) {
        // Accumulate the output size to global memory, only done once per block by
        // threadIdx.x == 0, and retrieve the previously stored value from the
        // global memory. The previous value is now current block's starting offset.
        // All writes from this block will be after this offset. Note: this does not
        // guarantee that the data written will be in any specific order.
        if (tid == 0) {
            // 原子计算得到当前线程块在输出前沿的偏移
            offset = atomicAdd(
                output_size, (index_t)aggregate_degree_per_block);
        }
    }
    __syncthreads();

    // 该线程所在线程块的末尾的长度
    // (threadIdx.x + blockDim.x * blockIdx.x) - threadIdx.x + blockDim.x
    // blockIdx<128> 0   1   2   3
    // length      128 216 384 512  
    index_t length = g_tid - tid + tparams::block_size;

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
        int id = archi::cuda::UpperBound(degrees + 0, degrees + length, i) - 1 - degrees;

        // If the id is greater than the width of the block or the input size, we
        // exit.
        if (id >= length) {
            continue;
        }

        // Fetch the vertex corresponding to the id.
        vertex_t src = vertices[id];
        if (!utils::is_vertex_valid<vstart>(src)) {
            continue;
        }

        // If the vertex is valid, get its edge, neighbor and edge weight.
        // 边索引 (i-degrees[id])表示当前线程在结点v的邻接表中的偏移
        auto e = sedges[id] + i - degrees[id];
        auto dst = graph.get_dst_vertex(e);   // 目标结点
        auto w = graph.get_edge_weight(e);          // 边权重

        // Use-defined advance condition.
        // advance操作 判断边是否满足条件
        bool cond = functor_t::advance(src, dst, e, w, d_status);

        archi::cuda::print("g_tid=%u src=%u, dst=%u, e=%u, w=%d d_status.dists[dst]=%f  cond=%d\n",
            g_tid, src, dst, e, w, d_status.dists[dst], cond);

        // Store [neighbor] into the output frontier.
        if constexpr (frontier_t::has_output) {
            // 满足条件则记录目标结点
            frontier.set(offset + i, cond ? dst : utils::invalid_vertex<vstart, vertex_t>());
        }
    }
}

template <typename functor_t, typename comp_t, typename frontier_t>
class AdvanceGC : public BaseEngine<comp_t, frontier_t> {
public:
    using base_t = BaseEngine<comp_t, frontier_t>;
    using graph_t = typename base_t::graph_type;
    using hstatus_t = typename base_t::hstatus_type;
    using dstatus_t = typename base_t::dstatus_type;
    using vertex_t = typename graph_t::vertex_type;
    using edge_t = typename graph_t::edge_type;
    using index_t = typename frontier_t::index_type;

    constexpr static arch_t arch = graph_t::arch_value;

    static_assert(frontier_t::kind == SPARSE_BASED);

    AdvanceGC(comp_t &comp, frontier_t& frontier)
        : base_t(comp, frontier), temp_buf_(1) {}
    
    void Forward() override {
        // nvtx3::scoped_range r{"Forward"};
        ResizeOuput();
        advance_engine(this->graph_, this->d_status_, this->frontier_, temp_buf_);
        filter_engine(this->graph_, this->d_status_, this->frontier_);
    }

    void ResizeOuput() {
        if constexpr (!frontier_t::has_output) {
            return;
        }

        auto graph_ref = this->graph_.ToArch();
        auto& frontier = this->frontier_;
        auto frontier_ref = frontier.ToArch();
        auto v_degree_func = [=] __GENLX_ARCH__ (const vertex_t &i) {
            auto vid = frontier_ref.get(i);
            // archi::cuda::print("vid=%u, degree=%u\n", vid, graph_ref.get_degree(vid));
            return (utils::is_vertex_valid<graph_t::vstart_value>(vid)
                        ? graph_ref.get_degree(vid) : 0);
        };

        auto sz = archi::transform_reduce<arch>(
            thrust::make_counting_iterator<vertex_t>(0),
            thrust::make_counting_iterator<vertex_t>(frontier.input_size()),
            v_degree_func,
            edge_t(0),
            thrust::plus<edge_t>()
        );
        frontier.reset_output(sz);
    }

    static void 
    filter_engine(const graph_t &graph, dstatus_t &d_status, frontier_t &frontier) {
        // nvtx3::scoped_range r{"filter_engine"};

        if constexpr (!frontier_t::has_output || !HasFilter::value) {
            return;
        } else {
            constexpr auto vstart = graph_t::vstart_value;
            using index_t = typename frontier_t::index_type;
            using vertex_t = typename frontier_t::vertex_type;

            LOG_DEBUG("filter begin");
            frontier.swap_inout();

            frontier.reset_output(frontier.input_size());
            auto frontier_ref = frontier.ToArch();
            auto bypass = [=] __GENLX_DEV__(const index_t &idx) {
                vertex_t v = frontier_ref.get(idx);
                if (!utils::is_vertex_valid<vstart>(v) ||
                    functor_t::filter(v, d_status)) {
                    return utils::invalid_vertex<vstart, vertex_t>();
                }
                return v;
            };

            archi::transform<arch>(
                thrust::make_counting_iterator<index_t>(0),
                thrust::make_counting_iterator<index_t>(frontier.input_size()),
                frontier.output().data(), bypass);

            LOG_DEBUG("filter end");
        }
    }

protected:
    static void 
    advance_engine(const graph_t &graph, dstatus_t &d_status, frontier_t &frontier,
                  Buffer<arch, index_t>& output_size) {
        // nvtx3::scoped_range r{"advance_engine"};

        using tparams = archi::LaunchTparams<arch>;
        constexpr auto kernel = advance_engine_kernel<tparams, functor_t,
                                                      typename graph_t::arch_ref_t,
                                                      dstatus_t, 
                                                      typename frontier_t::arch_ref_t>;

        checkCudaErrors((archi::LaunchKernel<arch, tparams>(
            {frontier.input_size()}, 
            kernel, 
            graph.ToArch(), 
            d_status,
            frontier.ToArch(), 
            output_size.data()
        )));
        archi::LaunchSync<arch>();
        
        LOG_DEBUG("output_size: ", output_size);
        LOG_DEBUG("output: ", frontier.output());
    }

    struct HasFilter {
        template <typename T>
        static std::true_type test(decltype(&T::filter));
        template <typename T>
        static std::false_type test(...);

        constexpr static bool value = 
            std::is_same<std::true_type, decltype(test<functor_t>(0))>::value;
    };

    Buffer<arch, index_t> temp_buf_;
};

} // namespce engine

constexpr graph_view_t AdvanceViews = graph_view_t::csr | graph_view_t::normal;

template <typename vertex_t, typename edge_t,
          typename weight_t, typename dstatus_t, typename vprop_t = empty_t>
struct AdvanceFunctor {
    template <typename functor_t, typename comp_t, typename frontier_t>
    using engine_type = engine::AdvanceGC<functor_t, comp_t, frontier_t>;

    __GENLX_DEV_INL__
    static bool advance(const vertex_t &src, const vertex_t &dst, const edge_t &edge,
                 const weight_t &weight, dstatus_t& d_status) {
        return false;
    }

    // __GENLX_DEV_INL__
    // static bool filter(const vertex_t& vid, const dstatus_t& d_status) { return true; }
};
    
} // namespace graph_genlx