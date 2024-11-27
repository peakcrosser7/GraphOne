#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>

#include "GraphOne/engine/base.h"
#include "GraphOne/vec/dense.h"
#include "GraphOne/archi/macro/macro.h"
#include "GraphOne/archi/thrust/thrust.h"
#include "GraphOne/archi/kernel/kernel.h"
#include "GraphOne/archi/blas/spmv/spmv.h"
#include "GraphOne/archi/only/cuda.cuh"


namespace graph_one {

namespace engine {

namespace {

template <typename tparams,
    typename functor_t,
    typename gather_t,
    typename dstatus_t,
    typename index_t,    
    typename info_t>
__ONE_CUDA_KERNEL__
static void apply_construct_kernel(const gather_t *opt_vec, 
    dstatus_t d_status, index_t size, info_t *ipt_vec, 
    index_t* num_applyed) { // apply to all vertices when num_applyed is nullptr

    const index_t g_tid = tparams::global_tid();
    const uint32_t tid = tparams::thread_id();
    const uint32_t bid = tparams::block_id();
    const index_t grid_threads = tparams::grid_threads();

    if (num_applyed) {
        index_t sum = index_t(0);
        if (g_tid == 0) {
            *num_applyed = 0;
        }

        for (index_t vid = g_tid; vid < size; vid += grid_threads) {
            gather_t res = opt_vec[vid];
            info_t info = functor_t::default_info();
            bool pred = (res != functor_t::default_result()
                        && functor_t::apply_each(vid, res, d_status));
            // archi::cuda::print("g_tid=%u, pred=%u\n", vid, pred);
            if (pred) {
                // archi::cuda::print("g_tid=%u pred=%u\n", vid, pred);
                info = functor_t::construct_each(vid, d_status);
            }
            ipt_vec[vid] = info;
            sum += pred;
        }

        __shared__ index_t sdata[32];
        sum = archi::cuda::BlockReduceSum(sum, sdata);
        if (tid == 0 && sum > 0) {
            // archi::cuda::print("g_tid=%u sum=%u\n", g_tid, sum);
            atomicAdd(num_applyed, sum);
        }
    } else {    // num_applyed == nullptr
        for (index_t vid = g_tid; vid < size; vid += grid_threads) {
            gather_t res = opt_vec[vid];
            info_t info = functor_t::default_info();
            bool pred = functor_t::apply_each(vid, res, d_status);
            // archi::cuda::print("g_tid=%u, pred=%u\n", vid, pred);
            if (pred) {
                // archi::cuda::print("g_tid=%u pred=%u\n", vid, pred);
                info = functor_t::construct_each(vid, d_status);
            }
            ipt_vec[vid] = info;
        }
    }
}

template <typename tparams,
          typename index_t>  
__ONE_CUDA_KERNEL__
static void second_reduce_kernel(index_t* num_applyed, index_t size) {
    __shared__ index_t sdata[32];
    const uint32_t tid = tparams::thread_id();
    const uint32_t bid = tparams::block_id();
    const uint32_t grid_threads = tparams::grid_threads();

    index_t sum = 0;
    for (uint32_t i = tid; i < size; i += grid_threads) {
        sum += num_applyed[i];
    }
    sum = archi::cuda::BlockReduceSum(sum, sdata);
    if (tid == 0) {
        num_applyed[tid] = sum;
    }
}

}   // namespace

template <typename functor_t, typename comp_t, typename frontier_t>
class BlasGcEngineBase : public GcEngine<comp_t, frontier_t> {
public:
    using base_t = GcEngine<comp_t, frontier_t>;
    using graph_t = typename base_t::graph_type;
    using hstatus_t = typename base_t::hstatus_type;
    using dstatus_t = typename base_t::dstatus_type;

    using vertex_t = typename graph_t::vertex_type;
    using edge_t = typename graph_t::edge_type;
    using weight_t = typename graph_t::weight_type;
    using info_t = typename functor_t::info_type;
    using gather_t = typename functor_t::gather_type;

    struct SpmvFunctor {
        __ONE_DEV_INL__
        static gather_t initialize() {
            return functor_t::default_result();
        }

        __ONE_DEV_INL__
        static gather_t combine(const weight_t& nonzero, const info_t& x) {
            return functor_t::gather_combine(nonzero, x);
        }

        __ONE_DEV_INL__
        static gather_t reduce(const gather_t& lhs, const gather_t& rhs) {
            return functor_t::gather_reduce(lhs, rhs);
        }
    };
    
    using spmv_t =
        blas::SpmvDispatcher<blas::SpmvCudaCsrMergeBased, SpmvFunctor, vertex_t,
                             edge_t, weight_t, info_t, gather_t>;

    constexpr static arch_t arch = graph_t::arch_value;

    using base_t::GcEngine;
    BlasGcEngineBase(comp_t &comp, frontier_t& frontier, vertex_t buf_size)
        : base_t(comp, frontier), ipt_vec_(comp.graph.num_vertices()),
          opt_vec_(comp.graph.num_vertices()),
          spmv_(make_spmv_(this->graph_, ipt_vec_.data(), opt_vec_.data())),
          temp_buf_(buf_size) {}

protected:
    static spmv_t make_spmv_(const graph_t& graph, info_t* ipt_vec, gather_t* opt_vec) {
        auto graph_ref = graph.ToArch();
        if constexpr (graph_t::both_mat) {
            return blas::MakeCsrSpMV<arch, blas::SpmvCudaCsrMergeBased, SpmvFunctor>(
                graph_ref.num_vertices, graph_ref.num_vertices, graph_ref.num_edges,
                graph_ref.trans_row_offsets, graph_ref.trans_col_indices, graph_ref.trans_values,
                ipt_vec, opt_vec
            );
        } else {
            return blas::MakeCsrSpMV<arch, blas::SpmvCudaCsrMergeBased, SpmvFunctor>(
                graph_ref.num_vertices, graph_ref.num_vertices, graph_ref.num_edges,
                graph_ref.row_offsets, graph_ref.col_indices, graph_ref.values,
                ipt_vec, opt_vec
            );
        }
    }

    DenseVec<arch, info_t, vertex_t> ipt_vec_;
    DenseVec<arch, gather_t, vertex_t> opt_vec_;
    spmv_t spmv_;
    Buffer<arch, vertex_t, vertex_t> temp_buf_;
};


template <typename functor_t, typename comp_t, typename frontier_t, typename enable_t = void>
class BlasGcEngine : public BlasGcEngineBase<functor_t, comp_t, frontier_t> {};

template <typename functor_t, typename comp_t, typename frontier_t>
class BlasGcEngine<functor_t, comp_t, frontier_t,
    typename std::enable_if_t<frontier_t::kind == FrontierKind::SPARSE_BASED>> 
    : public BlasGcEngineBase<functor_t, comp_t, frontier_t> {
public:
    using base_t = BlasGcEngineBase<functor_t, comp_t, frontier_t>;
    using graph_t = typename base_t::graph_type;
    using hstatus_t = typename base_t::hstatus_type;
    using dstatus_t = typename base_t::dstatus_type;
    using spmv_t = typename base_t::spmv_t;

    using vertex_t = typename graph_t::vertex_type;
    using edge_t = typename graph_t::edge_type;
    using weight_t = typename graph_t::weight_type;
    using info_t = typename functor_t::info_type;
    using gather_t = typename functor_t::gather_type;

    constexpr static arch_t arch = graph_t::arch_value;

    BlasGcEngine(comp_t &comp, frontier_t& frontier)
        : base_t(comp, frontier, comp.graph.num_vertices()) {}

    void Forward() override {
        Construct(this->d_status_, this->frontier_, this->ipt_vec_, this->temp_buf_);
        Gather(this->spmv_);
        Apply(this->d_status_, this->frontier_, this->opt_vec_, this->temp_buf_);
    }

    static void Construct(const dstatus_t &d_status, const frontier_t &frontier, 
                        DenseVec<arch, info_t, vertex_t>& ipt_vec,
                        // DenseVec<arch, gather_t, vertex_t>& opt_vec,
                        Buffer<arch, vertex_t, vertex_t>& temp_buf){
        LOG_DEBUG("Construct begin");
        info_t* buffer = reinterpret_cast<info_t *>(temp_buf.data());
        
        archi::fill<arch>(ipt_vec.begin(), ipt_vec.end(), functor_t::default_info());
        // archi::fill<arch>(opt_vec.begin(), opt_vec.end(), functor_t::default_result());

        // LOG_DEBUG("frontier_input:", BufferToString(frontier.input(), frontier.input_size()));
        archi::transform<arch>(frontier.input().begin(), frontier.input().end(), buffer,
            [=] __ONE_DEV__ (const vertex_t &vid) {
                return functor_t::construct_each(vid, d_status);
            });
        archi::scatter<arch>(buffer, buffer + frontier.input_size(),
                             frontier.input().begin(), ipt_vec.begin());
        LOG_DEBUG("ipt_vec: ", ipt_vec);
    }

    static void Gather(spmv_t& spmv) {
        LOG_DEBUG("Gather begin");
        spmv();
    }

    static void Apply(dstatus_t& d_status,
                      frontier_t &frontier,
                      const DenseVec<arch, gather_t, vertex_t>& opt_vec,
                      Buffer<arch, vertex_t, vertex_t>& temp_buf) {
        LOG_DEBUG("Apply begin");
        bool* buffer = reinterpret_cast<bool *>(temp_buf.data());
        archi::transform<arch>(opt_vec.begin(), opt_vec.end(), thrust::make_counting_iterator<vertex_t>(0), buffer,
            [=] __ONE_DEV__ (const gather_t& res, const vertex_t& vid) mutable {
                if (res != functor_t::default_result()
                    && functor_t::apply_each(vid, res, d_status)) {
                    return true;
                }
                return false;
            });
        auto output_sz = archi::copy_if<arch>(thrust::make_counting_iterator<edge_t>(0),
            thrust::make_counting_iterator<edge_t>(opt_vec.size()),
            buffer, frontier.output().begin(),
            thrust::identity<bool>()) - frontier.output().begin();
        frontier.reset_output(output_sz);
        LOG_DEBUG("input frontier sz: ", frontier.input_size(),
            " output frontier sz: ", frontier.output_size(), "\n");
    }

};


template <typename functor_t, typename comp_t, typename frontier_t>
class BlasGcEngine<functor_t, comp_t, frontier_t,
    typename std::enable_if_t<frontier_t::kind == FrontierKind::DENSE_BASED>> 
    : public BlasGcEngineBase<functor_t, comp_t, frontier_t> {
public:
    using base_t = BlasGcEngineBase<functor_t, comp_t, frontier_t>;
    using graph_t = typename base_t::graph_type;
    using hstatus_t = typename base_t::hstatus_type;
    using dstatus_t = typename base_t::dstatus_type;
    using spmv_t = typename base_t::spmv_t;

    using vertex_t = typename graph_t::vertex_type;
    using edge_t = typename graph_t::edge_type;
    using weight_t = typename graph_t::weight_type;
    using info_t = typename functor_t::info_type;
    using gather_t = typename functor_t::gather_type;
    using index_t = typename frontier_t::index_type;

    constexpr static arch_t arch = graph_t::arch_value;

    BlasGcEngine(comp_t &comp, frontier_t &frontier)
        : base_t(comp, frontier,
                 init_temp_buf_size_(comp, frontier)) {}

    void Forward() override {
        Construct(this->d_status_, this->frontier_, 
            this->ipt_vec_,
            // this->opt_vec_,
            this->temp_buf_);
        Gather(this->spmv_);
        Apply(this->d_status_, this->frontier_, 
            this->opt_vec_,
            this->ipt_vec_,
            this->temp_buf_);
    }

    static void Construct(const dstatus_t &d_status, frontier_t &frontier,
                        DenseVec<arch, info_t, vertex_t>& ipt_vec,
                        // DenseVec<arch, gather_t, vertex_t>& opt_vec,
                        Buffer<arch, vertex_t, vertex_t>& temp_buf){
        LOG_DEBUG("Construct begin");

        auto& init_buf = frontier.init_buf();
        if (!init_buf.empty()) {
            archi::fill<arch>(ipt_vec.begin(), ipt_vec.end(), functor_t::default_info());
            // archi::fill<arch>(opt_vec.begin(), opt_vec.end(), functor_t::default_result());

            info_t* buffer = reinterpret_cast<info_t *>(temp_buf.data());
            archi::transform<arch>(init_buf.begin(), init_buf.end(), buffer,
                [=] __ONE_DEV__ (const vertex_t &vid) {
                    return functor_t::construct_each(vid, d_status);
                });
            archi::scatter<arch>(buffer, buffer + init_buf.size(),
                                init_buf.begin(), ipt_vec.begin());
            
            // clear init_buf
            init_buf = std::move(std::remove_reference_t<decltype(init_buf)>());
        }

        LOG_DEBUG("ipt_vec: ", ipt_vec);
    }

    static void Gather(spmv_t& spmv) {
        LOG_DEBUG("Gather begin");
        spmv();
    }

    static void Apply(dstatus_t& d_status,
                      frontier_t& frontier,
                      const DenseVec<arch, gather_t, vertex_t>& opt_vec,
                      DenseVec<arch, info_t, vertex_t>& ipt_vec,
                      Buffer<arch, vertex_t, vertex_t>& temp_buf) {
        LOG_DEBUG("Apply begin");
        LOG_DEBUG("opt_vec: ", opt_vec);

        using tparams = archi::LaunchTparams<arch>;
        constexpr auto apply_constr_kernel = apply_construct_kernel<tparams, functor_t, 
                                                    gather_t, dstatus_t, index_t, info_t>;
        
        index_t* buffer = reinterpret_cast<index_t*>(temp_buf.data());
        checkArchErrors(arch, (archi::LaunchKernel<arch, tparams>(
            {kNumBlocks, true},
            apply_constr_kernel,
            opt_vec.data(), d_status, opt_vec.size(),
            ipt_vec.data(), buffer
        )));

        archi::memcpy<arch_t::cpu, arch, index_t>(&frontier.output_size(), buffer, 1);
        LOG_DEBUG("input frontier sz: ", frontier.input_size(),
            " output frontier sz: ", frontier.output_size(), "\n");
        LOG_DEBUG("ipt_vec: ", ipt_vec);
    }

private:
    static constexpr vertex_t kNumBlocks = archi::LaunchTparams<arch>::block_size * 4;

    vertex_t init_temp_buf_size_(comp_t &comp, frontier_t &frontier) {
        vertex_t init_size = frontier.init_buf().size();
        return (init_size * sizeof(info_t) + sizeof(vertex_t) - 1) / sizeof(vertex_t);
    }

};

template <typename functor_t, typename comp_t, typename frontier_t>
class BlasGcEngine<functor_t, comp_t, frontier_t,
    typename std::enable_if_t<frontier_t::kind == FrontierKind::ALL_ACTIVE>> 
    : public BlasGcEngineBase<functor_t, comp_t, frontier_t> {
public:
    using base_t = BlasGcEngineBase<functor_t, comp_t, frontier_t>;
    using graph_t = typename base_t::graph_type;
    using hstatus_t = typename base_t::hstatus_type;
    using dstatus_t = typename base_t::dstatus_type;
    using spmv_t = typename base_t::spmv_t;

    using vertex_t = typename graph_t::vertex_type;
    using edge_t = typename graph_t::edge_type;
    using weight_t = typename graph_t::weight_type;
    using info_t = typename functor_t::info_type;
    using gather_t = typename functor_t::gather_type;
    using index_t = typename frontier_t::index_type;

    constexpr static arch_t arch = graph_t::arch_value;

    BlasGcEngine(comp_t &comp, frontier_t &frontier)
        : base_t(comp, frontier, 0),
          is_first_construct_(true) {}

    void Forward() override {
        Construct(this->is_first_construct_, 
            this->d_status_, 
            this->frontier_, 
            this->ipt_vec_);
        Gather(this->spmv_);
        Apply(this->d_status_,
            this->opt_vec_,
            this->ipt_vec_);
    }

    static void Construct(bool& is_first_construct,
                        const dstatus_t &d_status, 
                        frontier_t& frontier,
                        DenseVec<arch, info_t, vertex_t>& ipt_vec){
        LOG_DEBUG("Construct begin");
        if (is_first_construct) {
            is_first_construct = false;
            archi::transform<arch>(thrust::make_counting_iterator<edge_t>(0),
                thrust::make_counting_iterator<edge_t>(frontier.size()),
                ipt_vec.begin(), [d_status] __ONE_DEV__ (const vid_t& vid)  {
                    return functor_t::construct_each(vid, d_status);
                });
        }

        LOG_DEBUG("ipt_vec: ", ipt_vec);
    }

    static void Gather(spmv_t& spmv) {
        LOG_DEBUG("Gather begin");
        spmv();
    }

    static void Apply(dstatus_t& d_status,
                      const DenseVec<arch, gather_t, vertex_t>& opt_vec,
                      DenseVec<arch, info_t, vertex_t>& ipt_vec) {
        LOG_DEBUG("Apply begin");
        LOG_DEBUG("opt_vec: ", opt_vec);

        using tparams = archi::LaunchTparams<arch>;
        constexpr auto apply_constr_kernel = apply_construct_kernel<tparams, functor_t, 
                                                    gather_t, dstatus_t, index_t, info_t>;
        
        checkArchErrors(arch, (archi::LaunchKernel<arch, tparams>(
            {kNumBlocks, true},
            apply_constr_kernel,
            opt_vec.data(), d_status, opt_vec.size(),
            ipt_vec.data(), 
            nullptr // not need to sum
        )));

    }

private:
    static constexpr vertex_t kNumBlocks = archi::LaunchTparams<arch>::block_size * 4;

    bool is_first_construct_;
};
    
} // namespace engine

constexpr graph_view_t BlasViews = graph_view_t::csr | graph_view_t::transpose;

template <typename vertex_t, typename weight_t, typename dstatus_t,
          typename info_t, typename gather_t>
struct BlasFunctor {
    template <typename functor_t, typename comp_t, typename frontier_t>
    using engine_type = engine::BlasGcEngine<functor_t, comp_t, frontier_t>;
    using info_type = info_t;
    using gather_type = gather_t;

    __ONE_ARCH_INL__
    static info_t default_info() {
        return info_t{0};
    }

    __ONE_ARCH_INL__
    static gather_t default_result() {
        return gather_t{0};
    }

    __ONE_DEV_INL__
    static info_t construct_each(const vertex_t& vid, const dstatus_t& d_status) {
        return info_t{};
    }

    __ONE_DEV_INL__
    static gather_t gather_combine(const weight_t& weight, const info_t& info) {
        return gather_t{};    
    }

    __ONE_DEV_INL__
    static gather_t gather_reduce(const gather_t& lhs, const gather_t& rhs) {}

    /// return true when activating this vertex
    __ONE_DEV_INL__
    static bool apply_each(const vertex_t& vid, const gather_t& res, dstatus_t& d_status) {}
};


} // namespace graph_one