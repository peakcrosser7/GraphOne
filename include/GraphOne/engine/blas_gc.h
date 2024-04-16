#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>

#include "GraphOne/engine/base.h"
#include "GraphOne/vec/dense.h"
#include "GraphOne/archi/macro/macro.h"
#include "GraphOne/archi/thrust/thrust.h"
#include "GraphOne/archi/kernel/kernel.h"
#include "GraphOne/archi/blas/SpMV/spmv.h"
#include "GraphOne/archi/only/cuda.cuh"


namespace graph_one {

namespace engine {

template <typename tparams,
    typename functor_t,
    typename gather_t,
    typename dstatus_t,
    typename index_t,    
    typename info_t>
__ONE_CUDA_KERNEL__
static void apply_construct_kernel(const gather_t *res_vec, 
    dstatus_t d_status, index_t size, info_t *info_vec, index_t* num_applyed) {
    const index_t g_tid = tparams::global_tid();
    const uint32_t tid = tparams::thread_id();
    const uint32_t bid = tparams::block_id();

    index_t sum = index_t(0);
    for (index_t vid = g_tid; vid < size; vid += tparams::grid_threads()) {
        gather_t res = res_vec[vid];
        info_t info = functor_t::default_info();
        index_t pred = (res != functor_t::default_result()
                    && functor_t::apply(vid, res, d_status));
        // archi::cuda::print("g_tid=%u, pred=%u\n", vid, pred);
        if (pred != 0) {
            archi::cuda::print("g_tid=%u pred=%u\n", vid, pred);
            info = functor_t::construct(vid, d_status);
        }
        info_vec[vid] = info;
        sum += pred;
    }

    __shared__ index_t sdata[32];
    sum = archi::cuda::BlockReduceSum(sum, sdata);
    if (tid == 0) {
        num_applyed[bid] = sum;
    }
}

template <typename tparams,
          typename index_t>  
__ONE_CUDA_KERNEL__
static void second_reduce_kernel(index_t* num_applyed, index_t size) {
    __shared__ index_t sdata[32];
    const uint32_t tid = tparams::thread_id();
    const uint32_t bid = tparams::block_id();

    index_t sum = tid < size ? num_applyed[tid] : index_t(0);
    sum = archi::cuda::BlockReduceSum(sum, sdata);
    if (tid == 0) {
        num_applyed[tid] = sum;
    }
}

template <arch_t arch,
          typename value_t,
          typename index_t>
std::string 
BufferToString(const Buffer<arch, value_t, index_t>& frontier, index_t size) {
    size = std::min(size, frontier.size());
    std::string str("[");
    if constexpr (arch != arch_t::cpu) {
        value_t* h_data = archi::memalloc<arch_t::cpu, value_t>(size);
        archi::memcpy<arch_t::cpu, arch, value_t>(h_data, frontier.data(), size);
        for (index_t i = 0; i < size; ++i) {
            str += utils::NumToString(h_data[i]);
            if (i < size - 1) {
                str += ",";
            }
        }
        archi::memfree<arch_t::cpu, value_t>(h_data);
    } else {
        const value_t* h_data = frontier.data();
        for (index_t i = 0; i < size; ++i) {
            str += utils::NumToString(h_data[i]);
            if (i < size - 1) {
                str += ",";
            }
        }
    }
    str += "]";
    return str;
}

template <typename functor_t, typename comp_t, typename frontier_t>
class BlasEngineBase : public BaseEngine<comp_t, frontier_t> {
public:
    using base_t = BaseEngine<comp_t, frontier_t>;
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
            return functor_t::combine(nonzero, x);
        }

        __ONE_DEV_INL__
        static gather_t reduce(const gather_t& lhs, const gather_t& rhs) {
            return functor_t::reduce(lhs, rhs);
        }
    };
    
    using spmv_t =
        blas::SpmvDispatcher<blas::SpmvCudaMergeBased, SpmvFunctor, vertex_t,
                             edge_t, weight_t, info_t, gather_t>;

    constexpr static arch_t arch = graph_t::arch_value;

    using base_t::BaseEngine;

protected:
    static spmv_t make_spmv_(const graph_t& graph, info_t* info_vec, gather_t* res_vec) {
        auto graph_ref = graph.ToArch();
        return blas::MakeSpMV<arch, blas::SpmvCudaMergeBased, SpmvFunctor>(
            graph_ref.num_vertices, graph_ref.num_vertices, graph_ref.num_edges,
            graph_ref.row_offsets, graph_ref.col_indices, graph_ref.values,
            info_vec, res_vec
        );
    }
};


template <typename functor_t, typename comp_t, typename frontier_t, typename enable_t = void>
class BlasEngine : public BlasEngineBase<functor_t, comp_t, frontier_t> {};

template <typename functor_t, typename comp_t, typename frontier_t>
class BlasEngine<functor_t, comp_t, frontier_t,
    typename std::enable_if_t<frontier_t::kind == FrontierKind::SPARSE_BASED>> 
    : public BlasEngineBase<functor_t, comp_t, frontier_t> {
public:
    using base_t = BlasEngineBase<functor_t, comp_t, frontier_t>;
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

    BlasEngine(comp_t &comp, frontier_t& frontier)
        : base_t(comp, frontier), info_vec_(comp.graph.num_vertices()),
          res_vec_(comp.graph.num_vertices()),
          temp_buf_(comp.graph.num_vertices()),
          spmv_(base_t::make_spmv_(this->graph_, info_vec_.data(), res_vec_.data())) {}

    void Forward() override {
        Construct(this->d_status_, this->frontier_, this->info_vec_, this->res_vec_, this->temp_buf_);
        Gather(spmv_);
        LOG_DEBUG("res_vec: ", this->res_vec_);
        Apply(this->d_status_, this->frontier_, this->res_vec_, this->temp_buf_);
    }

    static void Construct(const dstatus_t &d_status, const frontier_t &frontier, 
                        DenseVec<arch, info_t, vertex_t>& info_vec,
                        DenseVec<arch, gather_t, vertex_t>& res_vec,
                        Buffer<arch, vertex_t, vertex_t>& temp_buf){
        LOG_DEBUG("Construct begin");
        info_t* buffer = reinterpret_cast<info_t *>(temp_buf.data());
        
        archi::fill<arch>(info_vec.begin(), info_vec.end(), functor_t::default_info());
        archi::fill<arch>(res_vec.begin(), res_vec.end(), functor_t::default_result());

        // LOG_DEBUG("frontier_input:", BufferToString(frontier.input(), frontier.input_size()));
        archi::transform<arch>(frontier.input().begin(), frontier.input().end(), buffer,
            [=] __ONE_DEV__ (const vertex_t &vid) {
                return functor_t::construct(vid, d_status);
            });
        archi::scatter<arch>(buffer, buffer + frontier.input_size(),
                             frontier.input().begin(), info_vec.begin());
        LOG_DEBUG("info_vec: ", info_vec);
    }

    static void Gather(spmv_t& spmv) {
        LOG_DEBUG("Gather begin");
        spmv();
    }

    static void Apply(dstatus_t& d_status,
                      frontier_t &frontier,
                      const DenseVec<arch, gather_t, vertex_t>& res_vec,
                      Buffer<arch, vertex_t, vertex_t>& temp_buf) {
        LOG_DEBUG("Apply begin");
        bool* buffer = reinterpret_cast<bool *>(temp_buf.data());
        archi::transform<arch>(res_vec.begin(), res_vec.end(), thrust::make_counting_iterator<vertex_t>(0), buffer,
            [=] __ONE_DEV__ (const gather_t& res, const vertex_t& vid) mutable {
                if (res != functor_t::default_result()
                    && functor_t::apply(vid, res, d_status)) {
                    return true;
                }
                return false;
            });
        auto output_sz = archi::copy_if<arch>(thrust::make_counting_iterator<edge_t>(0),
            thrust::make_counting_iterator<edge_t>(res_vec.size()),
            buffer, frontier.output().begin(),
            thrust::identity<bool>()) - frontier.output().begin();
        frontier.reset_output(output_sz);
        // LOG_INFO("frontier_output: ", BufferToString(frontier.output(), frontier.output_size()));
        LOG_DEBUG("input frontier sz: ", frontier.input_size(),
            " output frontier sz: ", frontier.output_size(), "\n");
    }

protected:

    DenseVec<arch, info_t, vertex_t> info_vec_;
    DenseVec<arch, gather_t, vertex_t> res_vec_;

    Buffer<arch, vertex_t, vertex_t> temp_buf_;

    spmv_t spmv_;

};

template<arch_t arch,
    typename value_t,
    typename vertex_t,
    typename func_t>
std::string dense2sparse_print(const Buffer<arch, value_t, vertex_t>& dense_vec,
    func_t val_func) {
    Buffer<arch, value_t, vertex_t> buffer(dense_vec.size());
    value_t target = val_func();
    vertex_t output_sz = archi::copy_if<arch>(thrust::make_counting_iterator<vertex_t>(0),
        thrust::make_counting_iterator<vertex_t>(dense_vec.size()),
        dense_vec.begin(), buffer.begin(),
        [=] __ONE_DEV__ (const value_t& v) {
            return v != target;
        }) - buffer.begin();
    return BufferToString(buffer, output_sz);
}

template <typename functor_t, typename comp_t, typename frontier_t>
class BlasEngine<functor_t, comp_t, frontier_t,
    typename std::enable_if_t<frontier_t::kind == FrontierKind::DENSE_BASED>> 
    : public BlasEngineBase<functor_t, comp_t, frontier_t> {
public:
    using base_t = BlasEngineBase<functor_t, comp_t, frontier_t>;
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

    BlasEngine(comp_t &comp, frontier_t& frontier)
     : base_t(comp, frontier), 
       spmv_(base_t::make_spmv_(this->graph_, this->frontier_.input().data(), this->frontier_.output().data())),
       reduce_buf_(archi::cuda::GetNumBlock(frontier.input_size())) {}


    void Forward() override {
        Construct(this->d_status_, this->frontier_, 
            this->frontier_.input(),
            this->frontier_.output());
        Gather(spmv_);
        Apply(this->d_status_, this->frontier_, 
            this->frontier_.output(),
            this->frontier_.input(),
            reduce_buf_);
    }

    static void Construct(const dstatus_t &d_status, frontier_t &frontier,
                        Buffer<arch, info_t, vertex_t>& info_vec,
                        Buffer<arch, gather_t, vertex_t>& res_vec){
        LOG_DEBUG("Construct begin");

        auto& init_buf = frontier.init_buf();
        archi::fill<arch>(res_vec.begin(), res_vec.end(), functor_t::default_result());

        if (!init_buf.empty()) {
            archi::fill<arch>(info_vec.begin(), info_vec.end(), functor_t::default_info());
            // archi::fill<arch>(res_vec.begin(), res_vec.end(), functor_t::default_result());

            Buffer<arch, info_t, vertex_t> tmp_buf(init_buf.size());
            archi::transform<arch>(init_buf.begin(), init_buf.end(), tmp_buf.begin(),
                [=] __ONE_DEV__ (const vertex_t &vid) {
                    return functor_t::construct(vid, d_status);
                });
            archi::scatter<arch>(tmp_buf.begin(), tmp_buf.begin() + init_buf.size(),
                                init_buf.begin(), info_vec.begin());
            
            // clear init_buf
            init_buf = std::move(std::remove_reference_t<decltype(init_buf)>());
        }

        LOG_DEBUG("info_vec: ", info_vec);
        // LOG_INFO("frontier_input: ", dense2sparse_print(info_vec, functor_t::default_info));
    }

    static void Gather(spmv_t& spmv) {
        LOG_DEBUG("Gather begin");
        spmv();
    }

    static void Apply(dstatus_t& d_status,
                      frontier_t& frontier,
                      const Buffer<arch, gather_t, vertex_t>& res_vec,
                      Buffer<arch, info_t, vertex_t>& info_vec,
                      Buffer<arch, index_t, index_t>& reduce_buf) {
        LOG_DEBUG("Apply begin");
        LOG_DEBUG("res_vec: ", res_vec);

        using tparams = archi::LaunchTparams<arch>;
        constexpr auto constr_apply_kernel = apply_construct_kernel<tparams, functor_t, 
                                                    gather_t, dstatus_t, index_t, info_t>;
        constexpr auto reduce_kernel = second_reduce_kernel<tparams, index_t>;
        
        checkArchErrors(arch, (archi::LaunchKernel<arch, tparams>(
            {tparams::block_size * 4, true},
            constr_apply_kernel,

            res_vec.data(), d_status, frontier.input_size(),
            info_vec.data(), reduce_buf.data()
        )));

        LOG_DEBUG("reduce_buf: ", reduce_buf);

        checkArchErrors(arch, (archi::LaunchKernel<arch, tparams>(
            {1, true},
            reduce_kernel,
            reduce_buf.data(),
            reduce_buf.size()
        )));
        archi::LaunchSync<arch>();
        archi::memcpy<arch_t::cpu, arch, index_t>(&frontier.output_nnz(), reduce_buf.data(), 1);
        // LOG_INFO("frontier_output: ", dense2sparse_print(res_vec, functor_t::default_result));
        LOG_DEBUG("input frontier sz: ", frontier.input_nnz(),
            " output frontier sz: ", frontier.output_nnz(), "\n");
        LOG_DEBUG("info_vec: ", info_vec);
    }

protected:
    spmv_t spmv_;
    Buffer<arch, index_t, index_t> reduce_buf_;
};

    
} // namespace engine

constexpr graph_view_t BlasViews = graph_view_t::csr | graph_view_t::transpose;

template <typename vertex_t, typename weight_t, typename dstatus_t,
          typename info_t, typename gather_t>
struct BlasFunctor {
    template <typename functor_t, typename comp_t, typename frontier_t>
    using engine_type = engine::BlasEngine<functor_t, comp_t, frontier_t>;
    using info_type = info_t;
    using gather_type = gather_t;

    static constexpr bool sparse = false;

    __ONE_ARCH_INL__
    static info_t default_info() {
        return info_t{0};
    }

    __ONE_ARCH_INL__
    static gather_t default_result() {
        return gather_t{0};
    }

    __ONE_DEV_INL__
    static info_t construct(const vertex_t& vid, const dstatus_t& d_status) {
        return info_t{};
    }

    __ONE_DEV_INL__
    static gather_t combine(const weight_t& weight, const info_t& info) {
        return gather_t{};    
    }

    __ONE_DEV_INL__
    static gather_t reduce(const gather_t& lhs, const gather_t& rhs) {}

    /// return true when activating this vertex
    __ONE_DEV_INL__
    static bool apply(const vertex_t& vid, const gather_t& res, dstatus_t& d_status) {}
};


} // namespace graph_one