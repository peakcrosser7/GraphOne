#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>

#include "GraphGenlX/engine/base.h"
#include "GraphGenlX/vec/dense.h"
#include "GraphGenlX/archi/macro/macro.h"
#include "GraphGenlX/archi/thrust/thrust.h"
#include "GraphGenlX/archi/kernel/kernel.h"
#include "GraphGenlX/archi/blas/SpMV/spmv.h"

namespace graph_genlx {

namespace engine {

template <typename functor_t, typename comp_t, typename frontier_t>
class BlasEngine : public BaseEngine<comp_t, frontier_t> {
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
        __GENLX_DEV_INL__
        static gather_t initialize() {
            return functor_t::default_result();
        }

        __GENLX_DEV_INL__
        static gather_t combine(const weight_t& nonzero, const info_t& x) {
            return functor_t::gather(nonzero, x);
        }

        __GENLX_DEV_INL__
        static gather_t reduce(const gather_t& lhs, const gather_t& rhs) {
            return functor_t::reduce(lhs, rhs);
        }
    };
    
    using spmv_t =
        blas::SpmvDispatcher<blas::SpmvCudaMergeBased, SpmvFunctor, vertex_t,
                             edge_t, weight_t, info_t, gather_t>;

    constexpr static arch_t arch = graph_t::arch_value;

    BlasEngine(comp_t &comp, frontier_t& frontier)
        : base_t(comp, frontier), info_vec_(comp.graph.num_vertices()),
          res_vec_(comp.graph.num_vertices()),
          temp_buf_(comp.graph.num_vertices()),
          spmv_(make_spmv_(this->graph_, info_vec_.data(), res_vec_.data())) {}

    void Forward() {
        Construct(this->d_status_, this->frontier_, info_vec_, res_vec_, temp_buf_);
        Gather(spmv_);
        LOG_DEBUG("res_vec: ", res_vec_);
        Apply(this->d_status_, this->frontier_, res_vec_, temp_buf_);
    }

    static void Construct(const dstatus_t &d_status, const frontier_t &frontier, 
                        DenseVec<arch, info_t, vertex_t>& info_vec,
                        DenseVec<arch, gather_t, vertex_t>& res_vec,
                        Buffer<arch, vertex_t, vertex_t>& temp_buf){
        LOG_DEBUG("Construct begin");
        info_t* buffer = reinterpret_cast<info_t *>(temp_buf.data());
        
        archi::fill<arch>(info_vec.begin(), info_vec.end(), functor_t::default_info());
        archi::fill<arch>(res_vec.begin(), res_vec.end(), functor_t::default_result());

        LOG_DEBUG("frontier_input:", frontier.input());
        archi::transform<arch>(frontier.input().begin(), frontier.input().end(), buffer,
            [=] __GENLX_DEV__ (const vertex_t &vid) {
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
            [=] __GENLX_DEV__ (const gather_t& res, const vertex_t& vid) mutable {
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
    }

private:
    static spmv_t make_spmv_(const graph_t& graph, info_t* info_vec, gather_t* res_vec) {
        auto graph_ref = graph.ToArch();
        return blas::MakeSpMV<arch, blas::SpmvCudaMergeBased, SpmvFunctor>(
            graph_ref.num_vertices, graph_ref.num_vertices, graph_ref.num_edges,
            graph_ref.row_offsets, graph_ref.col_indices, graph_ref.values,
            info_vec, res_vec
        );
    }

protected:

    DenseVec<arch, info_t, vertex_t> info_vec_;
    DenseVec<arch, gather_t, vertex_t> res_vec_;

    Buffer<arch, vertex_t, vertex_t> temp_buf_;

    spmv_t spmv_;

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

    static info_t default_info() {
        return info_t{0};
    }

    __GENLX_ARCH_INL__
    static gather_t default_result() {
        return gather_t{0};
    }

    __GENLX_DEV_INL__
    static info_t construct(const vertex_t& vid, const dstatus_t& d_status) {
        return info_t{};
    }

    __GENLX_DEV_INL__
    static gather_t gather(const weight_t& weight, const info_t& info) {
        return gather_t{};    
    }

    __GENLX_DEV_INL__
    static gather_t reduce(const gather_t& lhs, const gather_t& rhs) {}

    /// return true when activating this vertex
    __GENLX_DEV_INL__
    static bool apply(const vertex_t& vid, const gather_t& res, dstatus_t& d_status) {}
};


} // namespace graph_genlx