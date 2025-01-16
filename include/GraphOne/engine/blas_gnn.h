#pragma once

#include <type_traits>
#include <memory>

#include "GraphOne/type.hpp"
#include "GraphOne/mat/dense.h"
#include "GraphOne/mat/csr.h"
#include "GraphOne/engine/base.h"
#include "GraphOne/frontier/gnn_frontier.h"
#include "GraphOne/archi/macro/macro.h"
#include "GraphOne/archi/blas/spmm/spmm.h"

namespace graph_one {

namespace engine {

template <typename functor_t, typename comp_t>
class BlasGnnEngine : public GnnEngine<comp_t> {
public:
    using base_t = GnnEngine<comp_t>;
    using graph_t = typename base_t::graph_type;
    using hstatus_t = typename base_t::hstatus_type;
    using dstatus_t = typename base_t::dstatus_type;

    using vertex_t = typename graph_t::vertex_type;
    using edge_t = typename graph_t::edge_type;
    using feat_t = typename functor_t::feat_type;
    using dim_t = typename comp_t::dim_type;

    static_assert(std::is_same_v<feat_t, typename graph_t::weight_type>);

    struct SpmmFunctor {
        static constexpr bool use_cusparse = functor_t::use_cusparse_spmm;

        __ONE_DEV_INL__
        static feat_t combine(const feat_t& weight, const feat_t& feat) {
            return feat_t{};
        }

        __ONE_DEV_INL__
        static feat_t reduce(const feat_t& lhs, const feat_t& rhs) {
            return feat_t{};
        }
    };

    using spmm_kind_t = blas::SpmmCudaCsrCusparse;
    using spmm_t = blas::SpmmDispatcher<spmm_kind_t, SpmmFunctor, 
                                        vertex_t, edge_t, feat_t,
                                        true, feat_t, true, feat_t, feat_t>;

    constexpr static arch_t arch = graph_t::arch_value;

    BlasGnnEngine(comp_t &comp)
        : base_t(comp), vertex_emb_(comp.vertex_emb), acc_emb_(gnn::tensor(comp.vertex_emb)) {
        auto graph_mat = this->graph_.graph_mat();
        static_assert(std::is_same_v<decltype(graph_mat), CsrMat<arch, feat_t, vertex_t>>);

        if constexpr (functor_t::need_construct_egdes) {
            vertex_t num_v = this->graph_.num_vertices();
            dim_t feat_dim = comp.feat_dim();

            ipt_emb_ = DenseMat<arch, feat_t, vertex_t>(num_v, feat_dim);
            ipt_transpose_emb_ = DenseMat<arch, feat_t, vertex_t>(feat_dim, num_v);
            ipt_edge_emb_ = graph_mat;
            edge_emb_ptr_ = &ipt_edge_emb_;
        } else {
            edge_emb_ptr_ = &graph_mat;
        }
        spmm_ = std::make_unique<spmm_t>(MakeCsrSpMM(new*(this->edge_emb_ptr_), this->vertex_emb_, this->acc_emb_));
    }

    void Forward() {
        Construct(this->graph_.graph_mat(), this->vertex_emb_.get(), this->ipt_edge_emb_, this->h_status_);
        Gather(*(this->spmm_.get()));
        Apply(this->acc_emb_, this->vertex_emb_.get(), this->h_status_);
    }

    void Construct(const CsrMat<arch, feat_t, vertex_t>& graph_mat,
            DenseMat<arch, feat_t, vertex_t>& ipt_vertex_emb,
            CsrMat<arch, feat_t, vertex_t>& opt_edge_emb,
            hstatus_t& h_status) {
        if constexpr (functor_t::need_construct_egdes) {
            functor_t::construct_sdmm_ipt_emb(ipt_vertex_emb, this->ipt_emb_, h_status);
            functor_t::construct_sdmm_ipt_trans_emb(ipt_vertex_emb, this->ipt_transpose_emb_, h_status);
            // SddmmCsr(this->ipt_emb_, this->ipt_transpose_emb_, graph_mat, opt_edge_emb);
        }
    }
    
    void Gather(spmm_t& spmm) {
        spmm();
    }

    void Apply(DenseMat<arch, feat_t, vertex_t>& ipt_acc_emb,
            DenseMat<arch, feat_t, vertex_t>& opt_vertex_emb,
            hstatus_t& h_status) {
        functor_t::apply(ipt_acc_emb, opt_vertex_emb, h_status);
    }

protected:
    static spmm_t make_spmm_(const graph_t& graph, 
                             gnn::tensor_t<arch, feat_t, vertex_t>& matB, 
                             gnn::tensor_t<arch, feat_t, vertex_t>& matC) {
        auto graph_ref = graph.ToArch();
        return blas::MakeCsrSpMM<arch, spmm_kind_t, true, true>(
            graph_ref.num_vertices, graph_ref.num_vertices, graph_ref.num_edges, matB->n_cols, 
            graph_ref.row_offsets, graph_ref.col_indices, graph_ref.values,
            matB->data(), matC->data()
        );
    }


    gnn::tensor_t<arch, feat_t, vertex_t> vertex_emb_;

    CsrMat<arch, feat_t, vertex_t>* edge_emb_ptr_{nullptr};
    DenseMat<arch, feat_t, vertex_t> ipt_emb_;
    DenseMat<arch, feat_t, vertex_t> ipt_transpose_emb_;
    CsrMat<arch, feat_t, vertex_t> ipt_edge_emb_;
    gnn::tensor_t<arch, feat_t, vertex_t> acc_emb_;

    std::unique_ptr<spmm_t> spmm_;
};

} // namespace engine

template <arch_t arch, typename vertex_t,
          typename feat_t, typename hstatus_t>
struct BlasGnnFunctor {
    template <typename functor_t, typename comp_t>
    using engine_type = engine::BlasGnnEngine<functor_t, comp_t>;
    using feat_type = feat_t;

    static constexpr bool need_construct_egdes = false;
    static constexpr bool need_construct_post = false;
    static constexpr bool use_cusparse_spmm = true;

    static void construct_sdmm_ipt_emb(const DenseMat<arch, feat_t, vertex_t>& ipt_vertex_emb,
                                       DenseMat<arch, feat_t, vertex_t>& ipt_emb,
                                       hstatus_t& h_status) {}

    static void construct_sdmm_ipt_trans_emb(DenseMat<arch, feat_t, vertex_t>& ipt_transpose_emb,
                                        hstatus_t& h_status) {}

    static feat_t construct_combine(const feat_t* lhs, const feat_t* rhs) {
        return feat_t{};
    }

    static feat_t construct_reduce(const feat_t* lhs, const feat_t* rhs) {
        return feat_t{};
    }

    static feat_t construct_post(const feat_t& feat, const feat_t& weight) {
        return feat_t{};
    }

    __ONE_DEV_INL__
    static feat_t gather_combine(const feat_t& weight, const feat_t& feat) {
        return feat_t{};
    }

    __ONE_DEV_INL__
    static feat_t gather_reduce(const feat_t& lhs, const feat_t& rhs) {
        return feat_t{};
    }

    static void apply(DenseMat<arch, feat_t, vertex_t>& ipt_acc_emb,
                      DenseMat<arch, feat_t, vertex_t>& opt_vertex_emb,
                      hstatus_t& h_status) {}
};

namespace gnn {

template <typename functor_t, typename comp_t>
void EngineForward(comp_t& comp) {
    using engine_t =
        typename functor_t::engine_type<functor_t, comp_t>;
    
    engine_t engine(comp);

    comp.BeforeEngine();
    engine.Forward();
    comp.AfterEngine();


}

} // namespace gnn

} // namespace graph_one


