
#include <limits>
#include <vector>

#include "GraphOne/base/buffer.h"
#include "GraphOne/loader/graph_loader.h"
#include "GraphOne/mat/convert.h"
#include "GraphOne/archi/blas/SpMV/spmv.h"
#include "GraphOne/archi/thrust/thrust.h"
#include "GraphOne/utils.h"

#include "test.hpp"


using namespace std;
using namespace graph_one;

constexpr arch_t arch = arch_t::cuda;
using dist_t = int;

constexpr dist_t kMaxDist = numeric_limits<dist_t>::max();

struct SSSPFunctor {
    static dist_t default_info() {
        return kMaxDist;
        // return 0;
    }

    __ONE_ARCH_INL__
    static dist_t default_result() {
        return kMaxDist;
        // return 0;
    }

    // __ONE_DEV_INL__
    // static dist_t construct(const vid_t& vid, const sssp_dstatus_t& d_status) {
    //     return d_status.dists[vid];
    // }

    __ONE_DEV_INL__
    static dist_t gather(const dist_t& weight, const dist_t& info) {
        return (info == kMaxDist) ? info : weight + info;
        // return weight * info;
    }

    __ONE_DEV_INL__
    static dist_t reduce(const dist_t& lhs, const dist_t& rhs) {
        return std::min(lhs, rhs);
        // return lhs + rhs;
    }

    // __ONE_DEV_INL__
    // static bool apply(const vid_t& vid, const dist_t& res, sssp_dstatus_t& d_status) {
    //     if (res < d_status.dists[vid]) {
    //         d_status.dists[vid] = res;
    //         return true;
    //     }
    //     return false;
    // }

};

struct SpmvFunctor {
    using functor_t = SSSPFunctor;

    __ONE_DEV_INL__
    static dist_t initialize() {
        return functor_t::default_result();
    }

    __ONE_DEV_INL__
    static dist_t combine(const dist_t& nonzero, const dist_t& x) {
        return functor_t::gather(nonzero, x);
    }

    __ONE_DEV_INL__
    static dist_t reduce(const dist_t& lhs, const dist_t& rhs) {
        return functor_t::reduce(lhs, rhs);
    }
};

using spmv_t =
    blas::SpmvDispatcher<blas::SpmvCudaMergeBased, SpmvFunctor, vid_t,
                            eid_t, dist_t, dist_t, dist_t>;


int main() {
    LoadEdgeOpts opts;
    opts.is_directed = true;
    auto csr = GraphLoader<vstart_t::FROM_0_TO_0>()
                .LoadEdgesFromTxt<dist_t>("../datasets/sample/sample.adj", opts)
                .ToCsr<arch>();
    vector<dist_t> h_x(csr.n_cols, kMaxDist);
    // vector<dist_t> h_x(csr.n_cols, 1);

    h_x[1] = 0;
    
    Buffer<arch, dist_t> x(h_x);
    Buffer<arch, dist_t> y(csr.n_rows);

    LOG_INFO("csr: ", csr);
    LOG_INFO("x: ", x);

    spmv_t spmv = blas::MakeSpMV<arch, blas::SpmvCudaMergeBased, SpmvFunctor>(
            csr.n_rows, csr.n_cols, csr.nnz,
            csr.row_offsets.data(), csr.col_indices.data(), csr.values.data(),
            x.data(), y.data()
        );

    spmv();


    LOG_INFO("y: ", y);

    return 0;
}