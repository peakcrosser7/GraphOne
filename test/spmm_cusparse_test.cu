
#include "GraphOne/type.hpp"
#include "GraphOne/loader/graph_loader.h"
#include "GraphOne/mat/dense.h"
#include "GraphOne/archi/blas/spmm/spmm.h"
#include "GraphOne/utils/log.hpp"

#include "test.hpp"

using namespace std;
using namespace graph_one;

constexpr arch_t arch = arch_t::cuda;
using feat_t = float;

struct SpmmFunctor {
    static constexpr bool use_cusparse = true;
};


int main() {
    LoadEdgeOpts opts;
    opts.is_directed = true;
    GraphLoader<vstart_t::FROM_0_TO_0> loader;
    auto csr = loader.LoadEdgesFromTxt<feat_t>("../datasets/sample/sample.adj", opts)
                .ToCsr<arch>();
    std::vector<std::vector<feat_t>> feats(csr.n_cols);
    loader.LoadVertexStatusFromTxt<feat_t>("../datasets/sample/sample_more.feat", [&](vid_t vid, std::vector<feat_t>& vdata) {
        feats[vid] = vdata;
        return true;
    });
    printx("csr: ", csr);

    DenseMat<arch, feat_t> matB(feats);
    DenseMat<arch, feat_t> matC(csr.n_rows, matB.n_cols);

    printx("matB: ", matB);

    auto spmm = blas::MakeCsrSpMM<arch, blas::SpmmCudaCsrCusparse, SpmmFunctor, true, true>(
            csr.n_rows, csr.n_cols, csr.nnz, matB.n_cols, 
            csr.row_offsets.data(), csr.col_indices.data(), csr.values.data(),
            matB.data(), matC.data());
    spmm();
    DenseMat<arch_t::cpu, feat_t> h_matC(matC);

    std::vector<std::vector<feat_t>> ref = {
        {21, 7, 12, 15, 15},
        {4, 2, 1, 1, 2},
        {10, 2, 10, 12, 8},
        {2,4, 6, 12, 14},
        {16, 20, 24, 28, 32}
    };
    DenseMat<arch_t::cpu, feat_t> matC_ref(ref);


    printx("matC: ", h_matC);
    printx("matC_ref: ", matC_ref);

    bool pass = true;
    for (int i = 0; i < matC_ref.n_rows; ++i) {
        for (int j = 0; j < matC_ref.n_cols; ++j) {
            if (h_matC.at(i, j) != matC_ref.at(i, j)) {
                LOG_WARNING("ERROR at (", i, ", ", j, ")");
                pass = false;
                break;
            }
        }
    }
    if (pass) {
        printx("PASS");
    }

    return 0;
}