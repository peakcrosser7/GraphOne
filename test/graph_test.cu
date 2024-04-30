#include "GraphOne/loader/graph_loader.h"
#include "GraphOne/vec/dense.h"
#include "GraphOne/mat/dense.h"
#include "GraphOne/graph/builder.h"
#include "GraphOne/domain.h"

#include "test.hpp"

using namespace std;
using namespace graph_one;

int main () {
    // install_oneshot_signal_handlers();
    
    GraphLoader<> loader;
    EdgeCache<vstart_t::FROM_0_TO_0, int, vid_t, eid_t> cache = 
        loader.LoadEdgesFromTxt<int>("../datasets/sample/sample.adj");
    CsrMat<arch_t::cuda, int, vid_t, eid_t, vstart_t::FROM_0_TO_0> csr = 
        cache.ToCsr<arch_t::cuda, true>();

    DenseMat<arch_t::cpu, double> feats(csr.n_rows, 4);
    DenseVec<arch_t::cpu, int32_t> labels(csr.n_rows);
    loader.LoadVertexStatusFromTxt<int32_t>("../datasets/sample/sample_more.feat",[&](vid_t vid, std::vector<int32_t>& vdata) {
        if (vdata.size() < 5) {
            return false;
        }
        for (int i = 0; i < 4; ++i) {
            feats[vid][i] = vdata[i];
        }
        labels[vid] = vdata.back();
        return true;
    });
    cout << feats.ToString() << endl;
    cout << labels.ToString() << endl;

    auto feat = loader.LoadVertexVecFromTxt<arch_t::cuda, int>("../datasets/sample/sample_single.feat", csr.n_rows);
    cout << feat.ToString() << endl;

    DenseVec<arch_t::cpu, float> feat2(csr.n_rows);
    loader.LoadVertexStatusFromTxt<float>("../datasets/sample/sample_single.feat", [&](vid_t vid, std::vector<float>& vdata) {
        feat2[vid] = vdata.front();
        return true;
    });
    cout << feat2.ToString() << endl;

    gnn::vprop_t<arch_t::cpu> vprops1(csr.n_rows, 4);
    loader.LoadVertexStatusFromTxt<int32_t>("../datasets/sample/sample_more.feat", [&](vid_t vid, std::vector<int32_t>& vdata) {
        if (vdata.size() < 5) {
            return false;
        }
        for (int i = 0; i < 4; ++i) {
            vprops1.features[vid][i] = vdata[i];
        }
        vprops1.labels[vid] = vdata.back();
        return true;
    });
    cout << "vprop1:" <<vprops1.ToString() << endl;

    gnn::vprop_t<arch_t::cuda> vprops2;
    vprops2.features = feats;
    vprops2.labels = labels;
    cout << "vprop2:" <<vprops2.ToString() << endl;

    vector<vector<double>> feat_vec(csr.n_rows);
    DenseVec<arch_t::cpu, int32_t> label_vec(csr.n_rows);
    loader.LoadVertexStatusFromTxt<double>("../datasets/sample/sample_more.feat", [&](vid_t vid, std::vector<double>& vdata) {
        label_vec[vid] = vdata.back();
        vdata.pop_back();
        feat_vec[vid] = std::move(vdata);
        return true;
    });
    gnn::vprop_t<arch_t::cuda> vprops3;
    vprops3.features = DenseMat<arch_t::cuda, double>(feat_vec);
    vprops3.labels = label_vec;
    cout << "vprop3:" << vprops3.ToString() << endl;


    // using csr_t = decltype(csr);
    // using graph_t = Graph<
    //         arch_t::cuda, graph_view_t::normal|graph_view_t::csr, 
    //         csr_t::vstart_value, 
    //         gnn::vprop_t<arch_t::cuda>,
    //         csr_t::index_type,
    //         csr_t::offset_type,
    //         csr_t::value_type
    //     >;
    // auto g = graph_t(std::move(csr), std::move(vprops2));
    auto g = graph::build<
            arch_t::cuda,
            graph_view_t::normal|graph_view_t::csr
        >(cache);
    cout << g.ToString() << endl;

    cout << csr.ToString() << endl;

    return 0;
}
