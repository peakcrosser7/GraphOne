#include "GraphGenlX/loader/loader.h"
#include "GraphGenlX/vec/dense.h"
#include "GraphGenlX/mat/dense.h"
#include "GraphGenlX/graph/builder.h"
#include "GraphGenlX/domain.h"

#include "test.hpp"

using namespace std;
using namespace graph_genlx;

int main () {
    // install_oneshot_signal_handlers();
    
    Loader<> loader;
    auto csr = loader.LoadCsrFromTxt<arch_t::cuda, int>("../data/sample/sample.adj");
    
    DenseMat<arch_t::cpu, double> feats(csr.n_rows, 4);
    DenseVec<arch_t::cpu, int32_t> labels(csr.n_rows);
    loader.LoadVertexStatusFromTxt<int32_t>("../data/sample/sample_more.feat",[&](vid_t vid, std::vector<int32_t>& vdata) {
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

    auto feat = loader.LoadVertexVecFromTxt<arch_t::cuda, int>("../data/sample/sample_single.feat", csr.n_rows);
    cout << feat.ToString() << endl;

    DenseVec<arch_t::cpu, float> feat2(csr.n_rows);
    loader.LoadVertexStatusFromTxt<float>("../data/sample/sample_single.feat", [&](vid_t vid, std::vector<float>& vdata) {
        feat2[vid] = vdata.front();
        return true;
    });
    cout << feat2.ToString() << endl;

    gnn::vprop_t<arch_t::cpu> vprops1(csr.n_rows, 4);
    loader.LoadVertexStatusFromTxt<int32_t>("../data/sample/sample_more.feat",[&](vid_t vid, std::vector<int32_t>& vdata) {
        if (vdata.size() < 5) {
            return false;
        }
        for (int i = 0; i < 4; ++i) {
            vprops1.features[vid][i] = vdata[i];
        }
        vprops1.labels[vid] = vdata.back();
        return true;
    });
    cout << vprops1.ToString() << endl;

    gnn::vprop_t<arch_t::cuda> vprops2;
    vprops2.features = feats;
    vprops2.labels = labels;
    cout << vprops2.ToString() << endl;

    auto g = graph::FromCsr<graph_view_t::csc>(std::move(csr), std::move(vprops2));
    cout << g.ToString() << endl;

    cout << csr.ToString() << endl;

    return 0;
}
