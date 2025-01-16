
#include <iostream>

#define DEBUG_LOG
#include "GraphOne/graph_one.h"
#include "GraphOne/engine/blas_gnn.h"

using namespace std;
using namespace graph_one;

constexpr arch_t arch = arch_t::cuda;
using feat_t = float;
using label_t = int32_t;

struct gcn_hstatus_t: gnn::status_t<arch, feat_t> {
    gnn::param_t<arch, feat_t> weight;
    gnn::param_t<arch, feat_t> bias;
};

template <typename graph_t>
struct GcnComp: gnn::Componment<graph_t, gcn_hstatus_t> {
    using gnn::Componment<graph_t, gcn_hstatus_t>::Componment;
    void Init() override {

    }
};

struct GcnFunctor: BlasGnnFunctor<arch, vid_t, feat_t, gcn_hstatus_t> {
    static constexpr bool need_construct_egdes = false;
    static constexpr bool use_cusparse_spmm = true;

    static void apply(DenseMat<arch, feat_t>& ipt_acc_emb,
                    DenseMat<arch, feat_t>& opt_vertex_emb,
                    gcn_hstatus_t& d_status) {
        auto weight = *d_status.weight;
        auto m = ipt_acc_emb.n_rows, n = weight.n_cols, k = ipt_acc_emb.n_cols;
        auto gemm = blas::MakeGemm<arch, blas::GemmCutlass, true, true, true>(m, n, k, 
            ipt_acc_emb.data(), weight.data(), opt_vertex_emb.data());
        gemm();
    }

};

class GraphConvolutionImpl : public gnn::Module<arch> {
private:
    int in_features;
    int out_features;
    gnn::param_t<arch, feat_t> weight;
    gnn::param_t<arch, feat_t> bias;

public:
    GraphConvolutionImpl(int in_dim, int out_dim, bool use_bias=true)
    : Module(), in_features(in_dim), out_features(out_dim),
      weight(register_parameter(NAME_OF(weight), gnn::tensor<arch>(in_dim, out_dim))) {
        if (use_bias) {
            bias = register_parameter(NAME_OF(bias), gnn::tensor<arch>(1, out_dim));
        } else {
            bias = register_parameter(NAME_OF(bias), nullptr);
        }
    }

    template <typename comp_t>
    gnn::tensor_t<arch, feat_t> forward(comp_t& comp) {
        comp.h_status.weight = weight;
        comp.h_status.bias = bias;
        gnn::EngineForward<GcnFunctor>(comp);
        return comp.h_status.vertex_emb;
    }

};
GRAPH_MODULE(GraphConvolution);


class GCNImpl : public gnn::Module<arch> {
private:
    GraphConvolution gc1;
    GraphConvolution gc2;
    float dropout;

public:
    static constexpr const char* ctor_args[] = {"nfeat", "nhid", "nclass", "drop"};

    GCNImpl(int nfeat, int nhid, int nclass, float drop) 
    : Module(), 
      gc1(register_module(NAME_OF(gc1), GraphConvolution(nfeat, nhid))), 
      gc2(register_module(NAME_OF(gc2), GraphConvolution(nhid, nclass))), 
      dropout(drop) {}

    template <typename comp_t>
    gnn::tensor_t<arch, feat_t> forward(comp_t& comp, ) {
        gc1(comp);
        auto x = gc2(comp);
        // gnn::log_softmax(x, dim=1);
        return x;
    }

};
GRAPH_MODULE(GCN);



int main(int argc, char *argv[]) {
    CLI::App app;
    string input_graph;
    string input_embedding;
    string input_model_prefix;
    string output_path;
    bool reoreder_vid = true;
    LoadEdgeOpts opts;
    add_common_args(app, input_graph, output_path, reoreder_vid, opts);
    app.add_option("-e,--input_embed", input_embedding, "input embedding dataset file")->required();
    app.add_option("-m,--input_model", input_model_prefix, "file prefix of the input GNN model, including json and onnx files")->required();
    CLI11_PARSE(app, argc, argv);

    GraphLoader<vstart_t::FROM_0_TO_0> loader(reoreder_vid);
    auto cache = loader.LoadEdgesFromTxt<feat_t>(input_graph, opts);
    auto g = graph::build<arch_t::cuda, BlasViews>(cache);
    using graph_t = decltype(g);

    std::vector<std::vector<feat_t>> feats(g.num_vertices());
    std::vector<label_t> labels(g.num_vertices());
    loader.LoadVertexStatusFromTxt<feat_t>(input_embedding, [&](vid_t vid, std::vector<feat_t>& vdata) {
        labels[vid] = static_cast<label_t>(vdata.back());
        vdata.pop_back();
        feats[vid] = vdata;
        return true;
    });
    gnn::vprop_t<arch, feat_t, label_t, vid_t> vprop(feats, labels);
    if (g.num_vertices() != vprop.num_vertices) {
        LOG_ERROR("the graph and vertex props should have the same number of vertices: "
            "graph.num_vertices()=", g.num_vertices(), ", vprop.num_vertices=", vprop.num_vertices);
    }

    auto model = gnn::build_module<GCN>(input_model_prefix + ".json");
    auto mp = OnnxLoader::LoadInitializer<arch, feat_t>(input_model_prefix + ".onnx");
    model->load_state_dict(std::move(mp));

    gcn_hstatus_t hstatus{};
    auto comp = GcnComp<graph_t>{g, hstatus};

    Run(comp, model);

    return 0;
}