#include <iostream>

#include "nlohmann/json.hpp"

#include "GraphOne/type.hpp"
#include "GraphOne/loader/onnx_loader.h"
#include "GraphOne/gnn/build.h"
#include "GraphOne/gnn/module.h"
#include "GraphOne/utils/string.hpp"

using namespace std;
using namespace graph_one;
using json = nlohmann::json;

constexpr arch_t arch = arch_t::cpu;

class GraphConvolutionImpl : public gnn::Module<arch> {
private:
    int in_features;
    int out_features;
    gnn::param_t<arch> weight;
    gnn::param_t<arch> bias;

public:
    GraphConvolutionImpl(int in_dim, int out_dim, bool use_bias=true)
    : Module(), in_features(in_dim), out_features(out_dim),
      weight(register_parameter(NAME_OF(weight), gnn::tensor_t<arch>(in_dim, out_dim))) {
        if (use_bias) {
            bias = register_parameter(NAME_OF(bias), gnn::tensor_t<arch>(1, out_dim));
        } else {
            bias = register_parameter(NAME_OF(bias), nullptr);
        }
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

};
GRAPH_MODULE(GCN);

int main(int argc, char *argv[]) {
    // GCN model(5, 3, 6, 0.5);

    // auto dict = model->named_parameters();
    // for (auto& [k, v]: dict) {
    //     cout << k << ":" << v->ToString() << endl;
    // }
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <json_file> <onnx_file>\n";
        return -1;
    }

    ifstream fin(argv[1]);
    json data = json::parse(fin);

    auto mp = OnnxLoader::LoadInitializer<arch_t::cpu, float>(argv[2]);
    cout << utils::HashmapToString(mp, [](const auto& p) {
        return "(" + utils::ToString(p.first) + ":" + 
            utils::ToString(p.second->n_rows) + "-" + utils::ToString(p.second->n_cols) + ")";
    }) << endl;

    cout << "-------------------------------\n";

    auto model = gnn::build_module<GCN>(data);
    model->load_state_dict(std::move(mp));

    auto dict = model->named_parameters();
    for (auto& [k, v]: dict) {
        cout << k << ":" << v->ToString() << endl;
    }

    cout << "-------------------------------\n";

    // cout << utils::HashmapToString(mp, [](const auto& p) {
    //     return "(" + utils::ToString(p.first) + ":" + 
    //         utils::ToString(*(p.second)) + ")";
    // }) << endl;

    return 0;
}
