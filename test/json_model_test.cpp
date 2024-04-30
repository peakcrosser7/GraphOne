#include <fstream>
#include <iostream>

#include "nlohmann/json.hpp"

#include "GraphOne/type.hpp"
#include "GraphOne/gnn/module.h"
#include "GraphOne/gnn/build.h"
#include "GraphOne/utils/refl.hpp"

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
      gc1(register_module(NAME_OF(gc1), GraphConvolution(nfeat, nhid, false))), 
      gc2(register_module(NAME_OF(gc2), GraphConvolution(nhid, nclass))), 
      dropout(drop) {}

};
GRAPH_MODULE(GCN);

// 基本的函数模板来打印tuple的内容（仅用于验证结果）
template<typename Tuple, std::size_t... Is>
void printTuple(const Tuple& t, std::index_sequence<Is...>) {
    ((std::cout << (Is == 0 ? "" : ", ") << std::get<Is>(t)), ...);
    std::cout << std::endl;
}

// 基本的函数模板来打印tuple的内容（仅用于验证结果）
template<typename Tuple>
void printTuple(const Tuple& t) {
    printTuple<Tuple>(t, std::make_index_sequence<std::tuple_size_v<Tuple> > {});
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <json_file>\n";
        return -1;
    }
    ifstream fin(argv[1]);
    json data = json::parse(fin);

    for (json::iterator it = data.begin(); it != data.end(); ++it) {
        std::cout << it.key() << " : " << it.value() << "\n";
    }

    // constexpr static auto ctor_nparams = refl::fields_number_ctor<GCNImpl>(0);
    // static_assert(ctor_nparams == 4);

    // Check the types of ctor parameters were detected correctly.
    // using ctor_param_type = refl::as_tuple<GCNImpl>;
    // static_assert(std::is_same_v<ctor_param_type,
    //                             std::tuple<int, int, int, float>>);

    // constexpr auto& json_map = GCNImpl::ctor_params;
    // auto t = gnn::JsonToTuple<ctor_param_type, sizeof(json_map) / sizeof(char*)>(data, GCNImpl::ctor_params);
    // printTuple<decltype(t)>(t);
    
    auto model = gnn::build_module<GCN>(data);

    // GCN model(data["nfeat"], data["nhid"], data["nclass"], data["drop"]);
    auto dict = model->named_parameters();
    for (auto& [k, v]: dict) {
        cout << k << ":" << v->ToString() << endl;
    }

    return 0;
}