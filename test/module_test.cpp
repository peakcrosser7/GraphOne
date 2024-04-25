
#include <iostream>

#include "GraphOne/gnn/module.h"

using namespace std;
using namespace graph_one;


constexpr arch_t arch = arch_t::cpu;

class DenseImpl : public gnn::Module<arch> {
private:
    gnn::param_t<arch> weight;

public:
    DenseImpl(int dim) : Module(), 
        weight(register_parameter(NAME_OF(weight), gnn::tensor_t<arch>(1, dim))) {}

    std::string ToString() const {
        return "Dense{ weight:" + weight->ToString() + " }"; 
    }
};
GRAPH_MODULE(Dense);

class LinearImpl: public gnn::Module<arch> {
private:
    gnn::param_t<arch> weight;
    gnn::param_t<arch> bias;
    Dense den_layer;

public:
    LinearImpl(int indim, int outdim)
     : Module(), 
       weight(register_parameter(NAME_OF(weight), gnn::tensor_t<arch>(indim, outdim))),
       bias(register_parameter(NAME_OF(bias), gnn::tensor_t<arch>(1, outdim))),
       den_layer(register_module(NAME_OF(den_layer), Dense(4))) {} 


    std::string ToString() const {
        return "Linear{ weight:" + weight->ToString() + ", "
            + "bias:" + bias->ToString() + " }";
    }
};
GRAPH_MODULE(Linear);

int main() {
    Linear model(5, 3);

    auto dict = model->named_parameters();
    for (auto& [k, v]: dict) {
        cout << k << ":" << v->ToString() << endl;
    }

    return 0;
}