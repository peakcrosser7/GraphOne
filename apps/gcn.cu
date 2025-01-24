#include <string>

#include "CLI11/CLI11.hpp"

#include "graph_one/graph_one.h"

using namespace graph_one;

class GCNLayerImpl : public graph_one::nn::Module {
public:
    GCNLayerImpl(int64_t in_features, int64_t out_features, bool bias = true) 
        : in_features_(in_features), out_features_(out_features) {
        
        // Initialize weight parameter
        weight_ = register_parameter("weight", torch::zeros({in_features, out_features}));

        // Initialize bias parameter if required
        if (bias) {
            bias_ = register_parameter("bias", torch::zeros({out_features}));
        }

        // Reset parameters to uniform initialization
        reset_parameters();
    }

    void reset_parameters() {
        // Compute standard deviation for uniform initialization
        double stdv = 1.0 / std::sqrt(weight_.size(1));

        // Uniform initialization for weight
        weight_.uniform_(-stdv, stdv);

        // Uniform initialization for bias if it exists
        if (bias_.defined()) {
            bias_.uniform_(-stdv, stdv);
        }
    }

    Tensor forward(GraphX& g, const Tensor& input) {

        // Compute support: matrix multiplication of input and weight
        Tensor support = mm(input, weight_);

        // Sparse matrix multiplication with adjacency matrix
        auto functor = make_functor(op::Mult{}, op::Add{});
        Tensor output = GraphForward(g, support, {}, functor);

        // Add bias if it exists
        if (bias_.defined()) {
            output += bias_;
        }

        return output;
    }

private:
    int64_t in_features_;
    int64_t out_features_;
    Tensor weight_;
    Tensor bias_;
};
TORCH_MODULE(GCNLayer);


class GCNImpl : public graph_one::nn::Module {
public:
    GCNImpl(int64_t nfeat, int64_t nhid, int64_t nclass) {
        
        // First Graph Convolution layer
        gc1_ = register_module("gc1", GCNLayer(nfeat, nhid, /*bias=*/false));

        // Second Graph Convolution layer
        gc2_ = register_module("gc2", GCNLayer(nhid, nclass));
    }

    Tensor forward(GraphX& g, const torch::Tensor& x) {
        // First graph convolution with ReLU activation
        Tensor h = torch::relu(gc1_->forward(g, x));

        // Second graph convolution
        h = gc2_->forward(g, h);

        // Log softmax for classification
        return torch::log_softmax(h, /*dim=*/1);
    }

private:
    GCNLayer gc1_{nullptr};
    GCNLayer gc2_{nullptr};
};
TORCH_MODULE(GCN);


int main(int argc, char *argv[]) {
    std::string input_graph;

    // GCN Parameters
    int nfeat = 96;
    int nclass = 10;
    int nhid = nclass * 2 + 2;

    CLI::App app;
    app.add_option("-i,--input_graph", input_graph, "input graph dataset file")->required();
    app.add_option("--nfeat", nfeat, "size of vertices' feature in GCN (default 96)");
    app.add_option("--ncls", nclass, "number of classes of each vertex in GCN (default 10)");
    CLI11_PARSE(app, argc, argv);    

    Device device(kCUDA);
    
    GraphX g = load_graph(input_graph, device);
    GCN model(nfeat, nhid, nclass);
    model->to(device);

    Tensor feat = make_rand<float>({g.num_vertices(), nfeat}, device);

    Tensor output = model->forward(g, feat);

    return 0;
}