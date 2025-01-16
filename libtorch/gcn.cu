#include <iostream>
#include <chrono>
#include <cstdio>

#include <torch/torch.h>

#include "load_data.hpp"

#include "spmm.cuh"

using namespace std;

// cub::CachingDeviceAllocator g_allocator(true);

torch::Tensor myspmm(const torch::Tensor& spmat, const torch::Tensor& dense) {
    auto m = spmat.size(0), k = spmat.size(1), n = dense.size(1);
    auto nnz = spmat._nnz();
    torch::Tensor output = torch::empty({m, n}, 
        torch::TensorOptions()
            .device(dense.device())
            .dtype(dense.dtype()));
    auto dispather = MakeCsrSpMM(m, k, nnz, n, 
                                 spmat.crow_indices().data_ptr<int64_t>(),
                                 spmat.col_indices().data_ptr<int64_t>(),
                                 spmat.values().data_ptr<float>(),
                                 dense.data_ptr<float>(),
                                 output.data_ptr<float>(), dense.device());
    dispather();
    return output;
}

class GCNLayerImpl : public torch::nn::Module {
public:
    GCNLayerImpl(int64_t in_features, int64_t out_features, bool bias = true) 
        : in_features_(in_features), out_features_(out_features) {
        
        // Initialize weight parameter
        weight_ = register_parameter("weight", torch::zeros({in_features, out_features}, 
            torch::TensorOptions().requires_grad(true)));

        // Initialize bias parameter if required
        if (bias) {
            bias_ = register_parameter("bias", torch::zeros({out_features}, 
                torch::TensorOptions().requires_grad(true)));
        }

        // Reset parameters to uniform initialization
        reset_parameters();
    }

    void reset_parameters() {
        // Compute standard deviation for uniform initialization
        double stdv = 1.0 / std::sqrt(weight_.size(1));

        // Uniform initialization for weight
        torch::NoGradGuard no_grad;
        weight_.uniform_(-stdv, stdv);

        // Uniform initialization for bias if it exists
        if (bias_.defined()) {
            bias_.uniform_(-stdv, stdv);
        }
    }

    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& adj) {

        // cout << "debug1\n";
        // Compute support: matrix multiplication of input and weight
        auto support = mm(input, weight_);
        // cout << "debug2\n";
        // Sparse matrix multiplication with adjacency matrix
        auto output = torch::mm(adj, support);
        // auto output = myspmm(adj, support);
        // cout << "debug3\n";
        // Add bias if it exists
        if (bias_.defined()) {
            output += bias_;
        }

        return output;
    }

private:
    int64_t in_features_;
    int64_t out_features_;
    torch::Tensor weight_;
    torch::Tensor bias_;
};
TORCH_MODULE(GCNLayer);


class GCNImpl : public torch::nn::Module {
public:
    GCNImpl(int64_t nfeat, int64_t nhid, int64_t nclass, double dropout)
        : dropout_(dropout) {
        
        // First Graph Convolution layer
        gc1_ = register_module("gc1", GCNLayer(nfeat, nhid, /*bias=*/false));

        // Second Graph Convolution layer
        gc2_ = register_module("gc2", GCNLayer(nhid, nclass));
    }

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& adj) {
        // First graph convolution with ReLU activation
        auto h = torch::relu(gc1_->forward(x, adj));

        // Dropout (only applied during training)
        if (is_training()) {
            h = torch::dropout(h, dropout_, /*train=*/true);
        }

        // Second graph convolution
        h = gc2_->forward(h, adj);

        // Log softmax for classification
        return torch::log_softmax(h, /*dim=*/1);
    }

private:
    GCNLayer gc1_{nullptr};
    GCNLayer gc2_{nullptr};
    double dropout_;
};
TORCH_MODULE(GCN);

struct TrainingOptions {
public:
    bool cuda = false;
    bool fastmode = false;
    int epochs = 200;
    double lr = 0.01;
    double weight_decay = 5e-4;
};

torch::Tensor accuracy(const torch::Tensor& output, const torch::Tensor& labels) {
    // 获取每行最大值的索引
    auto preds = std::get<1>(output.max(1)).to(labels.dtype());
    
    // 比较预测与真实标签是否相等
    auto correct = preds.eq(labels).to(torch::kDouble);
    
    // 计算正确预测的数量
    auto correct_count = correct.sum().item<double>();
    
    // 计算准确率
    return torch::tensor(correct_count / labels.size(0), torch::kDouble);
}

void train(GCN& model, GraphDataset& dataset, torch::optim::Optimizer& optimizer, int epoch) {

    torch::Device device(torch::kCUDA, 0);
    dataset.features = dataset.features.to(device);
    dataset.adj = dataset.adj.to(device);
    model->to(device);

    // Training mode
    // model->train();
    // optimizer.zero_grad();

    // // Forward pass
    // auto output = model->forward(dataset.features, dataset.adj);
    

    // cout << "debug4\n";

    // // Training loss and accuracy
    // auto loss_train = torch::nll_loss(
    //     output.index_select(0, dataset.idx_train), 
    //     dataset.labels.index_select(0, dataset.idx_train)
    // );
    // // cout << "debug5\n";
    // auto acc_train = accuracy(
    //     output.index_select(0, dataset.idx_train), 
    //     dataset.labels.index_select(0, dataset.idx_train)
    // );
    // // cout << "debug6\n";

    // // Backward pass
    // // loss_train.backward();
    // // cout << "debug7\n";
    // // optimizer.step();
    // // cout << "debug8\n";

    // // Validation 
    model->eval();
    auto start_time = std::chrono::high_resolution_clock::now();

    auto output = model->forward(dataset.features, dataset.adj);

    auto end_time = std::chrono::high_resolution_clock::now();

    // // cout << "debug9\n";

    // auto loss_val = torch::nll_loss(
    //     output.index_select(0, dataset.idx_val), 
    //     dataset.labels.index_select(0, dataset.idx_val)
    // );
    // auto acc_val = accuracy(
    //     output.index_select(0, dataset.idx_val), 
    //     dataset.labels.index_select(0, dataset.idx_val)
    // );

    // Calculate training time
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Print epoch statistics
    // printf("Epoch: %04d loss_train: %.4f acc_train: %.4f loss_val: %.4f acc_val: %.4f time: %.4fms\n",
    printf("Epoch: %04d  time: %.4fms\n",
            epoch + 1, 
            // loss_train.item<float>(), 
            // acc_train.item<float>(), 
            // loss_val.item<float>(), 
            // acc_val.item<float>(), 
            duration.count() / 1000.0);
}


int main() {
    auto before = std::chrono::high_resolution_clock::now();
    GraphDataset dataset = load_data("../data/citeseer/", "citeseer");
    int64_t nclass = dataset.labels.max().item<int64_t>() + 1;
    int64_t nhid = nclass * 2 + 2;
    
    GCN model(dataset.features.size(1), nhid, nclass, 0.5);
    auto start = std::chrono::high_resolution_clock::now();
    auto precost = std::chrono::duration_cast<std::chrono::milliseconds>(start - before);
    std::cout << "Prepare time: " << precost.count() << "ms\n";
    TrainingOptions training_options;

    // for (auto p: model->named_parameters()) {
    //     std::cout << p.key() << p.value().is_leaf() << std::endl;
    // }

    torch::optim::Adam optimizer(model->parameters(), 
                    torch::optim::AdamOptions(training_options.lr)
                        .weight_decay(training_options.weight_decay));

    for(int e = 0; e < 100; ++e) {
        train(model, dataset, optimizer, e);
    }
    
    model->eval();

    return 0;
}