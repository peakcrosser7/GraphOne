#pragma once

#include <torch/torch.h>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct GraphDataset {
    torch::Tensor adj;
    torch::Tensor features;
    torch::Tensor labels;
    torch::Tensor idx_train;
    torch::Tensor idx_val;
    torch::Tensor idx_test;
};

// One-hot encoding function
torch::Tensor encode_onehot(const std::vector<std::string>& unique_labels) {
    std::unordered_map<std::string, int> label_map;
    for (size_t i = 0; i < unique_labels.size(); ++i) {
        if (label_map.count(unique_labels[i]) == 0) {
            label_map[unique_labels[i]] = label_map.size();
        }
    }

    int num_classes = label_map.size();
    torch::Tensor one_hot = torch::zeros({static_cast<long>(unique_labels.size()), num_classes}, 
                                         torch::TensorOptions().dtype(torch::kFloat32));
    
    for (size_t i = 0; i < unique_labels.size(); ++i) {
        one_hot[i][label_map[unique_labels[i]]] = 1.0;
    }

    return one_hot;
}

// Sparse tensor normalization

torch::Tensor normalize_sparse_csr(const torch::Tensor& csr_tensor) {
    // 检查是否为 CSR 格式
    if (!csr_tensor.is_sparse_csr()) {
        throw std::runtime_error("Input tensor must be in sparse CSR format.");
    }

    // 获取 CSR 张量的组件
    auto crow_indices = csr_tensor.crow_indices(); // 行指针
    auto col_indices = csr_tensor.col_indices();   // 列索引
    auto values = csr_tensor.values();            // 非零值

    // 计算行和
    auto num_rows = crow_indices.size(0) - 1; // 行数
    torch::Tensor row_sums = torch::empty({num_rows}, values.options());

    for (int64_t i = 0; i < num_rows; ++i) {
        // 获取当前行的范围
        int64_t start = crow_indices[i].item<int64_t>();
        int64_t end = crow_indices[i + 1].item<int64_t>();

        // 计算当前行的和
        row_sums[i] = values.slice(0, start, end).sum();
    }

    // 避免除以零
    row_sums = row_sums.clamp_min(1e-10); // 防止行和为零

    // 对非零值进行归一化
    torch::Tensor normalized_values = values.clone();
    for (int64_t i = 0; i < num_rows; ++i) {
        int64_t start = crow_indices[i].item<int64_t>();
        int64_t end = crow_indices[i + 1].item<int64_t>();
        normalized_values.slice(0, start, end) /= row_sums[i];
    }

    // 构建新的归一化 CSR 张量
    return torch::sparse_csr_tensor(crow_indices, col_indices, normalized_values, 
        csr_tensor.sizes(), csr_tensor.options());
}

GraphDataset load_data(const std::string& path = "./data/cora/", 
                      const std::string& dataset_name = "cora") {
    std::cout << "Loading " << dataset_name << " dataset..." << std::endl;

    // Read content file
    std::ifstream content_file(path + dataset_name + ".content");
    std::string line;
    std::vector<std::vector<std::string>> idx_features_labels;
    std::vector<std::string> unique_labels;

    while (std::getline(content_file, line)) {
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;

        while (iss >> token) {
            tokens.push_back(token);
        }

        idx_features_labels.push_back(tokens);
        unique_labels.push_back(tokens.back());
    }

    std::cout << "debug2\n";

    // Prepare feature indices and values
    std::vector<long> feature_indices_vec;
    std::vector<float> feature_values_vec;
    int num_features = idx_features_labels[0].size() - 2;
    std::cout << "num_feats:" << num_features << "\n";
    std::cout << "num_v:" << idx_features_labels.size() << "\n";

    for (size_t i = 0; i < idx_features_labels.size(); ++i) {
        for (size_t j = 1; j < idx_features_labels[i].size() - 1; ++j) {
            float value = std::stof(idx_features_labels[i][j]);
            if (value > 0) {
                feature_indices_vec.push_back(i);
                feature_indices_vec.push_back(j - 1);
                feature_values_vec.push_back(value);
            }
        }
    }
    std::cout << "debug3\n";

    // Create feature sparse tensor
    auto feature_indices = torch::from_blob(feature_indices_vec.data(), 
        {2, static_cast<long>(feature_indices_vec.size() / 2)}, 
        torch::TensorOptions().dtype(torch::kLong)).clone();
    auto feature_values = torch::from_blob(feature_values_vec.data(), 
        {static_cast<long>(feature_values_vec.size())}, 
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
    
    std::cout << "debug3-1\n";
    torch::Tensor features = torch::sparse_coo_tensor(
        feature_indices, 
        feature_values, 
        {static_cast<long>(idx_features_labels.size()), num_features}
    );
    // std::cout << "debug3-2\n";
    // features = features.to_sparse_csr();
    features = features.to_dense();
    features = torch::nn::functional::normalize(features);

    std::cout << "debug4\n";

    // Create adjacency indices
    std::unordered_map<std::string, int> idx_map;
    for (size_t i = 0; i < idx_features_labels.size(); ++i) {
        idx_map[idx_features_labels[i][0]] = i;
    }

    std::vector<long> adj_indices_vec;
    std::vector<float> adj_values_vec;

    std::cout << "debug5\n";

    std::ifstream edges_file(path + dataset_name + ".cites");
    while (std::getline(edges_file, line)) {
        std::istringstream iss(line);
        std::string src, dest;
        iss >> src >> dest;

        if (idx_map.count(src) && idx_map.count(dest)) {
            int src_idx = idx_map[src];
            int dest_idx = idx_map[dest];
            
            // Add symmetric edges
            adj_indices_vec.push_back(src_idx);
            adj_indices_vec.push_back(dest_idx);
            adj_values_vec.push_back(1.0);
            
            adj_indices_vec.push_back(dest_idx);
            adj_indices_vec.push_back(src_idx);
            adj_values_vec.push_back(1.0);
        }
    }

    std::cout << "debug6\n";

    // Add self-loops
    for (size_t i = 0; i < idx_features_labels.size(); ++i) {
        adj_indices_vec.push_back(i);
        adj_indices_vec.push_back(i);
        adj_values_vec.push_back(1.0);
    }

    // std::cout << "debug7\n";

    // Create adjacency sparse tensor
    auto adj_indices = torch::from_blob(adj_indices_vec.data(), 
        {2, static_cast<long>(adj_indices_vec.size() / 2)}, 
        torch::TensorOptions().dtype(torch::kLong)).clone();
    auto adj_values = torch::from_blob(adj_values_vec.data(), 
        {static_cast<long>(adj_values_vec.size())}, 
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
    
    torch::Tensor adj = torch::sparse_coo_tensor(
        adj_indices, 
        adj_values, 
        {static_cast<long>(idx_features_labels.size()), 
         static_cast<long>(idx_features_labels.size())}
    ).to_sparse_csr();

    // std::cout << "debug8\n";

    // One-hot encode labels
    torch::Tensor labels = encode_onehot(unique_labels);

    // std::cout << "debug9\n";

    // Normalize features and adjacency
    // features = normalize_sparse_csr(features);
    // std::cout << "debug9-1\n";

    adj = normalize_sparse_csr(adj);

    // std::cout << "debug10\n";

    // Prepare indices
    std::vector<long> train_indices(140);
    std::vector<long> val_indices(300);
    std::vector<long> test_indices(1000);

    std::iota(train_indices.begin(), train_indices.end(), 0);
    std::iota(val_indices.begin(), val_indices.end(), 200);
    std::iota(test_indices.begin(), test_indices.end(), 500);

    // std::cout << "debug11\n";

    // Prepare dataset
    GraphDataset dataset;
    dataset.features = features;
    dataset.labels = torch::argmax(labels, 1);
    dataset.adj = adj;
    dataset.idx_train = torch::from_blob(train_indices.data(), {140}, torch::TensorOptions().dtype(torch::kLong)).clone();
    dataset.idx_val = torch::from_blob(val_indices.data(), {300}, torch::TensorOptions().dtype(torch::kLong)).clone();
    dataset.idx_test = torch::from_blob(test_indices.data(), {1000}, torch::TensorOptions().dtype(torch::kLong)).clone();

    std::cout << "loaded done\n";

    return dataset;
}