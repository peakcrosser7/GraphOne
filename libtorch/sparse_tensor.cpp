#include <torch/torch.h>

#include <iostream>

int main() {
    // 创建一个 COO 格式的稀疏张量
    torch::Tensor indices = torch::tensor({{0, 1, 1}, {2, 0, 2}});
    torch::Tensor values = torch::tensor({3.f, 4.f, 5.f});
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor coo_tensor = torch::sparse_coo_tensor(indices, values, {2,3}, options);

    // 创建一个 CSR 格式的稀疏张量
    torch::Tensor crow_indices = torch::tensor({0, 2, 2, 3});
    torch::Tensor col_indices = torch::tensor({2, 0, 2});
    torch::Tensor csr_values = torch::tensor({3.f, 4.f, 5.f});
    torch::Tensor csr_tensor = torch::sparse_csr_tensor(crow_indices, col_indices, csr_values, {2,3}, options);

    std::cout << coo_tensor.is_sparse() << std::endl;
    // 判断稀疏张量的布局
    if (coo_tensor.layout() == torch::kSparse) {
        std::cout << "coo_tensor is in COO format." << std::endl;
    } else if (coo_tensor.layout() == torch::kSparseCsr) {
        std::cout << "coo_tensor is in CSR format." << std::endl;
    }

    if (csr_tensor.layout() == torch::kSparse) {
        std::cout << "csr_tensor is in COO format." << std::endl;
    } else if (csr_tensor.layout() == torch::kSparseCsr) {
        std::cout << "csr_tensor is in CSR format." << std::endl;
    }



    torch::Tensor t;
    if (!t.defined()) {
        std::cout << "t is NOT defined\n"; 
    }
    

    return 0;
}