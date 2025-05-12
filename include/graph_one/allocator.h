#pragma once 

#include <unordered_map>

#include <torch/torch.h>


namespace graph_one{

class MemAllocator {
public:
    static MemAllocator& Get() {
        static MemAllocator instance;
        return instance;
    }

    MemAllocator(const MemAllocator&) = delete;
    MemAllocator& operator=(const MemAllocator&) = delete;

    void* Allocate(int64_t size, torch::Device device) {
        torch::Tensor tensor = torch::empty({size}, torch::dtype(torch::kByte).device(device));
        tensors_[tensor.data_ptr()] = tensor;
        return tensor.data_ptr();
    }

    void* CudaAllocate(int64_t size) {
        return Allocate(size, torch::kCUDA);
    }

    torch::Tensor GetTensor(void* ptr) {
        auto it = tensors_.find(ptr);
        if (it != tensors_.end()) {
            return it->second;
        } else {
            TORCH_CHECK(false, "Pointer not found in allocator");
            return torch::Tensor();
        }
    }

    torch::Tensor PopTensor(void* ptr) {
        auto it = tensors_.find(ptr);
        if (it != tensors_.end()) {
            torch::Tensor tensor = it->second;
            tensors_.erase(it);
            return tensor;
        } else {
            TORCH_CHECK(false, "Pointer not found in allocator");
            return torch::Tensor();
        }
    }


    torch::Tensor PopTensor(void* ptr, torch::Dtype dtype) {
        auto it = tensors_.find(ptr);
        if (it != tensors_.end()) {
            torch::Tensor tensor = it->second;
            tensors_.erase(it);
            
            int64_t target_element_size = torch::elementSize(dtype);
            auto sizes = tensor.sizes().vec();
            sizes.back() = sizes.back() / target_element_size;

            return tensor.view(sizes).to(dtype);          
        } else {
            TORCH_CHECK(false, "Pointer not found in allocator");
            return torch::Tensor();
        }
    }

    template <typename T>
    void Free(T* ptr) {
        auto it = tensors_.find(ptr);
        if (it != tensors_.end()) {
            tensors_.erase(it);
        } else {
            TORCH_CHECK(false, "Pointer not found in allocator");
        }
    }

private:
    MemAllocator() = default;

    std::unordered_map<void*, torch::Tensor> tensors_;
};


} // namespace graph_one
