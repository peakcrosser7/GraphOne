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

    template <typename T = void>
    T* Allocate(int64_t size, torch::Device device) {
        torch::ScalarType dtype;
        if constexpr (std::is_same_v<T, void>) {
            dtype = torch::kByte;
        } else {
            dtype = torch::CppTypeToScalarType<T>::value;
        }
        torch::Tensor tensor = torch::empty({size}, torch::dtype(dtype).device(device));
        tensors_[tensor.data_ptr()] = tensor;
        if constexpr (std::is_same_v<T, void>) {
            return tensor.data_ptr();
        } else {
            return tensor.data_ptr<T>();
        }
    }

    template <typename T = void>
    T* CudaAllocate(int64_t size) {
        return Allocate<T>(size, torch::kCUDA);
    }

    template <typename T>
    torch::Tensor GetTensor(T* ptr) {
        auto it = tensors_.find(ptr);
        if (it != tensors_.end()) {
            return it->second;
        } else {
            TORCH_CHECK(false, "Pointer not found in allocator");
            return torch::Tensor();
        }
    }

    template <typename T>
    torch::Tensor PopTensor(T* ptr) {
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
