#pragma once

#include <torch/torch.h>

namespace graph_one{

using Tensor = torch::Tensor;
using Device = torch::Device;
using DeviceType = torch::DeviceType;

constexpr DeviceType kCPU = torch::kCPU;
constexpr DeviceType kCUDA = torch::kCUDA;

torch::Tensor make_empty(torch::IntArrayRef size, torch::ScalarType dtype, torch::Device device) {
    return torch::empty(size, torch::TensorOptions().dtype(dtype).device(device));
}

template <typename T>
torch::Tensor make_empty(torch::IntArrayRef size, torch::Device device) {
    return torch::empty(size, torch::TensorOptions()
        .dtype(torch::CppTypeToScalarType<T>::value).device(device));
}

torch::Tensor make_zeros(torch::IntArrayRef size, torch::ScalarType dtype, torch::Device device) {
    return torch::zeros(size, torch::TensorOptions().dtype(dtype).device(device));
}

template <typename T>
torch::Tensor make_zeros(torch::IntArrayRef size, torch::Device device) {
    return torch::zeros(size, torch::TensorOptions()
        .dtype(torch::CppTypeToScalarType<T>::value).device(device));
}

torch::Tensor make_ones(torch::IntArrayRef size, torch::ScalarType dtype, torch::Device device) {
    return torch::ones(size, torch::TensorOptions().dtype(dtype).device(device));
}

template <typename T>
torch::Tensor make_ones(torch::IntArrayRef size, torch::Device device) {
    return torch::ones(size, torch::TensorOptions()
        .dtype(torch::CppTypeToScalarType<T>::value).device(device));
}

torch::Tensor make_full(torch::IntArrayRef size, const torch::Scalar & fill_value, 
                        torch::ScalarType dtype, torch::Device device) {
    return torch::full(size, fill_value, torch::TensorOptions().dtype(dtype).device(device));
}

template <typename T>
torch::Tensor make_full(torch::IntArrayRef size, const torch::Scalar & fill_value, torch::Device device) {
    return torch::full(size, fill_value, torch::TensorOptions()
        .dtype(torch::CppTypeToScalarType<T>::value).device(device));
}

torch::Tensor make_rand(torch::IntArrayRef size, torch::ScalarType dtype, torch::Device device) {
    return torch::rand(size, torch::TensorOptions().dtype(dtype).device(device));
}

template <typename T>
torch::Tensor make_rand(torch::IntArrayRef size, torch::Device device) {
    return torch::rand(size, torch::TensorOptions()
        .dtype(torch::CppTypeToScalarType<T>::value).device(device));
}


namespace nn {

class Module : public torch::nn::Module {
public:
    Module() { eval(); }
};

} // namespace nn 

} // namespace graph_one
