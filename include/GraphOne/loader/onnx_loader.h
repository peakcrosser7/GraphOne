#pragma once

#include <fstream>
#include <string>
#include <unordered_map>
#include <memory>
#include <utility>

#include "onnx/onnx.pb.h"

#include "GraphOne/gnn/tensor.h"
#include "GraphOne/utils/log.hpp"
#include "GraphOne/archi/mem/mem.h"

namespace graph_one {

template <typename T>
bool is_same_onnx_datatype(onnx::TensorProto_DataType type) {
    return false;
}

template<>
bool is_same_onnx_datatype<float>(onnx::TensorProto_DataType type) {
    return type == onnx::TensorProto_DataType_FLOAT;
}

template<>
bool is_same_onnx_datatype<uint32_t>(onnx::TensorProto_DataType type) {
    return type == onnx::TensorProto_DataType_UINT32;
}

template<>
bool is_same_onnx_datatype<int32_t>(onnx::TensorProto_DataType type) {
    return type == onnx::TensorProto_DataType_INT32;
}    

class OnnxLoader {
public:
    static constexpr const char * file_ext = ".onnx";

    template <arch_t arch, typename value_t=float>
    static std::unordered_map<std::string, gnn::param_t<arch, value_t>>
    LoadInitializer(const std::string& filepath) {
        if (!utils::StrEndWith(filepath, file_ext)) {
            LOG_ERROR("file extension does not match, it should \"", 
                file_ext, "\"");
        }

        std::fstream fin(filepath, std::ios::in | std::ios::binary);
        if (fin.fail()) {
            LOG_ERROR("cannot open graph adj file: ", filepath);
        }

        onnx::ModelProto model;
        if (!model.ParseFromIstream(&fin)) {
            LOG_ERROR("failed to parse onnx file");
        }
        std::unordered_map<std::string, gnn::param_t<arch, value_t>> state_dict;
        for (const onnx::TensorProto& data: model.graph().initializer()) {
            LoadParamter_(data, state_dict);
        }
        return state_dict;
    }

private:

    template <arch_t arch, typename value_t>
    static void LoadParamter_(const onnx::TensorProto& data, 
        std::unordered_map<std::string, gnn::param_t<arch, value_t>>& state_dict) {
        std::string name = data.name();
        auto shape = data.dims();
        int shape_sz = shape.size();
        if (shape_sz > 2) {
            LOG_ERROR("shape of parameter '", name, "' is ", shape_sz,
                ", which is greater than 2, currently not supported");
        }
        gnn::param_t<arch, value_t> parameter{nullptr};
        if (shape_sz == 2) {
            parameter = gnn::param_t<arch, value_t>(
                new gnn::tensor_t<arch, value_t>(shape[0], shape[1]));
        } else {
            parameter = gnn::param_t<arch, value_t>(
                new gnn::tensor_t<arch, value_t>(1, shape[0]));
        }
        if (!is_same_onnx_datatype<value_t>(onnx::TensorProto_DataType(data.data_type()))) {
            LOG_ERROR("data type of parameter '", name, "' is `", 
                onnx::TensorProto::DataType_Name(data.data_type()), 
                "`, which should be `", typeid(value_t).name(), "`");
        }
        auto raw_data = reinterpret_cast<const value_t*>(data.raw_data().c_str());
        value_t* param_data = parameter->values.data();
        auto sz = parameter->n_rows * parameter->n_cols;
        if (data.raw_data().size() != sz * sizeof(value_t)) {
            LOG_ERROR("data type of parameter '", name, "' is wrong," 
                "raw data size is ", data.raw_data().size(), 
                ", but `n_rows * n_cols * sizeof(value_t)` is ", sz * sizeof(value_t));
        }
        archi::memcpy<arch, arch_t::cpu, value_t>(param_data, raw_data, sz);
        state_dict.emplace(std::move(name), std::move(parameter));
    }
};

} // namespace graph_one