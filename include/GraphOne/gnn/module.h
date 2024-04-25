#pragma once

#include <string>
#include <memory>
#include <optional>
#include <unordered_map>

#include "GraphOne/type.hpp"
#include "GraphOne/utils/log.hpp"
#include "GraphOne/gnn/tensor.h"
#include "GraphOne/gnn/module_holder.h"

#define NAME_OF(v) (#v)

namespace graph_one::gnn {

template <arch_t arch, typename value_t = float>
class Module : public std::enable_shared_from_this<Module<arch, value_t>> {
public:

    using module_spec_t = Module<arch, value_t>;
    using param_spec_t = param_t<arch, value_t>;

    Module() = default;

    explicit Module(std::string name) 
        : name_(std::move(name)) {}
    
    virtual ~Module() = default;


    template <typename module_t>
    std::shared_ptr<module_t> register_module(std::string name, ModuleHolder<module_t> module) {
        if (name.empty()) {
            LOG_ERROR("Submodule name must not be empty");
        }
        if (name.find('.') != std::string::npos) {
            LOG_ERROR("Submodule name must not contain a dot (got '", name,"')");
        }
        auto [it, _] = children_.emplace(std::move(name), std::move(module.ptr()));
        return std::dynamic_pointer_cast<module_t>(it->second);        
    }

    param_spec_t register_parameter(std::string name, tensor_t<arch, value_t>&& param) {
        if (name.empty()) {
            LOG_ERROR("Parameter name must not be empty");
        }
        if (name.find('.') != std::string::npos) {
            LOG_ERROR("Parameter name must not contain a dot (got '", name,"')");
        }
        auto [it, _] = parameters_.emplace(std::move(name), 
            param_spec_t(new tensor_t<arch, value_t>(std::move(param))));
        return it->second;
    }

    std::unordered_map<std::string, param_spec_t>
    named_parameters(bool recurse = true) const {
        std::unordered_map<std::string, param_spec_t> result;
        named_parameters_(recurse, "", result);
        return result;
    }


    std::string ToString() const {
        return "Module { " + name_ + " }";
    }

private:
    void named_parameters_(bool recurse, std::string prefix,
        std::unordered_map<std::string, param_spec_t>& result
    ) const {
        for (auto& [name, param]: parameters_) {
            result.emplace(prefix + name, param);
        }
        if (recurse) {
            for (auto& [name, module]: children_) {
                module->named_parameters_(recurse, prefix + name + ".", result);
            }
        }
    }


protected:
    std::unordered_map<std::string, param_spec_t> parameters_;

    std::unordered_map<std::string, std::shared_ptr<module_spec_t>> children_;

    std::string name_;
};


} // namespace graph_one