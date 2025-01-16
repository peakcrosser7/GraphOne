#pragma once

#include <string>
#include <memory>
#include <vector>
#include <optional>
#include <unordered_map>

#include "GraphOne/type.hpp"
#include "GraphOne/utils/log.hpp"
#include "GraphOne/domain/gnn/tensor.h"
#include "GraphOne/domain/gnn/module_holder.h"

#define NAME_OF(v) (#v)

namespace graph_one::gnn {

template <arch_t arch, typename value_t = float, typename index_t = uint32_t>
class Module : public std::enable_shared_from_this<Module<arch, value_t, index_t>> {
public:

    using module_spec_t = Module<arch, value_t, index_t>;
    using param_spec_t = param_t<arch, value_t, index_t>;

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
            LOG_ERROR("Submodule name must not contain a dot (got '", name, "')");
        }
        if (children_.count(name) != 0) {
            LOG_ERROR("Submodule name must not be defined (got '", name, "')");
        }
        auto [it, _] = children_.emplace(std::move(name), std::move(module.ptr()));
        return std::dynamic_pointer_cast<module_t>(it->second);        
    }

    param_spec_t register_parameter(std::string name, tensor_t<arch, value_t, index_t> param) {
        if (name.empty()) {
            LOG_ERROR("Parameter name must not be empty");
        }
        if (name.find('.') != std::string::npos) {
            LOG_ERROR("Parameter name must not contain a dot (got '", name,"')");
        }
        if (parameters_.count(name) != 0) {
            LOG_ERROR("Parameter name must not be defined (got '", name, "')");
        }
        auto [it, _] = parameters_.emplace(std::move(name), 
            param_spec_t(param));
        return it->second;
    }

    param_spec_t register_parameter(std::string name, std::nullptr_t) {
        if (name.empty()) {
            LOG_ERROR("Parameter name must not be empty");
        }
        if (name.find('.') != std::string::npos) {
            LOG_ERROR("Parameter name must not contain a dot (got '", name,"')");
        }
        if (parameters_.count(name) != 0) {
            LOG_ERROR("Parameter name must not be defined (got '", name, "')");
        }
        auto [it, _] = parameters_.emplace(std::move(name), param_spec_t());
        return it->second;
    }

    std::unordered_map<std::string, param_spec_t>
    named_parameters(bool recurse = true) const {
        std::unordered_map<std::string, param_spec_t> result;
        named_parameters_(recurse, "", result);
        return result;
    }

    std::pair<std::vector<std::string>, std::vector<std::string>>
    load_state_dict(std::unordered_map<std::string, param_spec_t>&& state_dict, 
        bool strict = true) {
        
        std::unordered_map<std::string, param_spec_t> param_dict =
            named_parameters();  

        auto ret = load_state_dict_check_(param_dict, state_dict, strict);      

        for (auto& [name, param]: param_dict) {
            if (auto it = state_dict.find(name); it != state_dict.end()) {
                if (param->n_rows != it->second->n_rows 
                    || param->n_cols != it->second->n_cols) {
                    LOG_ERROR("shape of state `", name, "` is not matched, need (",
                        param->n_rows, ",", param->n_cols, "), but got (",
                        it->second->n_rows, ",", it->second->n_cols, ")");
                }
                *param = std::move(*(it->second));
            }
        } 
        return ret;
    }

    std::pair<std::vector<std::string>, std::vector<std::string>>
    load_state_dict(const std::unordered_map<std::string, param_spec_t>& state_dict, 
        bool strict = true) {
        
        std::unordered_map<std::string, param_spec_t> param_dict =
            named_parameters();  

        auto ret = load_state_dict_check_(param_dict, state_dict, strict);      

        for (auto& [name, param]: param_dict) {
            if (auto it = state_dict.find(name); it != state_dict.end()) {
                if (param->n_rows != it->second->n_rows 
                    || param->n_cols != it->second->n_cols) {
                    LOG_ERROR("shape of state `", name, "` is not matched, need (",
                        param->n_rows, ",", param->n_cols, "), but got (",
                        it->second->n_rows, ",", it->second->n_cols, ")");
                }
                *param = *(it->second);
            }
        }
        return ret;
    }

    std::string ToString() const {
        return "Module { " + name_ + " }";
    }

private:
    std::pair<std::vector<std::string>, std::vector<std::string>>
    load_state_dict_check_(
        const std::unordered_map<std::string, param_spec_t>& param_dict,
        const std::unordered_map<std::string, param_spec_t>& state_dict, 
        bool strict) {

        std::vector<std::string> missing_keys, unexpected_keys;
        for (const auto& [name, _]: param_dict) {
            if (state_dict.count(name) == 0) {
                missing_keys.push_back(name);
            }
        }
        for (const auto& [name, _]: state_dict) {
            if (param_dict.count(name) == 0) {
                unexpected_keys.push_back(name);
            }
        }
        if (strict && (!missing_keys.empty() || !unexpected_keys.empty())) {
            LOG_ERROR("load_state_dict() failed. "
                "missing_keys:", utils::VecToString(missing_keys), ", "
                "unexpected_keys:", utils::VecToString(unexpected_keys));
        }
        return std::make_pair(std::move(missing_keys), std::move(unexpected_keys));
    }

    void named_parameters_(bool recurse, std::string prefix,
        std::unordered_map<std::string, param_spec_t>& result
    ) const {
        for (auto& [name, param]: parameters_) {
            if (param) {
                result.emplace(prefix + name, param);
            }    
        }
        if (recurse) {
            for (auto& [name, module]: children_) {
                if (module) {
                    module->named_parameters_(recurse, prefix + name + ".", result);
                }
            }
        }
    }


protected:
    std::unordered_map<std::string, param_spec_t> parameters_;

    std::unordered_map<std::string, std::shared_ptr<module_spec_t>> children_;

    std::string name_;
};


} // namespace graph_one