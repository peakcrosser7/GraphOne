#pragma once

#include <string>
#include <vector>
#include <type_traits>

#include "GraphGenlX/type.hpp"
#include "GraphGenlX/archi.h"

namespace graph_genlx::utils {

bool StrEndWith(const std::string& str, const std::string& suffix) {
    auto suf_size = suffix.size();
    if (suf_size == 0) {
        return true;
    }
    auto str_size = str.size();
    if (str_size == 0 || suf_size > str_size) {
        return false;
    }
    
    for (int i = 0, j = str_size - suf_size; i < suf_size; ++i, ++j) {
        if (suffix[i] != str[j]) {
            return false;
        }
    }
    return true;
}


bool StrStartWith(const std::string& str, const std::string& prefix) {
    auto pref_size = prefix.size();
    if (pref_size == 0) {
        return true;
    }
    auto str_size = str.size();
    if (str_size == 0 || pref_size > str_size) {
        return false;
    }
    for (int i = 0; i < pref_size; ++i) {
        if (prefix[i] != str[i]) {
            return false;
        }
    }
    return true;
}

inline std::string ArchToString(arch_t arch) {
    switch (arch) {
    case arch_t::cpu:
        return "cpu";
    case arch_t::cuda:
        return "cuda";
    default:
        break;
    }
    return "undefine";
}

template <typename T>
using ToStrFuncType = decltype(std::declval<T>().ToString());

template <typename T, typename = void>
struct HasToStrMethod : std::false_type {};

template <typename T>
struct HasToStrMethod<T, std::void_t<ToStrFuncType<T>>> : std::true_type {};

template <typename T>
std::string ToString(const T& x) {
    if constexpr (std::is_arithmetic_v<T>) {
        return std::to_string(x);
    } else if constexpr (HasToStrMethod<T>::value){
        return x.ToString();
    } else {
        return std::string(x);
    }
}

template <typename T>
std::string NumToString(T num) {
    return ToString(num);
}

template <typename T, typename STR_FUNC = decltype(ToString<T>)>
std::string VecToString(const std::vector<T> &vec,
                        STR_FUNC str_func = ToString<T>) {
    std::string str = "[";
    size_t sz = vec.size();
    for (size_t i = 0; i < sz; ++i) {
        str += str_func(vec[i]) + ", ";
    }
    if (sz > 0) {
        str += "\b\b";
    }
    return str += "]";
}

template <typename T, typename STR_FUNC = decltype(ToString<T>)>
std::string VecToString(const thrust::host_vector<T> &vec,
                        STR_FUNC str_func = ToString<T>) {
    std::string str = "[";
    size_t sz = vec.size();
    for (size_t i = 0; i < sz; ++i) {
        str += str_func(vec[i]) + ", ";
    }
    if (sz > 0) {
        str += "\b\b";
    }
    return str += "]";
}

template <typename T, typename STR_FUNC = decltype(ToString<T>)>
std::string VecToString(const thrust::device_vector<T> &vec,
                        STR_FUNC str_func = ToString<T>) {
    thrust::host_vector<T> host_vec = vec;
    return VecToString(host_vec, str_func);
}

} // namespace graph_genl
