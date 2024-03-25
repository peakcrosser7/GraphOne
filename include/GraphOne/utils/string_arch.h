#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "GraphOne/utils/string.hpp"

namespace graph_one::utils {

template <typename T, typename STR_FUNC = decltype(ToString<T>)>
std::string VecToString(const thrust::host_vector<T> &vec,
                        STR_FUNC str_func = ToString<T>) {
    std::string str = "[";
    size_t sz = vec.size();
    for (size_t i = 0; i < sz; ++i) {
        str += str_func(vec[i]) + ",";
    }
    if (sz > 0) {
        str += "\b";
    }
    return str += "]";
}

template <typename T, typename STR_FUNC = decltype(ToString<T>)>
std::string VecToString(const thrust::device_vector<T> &vec,
                        STR_FUNC str_func = ToString<T>) {
    thrust::host_vector<T> host_vec = vec;
    return VecToString(host_vec, str_func);
}
    
} // namespace graph_one
