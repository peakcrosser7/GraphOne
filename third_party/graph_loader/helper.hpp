#pragma once

#include <cstdlib>
#include <type_traits>

#include "utils.hpp"

namespace graph_loader {

struct empty_t {};

template <typename T>
typename std::enable_if_t<std::is_integral<T>::value || std::is_floating_point<T>::value, T> 
general_weight_parse(const char* str) {
    if (!str) {
        return T(1);
    }
    return utils::StrToNum<T>(str);
}

// CANNOT use function-specialization for SFINAE
template <typename T>
typename std::enable_if_t<std::is_same<T, empty_t>::value, T> 
general_weight_parse(const char* str) {
    return {};
}

auto dummy_func = []() -> void {};


template<typename T>
struct DependentFalse : std::false_type {};


} // namespace graph_loader
