#pragma once

#include <string>
#include <iostream>

#include "GraphOne/debug/debug.hpp"
#include "GraphOne/utils/string.hpp"

namespace graph_one {

constexpr const char LOG_INFO_STR[]    = "[INFO]  ";
constexpr const char LOG_ERROR_STR[]   = "[ERROR] ";
constexpr const char LOG_DEBUG_STR[]   = "[DEBUG] ";
constexpr const char LOG_WARNING_STR[] = "[WARN]  ";


template<typename T>
std::enable_if_t<!std::is_class_v<T> || !utils::HasToStrMethod<T>::value, const T&>
print_helper(const T& value) {
    return value;
}

/// @brief log helper to call ToString() method automatically
template<typename T>
std::enable_if_t<std::is_class_v<T> && utils::HasToStrMethod<T>::value, std::string>
print_helper(const T& value) {
    return value.ToString();
}

// log() will call ToString() method automatically
template <const char* label, typename... args_t>
void log(args_t&&... args) {
    std::cout << label;
    (std::cout << ... << print_helper(args)) << std::endl;
}

template <typename... args_t>
void printx(args_t&&... args) {
    (std::cout << ... << print_helper(args)) << std::endl;
} 

template <typename... args_t>
void LOG_INFO(args_t&&... args) {
    log<LOG_INFO_STR>(std::forward<args_t>(args)...);
}

template <typename... args_t>
void LOG_ERROR(args_t&&... args) {
    log<LOG_ERROR_STR>(std::forward<args_t>(args)...);
    exit(1);
}

template <typename... args_t>
void LOG_WARNING(args_t&&... args) {
    log<LOG_WARNING_STR>(std::forward<args_t>(args)...);
}

template <typename... args_t>
void LOG_DEBUG(args_t&&... args) {
#ifdef DEBUG_LOG
    log<LOG_DEBUG_STR>(std::forward<args_t>(args)...);
#endif
}

} // namespace graph_genl
