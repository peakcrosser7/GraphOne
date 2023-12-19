#pragma once

#include <string>
#include <iostream>

namespace graph_genlx {

// #define DEBUG_LOG

constexpr const char LOG_INFO_STR[]    = "[INFO]  ";
constexpr const char LOG_ERROR_STR[]   = "[ERROR] ";
constexpr const char LOG_DEBUG_STR[]  = "[DEBUG] ";
constexpr const char LOG_WARNING_STR[] = "[WARN]  ";

template <typename... args_t>
void printx(args_t&&... args) {
    (std::cout << ... << args) << std::endl;
}

template <const char* label, typename... args_t>
void log(args_t&&... args) {
    std::cout << label; 
    (std::cout << ... << args) << std::endl;
}

template <typename... args_t>
void LOG_INFO(args_t&&... args) {
    log<LOG_INFO_STR>(args...);
}

template <typename... args_t>
void LOG_ERROR(args_t&&... args) {
    log<LOG_ERROR_STR>(args...);
    exit(1);
}

template <typename... args_t>
void LOG_WARNING(args_t&&... args) {
    log<LOG_WARNING_STR>(args...);
}

template <typename... args_t>
void LOG_DEBUG(args_t&&... args) {
#ifdef DEBUG_LOG
    log<LOG_DEBUG_STR>(args...);
#endif
}

} // namespace graph_genl
