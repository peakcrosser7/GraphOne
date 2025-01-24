#pragma once

#include <string>
#include <iostream>

namespace graph_one {

constexpr const char LOG_INFO_STR[]    = "[INFO]  ";
constexpr const char LOG_ERROR_STR[]   = "[ERROR] ";
constexpr const char LOG_DEBUG_STR[]   = "[DEBUG] ";
constexpr const char LOG_WARNING_STR[] = "[WARN]  ";


// log() will call ToString() method automatically
template <const char* label, typename... args_t>
void log(args_t&&... args) {
    std::cout << label;
    (std::cout << ... << args) << std::endl;
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
