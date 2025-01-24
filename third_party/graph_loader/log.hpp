#pragma once

#include <string>
#include <iostream>


namespace graph_loader {

constexpr const char LOG_INFO_STR[]    = "[INFO]  ";
constexpr const char LOG_ERROR_STR[]   = "[ERROR] ";
constexpr const char LOG_DEBUG_STR[]   = "[DEBUG] ";
constexpr const char LOG_WARNING_STR[] = "[WARN]  ";
constexpr const char LOG_FATAL_STR[]   = "[FATAL] ";

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
    log<LOG_INFO_STR>(std::forward<args_t>(args)...);
}

template <typename... args_t>
void LOG_ERROR(args_t&&... args) {
    log<LOG_ERROR_STR>(std::forward<args_t>(args)...);
}

template <typename... args_t>
void LOG_FATAL(args_t&&... args) {
    log<LOG_FATAL_STR>(std::forward<args_t>(args)...);
    std::abort();
}

template <typename... args_t>
void LOG_WARNING(args_t&&... args) {
    log<LOG_WARNING_STR>(std::forward<args_t>(args)...);
}

template <typename... args_t>
void LOG_DEBUG(args_t&&... args) {
#ifdef DEBUG_LOG_LOADER
    log<LOG_DEBUG_STR>(std::forward<args_t>(args)...);
#endif
}

} // namespace graph_loader
