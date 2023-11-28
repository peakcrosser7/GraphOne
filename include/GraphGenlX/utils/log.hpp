#pragma once

#include <string>
#include <iostream>

namespace graph_genlx {

constexpr const char* LOG_ERROR_STR = "[ERROR] ";
constexpr const char* LOG_DEBUG_STR = "[DEBUG] ";
constexpr const char* LOG_WARNING_STR = "[WARNING] ";

#define LOG_ERROR std::cout << LOG_ERROR_STR
#define LOG_DEBUG std::cout << LOG_DEBUG_STR
#define LOG_WARNING std::cout << LOG_WARNING_STR

} // namespace graph_genl
