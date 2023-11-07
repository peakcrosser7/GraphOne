#pragma once

#include <string>
#include <iostream>

namespace graph_genlx {

constexpr const char* LOG_ERROR_STR = "[ERROR] ";
constexpr const char* LOG_DEBUG_STR = "[DEBUG] ";


#define LOG_ERROR std::cout << LOG_ERROR_STR
#define LOG_DEBUG std::cout << LOG_DEBUG_STR

} // namespace graph_genl
