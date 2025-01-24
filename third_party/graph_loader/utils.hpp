#pragma once

#include <string>
#include <cstdlib>
#include <cerrno>
#include <limits>

#include "exception.hpp"

namespace graph_loader::utils {

inline bool StrEndWith(const std::string& str, const std::string& suffix) {
    auto suf_size = suffix.size();
    if (suf_size == 0) {
        return true;
    }
    auto str_size = str.size();
    if (str_size == 0 || suf_size > str_size) {
        return false;
    }
    
    for (size_t i = 0, j = str_size - suf_size; i < suf_size; ++i, ++j) {
        if (suffix[i] != str[j]) {
            return false;
        }
    }
    return true;
}

inline bool StrStartWith(const std::string& str, const std::string& prefix) {
    auto pref_size = prefix.size();
    if (pref_size == 0) {
        return true;
    }
    auto str_size = str.size();
    if (str_size == 0 || pref_size > str_size) {
        return false;
    }
    for (size_t i = 0; i < pref_size; ++i) {
        if (prefix[i] != str[i]) {
            return false;
        }
    }
    return true;
}

inline std::string StrStrip(const std::string& str, const std::string& chars = " \t\n\r") {
    size_t start = str.find_first_not_of(chars);
    if (start == std::string::npos) {
        return "";
    }

    size_t end = str.find_last_not_of(chars);
    return str.substr(start, end - start + 1);
}

template <typename T = unsigned long>
typename std::enable_if_t<std::is_integral_v<T>, T> 
StrToNum(const char* str) {
    char* endptr;
    auto value = std::strtoul(str, &endptr, 10);

    throw_if_exception(endptr == str, 
                       "StrToNum failed due to no digits were found, num_str=" + std::string(str));
    throw_if_exception(errno == ERANGE && value == std::numeric_limits<decltype(value)>::max(),
                       "StrToNum failed due to result out of return type of `strtoul/strtod` range, num_str=" + std::string(str));
    throw_if_exception(value > std::numeric_limits<T>::max(),
                       "StrToNum failed due to result out of type T range, num_str=" + std::string(str));
    return static_cast<T>(value);
}

template <typename T>
typename std::enable_if_t<std::is_floating_point_v<T>, T> 
StrToNum(const char* str) {
    char* endptr;
    auto value = std::strtod(str, &endptr);

    throw_if_exception(endptr == str, 
                      "StrToNum failed due to no digits were found, num_str=" + std::string(str));
    throw_if_exception(errno != 0, 
                       "StrToNum failed due to error occurred during conversion, num_str=" + std::string(str));
    throw_if_exception(value > std::numeric_limits<T>::max(),
                       "StrToNum failed due to result out of type T range, num_str=" + std::string(str));
    return static_cast<T>(value);
}

} // namespace graph_loader::utils
