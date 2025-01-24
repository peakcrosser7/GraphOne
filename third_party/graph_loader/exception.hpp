#pragma once

#include <string>

namespace graph_loader {

struct Exception : std::exception {
    const char* report;

    Exception(const char* message) : report(message) {}
    virtual const char *what() const noexcept { return report; }
};

inline void throw_if_exception(bool is_exception, const char* message) {
    if (is_exception) {
        throw Exception(message);
    }
}

inline void throw_if_exception(bool is_exception, std::string message = "") {
    if (is_exception) {
        throw Exception(message.c_str());
    }
}

} // namespace graph_loader 