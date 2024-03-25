#pragma once

#include "GraphOne/debug/debug.hpp"

#ifdef DEBUG_THRUST
    #include <exception>

    #define checkThrustErrors_no_ret(stmt)                                         \
        try {                                                                      \
            stmt;                                                                  \
        } catch (const std::exception &e) {                                        \
            std::cerr << "Thrust exception at " << __FILE__ << ":" << __LINE__     \
                    << " \"" << #stmt << "\": " << e.what() << std::endl;          \
            abort();                                                               \
        }

    #define checkThrustErrors_with_ret(stmt)                                       \
        [&]() -> decltype(stmt) {                                                  \
            try {                                                                  \
                return (stmt);                                                     \
            } catch (const std::exception &e) {                                    \
                std::cerr << "Thrust exception at " << __FILE__ << ":" << __LINE__ \
                        << " \"" << #stmt << "\": " << e.what() << std::endl;      \
                abort();                                                           \
            }                                                                      \
        }()
#else
    #define checkThrustErrors_no_ret(stmt) (stmt)

    #define checkThrustErrors_with_ret(stmt) (stmt)
#endif