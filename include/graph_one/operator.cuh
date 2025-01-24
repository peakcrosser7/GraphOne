#pragma once

#include <cuda.h>

namespace graph_one::op {

struct Mult {

    template <typename T>
    __device__ __host__ 
    constexpr static T identity() {
        return T(1);
    }

    template <typename T>
    __host__ __device__ 
    constexpr static T call(const T& lhs, const T& rhs) {
        return lhs * rhs;
    }
};


struct Add {

    template <typename T>
    __device__ __host__ 
    constexpr static T identity() {
        return T(0);
    }

    template <typename T>
    __host__ __device__ 
    constexpr static T call(const T& lhs, const T& rhs) {
        return lhs + rhs;
    }
};
    
} // namespace graph_one