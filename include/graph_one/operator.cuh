#pragma once

#include <cuda.h>

namespace graph_one::op {

struct Mult {

    template <typename T>
    __device__ __host__ __forceinline__
    T identity() const {
        return T(1);
    }

    template <typename T>
    __host__ __device__ __forceinline__
    T operator() (const T& lhs, const T& rhs) const {
        return lhs * rhs;
    }
};


struct Add {

    template <typename T>
    __device__ __host__ __forceinline__
    T identity() const {
        return T(0);
    }

    template <typename T>
    __host__ __device__ __forceinline__
    T operator() (const T& lhs, const T& rhs) const {
        return lhs + rhs;
    }
};
    
} // namespace graph_one