#pragma once

#include <limits>

#include <cuda.h>

namespace graph_one::op {

struct Mult {

    template <typename T>
    __host__ __device__ __forceinline__
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
    __host__ __device__ __forceinline__
    T identity() const {
        return T(0);
    }

    template <typename T>
    __host__ __device__ __forceinline__
    T operator() (const T& lhs, const T& rhs) const {
        return lhs + rhs;
    }
};

struct SafeAdd {

    template <typename T>
    __host__ __device__ __forceinline__
    T identity() const {
        return T(0);
    }

    template <typename T>
    __host__ __device__ __forceinline__
    T operator() (const T& lhs, const T& rhs) const {
        if (lhs > std::numeric_limits<T>::max() - rhs) {
            return std::numeric_limits<T>::max();
        }
        return lhs + rhs;
    }
};


struct Min {

    template <typename T>
    __host__ __device__ __forceinline__
    T identity() const {
        return std::numeric_limits<T>::max();
    }

    template <typename T>
    __host__ __device__ __forceinline__
    T operator() (const T& lhs, const T& rhs) const {
        return min(lhs, rhs);
    }
};
    
} // namespace graph_one