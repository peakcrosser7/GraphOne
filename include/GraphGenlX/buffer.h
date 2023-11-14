#pragma once

#include <vector>
#include <string>

#include "GraphGenlX/type.hpp"
#include "GraphGenlX/utils.h"
#include "GraphGenlX/mem/cpu.hpp"
#include "GraphGenlX/mem/cuda.cuh"

namespace graph_genlx {

template<typename value_t, arch_t arch = arch_t::cpu, typename index_t = std::size_t>
class Buffer {

protected:
    static constexpr auto arch_memalloc = archi::memalloc<arch>::template call<value_t>;
    static constexpr auto arch_memfree = archi::memfree<arch>::template call<value_t>;
    static constexpr auto arch_memset = archi::memset<arch>::template call<value_t>;

    template<arch_t from_arch>
    static constexpr auto arch_memcpy = archi::memcpy<arch, from_arch>::template call<value_t>;

public:
    Buffer() = default;

    Buffer(index_t size) : size_(size), data_(arch_memalloc(size)) {
        arch_memset(data_, 0, size);
    }

    // copy ctor from std::vector
    Buffer(const std::vector<value_t>& vec) : size_(vec.size()), data_(arch_memalloc(vec.size())) {
        arch_memcpy<arch_t::cpu>(data_, vec.data(), size_);
    }

    template <arch_t from_arch>
    Buffer(const Buffer<value_t, from_arch, index_t>& rhs) 
        : size_(rhs.size()), data_(arch_memalloc(rhs.size())) {
        arch_memcpy<from_arch>(data_, rhs.data(), size_);
    }

    Buffer(Buffer<value_t, arch, index_t>&& rhs) : size_(rhs.size_), data_(rhs.data_) {
        rhs.size_ = 0;
        rhs.data_ = nullptr;
    }

    Buffer& operator= (const Buffer<value_t, arch, index_t>& rhs) {
        if (this == &rhs) {
            return *this;
        }
        size_ = rhs.size();
        arch_memfree(data_);
        data_ = arch_memalloc(size_);
        arch_memcpy<arch>(data_, rhs.data(), size_);
        return *this;
    }

    template<arch_t from_arch>
    Buffer& operator= (const Buffer<value_t, from_arch, index_t>& rhs) {
        size_ = rhs.size();
        arch_memfree(data_);
        data_ = arch_memalloc(size_);
        arch_memcpy<from_arch>(data_, rhs.data(), size_);
        return *this;
    }

    Buffer& operator= (Buffer<value_t, arch, index_t>&& rhs) {
        if (this != &rhs) {
            size_ = rhs.size_;
            data_ = rhs.data_;
            rhs.size_ = 0;
            rhs.data_ = nullptr;
        }
        return *this;
    }

    value_t& operator[](index_t i) {
        return data_[i];
    }

    const value_t& operator[](index_t i) const {
        return data_[i];
    }

    ~Buffer() {
        arch_memfree(data_);
    }

    const value_t* data() const {
        return data_;
    }

    value_t* data() {
        return data_;
    }

    index_t size() const {
        return size_;
    }

    std::string ToString() const {
        std::string str;
        str += "Buffer{ ";
        str += "size_:" + utils::NumToString(size_) + ", ";
        str += "data_:[";

        for (index_t i = 0; i < size_; ++i) {
            str += utils::ToString(data_[i]);
            if (i < size_ - 1) {
                str += ",";
            }
        }
        str += "] }";
        return str;
    }

protected:
    index_t size_{0};
    value_t* data_{nullptr};
};

}