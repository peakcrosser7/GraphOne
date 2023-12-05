#pragma once

#include <vector>
#include <string>

#include "GraphGenlX/type.hpp"
#include "GraphGenlX/utils.h"
#include "GraphGenlX/archi/mem/mem.h"

namespace graph_genlx {

template<arch_t arch, typename value_t, typename index_t = std::size_t>
class Buffer {
public:
    Buffer() = default;

    explicit Buffer(index_t size)
        : size_(size), data_(archi::memalloc<arch, value_t>(size)) {
        archi::memset<arch, value_t>(data_, 0, size);
    }

    // copy ctor from std::vector
    explicit Buffer(const std::vector<value_t> &vec)
        : size_(vec.size()), data_(archi::memalloc<arch, value_t>(vec.size())) {
        archi::memcpy<arch, arch_t::cpu, value_t>(data_, vec.data(), size_);
    }

    Buffer(const Buffer& rhs) 
        : size_(rhs.size()), data_(archi::memalloc<arch, value_t>(rhs.size())) {
        archi::memcpy<arch, arch, value_t>(data_, rhs.data(), size_);
        // LOG_DEBUG << "buffer copy ctor\n";
    }

    template <arch_t from_arch>
    Buffer(const Buffer<from_arch, value_t, index_t>& rhs) 
        : size_(rhs.size()), data_(archi::memalloc<arch, value_t>(rhs.size())) {
        archi::memcpy<arch, from_arch, value_t>(data_, rhs.data(), size_);
        // LOG_DEBUG << "buffer copy ctor\n";
    }

    Buffer(Buffer&& rhs) : size_(rhs.size_), data_(rhs.data_) {
        rhs.size_ = 0;
        rhs.data_ = nullptr;
        // LOG_DEBUG << "buffer move ctor\n";
    }

    void copy_from(const std::vector<value_t>& vec, index_t start = 0) {
        index_t len = std::min(index_t(vec.size()), size_ - start);
        archi::memcpy<arch, arch_t::cpu, value_t>(data_ + start, vec.data(), len);
    }

    // template <arch_t from_arch>
    // void copy_from(const Buffer<from_arch, value_t, index_t>& src_buf) {
    //     *this = src_buf;
    // }

    // void move_from(Buffer<arch, value_t, index_t>& src_buf) {
    //     *this = std::move(src_buf);
    // }

    Buffer& operator= (const Buffer& rhs) {
        if (this == &rhs) {
            return *this;
        }
        size_ = rhs.size();
        archi::memfree<arch, value_t>(data_);
        data_ = archi::memalloc<arch, value_t>(size_);
        archi::memcpy<arch, arch, value_t>(data_, rhs.data(), size_);
        return *this;
    }

    template<arch_t from_arch>
    Buffer& operator= (const Buffer<from_arch, value_t, index_t>& rhs) {
        size_ = rhs.size();
        archi::memfree<arch, value_t>(data_);
        data_ = archi::memalloc<arch, value_t>(size_);
        archi::memcpy<arch, from_arch, value_t>(data_, rhs.data(), size_);
        return *this;
    }

    Buffer& operator= (Buffer&& rhs) {
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
        archi::memfree<arch, value_t>(data_);
    }

    const value_t *begin() const {
        return data_;
    }

    value_t *begin() {
        return data_;
    }
    
    const value_t *end() const {
        return data_ + size_;
    }

    value_t *end() {
        return data_ + size_;
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

    void reset() {
        archi::memset<arch, value_t>(data_, size_, 0);
    }

    void reset(index_t size) {
        if (size != size_) {
            archi::memfree<arch, value_t>(data_);
            data_ = archi::memalloc<arch, value_t>(size);
            size_ = size;
        }
        archi::memset<arch, value_t>(data_, size, 0);
    }

    std::string ToString() const {
        std::string str;
        str += "Buffer{ ";
        str += "size_:" + utils::NumToString(size_) + ", ";
        str += "data_:[";

        value_t* host_data = data_;
        if constexpr (arch != arch_t::cpu) {
            host_data = archi::memalloc<arch_t::cpu, value_t>(size_);
            archi::memcpy<arch_t::cpu, arch, value_t>(host_data, data_, size_);
        }

        for (index_t i = 0; i < size_; ++i) {
            str += utils::ToString(host_data[i]);
            if (i < size_ - 1) {
                str += ",";
            }
        }
        str += "] }";
        if constexpr (arch != arch_t::cpu) {
            archi::memfree<arch_t::cpu, value_t>(host_data);
        }

        return str;
    }

protected:
    index_t size_{0};
    value_t* data_{nullptr};
};

}