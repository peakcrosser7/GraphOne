#pragma once

#include "GraphGenlX/base/buffer.h"

namespace graph_genlx {

template <arch_t arch, 
          typename value_t, 
          typename index_t = uint32_t>
class DenseVec : public Buffer<arch, value_t, index_t> {
public:
    using Buffer<arch, value_t, index_t>::Buffer;

    constexpr static arch_t arch_type = arch;

    std::string ToString() const {
        std::string str;
        str += "DenseVec{ ";
        str += "arch_type:" + utils::ToString(arch_type) + ", ";
        str += "size_:" + utils::NumToString(this->size_) + ", ";
        str += "data_:[";

        value_t* host_data = this->data_;
        if constexpr (arch != arch_t::cpu) {
            host_data = archi::memalloc<arch_t::cpu>::template call<value_t>(this->size_);
            archi::memcpy<arch_t::cpu, arch>::template call<value_t>(host_data, this->data_, this->size_);
        }

        for (index_t i = 0; i < this->size_; ++i) {
            str += utils::ToString(host_data[i]);
            if (i < this->size_ - 1) {
                str += ",";
            }
        }
        str += "] }";
        if constexpr (arch != arch_t::cpu) {
            archi::memfree<arch_t::cpu>::template call<value_t>(host_data);
        }

        return str;
    }
};

} // namespace graph_genlx