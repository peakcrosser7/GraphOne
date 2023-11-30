#pragma once

#include <vector>

#include "GraphGenlX/type.hpp"
#include "GraphGenlX/utils.h"
#include "GraphGenlX/base/buffer.h"

namespace graph_genlx {

enum class layout_t {
    ROW_MAJOR,
    COL_MAJOR
};

template <arch_t arch, 
          typename value_t,
          typename index_t = uint32_t,
          typename tile_t = uint32_t>
struct StridedMat {
    StridedMat() = default;

    StridedMat(index_t n_stride, index_t n_contiguous)
        : n_stride(n_stride), n_contiguous(n_contiguous), values(n_stride * n_contiguous) {}
    
    StridedMat(const std::vector<std::vector<value_t>>& mat_vec)
        : n_stride(mat_vec.size()), n_contiguous(mat_vec.front().size()), values(n_stride * n_contiguous) {
        for (index_t i = 0; i < n_stride; ++i) {
            values.copy_from(mat_vec[i], i * n_contiguous);
        }
    }

    StridedMat(const StridedMat& rhs)
        : n_stride(rhs.n_stride), n_contiguous(rhs.n_contiguous), values(rhs.values) {}

    StridedMat(StridedMat&& rhs)
        : n_stride(rhs.n_stride), n_contiguous(rhs.n_contiguous), values(std::move(rhs.values)) {}

    template <arch_t from_arch>
    StridedMat(const StridedMat<from_arch, value_t, index_t, tile_t>& rhs) 
        : n_stride(rhs.n_stride), n_contiguous(rhs.n_contiguous), values(rhs.values) {}

    StridedMat& operator= (const StridedMat& rhs) {
        if (this != &rhs) {
            n_stride = rhs.n_stride;
            n_contiguous = rhs.n_contiguous;
            values = rhs.values;
        }
        return *this;
    }

    template <arch_t from_arch>
    StridedMat& operator= (const StridedMat<from_arch, value_t, index_t, tile_t>& rhs) {
        n_stride = rhs.n_stride;
        n_contiguous = rhs.n_contiguous;
        values = rhs.values;
        return *this;
    }   

    StridedMat& operator= (StridedMat&& rhs) {
        if (this != &rhs) {
            n_stride = rhs.n_stride;
            n_contiguous = rhs.n_contiguous;
            values = std::move(rhs.values);
        }
        return *this;
    }    

    value_t* operator[](index_t i) {
        return values.data() + (i * n_contiguous);
    }

    const value_t* operator[](index_t i) const {
        return values.data() + (i * n_contiguous);
    }

    value_t& at(index_t stride, index_t contiguous) {
        return values[stride * n_contiguous + contiguous];
    }

    const value_t& at(index_t stride, index_t contiguous) const {
        return values[stride * n_contiguous + contiguous];
    }

    tile_t capacity() const {
        return n_stride * n_contiguous;
    }

    std::string ToString() const {
        std::string str;
        str += "[";
        if constexpr (arch == arch_t::cpu) {
            for (index_t i = 0; i < n_stride; ++i) {
                str += utils::VecToString(operator[](i), n_contiguous);
                str += " ";
            }
        } else {
            Buffer<arch_t::cpu, value_t, tile_t> h_values(values);
            for (index_t i = 0; i < n_stride; ++i) {
                str += utils::VecToString(h_values.data() + (i * n_contiguous), n_contiguous);
                str += " ";
            }            
        }

        str += "]";
        return str;
    }

    index_t n_stride;
    index_t n_contiguous;

    Buffer<arch, value_t, tile_t> values;
};

template <arch_t arch, 
          typename value_t,
          typename index_t = uint32_t,
          layout_t layout = layout_t::ROW_MAJOR,
          typename tile_t = uint32_t>
struct DenseMat {};

template <arch_t arch, 
          typename value_t,
          typename index_t,
          typename tile_t>
struct DenseMat<arch, value_t, index_t, layout_t::ROW_MAJOR, tile_t>
    : public StridedMat<arch, value_t, index_t, tile_t> {
    using strided_mat_t = StridedMat<arch, value_t, index_t, tile_t>;

    constexpr static arch_t arch_type = arch;

    DenseMat() = default;

    DenseMat(index_t n_rows, index_t n_cols) 
        : strided_mat_t(n_rows, n_cols), n_rows(n_rows), n_cols(n_cols) {}
    
    DenseMat(const std::vector<std::vector<value_t>>& mat_vec)
        : strided_mat_t(mat_vec), n_rows(mat_vec.size()), n_cols(mat_vec.front().size()) {}

    DenseMat(const DenseMat& rhs)
        : strided_mat_t(rhs), n_rows(rhs.n_rows), n_cols(rhs.n_cols) {}

    DenseMat(DenseMat&& rhs)
        : strided_mat_t(std::move(rhs)), n_rows(rhs.n_rows), n_cols(rhs.n_cols) {}

    template <arch_t from_arch>
    DenseMat(const DenseMat<from_arch, value_t, index_t, layout_t::ROW_MAJOR, tile_t>& rhs) 
        : strided_mat_t(rhs), n_rows(rhs.n_rows), n_cols(rhs.n_cols) {}

    DenseMat& operator= (const DenseMat& rhs) {
        if (this != &rhs) {
            strided_mat_t::operator=(rhs);
            n_rows = rhs.n_rows;
            n_cols = rhs.n_cols;
        }
        return *this;
    }

    template <arch_t from_arch>
    DenseMat& operator= (const DenseMat<from_arch, value_t, index_t, layout_t::ROW_MAJOR, tile_t>& rhs) {
        strided_mat_t::operator=(rhs);
        n_rows = rhs.n_rows;
        n_cols = rhs.n_cols;
        return *this;
    }   

    DenseMat& operator= (DenseMat&& rhs) {
        if (this != &rhs) {
            strided_mat_t::operator=(std::move(rhs));
            n_rows = rhs.n_rows;
            n_cols = rhs.n_cols;
        }
        return *this;
    }   

    std::string ToString() const {
        return "DenseMat{ " 
            "arch_type:" + utils::ToString(arch_type) + ", " +
            "layout:ROW_MAJOR, " +
            "n_rows:" + utils::NumToString(n_rows) + ", " +
            "n_cols:" + utils::NumToString(n_cols) + ", " +
            "values:" + strided_mat_t::ToString() +
            " }";
    }
    
    index_t n_rows;
    index_t n_cols;
};
    
template <arch_t arch, 
          typename value_t,
          typename index_t,
          typename tile_t>
struct DenseMat<arch, value_t, index_t, layout_t::COL_MAJOR, tile_t>
    : public StridedMat<arch, value_t, index_t, tile_t> {
    using strided_mat_t = StridedMat<arch, value_t, index_t, tile_t>;

    constexpr static arch_t arch_type = arch;

    DenseMat() = default;

    DenseMat(index_t n_rows, index_t n_cols) 
        : strided_mat_t(n_cols, n_rows), n_rows(n_rows), n_cols(n_cols) {}

    DenseMat(const std::vector<std::vector<value_t>>& mat_vec)
        : strided_mat_t(mat_vec), n_rows(mat_vec.front().size()), n_cols(mat_vec.size()) {}

    DenseMat(const DenseMat& rhs)
        : strided_mat_t(rhs), n_rows(rhs.n_rows), n_cols(rhs.n_cols) {}

    DenseMat(DenseMat&& rhs)
        : strided_mat_t(std::move(rhs)), n_rows(rhs.n_rows), n_cols(rhs.n_cols) {}

    template <arch_t from_arch>
    DenseMat(const DenseMat<from_arch, value_t, index_t, layout_t::COL_MAJOR, tile_t>& rhs) 
        : strided_mat_t(rhs), n_rows(rhs.n_rows), n_cols(rhs.n_cols) {}

    DenseMat& operator= (const DenseMat& rhs) {
        if (this != &rhs) {
            strided_mat_t::operator=(rhs);
            n_rows = rhs.n_rows;
            n_cols = rhs.n_cols;
        }
        return *this;
    }

    template <arch_t from_arch>
    DenseMat& operator= (const DenseMat<from_arch, value_t, index_t, layout_t::COL_MAJOR, tile_t>& rhs) {
        strided_mat_t::operator=(rhs);
        n_rows = rhs.n_rows;
        n_cols = rhs.n_cols;
        return *this;
    }   

    DenseMat& operator= (DenseMat&& rhs) {
        if (this != &rhs) {
            strided_mat_t::operator=(std::move(rhs));
            n_rows = rhs.n_rows;
            n_cols = rhs.n_cols;
        }
        return *this;
    }  

    std::string ToString() const {
        return "DenseMat{ "
            "arch_type:" + utils::ToString(arch_type) + ", " +
            "layout:COL_MAJOR, " +
            "n_rows:" + utils::NumToString(n_rows) + ", " +
            "n_cols:" + utils::NumToString(n_cols) + ", " +
            "values:" + strided_mat_t::ToString() +
            " }";
    }
    
    index_t n_rows;
    index_t n_cols;
};

} // namespace graph_genlx