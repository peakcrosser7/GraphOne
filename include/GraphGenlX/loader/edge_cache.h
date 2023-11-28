#pragma once

#include <vector>

#include "GraphGenlX/mat/csr.h"

namespace graph_genlx {

template <typename edata_t, 
          typename index_t>
struct EdgeUnit {
    index_t src;
    index_t dst;
    edata_t edata;

    bool operator< (const EdgeUnit& rhs) const {
        return src < rhs.src || (src == rhs.src && dst < rhs.dst);
    }

    bool operator== (const EdgeUnit& rhs) const {
        return src == rhs.src && dst == rhs.dst;
    }
};

template <typename edata_t, 
          typename index_t,
          template<typename> class alloc_t = std::allocator>
class EdgeCache : public std::vector<EdgeUnit<edata_t, index_t>, alloc_t<EdgeUnit<edata_t, index_t>>> {
public:
    void push_back(const EdgeUnit<edata_t, index_t>& edge) {
        vec_spec_t::push_back(edge);
        max_vid_ = std::max(max_vid_, std::max(edge.src, edge.dst));
    }

    index_t max_vid() const {
        return max_vid_;
    }

    index_t num_edges() const {
        return vec_spec_t::size();
    }

    index_t num_vertices() const {
        return max_vid_ + 1;
    }

    template <arch_t arch, // cannot deduce
              typename offset_t = eid_t>
    CsrMat<arch, edata_t, index_t, offset_t> ToCsr() const {
        auto edge_cache = *this;

        index_t n_rows = edge_cache.num_vertices();
        index_t n_cols = edge_cache.num_vertices();
        offset_t nnz = edge_cache.num_edges();

        Buffer<arch_t::cpu, offset_t, index_t> row_offsets(n_rows + 1);
        Buffer<arch_t::cpu, index_t, offset_t> col_indices(nnz);
        Buffer<arch_t::cpu, edata_t, offset_t> values(nnz);

        // compute number of non-zero entries per row
        for (offset_t i = 0; i < nnz; ++i) {
            ++row_offsets[edge_cache[i].src];
        }
        
        // cumulative sum the nnz per row to get row_offsets[]
        for (index_t r = 0, total = 0; r <= n_rows; ++r) {
            index_t tmp = row_offsets[r];
            row_offsets[r] = total;
            total += tmp;
        }
        row_offsets[n_rows] = nnz;

        for (offset_t i = 0; i < nnz; ++i) {
            auto edge = edge_cache[i];
            index_t row = edge.src;
            index_t row_off = row_offsets[row];
            col_indices[row_off] = edge.dst;
            values[row_off] = edge.edata;
            ++row_offsets[row];
        }

        for (index_t r = 0, pre = 0; r <= n_rows; ++r) {
            index_t tmp = row_offsets[r];
            row_offsets[r] = pre;
            pre = tmp;
        }

        if constexpr (arch == arch_t::cpu) {
            return CsrMat<arch, edata_t, index_t, offset_t>(
                n_rows, n_cols, nnz,
                std::move(row_offsets), 
                std::move(col_indices),
                std::move(values)
            );
        }
        return CsrMat<arch, edata_t, index_t, offset_t>(
            n_rows, n_cols, nnz,
            row_offsets,    // use Buffer's copy ctor to convert from arch_t::cpu to arch 
            col_indices,
            values
        );
    }

protected:
    using vec_spec_t = std::vector<EdgeUnit<edata_t, index_t>, alloc_t<EdgeUnit<edata_t, index_t>>>;

    index_t max_vid_{0};
};
    
} // namespace graph_genlx