#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <vector>

#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>

#include "GraphGenlX/type.hpp"
#include "GraphGenlX/archi.h"
#include "GraphGenlX/utils.h"
#include "GraphGenlX/buffer.h"
#include "GraphGenlX/mat/coo.h"
#include "GraphGenlX/mat/csr.h"
#include "GraphGenlX/mat/csc.h"
#include "GraphGenlX/mat/convert.h"

namespace graph_genlx::mat {

template<typename value_t = double,
         typename index_t = vid_t,
         typename nnz_t = eid_t>
CooMat<arch_t::cpu, value_t, index_t, nnz_t> LoadCooFromTxt(const std::string& filepath,
    bool is_directed = true,
    const std::string& file_ext = ".adj",
    const std::string& comment_prefix = "#",
    char line_sep = ' ') {

    if (!utils::StrEndWith(filepath, file_ext)) {
        LOG_ERROR << "file extension does not match, it should \""
            << file_ext << "\"\n";
        exit(1);
    }

    std::fstream fin(filepath);
    if (fin.fail()) {
        LOG_ERROR << "cannot open graph adj file: " << filepath << std::endl;
        exit(1);
    }


    vector_t<arch_t::cpu, index_t> srcs;
    vector_t<arch_t::cpu, index_t> dsts;
    vector_t<arch_t::cpu, value_t> weights;
    
    constexpr int MAX_CNT = std::is_same_v<empty_t, value_t> ? 2 : 3;
    
    index_t max_id = 0;
    index_t src, dst;
    value_t val{};
    std::string words[MAX_CNT];

    std::string line;
    while (std::getline(fin, line)) {
        if (utils::StrStartWith(line, comment_prefix)) {
            continue;
        }
        std::stringstream ss;
        ss << line;

        int cnt = 0;
        while (cnt < MAX_CNT && std::getline(ss, words[cnt], line_sep)) {
            ++cnt;
        }
        if (cnt < 2) {
            LOG_ERROR << "read edge error: " << line << std::endl;
        }

        if constexpr (MAX_CNT == 3) {
            if (cnt == 3) {
                val = value_t(std::stod(words[2]));
            }
        } 

        src = index_t(std::stoi(words[0]));
        dst = index_t(std::stoi(words[1]));
        max_id = std::max<index_t>(max_id, std::max<index_t>(src, dst));

        srcs.push_back(src);
        dsts.push_back(dst);
        weights.push_back(val);
        if (!is_directed) {
            srcs.push_back(dst);
            dsts.push_back(src);
            weights.push_back(val);
        }
    }

    CooMat<arch_t::cpu, value_t, index_t, nnz_t> coo(
        max_id + 1, max_id + 1, 
        std::move(srcs), std::move(dsts), std::move(weights));
    return coo;
}

template <arch_t arch,  // cannot deduce
        typename value_t,
        typename index_t,
        typename offset_t>
CsrMat<arch, value_t, index_t, offset_t>
ToCsr(const CooMat<arch_t::cpu, value_t, index_t, offset_t>& coo) {
    index_t n_rows = coo.n_rows;
    index_t n_cols = coo.n_cols;
    offset_t nnz = coo.nnz;


    Buffer<offset_t, arch_t::cpu, index_t> row_offsets(coo.n_rows + 1);
    Buffer<index_t, arch_t::cpu, offset_t> col_indices(coo.nnz);
    Buffer<value_t, arch_t::cpu, offset_t> values(coo.nnz);

    // compute number of non-zero entries per row
    for (offset_t i = 0; i < nnz; ++i) {
        ++row_offsets[coo.row_indices[i]];
    }
    
    // cumulative sum the nnz per row to get row_offsets[]
    for (index_t r = 0, total = 0; r <= n_rows; ++r) {
        index_t tmp = row_offsets[r];
        row_offsets[r] = total;
        total += tmp;
    }
    row_offsets[n_rows] = nnz;

    for (offset_t i = 0; i < nnz; ++i) {
        index_t row = coo.row_indices[i];
        index_t row_off = row_offsets[row];
        col_indices[row_off] = coo.col_indices[i];
        values[row_off] = coo.values[i];
        ++row_offsets[row];
    }

    for (index_t r = 0, pre = 0; r <= n_rows; ++r) {
        index_t tmp = row_offsets[r];
        row_offsets[r] = pre;
        pre = tmp;
    }

    if constexpr (arch == arch_t::cpu) {
        return CsrMat<arch, value_t, index_t, offset_t>(
            n_rows, n_cols, nnz,
            std::move(row_offsets), 
            std::move(col_indices),
            std::move(values)
        );
    }
    return CsrMat<arch, value_t, index_t, offset_t>(
        n_rows, n_cols, nnz,
        row_offsets,    // use Buffer's copy ctor to convert from arch_t::cpu to arch 
        col_indices,
        values
    );

}

template <arch_t arch,
          typename value_t, 
          typename index_t,
          typename offset_t>
CooMat<arch, value_t, index_t, offset_t>
ToCoo(const CsrMat<arch, value_t, index_t, offset_t>& csr) {
    return {};
}

template <arch_t arch,
        typename value_t,
        typename index_t,
        typename offset_t>
CscMat<arch, value_t, index_t, offset_t>
ToCsc(const CsrMat<arch, value_t, index_t, offset_t>& csr) {
    Buffer<index_t, arch, offset_t> row_indices = 
        mat::OffsetsToIndices(csr.row_offsets, csr.nnz);
    Buffer<index_t, arch, offset_t> col_indices = csr.col_indices;
    Buffer<value_t, arch, offset_t> values = csr.values;

    auto zip_it = thrust::make_zip_iterator(
        thrust::make_tuple(row_indices.begin(), values.begin())
    );
    thrust::sort_by_key(archi::exec_policy<arch>, col_indices.begin(), col_indices.end(), zip_it);

    Buffer<offset_t, arch, index_t> col_offsets = 
        mat::IndicesToOffsets(col_indices, csr.n_cols + 1);

    return CscMat<arch, value_t, index_t, offset_t>(
        csr.n_rows, csr.n_cols, csr.nnz,
        std::move(col_offsets), 
        std::move(row_indices),
        std::move(values)
    );
}

} // namespace graph_genlx