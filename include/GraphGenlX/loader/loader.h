#pragma once

#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>

#include "GraphGenlX/type.hpp"
#include "GraphGenlX/utils/string.hpp"
#include "GraphGenlX/utils/log.hpp"
#include "GraphGenlX/loader/edge_cache.h"
#include "GraphGenlX/vec/dense.h"

namespace graph_genlx {

enum class vstart_t {
    FROM_0_TO_0,
    FROM_0_TO_1,
    FROM_1_TO_1
};

struct LoadEdgeOpts {
    std::string file_ext = ".adj";
    std::string comment_prefix = "#";
    std::string line_sep = " ";

    /// 是否是有向图,默认是
    bool is_directed = true;
    /// 是否保留自环边,默认不保留
    bool keep_self_loop = false;
    /// 是否保留重边,默认不保留
    bool keep_duplicate_edges = false;
};

struct LoadVertexOpts {
    std::string comment_prefix = "#";

    std::string vertex_sep = " ";
    std::string vdata_sep = " ";
};

template <typename index_t = vid_t>
class Loader {
public:
    explicit Loader(bool reorder_vid = true, vstart_t vid_start = vstart_t::FROM_0_TO_0) 
        : reorder_vid_(reorder_vid), vid_start_(vid_start) {
        if (vid_start != vstart_t::FROM_0_TO_0) {
            ++new_vid_;
        }
    }

    template <typename edata_t> // cannot deduce
    EdgeCache<edata_t, index_t> LoadEdgesFromTxt(const std::string& filepath,
        const LoadEdgeOpts& opts = LoadEdgeOpts()) {
        if (!utils::StrEndWith(filepath, opts.file_ext)) {
            LOG_ERROR << "file extension does not match, it should \""
                << opts.file_ext << "\"\n";
            exit(1);
        }

        std::fstream fin(filepath);
        if (fin.fail()) {
            LOG_ERROR << "cannot open graph adj file: " << filepath << std::endl;
            exit(1);
        }

        EdgeCache<edata_t, index_t> edge_cache;
        size_t init_cap = (1 << 12) / sizeof(EdgeUnit<edata_t, index_t>);
        edge_cache.reserve(init_cap);

        constexpr int MAX_CNT = std::is_same_v<empty_t, edata_t> ? 2 : 3;
        
        EdgeUnit<edata_t, index_t> edge;

        std::string line;
        char* pSave  = nullptr;
        char* pToken = nullptr;
        char* pLog   = nullptr;
        while (fin.good() && !fin.eof()) {
            std::getline(fin, line);
            if (utils::StrStartWith(line, opts.comment_prefix)) {
                continue;
            }

            pLog   = line.data();
            pToken = strtok_r(line.data(), opts.line_sep.c_str(), &pSave);
            if (nullptr == pToken) {
                LOG_WARNING << "can not extract source from (" << pLog << ")\n";
                continue;
            }
            edge.src = index_t(std::strtoul(pToken, nullptr, 10));

            pLog = pToken;
            pToken = strtok_r(nullptr, opts.line_sep.c_str(), &pSave);
            if (nullptr == pToken) {
                LOG_WARNING << "can not extract destination from (" << pLog << ")\n";
                continue;
            }
            edge.dst = index_t(std::strtoul(pToken, nullptr, 10));

            if constexpr (MAX_CNT == 3) {
                pLog = pToken;
                pToken = strtok_r(nullptr, opts.line_sep.c_str(), &pSave);
                if (pToken != nullptr) {
                    static_assert(std::is_integral_v<edata_t> || std::is_floating_point_v<edata_t>);
                    if constexpr (std::is_integral_v<edata_t>) {
                        edge.edata = edata_t(std::strtoul(pToken, nullptr, 10));
                    } else {
                        edge.edata = edata_t(std::strtod(pToken, nullptr));
                    }
                }
            }

            if (BeforeEdge_(edge, opts) == false) {
                continue;
            }
            edge_cache.push_back(edge);

            if (AfterEdge_(edge, opts)) {
                edge_cache.push_back(edge);
            }
        }

        AfterAllEdges_(edge_cache, opts);
        return edge_cache;      
    }

    template <arch_t arch, typename edata_t = double, typename offset_t = eid_t> // cannot deduce
    CsrMat<arch, edata_t, index_t, offset_t> 
    LoadCsrFromTxt(const std::string& filepath, const LoadEdgeOpts& opts = LoadEdgeOpts()) {
        return LoadEdgesFromTxt<edata_t>(filepath, opts)
            .template ToCsr<arch, offset_t>();
    }

    template <typename parse_t, typename parse_func>
    void LoadVertexStatusFromTxt(
        const std::string& filepath, 
        parse_func&& parser,    // bool parser(indext_t vid, char* vdata) or bool parser(index_t vid, std::vector<parse_t>& vdata)
        const LoadVertexOpts& opts = LoadVertexOpts()) {

        std::fstream fin(filepath);
        if (fin.fail()) {
            LOG_ERROR << "cannot open graph vertex file: " << filepath << std::endl;
            exit(1);
        }

        std::string line;
        char* pSave  = nullptr;
        char* pToken = nullptr;
        char* pLog   = nullptr;
        std::vector<parse_t> splits;
        while (fin.good() && !fin.eof()) {
            std::getline(fin, line);
            if (utils::StrStartWith(line, opts.comment_prefix)) {
                continue;
            }

            pLog   = line.data();
            pToken = strtok_r(line.data(), opts.vertex_sep.c_str(), &pSave);
            if (nullptr == pToken) {
                LOG_WARNING << "can not extract source from (" << pLog << ")\n";
                continue;
            }
            auto vertex = index_t(std::strtoul(pToken, nullptr, 10));
            if (reorder_vid_) {
                vertex = ReorderVID_(vertex);
            }

            if constexpr (std::is_same_v<parse_t, char*>) {
                if (parser(vertex, pSave) == false) {
                    LOG_WARNING << "can not decode vertex data from (" << pLog << ")\n";
                    continue;
                }
            } else {
                splits.clear();

                pLog   = pToken;   
                pToken = strtok_r(nullptr, opts.vdata_sep.c_str(), &pSave);
                while (pToken) {
                    parse_t val;
                    static_assert(std::is_integral_v<parse_t> || std::is_floating_point_v<parse_t>);
                    if constexpr (std::is_integral_v<parse_t>) {
                        val = parse_t(std::strtoul(pToken, nullptr, 10));
                    } else {
                        val = parse_t(std::strtod(pToken, nullptr));
                    }
                    splits.push_back(val);

                    pLog   = pToken;   
                    pToken = strtok_r(nullptr, opts.vdata_sep.c_str(), &pSave);  
                }

                if (parser(vertex, splits) == false) {
                    LOG_WARNING << "can not decode vertex data from  vertex:" 
                        << utils::VecToString(splits) << "\n";
                    continue;
                }
            }
        }
    }

    template <arch_t arch,
             typename vdata_t>
    DenseVec<arch, vdata_t, index_t> LoadVertexVecFromTxt(
        const std::string& filepath, 
        index_t num_vertices,
        const LoadVertexOpts& opts = LoadVertexOpts()
    ) {
        DenseVec<arch_t::cpu, vdata_t, index_t> h_vprops(num_vertices);
        LoadVertexStatusFromTxt<char*>(filepath, 
            [&](index_t vid, char* vdata) {
                static_assert(std::is_integral_v<vdata_t> || std::is_floating_point_v<vdata_t>);
                if constexpr (std::is_integral_v<vdata_t>) {
                    h_vprops[vid] = vdata_t(std::strtoul(vdata, nullptr, 10));
                } else {
                    h_vprops[vid] = vdata_t(std::strtod(vdata, nullptr));
                }
                return true;
            }, opts);
        
        if constexpr (arch == arch_t::cpu) {
            return h_vprops;
        }
        return DenseVec<arch, vdata_t, index_t>(h_vprops);
    }
    

protected:

    index_t ReorderVID_(index_t vid) {
        auto it = vid_map_.find(vid);
        if (it != vid_map_.end()) {
            return it->second;
        }
        vid_map_[vid] = new_vid_;
        return new_vid_++;
    }    

    template <typename edata_t>
    bool BeforeEdge_(EdgeUnit<edata_t, index_t>& edge, const LoadEdgeOpts& opts) {
        if (opts.keep_self_loop == false
                && edge.src == edge.dst) {
            return false;
        }
        if (vid_start_ == vstart_t::FROM_0_TO_1) {
            ++edge.src;
            ++edge.dst;
        }
        // 对于从1起始的结点过滤0号结点
        if (vid_start_ == vstart_t::FROM_1_TO_1
                && (edge.src == 0 || edge.dst == 0)) {
            return false;
        }

        if (reorder_vid_ == true) {
            edge.src = ReorderVID_(edge.src);
            edge.dst = ReorderVID_(edge.dst);
        }

        return true;
    }

    template <typename edata_t>
    bool AfterEdge_(EdgeUnit<edata_t, index_t>& edge, const LoadEdgeOpts& opts) {
        if (opts.is_directed == false) {
            std::swap(edge.src, edge.dst);
            return true;
        }

        return false;   // default: not push edge
    }

    template <typename edata_t>
    void AfterAllEdges_(EdgeCache<edata_t, index_t>& edge_cache, const LoadEdgeOpts& opts) {
        if (opts.keep_duplicate_edges == false) {
            std::sort(edge_cache.begin(), edge_cache.end());
            edge_cache.erase(std::unique(edge_cache.begin(), edge_cache.end()), edge_cache.end());
        }
    }

    /// 是否对结点重排序,默认重排
    bool reorder_vid_{true};
    /// 结点ID从0起始转换为从1起始,默认不是(即0到0或1到1,无需额外调整)
    vstart_t vid_start_{vstart_t::FROM_0_TO_0};

    index_t new_vid_{0};
    std::unordered_map<index_t, index_t> vid_map_{};
};
    
} // namespace graph_genlx