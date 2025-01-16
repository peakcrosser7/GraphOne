#pragma once

#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>

#include "GraphOne/type.hpp"
#include "GraphOne/utils/string.hpp"
#include "GraphOne/utils/log.hpp"
#include "GraphOne/loader/opts.hpp"
#include "GraphOne/loader/edge_cache.h"
#include "GraphOne/vec/dense.h"

namespace graph_one {

template <vstart_t v_start = vstart_t::FROM_0_TO_0,
          typename index_t = vid_t,
          typename offset_t = eid_t>
class GraphLoader {
  public:
    GraphLoader(bool reorder_vid = false) 
    : reorder_vid_(reorder_vid), new_vid_(0), vid_map_() {
        if constexpr (v_start != vstart_t::FROM_0_TO_0) {
            new_vid_ = 1;
        }
    }

    template <typename edata_t> // cannot deduce
    EdgeCache<v_start, edata_t, index_t, offset_t> 
    LoadEdgesFromTxt(const std::string& filepath,
        const LoadEdgeOpts& opts = LoadEdgeOpts()) {
        LOG_DEBUG("LoadEdgesFromTxt: opts=", opts);
        
        if (!utils::StrEndWith(filepath, opts.file_ext)) {
            LOG_ERROR("file extension does not match, it should \"", 
                opts.file_ext, "\"");
        }

        std::fstream fin(filepath);
        if (fin.fail()) {
            LOG_ERROR("cannot open graph adj file: ", filepath);
        }

        EdgeCache<v_start, edata_t, index_t, offset_t> edge_cache;
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
            line = utils::StrStrip(line);

            if (line.empty()) {
                continue;
            }
            if (utils::StrStartWith(line, opts.comment_prefix)) {
                continue;
            }

            pLog   = line.data();
            pToken = strtok_r(line.data(), opts.line_sep.c_str(), &pSave);
            if (nullptr == pToken) {
                LOG_WARNING("LoadEdgesFromTxt: can not extract source from (", pLog, ")");
                continue;
            }
            edge.src = index_t(std::strtoul(pToken, nullptr, 10));

            pLog = pToken;
            pToken = strtok_r(nullptr, opts.line_sep.c_str(), &pSave);
            if (nullptr == pToken) {
                LOG_WARNING("can not extract destination from (", pLog, ")");
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
                } else {
                    if constexpr (std::is_arithmetic_v<edata_t>) {
                        edge.edata = edata_t(1);
                    } else {
                        edge.edata = edata_t{};
                    }
                }
            }

            // LOG_DEBUG("loaded-edge: src=", edge.src, ", dst=", edge.dst);
            if (BeforeEdge_(edge, opts) == false) {
                continue;
            }
            // LOG_DEBUG("pushed-edge: src=", edge.src, ", dst=", edge.dst);
            edge_cache.push_back(edge);

            if (AfterEdge_(edge, opts)) {
                edge_cache.push_back(edge);
            }
        }

        AfterAllEdges_(edge_cache, opts);

        LOG_DEBUG("edge_info: ", "num_v=", edge_cache.num_vertices(), " num_e=", edge_cache.num_edges());
        return edge_cache;      
    }

    template <arch_t arch, typename edata_t = double> // cannot deduce
    CsrMat<arch, edata_t, index_t, offset_t, v_start> 
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
            LOG_ERROR("cannot open graph vertex file: ", filepath);
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
                LOG_WARNING("LoadVertexStatusFromTxt: can not extract source from (", pLog, ")");
                continue;
            }
            auto vertex = index_t(std::strtoul(pToken, nullptr, 10));
            // LOG_DEBUG("loaded vertex: ", vertex);
            if (reorder_vid_) {
                vertex = ReorderVid_(vertex);
            }
            // LOG_DEBUG("reordered vertex: ", vertex);


            if constexpr (std::is_same_v<parse_t, char*>) {
                if (parser(vertex, pSave) == false) {
                    LOG_WARNING("LoadVertexStatusFromTxt: can not decode vertex data from (", pLog, ")");
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
                    LOG_WARNING("can not decode vertex data from  vertex:", 
                        utils::VecToString(splits));
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

    /// @return wether find the vertex and reoreder
    bool ReorderedVid(index_t& vid) const {
        if (reorder_vid_ == false) {
            return true;
        }
        auto it = vid_map_.find(vid);
        if (it != vid_map_.end()) {
            vid = it->second;
            return true;
        } 
        return false;
    }

    const std::unordered_map<index_t, index_t>& get_vid_map() const {
        return vid_map_;
    }
    
    
protected:

    index_t ReorderVid_(index_t vid) {
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
        // 对于从1起始的结点过滤0号结点
        if (v_start == vstart_t::FROM_1_TO_0
                && (edge.src == 0 || edge.dst == 0)) {
            LOG_WARNING("BeforeEdge_: loaded an egde with vertex-0 but starting from 1: edge=(", 
                        edge.src, ", ", edge.dst, ")");
            return false;
        }
        if constexpr (v_start == vstart_t::FROM_1_TO_0) {
            --edge.src;
            --edge.dst;
        }

        if (reorder_vid_ == true) {
            edge.src = ReorderVid_(edge.src);
            edge.dst = ReorderVid_(edge.dst);
        }

        return true;
    }

    template <typename edata_t>
    bool AfterEdge_(EdgeUnit<edata_t, index_t>& edge, const LoadEdgeOpts& opts) {
        if (opts.is_directed == false && edge.src != edge.dst) {
            std::swap(edge.src, edge.dst);
            return true;
        }

        return false;   // default: not push edge
    }

    template <typename edata_t>
    void AfterAllEdges_(EdgeCache<v_start, edata_t, index_t, offset_t>& edge_cache, const LoadEdgeOpts& opts) {
        if (opts.keep_duplicate_edges == false) {
            std::sort(edge_cache.begin(), edge_cache.end());
            edge_cache.erase(std::unique(edge_cache.begin(), edge_cache.end()), edge_cache.end());
        }
    }

    // /// 结点ID从0起始转换为从1起始,默认不是(即0到0或1到1,无需额外调整)
    // vstart_t vid_start_;

    /// 是否对结点重排序,默认重排
    bool reorder_vid_;
    index_t new_vid_;
    std::unordered_map<index_t, index_t> vid_map_;
};

} // namespace graph_one