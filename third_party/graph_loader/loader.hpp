#pragma once

#include <string>
#include <fstream>
#include <unordered_map>
#include <type_traits>

// #define DEBUG_LOG_LOADER
#include "exception.hpp"
#include "opts.hpp"
#include "helper.hpp"
#include "log.hpp"
#include "utils.hpp"
#include "opts_factory.hpp"

namespace graph_loader {

template <typename vertex_t, typename edge_t, typename weight_t>
class CoreLoader { 
public:
    template <typename edge_load_func_t, 
              typename pre_load_func_t = decltype(dummy_func), 
              typename weight_parse_func_t = decltype(general_weight_parse<weight_t>)>
    static void Load(
        const std::string& filepath, 
        LoaderOpts& opts,
        const edge_load_func_t& edge_load_func, // bool FUNC(edge_t& eidx, vertex_t& src, vertex_t& dst, weight_t& val)
        const pre_load_func_t& pre_load_func = dummy_func,  // void FUNC() OR void FUNC(vertex_t num_v, edge_t num_e);
        const weight_parse_func_t& weight_parse_func = general_weight_parse<weight_t>    // weight_t FUNC(const char* str)
    ) {
        if (opts.header_cnt == 0) {
            LoadWithoutHeader(filepath, opts, edge_load_func, pre_load_func, weight_parse_func);
        } else {
            LoadWithHeader(filepath, opts, edge_load_func, pre_load_func, weight_parse_func);
        }
    }

    template <typename edge_load_func_t, 
              typename pre_load_func_t = decltype(dummy_func), 
              typename weight_parse_func_t = decltype(general_weight_parse<weight_t>)>
    static void LoadWithoutHeader(
        const std::string& filepath, 
        LoaderOpts& opts,
        const edge_load_func_t& edge_load_func, // bool FUNC(edge_t& eidx, vertex_t& src, vertex_t& dst, weight_t& val)
        const pre_load_func_t& pre_load_func = dummy_func,   // void FUNC() OR void FUNC(vertex_t num_v, edge_t num_e)
        const weight_parse_func_t& weight_parse_func = general_weight_parse<weight_t>    // weight_t FUNC(const char* str)
    ) {
        std::ifstream fin = LoadPrepare_(filepath, opts);

        std::unordered_map<vertex_t, vertex_t> reordered_map;
        vertex_t num_v = LoadHeaderSimulating_(fin, opts, pre_load_func, reordered_map);
        LOG_DEBUG("LoadWithoutHeader: num_v=", num_v);

        LoadEdgesWithOpts_(fin, opts, num_v, reordered_map, edge_load_func, weight_parse_func);  
    }

    template <typename edge_load_func_t, 
              typename pre_load_func_t = decltype(dummy_func), 
              typename weight_parse_func_t = decltype(general_weight_parse<weight_t>)>
    static void LoadWithHeader(
        const std::string& filepath, 
        LoaderOpts& opts,
        const edge_load_func_t& edge_load_func, // bool FUNC(edge_t& eidx, vertex_t& src, vertex_t& dst, weight_t& val)
        const pre_load_func_t& pre_load_func = dummy_func,   // void FUNC() OR void FUNC(vertex_t num_v, edge_t num_e)
        const weight_parse_func_t& weight_parse_func = general_weight_parse<weight_t>    // weight_t FUNC(const char* str)
    ) {
        std::ifstream fin = LoadPrepare_(filepath, opts);

        std::unordered_map<vertex_t, vertex_t> reordered_map;
        vertex_t num_v = LoadHeader_(fin, opts, pre_load_func);

        LoadEdgesWithOpts_(fin, opts, num_v, reordered_map, edge_load_func, weight_parse_func);  
    }

private:
    template <typename edge_load_func_t>
    static auto MakeReorderedEdgeLoader_(
        BasedIndex based_index,
        std::unordered_map<vertex_t, vertex_t>& reordered_map,
        const edge_load_func_t& edge_load_func    // bool FUNC(edge_t& eidx, vertex_t& src, vertex_t& dst, weight_t& val)
    ) {
        vertex_t base = (based_index == BasedIndex::BASED_0_TO_0 || based_index == BasedIndex::BASED_1_TO_0) ? 0 : 1;
        auto reorder_vid = [&reordered_map, base](vertex_t vid) -> vertex_t {
            auto [it,_] = reordered_map.insert({vid, vertex_t(reordered_map.size() + base)});
            return it->second;
        };

        if constexpr (std::is_same_v<weight_t, empty_t>) {
            return [&edge_load_func, reorder_vid](edge_t eidx, vertex_t& src, vertex_t& dst) -> bool {
                // vertex_t old_src = src, old_dst = dst;
                src = reorder_vid(src);
                dst = reorder_vid(dst);
                // LOG_DEBUG("reorder: src=", old_src, "->", src, " dst=", old_dst, "->", dst);
                return edge_load_func(eidx, src, dst);
            };            
        } else {
            return [&edge_load_func, reorder_vid](edge_t eidx, vertex_t& src, vertex_t& dst, weight_t& val) -> bool {
                src = reorder_vid(src);
                dst = reorder_vid(dst);
                return edge_load_func(eidx, src, dst, val);
            };
        }
    }

    template <typename edge_load_func_t>
    static auto MakeChekingEdgeLoader_(const edge_load_func_t& edge_load_func, vertex_t num_v) {
        if constexpr (std::is_same_v<weight_t, empty_t>) {
            return [num_v, &edge_load_func](edge_t eidx, vertex_t& src, vertex_t& dst) -> bool {
                throw_if_exception(src >= num_v, 
                                   "src-vertex should smaller than number of vertices, "
                                   "you need to set `do_reorder=true` in LoaderOpts");
                throw_if_exception(dst >= num_v, 
                                   "dst-vertex should smaller than number of vertices, "
                                   "you need to set `do_reorder=true` in LoaderOpts");
                return edge_load_func(eidx, src, dst);
            };
        } else {
            return [num_v, &edge_load_func](edge_t eidx, vertex_t& src, vertex_t& dst, weight_t& val) -> bool {
                throw_if_exception(src >= num_v, 
                                   "src-vertex should smaller than number of vertices, "
                                   "you need to set `do_reorder=true` in LoaderOpts");
                throw_if_exception(dst >= num_v, 
                                   "dst-vertex should smaller than number of vertices, "
                                   "you need to set `do_reorder=true` in LoaderOpts");
                return edge_load_func(eidx, src, dst, val);
            };
        }  
    }

    template <typename edge_load_func_t>
    static auto MakeBased1to0AndCheckingEdgeLoader_(const edge_load_func_t& edge_load_func, vertex_t num_v) {
        if constexpr (std::is_same_v<weight_t, empty_t>) {
            return [num_v, &edge_load_func](edge_t eidx, vertex_t& src, vertex_t& dst) -> bool {
                throw_if_exception(src == 0, "file is one-indexed but got vertex 0");
                throw_if_exception(dst == 0, "file is one-indexed but got vertex 0");
                --src, --dst;
                throw_if_exception(src >= num_v, 
                                   "src-vertex should smaller than number of vertices, "
                                   "you need to set `do_reorder=true` in LoaderOpts");
                throw_if_exception(dst >= num_v, 
                                   "dst-vertex should smaller than number of vertices, "
                                   "you need to set `do_reorder=true` in LoaderOpts");
                return edge_load_func(eidx, src, dst);
            };
        } else {
            return [num_v, &edge_load_func](edge_t eidx, vertex_t& src, vertex_t& dst, weight_t& val) -> bool {
                throw_if_exception(src == 0, "file is one-indexed but got vertex 0");
                throw_if_exception(dst == 0, "file is one-indexed but got vertex 0");
                --src, --dst;
                throw_if_exception(src >= num_v, 
                                   "src-vertex should smaller than number of vertices, "
                                   "you need to set `do_reorder=true` in LoaderOpts");
                throw_if_exception(dst >= num_v, 
                                   "dst-vertex should smaller than number of vertices, "
                                   "you need to set `do_reorder=true` in LoaderOpts");
                return edge_load_func(eidx, src, dst, val);
            };
        }
    }

    static bool LoadLine_(
        std::ifstream& fin, 
        std::string& line, 
        const LoaderOpts& opts,
        bool do_check = true
    ) {
        while(fin.good() && !fin.eof()) {
            std::getline(fin, line);
            if (do_check) {
                line = utils::StrStrip(line, opts.line_strips);
                if (line.empty() 
                    || (!opts.comment_prefix.empty() && utils::StrStartWith(line, opts.comment_prefix))) {
                    continue;
                }
            }
            return true;
        }
        return false;
    }

    static std::ifstream LoadPrepare_(const std::string& filepath, LoaderOpts& opts) {
        throw_if_exception(!utils::StrEndWith(filepath, opts.file_ext),
                           "File should have \"" + opts.file_ext + "\" file extension.");
        
        std::ifstream fin(filepath);
        throw_if_exception(fin.fail(), "Cannot open file: " + filepath);

        if (opts.load_banner_func) {
            std::string line;
            bool ok = LoadLine_(fin, line, opts, false);
            throw_if_exception(!ok, "Load file when calling pre_load_func");
            opts.load_banner_func(line, opts);
        }
        return fin;
    }

    template <typename pre_load_func_t>
    static vertex_t LoadHeaderSimulating_(
        std::ifstream& fin,
        LoaderOpts& opts,
        const pre_load_func_t& pre_load_func,  // void FUNC(vertex_t num_v, edge_t num_e)
        std::unordered_map<vertex_t, vertex_t>& reordered_map
    ) {
        vertex_t num_v = std::numeric_limits<vertex_t>::max();

        if constexpr (std::is_invocable_r_v<void, pre_load_func_t, vertex_t, edge_t>) {
            edge_t num_e = 0;

            if constexpr (std::is_same_v<weight_t, empty_t>) {
                if (opts.do_reorder) {
                    auto get_ve_edge_load_func = [&](edge_t eidx, vertex_t& src, vertex_t& dst) {
                            ++num_e;
                            return true;
                    };
                    auto get_ve_edge_load_wrapper = MakeReorderedEdgeLoader_(opts.based_index, reordered_map, get_ve_edge_load_func);
                    LoadEdges_(fin, opts, get_ve_edge_load_wrapper);

                    num_v = reordered_map.size();
                } else {
                    vertex_t max_id = -1;
                    auto get_ve_edge_load_func = [&](edge_t eidx, vertex_t& src, vertex_t& dst) {
                        max_id = std::max(max_id, src);
                        max_id = std::max(max_id, dst);
                        ++num_e;
                        return true;
                    };
                    LoadEdges_(fin, opts, get_ve_edge_load_func);
                    
                    num_v = max_id + (opts.based_index == BasedIndex::BASED_0_TO_0);
                }
            } else {
                if (opts.do_reorder) {
                    auto get_ve_edge_load_func = [&](edge_t eidx, vertex_t& src, vertex_t& dst, weight_t& val) {
                            ++num_e;
                            return true;
                    };
                    auto get_ve_edge_load_wrapper = MakeReorderedEdgeLoader_(opts.based_index, reordered_map, get_ve_edge_load_func);
                    LoadEdges_(fin, opts, get_ve_edge_load_wrapper);

                    num_v = reordered_map.size();
                } else {
                    vertex_t max_id = -1;
                    auto get_ve_edge_load_func = [&](edge_t eidx, vertex_t& src, vertex_t& dst, weight_t& val) {
                        max_id = std::max(max_id, src);
                        max_id = std::max(max_id, dst);
                        ++num_e;
                        return true;
                    };
                    LoadEdges_(fin, opts, get_ve_edge_load_func);
                    
                    num_v = max_id + (opts.based_index == BasedIndex::BASED_0_TO_0);
                }
            }

            pre_load_func(num_v, num_e);

        } else if constexpr (std::is_invocable_r_v<void, pre_load_func_t>) {
            pre_load_func();
        } else {
            static_assert(DependentFalse<pre_load_func_t>::value, 
                          "pre_load_func must be callable with either no arguments or two arguments: (vertex_t, edge_t)");
        }

        fin.clear();
        fin.seekg(0, std::ios::beg);

        return num_v;
    }
    
    static void ParseHeader2_(
        std::string& line, 
        const std::string& sep, 
        vertex_t& num_v, 
        edge_t& num_e
    ) {
        char* pSave  = nullptr;
        char* pToken = nullptr;

        pToken = strtok_r(line.data(), sep.c_str(), &pSave);
        throw_if_exception(pToken == nullptr, "fail to load num_v when calling ParseHeader2_");
        auto num_v_ = utils::StrToNum<>(pToken);
        throw_if_exception(num_v_ >= std::numeric_limits<vertex_t>::max(),
                        "vertex_t overflow when calling ParseHeader2_");

        pToken = strtok_r(nullptr, sep.c_str(), &pSave);
        throw_if_exception(pToken == nullptr, "fail to~ load num_e when calling ParseHeader2_");
        auto num_e_ = utils::StrToNum<>(pToken);
        throw_if_exception(num_e_ >= std::numeric_limits<edge_t>::max(), "edge_t overflow when calling ParseHeader2_");

        num_v = static_cast<vertex_t>(num_v_);
        num_e = static_cast<edge_t>(num_e_);
    }

    static void ParseHeader3_(
        std::string& line, 
        const std::string& sep, 
        vertex_t& num_v, 
        edge_t& num_e
    ) {
        char* pSave  = nullptr;
        char* pToken = nullptr;

        pToken = strtok_r(line.data(), sep.c_str(), &pSave);
        throw_if_exception(pToken == nullptr, "fail to load num_v when calling ParseHeader3_");
        auto num_rows = utils::StrToNum<>(pToken);

        pToken = strtok_r(nullptr, sep.c_str(), &pSave);
        throw_if_exception(pToken == nullptr, "fail to load num_cols when calling ParseHeader3_");
        auto num_cols = utils::StrToNum<>(pToken);

        throw_if_exception(num_rows >= std::numeric_limits<vertex_t>::max() ||
            num_cols >= std::numeric_limits<vertex_t>::max(),
            "vertex_t overflow when calling ParseHeader3_");
    
        throw_if_exception(num_rows != num_cols, "num_rows != num_cols i.e. the file is NOT a graph when calling ParseHeader3_."); 

        pToken = strtok_r(nullptr, sep.c_str(), &pSave);
        throw_if_exception(pToken == nullptr, "fail to load num_e when calling ParseHeader3_");
        auto nnz = utils::StrToNum<>(pToken);
        throw_if_exception(nnz >= std::numeric_limits<edge_t>::max(), "edge_t overflow when calling ParseHeader3_");

        num_v = static_cast<vertex_t>(num_rows);
        num_e = static_cast<edge_t>(nnz);    
    }

    template <typename pre_load_func_t>
    static vertex_t LoadHeader_(
        std::ifstream& fin, 
        const LoaderOpts& opts, 
        const pre_load_func_t& pre_load_func   // void FUNC(vertex_t num_v, edge_t num_e)
    ) {
        std::string line;
        bool ok = LoadLine_(fin, line, opts);
        throw_if_exception(!ok, "LoadLine_ failed when calling LoadHeader_");

        vertex_t num_v;
        edge_t num_e;
        if (opts.header_cnt == 2) {
            ParseHeader2_(line, opts.line_sep, num_v, num_e);
        } else if (opts.header_cnt == 3) {
            ParseHeader3_(line, opts.line_sep, num_v, num_e);
        } else {
            throw_if_exception(true, "unsupport header cnt: " + std::to_string(opts.header_cnt));
        }
        LOG_DEBUG("LoadHeader_: num_v=", num_v, " num_e=", num_e);

        if constexpr (std::is_invocable_r_v<void, pre_load_func_t, vertex_t, edge_t>) {
            pre_load_func(num_v, num_e);
        } else if constexpr (std::is_invocable_r_v<void, pre_load_func_t>) {
            pre_load_func();
        } else {
            static_assert(DependentFalse<pre_load_func_t>::value, 
                          "pre_load_func must be callable with either no arguments or two arguments: (vertex_t, edge_t)");
        }

        return num_v;
    }  

    template <typename edge_load_func_t, 
              typename weight_parse_func_t = decltype(general_weight_parse<weight_t>)>
    static void LoadEdges_(
        std::ifstream& fin, 
        const LoaderOpts& opts, 
        const edge_load_func_t& edge_load_func,
        const weight_parse_func_t& weight_parse_func = general_weight_parse<weight_t>
    ) {
        std::string line;
        char* pSave  = nullptr;
        char* pToken = nullptr;
        char* pLog = nullptr;
        for (edge_t eidx = 0; true;) {
            if (!LoadLine_(fin, line, opts)) {
                break;
            }

            pLog = line.data();
            pToken = strtok_r(line.data(), opts.line_sep.c_str(), &pSave);
            if (nullptr == pToken) {
                LOG_WARNING("can not extract source from (", pLog, ")");
                continue;
            }
            vertex_t src = utils::StrToNum<vertex_t>(pToken);

            pLog = pToken;
            pToken = strtok_r(nullptr, opts.line_sep.c_str(), &pSave);
            if (nullptr == pToken) {
                LOG_WARNING("can not extract destination from (", pLog, ")");
                continue;
            }
            vertex_t dst = utils::StrToNum<vertex_t>(pToken);

            // LOG_DEBUG("LoadEdges_: eidx=", eidx, " src=", src, " dst=", dst);
            if constexpr (std::is_same_v<weight_t, empty_t>) {
                if (edge_load_func(eidx, src, dst)) {
                    ++eidx;
                }                
            } else {
                pToken = strtok_r(nullptr, opts.line_sep.c_str(), &pSave);
                weight_t val = weight_parse_func(pToken);
                if (edge_load_func(eidx, src, dst, val)) {
                    ++eidx;
                }
            }
        }
        LOG_DEBUG("end of LoadEdges_()");
    }

    template <typename edge_load_func_t, typename weight_parse_func_t>
    static void LoadEdgesWithOpts_(
        std::ifstream& fin, 
        const LoaderOpts& opts, 
        vertex_t num_v,
        std::unordered_map<vertex_t, vertex_t>& reordered_map,
        const edge_load_func_t& edge_load_func,
        const weight_parse_func_t& weight_parse_func
    ) {
        if (opts.do_reorder) {
            LOG_DEBUG("LoadEdgesWithOpts_: will reorder");
            auto edge_load_wrapper = MakeReorderedEdgeLoader_(opts.based_index, reordered_map, edge_load_func);
            LoadEdges_(fin, opts, edge_load_wrapper, weight_parse_func);
        } else if (opts.based_index == BasedIndex::BASED_1_TO_0) {
            auto edge_load_wrapper = MakeBased1to0AndCheckingEdgeLoader_(edge_load_func, num_v);
            LoadEdges_(fin, opts, edge_load_wrapper, weight_parse_func);
        } else {
            auto edge_load_wrapper = MakeChekingEdgeLoader_(edge_load_func, num_v);
            LoadEdges_(fin, opts, edge_load_wrapper, weight_parse_func);
        }   
    }
};

} // namespace graph_loader



