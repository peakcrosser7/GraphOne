#pragma once

#include <string>

namespace graph_one {

struct LoadEdgeOpts {
    /// @brief 文件后缀名
    std::string file_ext = ".adj";
    /// @brief 文件注释行前缀
    std::string comment_prefix = "%";
    /// @brief 数据分隔符
    std::string line_sep = " \t";

    /// @brief 是否是有向图,默认无向图
    bool is_directed = false;
    /// @brief 是否保留自环边,默认不保留
    bool keep_self_loop = false;
    /// @brief 是否保留重边,默认不保留
    bool keep_duplicate_edges = false;

    std::string ToString() const {
        std::string str;
        str += "LoadEdgeOpts{ ";
        str += "file_ext: " + file_ext + ", ";
        str += "comment_prefix: " + comment_prefix + ", ";
        str +="line_sep: " + line_sep + ", ";
        str +="is_directed: " + utils::ToString(is_directed) + ", ";
        str +="keep_self_loop: " + utils::ToString(keep_self_loop) + ", ";
        str +="keep_duplicate_edges: " + utils::ToString(keep_duplicate_edges);
        str +=" }";
        return str;
    }
};

struct LoadVertexOpts {
    /// @brief 文件注释行前缀
    std::string comment_prefix = "%";

    /// @brief 结点分隔符
    std::string vertex_sep = " \t";
    /// @brief 结点属性数据分隔符
    std::string vdata_sep = " \t";
};

} // namespace graph_one