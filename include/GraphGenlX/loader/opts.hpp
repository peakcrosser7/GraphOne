#pragma once

#include <string>

namespace graph_genlx {

struct LoadEdgeOpts {
    /// @brief 文件后缀名
    std::string file_ext = ".adj";
    /// @brief 文件注释行前缀
    std::string comment_prefix = "%";
    /// @brief 数据分隔符
    std::string line_sep = " ";

    /// @brief 是否是有向图,默认无向图
    bool is_directed = false;
    /// @brief 是否保留自环边,默认不保留
    bool keep_self_loop = false;
    /// @brief 是否保留重边,默认不保留
    bool keep_duplicate_edges = false;
};

struct LoadVertexOpts {
    /// @brief 文件注释行前缀
    std::string comment_prefix = "%";

    /// @brief 结点分隔符
    std::string vertex_sep = " ";
    /// @brief 结点属性数据分隔符
    std::string vdata_sep = " ";
};

} // namespace graph_genlx