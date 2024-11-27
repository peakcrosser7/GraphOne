#pragma once

#include <string>

#include "CLI11/CLI11.hpp"

#include "GraphOne/loader/opts.hpp"

namespace graph_one {

namespace {
inline std::string desc_with_default(std::string&& desc, const std::string& default_val) {
    return desc + " (default '" + default_val + "')";
}

}

inline void add_common_args(CLI::App& app, 
    std::string& input_graph, std::string &output_path,
    bool& reorder_vid, LoadEdgeOpts& opts) {

    app.add_option("-i,--input_graph", input_graph, "input graph dataset file")->required();
    app.add_option("-o,--output", output_path, "output path to save result");
    app.add_flag("-r,--reorder", reorder_vid, desc_with_default("reorder vertex id in graph dataset to compress", reorder_vid ? "true" : "false"));

    app.add_option("--file_ext", opts.file_ext, desc_with_default("graph dataset file extension", opts.file_ext));
    app.add_option("--comment_prefix", opts.comment_prefix, desc_with_default("graph dataset file comment line prefix", opts.comment_prefix));
    app.add_option("--line_sep", opts.line_sep, desc_with_default("graph dataset file line data delimiter", opts.line_sep));
    app.add_flag("-d,--is_directed", opts.is_directed, desc_with_default("directed graph dataset", opts.is_directed ? "true" : "false"));
    app.add_flag("--keep_self", opts.keep_self_loop, desc_with_default("keep self loop edges in graph", opts.keep_self_loop ? "true" : "false"));
    app.add_flag("--keep_dup", opts.keep_duplicate_edges, desc_with_default("keep duplicate edges in graph", opts.keep_duplicate_edges ? "true" : "false"));
}

}