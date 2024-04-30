#pragma once

#include <string>

#include "CLI11/CLI11.hpp"

#include "GraphOne/loader/opts.hpp"

namespace graph_one {

inline void add_common_args(CLI::App& app, 
    std::string& input_graph, std::string &output_path,
    bool& reorder_vid, LoadEdgeOpts& opts) {

    app.add_option("-i,--input_graph", input_graph, "input graph dataset file")->required();
    app.add_option("-o,--output", output_path, "output path to save result");
    app.add_flag("-r,--reorder", reorder_vid, "reorder vertex id in graph dataset to compress (defalut 'false')");

    app.add_option("--file_ext", opts.file_ext, "graph dataset file extension (defalut '.adj')");
    app.add_option("--comment_prefix", opts.comment_prefix, "graph dataset file comment line prefix (default '%')");
    app.add_option("--line_sep", opts.line_sep, "graph dataset file line data delimiter (default ' ')");
    app.add_flag("-d,--is_directed", opts.is_directed, "directed graph dataset (defalut 'false')");
    app.add_flag("--keep_self", opts.keep_self_loop, "keep self loop edges in graph (defalut 'false')");
    app.add_flag("--keep_dup", opts.keep_duplicate_edges, "keep duplicate edges in graph (defalut 'false')");
}

}