#pragma once

#include "opts.hpp"


namespace graph_loader::OptsFactory {
inline LoaderOpts MatrixMarket(LoaderOpts opts = LoaderOpts()) {
    opts.comment_prefix = "%";
    opts.file_ext = ".mtx";
    opts.based_index = BasedIndex::BASED_1_TO_0;
    opts.header_cnt = 3;

    opts.load_banner_func = [&](std::string& line, LoaderOpts& opts_) {
        MMTypecode code = MMTypecode::FromString(line);    
        throw_if_exception(code.is_dense(), "File is not a sparse matrix");

        opts_.is_directed = !code.is_symmetric();
    };

    return opts;
}

inline LoaderOpts Snap(LoaderOpts opts = LoaderOpts()) {
    opts.file_ext = ".txt";
    opts.comment_prefix = "#";
    opts.header_cnt = 0;
    opts.do_reorder = true;

    return opts;
}

inline LoaderOpts WithHeader(LoaderOpts opts = LoaderOpts()) {
    opts.header_cnt = 2;

    return opts;
}

inline LoaderOpts WithoutHeader(LoaderOpts opts = LoaderOpts()) {
    opts.header_cnt = 0;

    return opts;
}

    
} // namespace graph_loader
