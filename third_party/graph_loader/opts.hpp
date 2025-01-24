#pragma once

#include <string>
#include <functional>

#include "mm_typecode.hpp"

namespace graph_loader {

enum class BasedIndex {
    BASED_0_TO_0,
    BASED_1_TO_0
};

struct LoaderOpts {
    std::string file_ext = "";
    std::string comment_prefix = "";
    std::string line_sep = " \t";
    std::string line_strips = " \t\n\r";
    BasedIndex based_index = BasedIndex::BASED_0_TO_0;
    int header_cnt = 2;
    bool is_directed = false;
    bool do_reorder = false;

    std::function<void(std::string&, LoaderOpts&)> load_banner_func;


    bool directed() const {
        return is_directed;
    }

    bool undirected() const {
        return !is_directed;
    }

    LoaderOpts& set_is_directed(bool directed) {
        is_directed = directed;
        return *this;
    }

    LoaderOpts& set_do_reoder(bool reorder) {
        do_reorder = reorder;
        return *this;
    }

    LoaderOpts& set_comment_prefix(const std::string& comment) {
        comment_prefix = comment;
        return *this;
    }

    LoaderOpts& set_file_ext(const std::string& file_extension) {
        file_ext = file_extension;
        return *this;
    }  
};


} // namespace graph_loader
