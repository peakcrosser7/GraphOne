#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

#include "GraphGenlX/utils.h"
#include "GraphGenlX/mat/coo.h"

namespace graph_genlx {

class DataLoader {
public:
    template<typename value_t = double>
    static CooMat<arch_t::cpu, value_t, vid_t, eid_t> LoadFromTxt(const std::string& filepath,
        const std::string& file_ext = ".txt",
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


        std::vector<vid_t> srcs;
        std::vector<vid_t> dsts;
        std::vector<value_t> weights;
        
        constexpr int MAX_CNT = std::is_same_v<empty_t, value_t> ? 2 : 3;
        
        vid_t max_id = 0;
        vid_t src, dst;
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

            src = vid_t(std::stoi(words[0]));
            dst = vid_t(std::stoi(words[1]));
            max_id = std::max<vid_t>(max_id, std::max<vid_t>(src, dst));

            srcs.push_back(src);
            dsts.push_back(dst);
            weights.push_back(val);
        }

        CooMat<arch_t::cpu, value_t, vid_t, eid_t> coo(max_id + 1, max_id + 1, 
            std::move(srcs), std::move(dsts), std::move(weights));
        return coo;
    }

private:

};

    
} // namespace graph_genlx 