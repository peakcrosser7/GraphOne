
#include <cstdio>
#include <iostream>
#include <limits>
#include <chrono>
#include <array>

#include "GraphOne/graph_one.h"

using namespace std;
using namespace graph_one;

constexpr arch_t arch = arch_t::cuda;
using prop_t = float;

struct pr_hstatus_t {
    prop_t alpha;
    DenseVec<arch, prop_t>& iweights;    // alpha * (1 / out_degrees)
    std::array<DenseVec<arch, prop_t>, 2>& ranks;   // pageranks
};

struct pr_dstatus_t {
    prop_t* iweights;
    prop_t* ranks[2];
    int opt_idx = 0;
    prop_t alpha_div_v = 0.;
};

template <typename graph_t>
struct PRComp : ComponentX<graph_t, pr_hstatus_t, pr_dstatus_t> {
    using comp_t = ComponentX<graph_t, pr_hstatus_t, pr_dstatus_t>;
    using comp_t::ComponentX;
    using vertex_t = typename comp_t::vertex_type;
    using edge_t = typename comp_t::edge_type;

    void Init() override {
        auto& iweights = this->h_status.iweights;
        auto graph_ref = this->graph.ToArch();
        prop_t alpha = this->h_status.alpha;

        archi::transform<arch>(thrust::make_counting_iterator<vertex_t>(0),
            thrust::make_counting_iterator<vertex_t>(graph_ref.num_vertices),
            iweights.begin(), 
            [graph_ref, alpha] __ONE_DEV__ (const vid_t& vid)  {
                prop_t val = 0;
                edge_t start = graph_ref.get_starting_out_edge(vid);
                edge_t end = start + graph_ref.get_out_degree(vid);
                for (edge_t offset = start; offset < end; offset++) {
                    val += graph_ref.get_out_edge_weight(offset);
                }

                return val != 0 ? alpha / val : 0;
            });

        this->d_status.alpha_div_v = (1.0 - alpha) / graph_ref.num_vertices;

        int& idx = this->d_status.opt_idx;
        auto& ranks = this->h_status.ranks;
        archi::fill<arch>(ranks[idx].begin(), ranks[idx].end(), 1.0 / graph_ref.num_vertices);
        archi::fill<arch>(ranks[idx ^ 1].begin(), ranks[idx ^ 1].end(), 0);
    }
    
    bool IsConvergent() override {
        int idx = this->d_status.opt_idx;
        prop_t* pre_rank = this->d_status.ranks[idx];
        prop_t* cur_rank = this->d_status.ranks[idx ^ 1];
        prop_t diff = archi::transform_reduce<arch>(
            thrust::counting_iterator<vertex_t>(0),
            thrust::counting_iterator<vertex_t>(this->graph.num_vertices()),
            [=] __ONE_DEV__ (const vertex_t& vid) {
                return std::abs(cur_rank[vid] - pre_rank[vid]);
            },
            prop_t(0.), thrust::maximum<prop_t>()
        );

        LOG_DEBUG("diff: ", diff);
        return diff <= 1e-6;
    }

    void AfterEngine() override {
        int idx = this->d_status.opt_idx;
        this->d_status.opt_idx ^= 1;
    }
};

struct PRFunctor : BlasFunctor<vid_t, prop_t, pr_dstatus_t, prop_t, prop_t> {

    __ONE_ARCH_INL__
    static prop_t default_info() {
        return prop_t(0);
    }

    __ONE_ARCH_INL__
    static prop_t default_result() {
        return prop_t(0);
    }

    __ONE_DEV_INL__
    static prop_t construct_each(const vid_t& vid, const pr_dstatus_t& d_status) {
        int idx = d_status.opt_idx;
        prop_t* ranks = d_status.ranks[idx];
        prop_t* iweights = d_status.iweights;
        return ranks[vid] * iweights[vid];
    }

    __ONE_DEV_INL__
    static prop_t gather_combine(const prop_t& weight, const prop_t& info) {
        return info * weight;
    }

    __ONE_DEV_INL__
    static prop_t gather_reduce(const prop_t& lhs, const prop_t& rhs) {
        return lhs + rhs;
    }

    __ONE_DEV_INL__
    static bool apply_each(const vid_t& vid, const prop_t& res, pr_dstatus_t& d_status) {
        int idx = d_status.opt_idx;
        prop_t* ranks = d_status.ranks[idx];
        prop_t alpha_div_v = d_status.alpha_div_v;
        ranks[vid] = alpha_div_v + res;
        return true;
    }

};


int main(int argc, char *argv[]) {
    CLI::App app;
    string input_graph;
    string output_path;
    bool reoreder_vid = false;
    LoadEdgeOpts opts;
    add_common_args(app, input_graph, output_path, reoreder_vid, opts);
    prop_t alpha = 0.85;
    app.add_option("--alpha", alpha, "alpha (factor) in PageRank");
    CLI11_PARSE(app, argc, argv);

    GraphLoader<vstart_t::FROM_0_TO_0> loader;
    auto cache = loader.LoadEdgesFromTxt<prop_t>(input_graph, opts);
    auto g = graph::build<arch_t::cuda, BlasViews | graph_view_t::normal>(cache);
    using graph_t = decltype(g);
    
    std::array<DenseVec<arch, prop_t>, 2> ranks{
        DenseVec<arch, prop_t>(g.num_vertices()),
        DenseVec<arch, prop_t>(g.num_vertices())
    };
    DenseVec<arch, prop_t> iweights(g.num_vertices());

    pr_hstatus_t h_status{alpha, iweights, ranks};
    pr_dstatus_t d_status{iweights.data(), {ranks[0].data(), ranks[1].data()}};
    AllActiveFrontier<arch, vid_t> frontier(g.num_vertices(), {});

    PRComp<graph_t> comp(g, h_status, d_status);

    auto start = std::chrono::high_resolution_clock::now();
    Run<PRFunctor>(comp, frontier);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printx("Elapsed time: ", duration.count(), " ms");
 
    if (!output_path.empty()) {
        FILE* fp;
        if ((fp = fopen(output_path.c_str(), "w")) == nullptr) {
            LOG_ERROR("open output file failed");
        }
        fprintf(fp, "Elapsed time: %lld ms\n", duration.count());
        auto h_ranks = ranks[d_status.opt_idx].to<arch_t::cpu>();
        for (vid_t i = 0; i < h_ranks.size(); ++i) {
            fprintf(fp, "%u-%f\n", i, h_ranks[i]);
        }
        fclose(fp);
    }

    return 0;
}