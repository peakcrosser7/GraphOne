
#include <cstdio>
#include <iostream>
#include <limits>
#include <chrono>

#include "GraphOne/graph_one.h"

using namespace std;
using namespace graph_one;

constexpr arch_t arch = arch_t::cuda;
using dist_t = float;

constexpr dist_t kMaxDist = std::numeric_limits<dist_t>::max();

struct sssp_hstatus_t {
    vid_t src_vid;

    DenseVec<arch, dist_t>& dists;
    DenseVec<arch, uint32_t>& visited;
};

struct sssp_dstatus_t {
    uint32_t iter;
    dist_t * dists;
    uint32_t* visited;
};

template <typename graph_t>
struct SSSPComp : ComponentX<graph_t, sssp_hstatus_t, sssp_dstatus_t> {
    using comp_t = ComponentX<graph_t, sssp_hstatus_t, sssp_dstatus_t>;
    using comp_t::ComponentX;

    void Init() override {
        auto& src_vid = this->h_status.src_vid;
        auto& dists = this->h_status.dists;

        archi::fill<arch>(dists.begin(), dists.end(), kMaxDist);
        dists.set(src_vid, 0);
    }

    void BeforeEngine() override {
        ++this->d_status.iter;
    }
};

struct SSSPFunctor : AdvanceFunctor<vid_t, eid_t, dist_t, sssp_dstatus_t> {
    __ONE_DEV_INL__
    static bool advance(const vid_t &src, const vid_t &dst, const eid_t &edge,
                        const dist_t &weight, sssp_dstatus_t &d_status) {
        auto* const dists = d_status.dists;
        dist_t src_dist = dists[src];
        dist_t dist_to_dst = src_dist + weight;
        // Check if the destination node has been claimed as someone's child
        dist_t recover_dist = archi::cuda::AtomicMin(&dists[dst], dist_to_dst);

        archi::cuda::print("tid=%u src=%u, dst=%u, e=%u, w=%d dist_to_dst=%f recover_dist=%f\n", 
            threadIdx.x, src, dst, edge, weight, dist_to_dst, recover_dist);
        // 距离更小则更新输出前沿
        return dist_to_dst < recover_dist;
    }

    __ONE_DEV_INL__
    static bool filter(const vid_t& vid, const sssp_dstatus_t& d_status) {
        auto& visited = d_status.visited;
        auto& iter = d_status.iter;
        if (visited[vid] == iter) {
            return true;
        }
        visited[vid] = iter;
        return false;
    }
};


int main(int argc, char *argv[]) {
    CLI::App app;
    string input_graph;
    string output_path;
    bool reoreder_vid = false;
    LoadEdgeOpts opts;
    add_common_args(app, input_graph, output_path, reoreder_vid, opts);
    vid_t src;
    app.add_option("--src", src, "source vertex in SSSP")->required();
    CLI11_PARSE(app, argc, argv);

    Loader<vstart_t::FROM_1_TO_1> loader(reoreder_vid);
    auto cache = loader.LoadEdgesFromTxt<dist_t>(input_graph, opts);
    auto g = graph::build<arch_t::cuda, AdvanceViews>(cache);
    using graph_t = decltype(g);

    if (!loader.ReorderedVid(src)) {
        LOG_ERROR("src vertex \"", src, "\" is not exist");
    }
    
    DenseVec<arch, dist_t> dists(g.num_vertices());
    DenseVec<arch, vid_t> visited(g.num_vertices());

    sssp_hstatus_t h_status{src, dists, visited};
    sssp_dstatus_t d_status{0, dists.data(), visited.data()};
    SpDblFrontier<arch, vid_t> frontier(g.num_vertices(), {src});

    SSSPComp<graph_t> comp(g, h_status, d_status);

    auto start = std::chrono::high_resolution_clock::now();
    Run<SSSPFunctor>(comp, frontier);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printx("Elapsed time: ", duration.count(), "ms");
    
    if (!output_path.empty()) {
        FILE* fp;
        if ((fp = fopen(output_path.c_str(), "w")) == nullptr) {
            LOG_ERROR("open output file failed");
        }
        fprintf(fp, "Elapsed time: %lld ms\n", duration.count());
        auto h_dists = dists.to<arch_t::cpu>();
        for (vid_t i = 0; i < h_dists.size(); ++i) {
            fprintf(fp, "%u-%f\n", i, h_dists[i]);
        }
        fclose(fp);
    }

    return 0;
}