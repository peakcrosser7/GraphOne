
#include <iostream>
#include <limits>
#include <chrono>

#include "GraphGenlX/graph_genlx.h"

using namespace std;
using namespace graph_genlx;

constexpr arch_t arch = arch_t::cuda;
using dist_t = float;

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

template <typename graph_t,
        typename hstatus_t,
        typename dstatus_t,
        typename frontier_t>
struct SSSPComp : ComponentX<graph_t, hstatus_t, dstatus_t, frontier_t> {
    using comp_t = ComponentX<graph_t, hstatus_t, dstatus_t, frontier_t>;

    void Init() {
        auto& src_vid = this->h_status.src_vid;
        auto& dists = this->h_status.dists;

        archi::fill<arch>(dists.begin(), dists.end(), std::numeric_limits<dist_t>::max());
        dists.set(src_vid, 0);
        this->frontier.Init(this->graph.num_vertices(), src_vid);
    }

    void BeforeEngine() {
        comp_t::BeforeEngine();
        ++this->d_status.iter;
    }
};

struct SSSPFactor : AdvanceFactor<vid_t, eid_t, dist_t, sssp_dstatus_t> {

    using engine_type = AdvanceFactor<vid_t, eid_t, dist_t, sssp_dstatus_t>::engine_type;

    __GENLX_DEV_INL__
    static bool advance(const vid_t &src, const vid_t &dst, const eid_t &edge,
                        const dist_t &weight, sssp_dstatus_t &d_status) {
        auto* const dists = d_status.dists;
        dist_t src_dist = dists[src];
        dist_t dist_to_dst = src_dist + weight;
        // Check if the destination node has been claimed as someone's child
        dist_t recover_dist = archi::cuda::atomicMin(&dists[dst], dist_to_dst);

        archi::cuda::print("tid=%u src=%u, dst=%u, e=%u, w=%d dist_to_dst=%f recover_dist=%f\n", 
            threadIdx.x, src, dst, edge, weight, dist_to_dst, recover_dist);
        // 距离更小则更新输出前沿
        return dist_to_dst < recover_dist;
    }

    __GENLX_DEV_INL__
    static bool filter(const vid_t& vid, const sssp_dstatus_t& d_status) {
        auto& visited = d_status.visited;
        auto& iter = d_status.iter;
        if (visited[vid] == iter) {
            return false;
        }
        visited[vid] = iter;
        return true;
    }
};

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printx("Usage: ", argv[0], " <graph_file> <src_vertex>");
        return -1;
    }
    
    vid_t src = std::stoi(argv[2]);

    Loader<vstart_t::FROM_1_TO_1, false> loader;
    LoadEdgeOpts opts;
    opts.comment_prefix = "%";
    // opts.is_directed = true;
    auto csr = loader.LoadCsrFromTxt<arch, int>(argv[1], opts);
    auto g = graph::FromCsr<>(std::move(csr));

    if (!loader.ReorderedVid(src)) {
        LOG_ERROR("src vertex \"", src, "\" is not exist");
    }
    
    DenseVec<arch, dist_t> dists(g.num_vertices());
    DenseVec<arch, vid_t> parents(g.num_vertices());

    sssp_hstatus_t h_status{src, dists, parents};
    sssp_dstatus_t d_status{0, dists.data(), parents.data()};
    ActiveFrontier<arch, vid_t> frontier;

    auto comp = compx::build<SSSPComp>(g, h_status, d_status, frontier);

    auto start = std::chrono::high_resolution_clock::now();
    Run<SSSPFactor>(comp);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printx("Elapsed time: ", duration.count(), "ms");
 
    auto h_dists = dists.to<arch_t::cpu>();
    for (int i = 0; i < min(500, h_dists.size()); ++i) {
        printx(i, "-", h_dists[i]);
    }
    return 0;
}