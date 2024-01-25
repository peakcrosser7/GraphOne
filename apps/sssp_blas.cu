
#include <iostream>
#include <limits>
#include <chrono>

#include "GraphGenlX/graph_genlx.h"

using namespace std;
using namespace graph_genlx;

constexpr arch_t arch = arch_t::cuda;
using dist_t = float;

constexpr dist_t kMaxDist = std::numeric_limits<dist_t>::max();

struct sssp_hstatus_t {
    vid_t src_vid;

    DenseVec<arch, dist_t>& dists;
    // DenseVec<arch, uint32_t>& visited;
};

struct sssp_dstatus_t {
    uint32_t iter;
    dist_t * dists;
    // uint32_t* visited;
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

struct SSSPFunctor : BlasFunctor<vid_t, dist_t, sssp_dstatus_t, dist_t, dist_t> {
    static dist_t default_info() {
        return kMaxDist;
    }

    __GENLX_ARCH_INL__
    static dist_t default_result() {
        return kMaxDist;
    }

    __GENLX_DEV_INL__
    static dist_t construct(const vid_t& vid, const sssp_dstatus_t& d_status) {
        return d_status.dists[vid];
    }

    __GENLX_DEV_INL__
    static dist_t gather(const dist_t& weight, const dist_t& info) {
        return (info == kMaxDist) ? info : weight + info;
    }

    __GENLX_DEV_INL__
    static dist_t reduce(const dist_t& lhs, const dist_t& rhs) {
        return std::min(lhs, rhs);
    }

    __GENLX_DEV_INL__
    static bool apply(const vid_t& vid, const dist_t& res, sssp_dstatus_t& d_status) {
        if (res < d_status.dists[vid]) {
            d_status.dists[vid] = res;
            return true;
        }
        return false;
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
    auto cache = loader.LoadEdgesFromTxt<dist_t>(argv[1], opts);
    auto g = graph::build<arch_t::cuda, BlasViews>(cache);
    using graph_t = decltype(g);

    if (!loader.ReorderedVid(src)) {
        LOG_ERROR("src vertex \"", src, "\" is not exist");
    }
    
    DenseVec<arch, dist_t> dists(g.num_vertices());
    DenseVec<arch, vid_t> visited(g.num_vertices());

    sssp_hstatus_t h_status{src, dists/*, visited*/};
    sssp_dstatus_t d_status{0, dists.data()/*, visited.data()*/};
    DblBufFrontier<arch, vid_t> frontier(g.num_vertices(), src);

    SSSPComp<graph_t> comp(g, h_status, d_status);

    auto start = std::chrono::high_resolution_clock::now();
    Run<SSSPFunctor>(comp, frontier);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printx("Elapsed time: ", duration.count(), "ms");
 
    auto h_dists = dists.to<arch_t::cpu>();
    for (int i = 0; i < min(500, h_dists.size()); ++i) {
        printx(i, "-", h_dists[i]);
    }
    return 0;
}