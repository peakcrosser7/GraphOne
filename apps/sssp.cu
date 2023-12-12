
#include <iostream>
#include <limits>

#include "GraphGenlX/graph_genlx.h"

using namespace std;
using namespace graph_genlx;

constexpr arch_t arch = arch_t::cuda;
using dist_t = uint32_t;

struct Status {
    vid_t src_vid;
    DenseVec<arch, dist_t>& dists;
    DenseVec<arch, vid_t>& parents;
};

template <typename graph_t,
        typename status_t,
        typename frontier_t>
struct SSSPComp : ComponentX<graph_t, status_t, frontier_t> {
    void Init() {
        auto& src_vid = this->status.src_vid;
        auto& dists = this->status.dists;

        archi::fill<arch>(dists.begin(), dists.end(), std::numeric_limits<dist_t>::max());
        dists.set(src_vid, 0);
        this->frontier.Init(src_vid);
    }
};

struct SSSPFactor : AdvanceFactor<vid_t, eid_t, dist_t> {
    using engine_type = AdvanceFactor<vid_t, eid_t, dist_t>::engine_type;

    SSSPFactor(dist_t* dists) : dists(dists) {}

    __GENLX_ARCH_INL__ bool advance(const vid_t &src, const vid_t &dst, const eid_t &edge,
                 const dist_t &weight) {
        dist_t src_dist = dists[src];
        dist_t dist_to_dist = src_dist + weight;

        // Check if the destination node has been claimed as someone's child
        // todo:min should atomic
        dist_t recover_dist =
            (dists[dst] < dist_to_dist ? dists[dst] : dist_to_dist);

        // 距离更小则更新输出前沿
        return (dist_to_dist < recover_dist);
    }

    dist_t* dists;
};

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <graph_file> <src_vertex>\n";
        return -1;
    }
    vid_t src = std::stoi(argv[2]);

    auto csr = Loader<>().LoadCsrFromTxt<arch, int>("../data/sample/sample.adj");
    auto g = graph::FromCsr(std::move(csr));
    
    DenseVec<arch, dist_t> dists(g.num_vertices());
    DenseVec<arch, vid_t> parents(g.num_vertices());

    Status status{src, dists, parents};
    ActiveFrontier<arch, vid_t> frontier;

    auto comp = compx::build<SSSPComp>(g, status, frontier);
    SSSPFactor factor(dists.data());

    Run(comp, factor);
    
    return 0;
}