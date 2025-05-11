#include <chrono>

#include "CLI11/CLI11.hpp"

#include "graph_one/graph_one.h"

using namespace graph_one;

using dist_t = float;

Tensor sssp(GraphX& g, vid_t src) {

    vid_t num_v = g.num_vertices();
    Device device = g.device();

    constexpr auto kMaxDist = std::numeric_limits<dist_t>::max();

    Tensor dists = make_full<dist_t>({num_v}, kMaxDist, device);
    dists[src] = 0;
    Tensor active_dists = dists.clone();

    Tensor dists_prev;

    auto functor = make_functor(op::SafeAdd{}, op::Min{});

    bool any_active = true;
    for (int iter = 1; any_active; ++iter) {
        Tensor result = GraphForward(functor, g, active_dists);

        Tensor mask = result < dists;
        dists = torch::where(mask, result, dists);
        active_dists = torch::where(mask, result, active_dists);

        any_active = torch::any(mask).item<bool>();
    }

    return dists;
}


int main(int argc, char *argv[]) {
    std::string input_graph;

    // SSSP Parameters
    vid_t src;

    CLI::App app;
    app.add_option("-i,--input_graph", input_graph, "input graph dataset file")->required();
    app.add_option("-s,--src", src, "source vertex id for SSSP")->required();
    CLI11_PARSE(app, argc, argv);

    GraphX g = load_graph(input_graph, kCUDA);

    auto start = std::chrono::high_resolution_clock::now();
    Tensor dists = sssp(g, src);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start) / 10;

    printx("Elapsed time: ", duration.count(), " ms");

    dists = dists.to(kCPU);
    printx("SSSP:");
    for (vid_t i = 0; i < dists.size(0); ++i) {
        printf("%d-%f\n", i, dists[i].item<dist_t>());
    }

    return 0;
}