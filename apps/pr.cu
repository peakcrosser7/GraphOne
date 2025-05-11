
#include <iostream>

#include "CLI11/CLI11.hpp"

#include "graph_one/graph_one.h"

using namespace graph_one;

Tensor pr(GraphX& g, float alpha, float eps) {

    vid_t num_v = g.num_vertices();
    Device device = g.device();

    Tensor p = make_full<float>({num_v}, 1.f / g.num_vertices(), device);

    Tensor p_prev;

    float error = 1.f;

    auto functor = make_functor(op::Mult{}, op::Add{});
    for (int iter = 1; error > eps && iter <= 100; ++iter) {
        p_prev = p;

        Tensor p_swap = GraphForward(g, p_prev, {}, functor);

        p = p_swap + (1.f - alpha) / num_v;

        Tensor r = p - p_prev;
        
        error = torch::sum(r * r).item<float>();
        error = std::sqrt(error);

        LOG_DEBUG("PageRank Iteration: ", iter, " Error: ", error);
    }

    return p;
}

int main(int argc, char *argv[]) {
    std::string input_graph;

    // PageRank Parameters
    float alpha = 0.85;
    float eps   = 1e-8;

    CLI::App app;
    app.add_option("-i,--input_graph", input_graph, "input graph dataset file")->required();
    app.add_option("--alpha", alpha, "alpha (factor) in PageRank (default 0.85)");
    app.add_option("--eps", eps, "epsilon in PageRank (default 1e-8)");
    CLI11_PARSE(app, argc, argv);

    GraphX g = load_graph(input_graph, kCUDA);

    Tensor new_inedge_weights = make_full<float>({g.num_edges()}, 1.f, g.device());
    Tensor out_degrees = GraphReduce(op::Add{}, g, new_inedge_weights,
                                     ReduceOpts().use_out_edges());
    out_degrees = alpha / out_degrees;
    new_inedge_weights = GraphWise(op::Mult{}, g, new_inedge_weights, out_degrees, 
                                   ElementWiseOpts().use_in_edges().use_src_vertex());
    g.set_inedge_weights(new_inedge_weights);

    auto start = std::chrono::high_resolution_clock::now();
    Tensor ranks = pr(g, alpha, eps);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start) / 10;

    printx("Elapsed time: ", duration.count(), " ms");

    ranks = ranks.to(kCPU);
    printx("PageRank:");
    for (vid_t i = 0; i < ranks.size(0); ++i) {
        printf("%d-%f\n", i, ranks[i].item<float>());
    }

    return 0;
}