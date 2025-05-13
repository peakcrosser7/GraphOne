#include <chrono>

#include "CLI11/CLI11.hpp"

#include "graph_one/graph_one.h"

using namespace graph_one;

int64_t tc(GraphX& g) {
    TORCH_CHECK(g.is_undirected(), "Triangle counting only works on undirected graphs");

    GraphX g_tril = Tril(g);
    g_tril.set_outedge_weights(make_ones<float>({g_tril.num_edges()}, g_tril.device()), true);

    auto functor = make_functor(op::Mult{}, op::Add{}, MaskApplier(g_tril.adj()));
    Tensor output = GraphForward(functor, g_tril, g_tril.adj_trans(), {}, ForwardOpts().use_outedge_weights());

    Tensor triangles = ReduceCSR(op::Add{}, output);
    int64_t num_triangles = torch::sum(triangles).item<int64_t>();
    return num_triangles;
}

int main(int argc, char *argv[]) {
    std::string input_graph;

    CLI::App app;
    app.add_option("-i,--input_graph", input_graph, "input graph dataset file")->required();
    CLI11_PARSE(app, argc, argv);

    GraphX g = load_graph(input_graph, kCUDA);

    auto start = std::chrono::high_resolution_clock::now();
    int64_t num_triangles = tc(g);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printx("Elapsed time: ", duration.count(), " ms");
    printx("TC: ", num_triangles);

    return 0;
}