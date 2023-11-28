#include "test.hpp"

#include "GraphGenlX/loader/loader.h"
#include "GraphGenlX/utils/string.hpp"

using namespace std;
using namespace graph_genlx;

int main() {
    Loader<> loader(true, vstart_t::FROM_1_TO_1);
    auto edges = loader.LoadEdgesFromTxt<int>("../data/sample/sample.adj");
    
    cout << "edge_cache:" <<
        utils::VecToString(edges, [](const auto edge) {
        return utils::NumToString(edge.src) + "-" + 
            utils::NumToString(edge.dst) + "-" + 
            utils::ToString(edge.edata);
    }) << endl;

    cout << "edge.max_id: " << edges.max_vid() << endl;
    cout << "edge.num_vertices:" << edges.num_vertices() << endl;

    auto csr = edges.ToCsr<arch_t::cpu>();
    cout << csr.ToString() << endl;

    return 0;
}