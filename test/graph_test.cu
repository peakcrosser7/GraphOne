#include "GraphGenlX/graph/builder.h"

#include "test.hpp"

using namespace std;
using namespace graph_genlx;

int main () {
    // install_oneshot_signal_handlers();
    
    auto coo = mat::LoadCooFromTxt<int>("../data/bfs_test/bfs_test.adj");
    auto csr = mat::ToCsr<arch_t::cuda>(coo);

    auto g = graph::FromCsr<graph_view_t::csr>(csr);
    cout << g.ToString() << endl;

    return 0;
}
