#include "GraphGenlX/DataLoader.h"
#include "GraphGenlX/mat/builder.h"

#include "test.hpp"

using namespace std;
using namespace graph_genlx;

int main() {
    auto coo = DataLoader::LoadFromTxt<int>("../data/bfs_test/bfs_test.adj", ".adj");
    cout << coo.ToString() << endl;

    auto csr = MatBuilder::ToCsr<arch_t::cpu>(coo);
    cout << csr.ToString() << endl;

    return 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
}