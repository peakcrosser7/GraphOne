#include "GraphGenlX/mat/builder.h"

#include "test.hpp"

using namespace std;
using namespace graph_genlx;

int main() {
    auto coo = mat::LoadCooFromTxt<int>("../data/bfs_test/bfs_test.adj");
    cout << coo.ToString() << endl;

    auto csr = mat::ToCsr<arch_t::cpu>(coo);
    cout << csr.ToString() << endl;

    auto csc = mat::ToCsc(csr);
    cout << csc.ToString() << endl;

    return 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
}