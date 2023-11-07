#include <iostream>

#include "GraphGenlX/DataLoader.hpp"

using namespace std;
using namespace graph_genlx;

int main() {
    auto coo = DataLoader::LoadFromTxt<empty_t>("../data/bfs_test/bfs_test.adj", ".adj");
    cout << coo.ToString() << endl;

    return 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
}