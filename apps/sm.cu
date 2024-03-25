
#include <iostream>

#include "GraphOne/graph_one.h"

using namespace std;
using namespace graph_one;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <graph_file> <src_vertex>\n";
        return -1;
    }
    unsigned src = std::stoi(argv[2]);

    auto csr = mat::LoadCsrFromTxt<arch_t::cuda, int>(argv[1]);
    auto g = graph::FromCsr(std::move(csr));
    



    

    return 0;
}