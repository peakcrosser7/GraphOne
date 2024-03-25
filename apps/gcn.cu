
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

    constexpr arch_t arch = arch_t::cuda;

    gnn::vprop_t<arch> vprops;   



    

    return 0;
}